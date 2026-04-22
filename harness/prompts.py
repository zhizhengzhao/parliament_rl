"""Prompt and persona machinery — Parliament side.

All "what the LLM sees" lives here:
  - Loads `context_configs/<PRL_CONTEXT>_context/` (default: Parliament)
  - Per-session persona sampling (no replacement) from a flat pool
  - System prompt rendering
  - Formatting new posts/comments/votes for context injection

Switch the 2×2 ablation cell with:
    PRL_CONTEXT=Parliament   # coupled (default)
    PRL_CONTEXT=Solo         # independent
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONTEXT_NAME = os.environ.get("PRL_CONTEXT", "Parliament")
PARLIAMENT_CONTEXT = PROJECT_DIR / "context_configs" / f"{CONTEXT_NAME}_context"

_MAX_AGENTS_PER_ROLE = 32  # per-role cap on `rng.sample` size


# ── Config loading ────────────────────────────────────────

def load_context_config() -> dict:
    config = json.loads((PARLIAMENT_CONTEXT / "config.json").read_text())
    config["_dir"] = str(PARLIAMENT_CONTEXT)
    config["actor_prompt"] = (PARLIAMENT_CONTEXT / "actor_prompt.txt").read_text()
    config["judge_prompt"] = (PARLIAMENT_CONTEXT / "judge_prompt.txt").read_text()
    return config


_CFG: dict | None = None


def get_config() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = load_context_config()
    return _CFG


# ── Persona sampling ─────────────────────────────────────

def _slot_index(name: str) -> int:
    """Extract 1-indexed slot from 'Scientist_2' → 1, 'Judge_3' → 2."""
    try:
        return int(name.rsplit("_", 1)[-1]) - 1
    except (ValueError, IndexError):
        return -1


def pick_persona(name: str, role: str, session_id: str) -> str:
    """Sample a persona for (name, role) seeded by session_id.

    Each session draws N distinct personas without replacement (rng.sample)
    seeded by `session_id:role`. Deterministic per (session_id, role, slot)
    — same agent in same session always sees the same persona, but
    different sessions see different draws and different agents in one
    session see different personas.
    """
    pools = get_config().get("persona_pools", {})
    role_key = "judge" if role == "judge" else "scientist"
    pool = pools.get(role_key)
    if not pool:
        return ""

    slot = _slot_index(name)
    if slot < 0:
        return ""

    rng = random.Random(f"{session_id}:{role_key}")
    k = min(len(pool), _MAX_AGENTS_PER_ROLE)
    sampled = rng.sample(pool, k)
    return sampled[slot] if slot < k else ""


# ── Actor & judge name sampling ──────────────────────────

def assign_session_names(session_id: str) -> dict[str, str]:
    """Map Scientist_1..n and Judge_1..n to deterministic per-session names.

    A single `rng.sample(pool, 2N)` draw partitions into:
        Scientist_1..N  ← sampled[0 : N]
        Judge_1..N      ← sampled[N : 2N]
    where N = `_MAX_AGENTS_PER_ROLE` (32). The fixed layout means any
    given (session_id, slot) always maps to the same name regardless of
    how many actors/judges the session actually has.

    The single-sample guarantees no two agents — actor or judge — share
    a name within one session. Judges are named because
    `"You are Elena, a judge on Science Parliament"` reads cleaner than
    `"You are Judge_1, a judge …"`. Judge names never leak to actors
    (judge votes are anonymized as "Anonymous Scientist").
    """
    pool = get_config().get("name_pool", [])
    if not pool:
        return {}
    rng = random.Random(f"name:{session_id}")
    total = min(2 * _MAX_AGENTS_PER_ROLE, len(pool))
    sampled = rng.sample(pool, total)
    mapping: dict[str, str] = {}
    for i in range(min(_MAX_AGENTS_PER_ROLE, total)):
        mapping[f"Scientist_{i + 1}"] = sampled[i]
    for j in range(max(0, total - _MAX_AGENTS_PER_ROLE)):
        mapping[f"Judge_{j + 1}"] = sampled[_MAX_AGENTS_PER_ROLE + j]
    return mapping


def apply_name_map(items: list[dict], name_map: dict[str, str]) -> None:
    """Rewrite `author` fields in-place so every actor shows their session name."""
    if not name_map:
        return
    for item in items:
        author = item.get("author")
        if author in name_map:
            item["author"] = name_map[author]


# ── System prompt ────────────────────────────────────────

def build_system_prompt(name: str, role: str, session_title: str,
                        reference_solution: str = "",
                        session_id: str = "") -> str:
    """Build the system prompt — both actors and judges see anonymized names."""
    cfg = get_config()
    persona = pick_persona(name, role, session_id)
    display_name = assign_session_names(session_id).get(name, name)
    if role == "judge":
        return cfg["judge_prompt"].format(
            name=display_name, session_title=session_title,
            reference_solution=reference_solution, persona=persona)
    return cfg["actor_prompt"].format(
        name=display_name, session_title=session_title, persona=persona)


# ── Content formatting (for runtime context injection) ───

_TYPE_ORDER = {"post": 0, "comment": 1, "vote": 2}


def _fmt_score(s: int) -> str:
    return f"+{s}" if s > 0 else str(s)


def _fmt_post(item: dict) -> str:
    return f'[P_{item["id"]}] by {item["author"]}:\n{item["content"]}'


def _fmt_comment(item: dict) -> str:
    return (f'[C_{item["id"]}] on P_{item["post_id"]} '
            f'by {item["author"]}:\n{item["content"]}')


def _fmt_vote(item: dict) -> str:
    tt = "P" if item["target_type"] == "post" else "C"
    target = f'{tt}_{item["target_id"]}'
    val = _fmt_score(item["value"])
    score = _fmt_score(item.get("target_score", 0))
    prev = item.get("previous_value")
    change = (f"changed {_fmt_score(prev)} → {val}"
              if prev is not None else f"{val} vote")
    return f'[V on {target}] by {item["author"]}: {change}, current score of {target}: {score}'


_FMT = {"post": _fmt_post, "comment": _fmt_comment, "vote": _fmt_vote}


def format_new_content(items: list[dict]) -> str:
    """Render new posts/comments/votes for injection into a user message.

    Sort posts → comments → votes (then by id within type) so the LLM
    sees the full picture (posts) before reactions (comments) and finally
    the signals (votes).
    """
    if not items:
        return ""
    items.sort(key=lambda x: (_TYPE_ORDER.get(x["type"], 9), x["id"]))
    return "\n\n".join(_FMT[item["type"]](item) for item in items)
