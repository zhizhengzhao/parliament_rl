#!/usr/bin/env python3
"""Extract RL training data from parliament.db.

One training sample = one actor's complete multi-turn trajectory in
one session.  The chat alternates between user (problem + intervening
discussion) and assistant (this actor's posts), so each sample contains
several reward-bearing assistant turns:

  user      problem + initial context
  assistant actor's 1st post
  user      what other agents posted/commented/voted on since
  assistant actor's 2nd post
  …

Per-turn rewards / advantages / trainability flags live in
`turn_rewards` / `turn_advantages` / `turn_trainable` arrays alongside
the messages; `rl/train.py` broadcasts each advantage to the matching
assistant-turn tokens via segment masks and zeroes the response mask
on non-trainable turns.

  reward     = sum of judge votes on the actor's post (judges only)
  position   debias linearly removed before normalization (later posts
             tend to score higher; we don't want the model to learn
             "post late = good")
  advantage  = (debiased_reward - baseline) / scale, both configurable
             (default mean_session / session_std → classic GRPO)

Cell-aware rendering: extract reads `experiment.json` next to the DB
to determine `actor_context_coupled` and `judge_votes_visible`, then
*for each actor* builds an isolated view of what they actually saw at
rollout time:

  | cell                       | what enters this actor's timeline                               |
  |---------------------------|-----------------------------------------------------------------|
  | A — Parliament             | every post + every comment + actor votes + judge votes (anon)   |
  | B — BlindParliament        | every post + every comment + actor votes (no judge votes)       |
  | C — Solo + judge visible   | only this actor's own posts + judge votes targeting those posts |
  | D — BlindSolo              | only this actor's own posts                                     |

Solo-cell isolation goes beyond filtering: each solo actor's posts are
re-numbered ``P_1, P_2, …`` from their own perspective so the local
post IDs never reveal that peers exist (no gaps in the sequence).
Judge votes in solo cells only carry references to posts this actor
authored — votes targeting peer posts are dropped at the scope-build
step.  This mirrors the per-actor IdMap topology that ``runner.py``
uses at rollout time.  Score-related events (vote lines, score
numbers) appear *only* when this actor's own content references
voting; vote-language-free actors stay completely score-free.

Template augmentation: every wrapping string (section header, post
header, vote event line, anonymous voter label, etc.) is sampled from
a multi-variant pool (`TEMPLATE_POOL`) seeded by `session.title`, so
the model sees diverse wrappings of the same semantic content and
learns to read information fields rather than memorise wrapper
strings.  4 cells share the same seed → no wrapper-choice confounding.
`--no-template-augment` disables this for ablation.

Identity anonymization: actor / judge names are mapped from the DB's
`Scientist_N` / `Judge_N` slots to per-session draws from a 406-name
pool, seeded by `session.title` so the same problem always shows the
same cast across all 4 cells.

Usage:
    python -m rl.extract --db data/run/parliament.db --output data/run/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path
from statistics import pstdev

PROJECT_DIR = Path(__file__).resolve().parent.parent
RL_CONTEXT = PROJECT_DIR / "context_configs" / "RL_context"
PARLIAMENT_CONTEXT = PROJECT_DIR / "context_configs" / "Parliament_context"

# Make `harness.prompts` importable when extract.py is run as `python -m rl.extract`
# from the project root. Parliament's session name map is the single source
# of truth; we import it directly so every Scientist_N mapping here is
# bit-identical to what the harness used at generation time.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from harness.prompts import assign_session_names  # noqa: E402

TYPE_ORDER = {"post": 0, "comment": 1, "vote": 2}

# Detects whether a post's content actually meta-references Parliament scoring.
SCORE_META_RE = re.compile(
    r"(?i)(score\s*of|scored|\bvote[sd]?\b|high[\-]?scoring|"
    r"negative[\-]?scoring|anonymous\s+(?:vote|scientist)|consensus)"
)
SCIENTIST_RE = re.compile(r"\bScientist_(\d+)\b")


# ── Config loading ───────────────────────────────────────

def load_rl_config() -> dict:
    cfg = json.loads((RL_CONTEXT / "config.json").read_text())
    # `name_pool` is the single source of truth for session casting and
    # is identical across all 2×2 cells (Parliament_context and
    # Solo_context share it), so we always read Parliament_context here
    # regardless of which cell produced the data.
    parl_cfg = json.loads((PARLIAMENT_CONTEXT / "config.json").read_text())
    cfg["name_pool"] = parl_cfg.get("name_pool", [])
    return cfg


_CFG: dict | None = None


def cfg() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = load_rl_config()
    return _CFG


# ── DB access ─────────────────────────────────────────────

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_session_ids(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT s.session_id FROM sessions s "
        "JOIN posts p ON s.session_id = p.session_id "
        "ORDER BY s.session_id").fetchall()]


def load_session_data(conn: sqlite3.Connection, session_id: str
                      ) -> tuple[dict, list[dict], list[dict], list[dict]]:
    session = dict(conn.execute(
        "SELECT session_id, title FROM sessions WHERE session_id = ?",
        (session_id,)).fetchone())

    posts = [dict(r) for r in conn.execute(
        "SELECT p.post_id, p.content, p.created_at, "
        "u.name AS author, u.role AS author_role "
        "FROM posts p JOIN users u ON p.user_id = u.user_id "
        "WHERE p.session_id = ? ORDER BY p.post_id",
        (session_id,)).fetchall()]

    comments = [dict(r) for r in conn.execute(
        "SELECT c.comment_id, c.post_id, c.content, c.created_at, "
        "u.name AS author, u.role AS author_role "
        "FROM comments c JOIN users u ON c.user_id = u.user_id "
        "JOIN posts p ON c.post_id = p.post_id "
        "WHERE p.session_id = ? ORDER BY c.comment_id",
        (session_id,)).fetchall()]

    votes = [dict(r) for r in conn.execute(
        "SELECT v.vote_id, v.post_id, v.comment_id, v.value, v.created_at, "
        "v.previous_value, u.name AS author, u.role AS author_role "
        "FROM votes v JOIN users u ON v.user_id = u.user_id "
        "LEFT JOIN posts p ON v.post_id = p.post_id "
        "LEFT JOIN comments c ON v.comment_id = c.comment_id "
        "LEFT JOIN posts p2 ON c.post_id = p2.post_id "
        "WHERE COALESCE(p.session_id, p2.session_id) = ? "
        "ORDER BY v.vote_id",
        (session_id,)).fetchall()]

    return session, posts, comments, votes


# ── Local IDs & timeline ─────────────────────────────────

def assign_local_ids_view(posts: list[dict], comments: list[dict]
                          ) -> tuple[list[dict], list[dict]]:
    """Return *deep-copied* posts/comments with per-view local IDs.

    Each actor's view of the session is built independently: in coupled
    cells ``posts`` is the global post list, in solo cells it is just
    this actor's own posts.  We deep-copy so different actors processed
    in the same call don't trample each other's ``local_id`` fields.

    Order follows DB-id (= creation chronology).  Comments additionally
    receive ``local_post_id`` resolved against this view's posts.
    """
    v_posts = [{**p, "local_id": i + 1} for i, p in enumerate(posts)]
    v_comments = [{**c, "local_id": i + 1} for i, c in enumerate(comments)]
    post_id_to_local = {p["post_id"]: p["local_id"] for p in v_posts}
    for c in v_comments:
        c["local_post_id"] = post_id_to_local.get(c["post_id"], 0)
    return v_posts, v_comments


def build_timeline(posts: list[dict], comments: list[dict],
                   votes: list[dict],
                   include_comments: bool = True,
                   include_vote_events: bool = False) -> list[dict]:
    """Merge posts, comments, and (optionally) vote events into a
    chronological timeline.

    Cell flags drive what enters:
      - ``include_comments=False`` for Solo cells (no comment concept).
      - ``include_vote_events=True`` only when the session's actor-side
        content references voting; otherwise vote events are absent from
        the rendered timeline (the trainer never sees a stray score).

    Vote entries carry ``target_type`` ∈ {"post", "comment"} and
    ``target_local_id`` so the renderer can produce the same
    ``[V on P_3]`` format the LLM saw at rollout time without
    threading id maps through call signatures.
    """
    post_id_to_local = {p["post_id"]: p["local_id"] for p in posts}
    comment_id_to_local = {c["comment_id"]: c["local_id"] for c in comments}

    timeline: list[dict] = []
    for p in posts:
        timeline.append({
            "type": "post",
            "local_id": p["local_id"],
            "post_id": p["post_id"],
            "author": p["author"],
            "author_role": p["author_role"],
            "content": p["content"],
            "created_at": p["created_at"],
        })
    if include_comments:
        for c in comments:
            timeline.append({
                "type": "comment",
                "local_id": c["local_id"],
                "comment_id": c["comment_id"],
                "local_post_id": c["local_post_id"],
                "author": c["author"],
                "author_role": c["author_role"],
                "content": c["content"],
                "created_at": c["created_at"],
            })
    if include_vote_events:
        for v in votes:
            if v.get("post_id"):
                target_type, target_global = "post", v["post_id"]
                target_local = post_id_to_local.get(target_global)
            elif v.get("comment_id"):
                target_type, target_global = "comment", v["comment_id"]
                target_local = comment_id_to_local.get(target_global)
            else:
                continue
            if target_local is None:
                continue                        # vote on an entity outside this session
            timeline.append({
                "type": "vote",
                "vote_id": v["vote_id"],
                "target_type": target_type,
                "target_global_id": target_global,
                "target_local_id": target_local,
                "value": v["value"],
                "previous_value": v.get("previous_value"),
                "author": v["author"],
                "author_role": v["author_role"],
                "created_at": v["created_at"],
            })
    timeline.sort(key=lambda x: (
        x["created_at"], TYPE_ORDER.get(x["type"], 9),
        x.get("local_id", x.get("vote_id", 0))))
    return timeline


# ── Scoring ──────────────────────────────────────────────

def fmt_score(s: int) -> str:
    return f"+{s}" if s > 0 else str(s)


def _fold_vote(post_scores: dict[int, int],
               comment_scores: dict[int, int], v: dict) -> None:
    """Add a single vote into the running score dicts."""
    if v["post_id"]:
        post_scores[v["post_id"]] = post_scores.get(v["post_id"], 0) + v["value"]
    elif v["comment_id"]:
        comment_scores[v["comment_id"]] = comment_scores.get(v["comment_id"], 0) + v["value"]


# ── Identity anonymization ───────────────────────────────

def sub_names_in_text(text: str, name_map: dict[str, str]) -> str:
    """Rewrite every `Scientist_N` token in free-form text per `name_map`.

    Belt-and-braces for legacy DBs whose content literally contains
    `Scientist_N`. New Parliament runs (which apply the same name_map
    at LLM time) never produce such strings in the first place.
    """
    if not name_map:
        return text
    return SCIENTIST_RE.sub(
        lambda m: name_map.get(f"Scientist_{m.group(1)}",
                               f"Scientist_{m.group(1)}"),
        text,
    )


# ── Cell config detection ────────────────────────────────

def detect_cell_flags(db_path: str) -> tuple[bool, bool]:
    """Read ``actor_context_coupled`` / ``judge_votes_visible`` from the
    rollout's ``experiment.json`` next to the DB.

    Falls back to (True, True) — the Parliament-A defaults — when no
    experiment.json is present (e.g. tests, legacy data).  Returns
    ``(actor_coupled, judge_visible)``.
    """
    exp_path = Path(db_path).parent / "experiment.json"
    if not exp_path.exists():
        return True, True
    try:
        cfg_dict = json.loads(exp_path.read_text()).get("config", {})
    except Exception:
        return True, True
    return (bool(cfg_dict.get("actor_context_coupled", True)),
            bool(cfg_dict.get("judge_votes_visible", True)))


# ── Vote-language detection (cell-aware) ─────────────────

def session_uses_vote_language(posts: list[dict], comments: list[dict],
                               actor_coupled: bool) -> bool:
    """True iff any actor-authored content in this session references voting.

    Decides whether vote events should appear as standalone items in
    the rendered training timeline.  When the actor never mentioned
    scoring, the training context stays vote-free; the moment the
    actor talks about "the +2 on P_3" the corresponding vote events
    must appear so the reference resolves.

    Coupled cells (Parliament/BlindParliament) check both posts and
    comments — both can land in the actor's context.  Solo cells
    check posts only since comments don't exist there.
    """
    targets = [p for p in posts if p["author_role"] == "actor"]
    if actor_coupled:
        targets += [c for c in comments if c["author_role"] == "actor"]
    return any(SCORE_META_RE.search(t["content"]) for t in targets)


# ── Rewards ──────────────────────────────────────────────

def compute_post_rewards(actor_posts: list[dict],
                         votes: list[dict]) -> dict[int, float]:
    """Sum judge votes on each actor post.

    Only judges contribute to the reward signal — actor votes are noisy
    peer guesses (actors are also trying to find the answer). Actor votes
    still appear in the discussion context the model conditions on; they
    just don't bias the reward.
    """
    rewards = {p["post_id"]: 0.0 for p in actor_posts}
    for v in votes:
        if v.get("author_role") != "judge":
            continue
        pid = v.get("post_id")
        if pid in rewards:
            rewards[pid] += v["value"]
    return rewards


# ── Template-augmentation pool ───────────────────────────
#
# The actor's training context is rebuilt from the DB by
# `extract_session`/`_render_update`.  Every wrapping string (section
# header, post header, comment header, vote event line, anonymous
# voter label, etc.) is sampled from a multi-variant pool seeded by
# the session (and by the entry id for repeated entries) so the model
# never sees a single fixed wrapper across the dataset.  Information
# fields ({local_id}, {author}, {value}, {cur}, ...) are always
# present and unchanged — only the surrounding text varies.
#
# Why augment:
#   With a single fixed wrapper, the model learns a brittle binding
#   between literal strings ("[V on P_3] by Anonymous Scientist:") and
#   their semantics.  Diverse wrappers force the model to read the
#   information fields rather than memorise the wrapper, which both
#   reduces overfitting (Khan et al., 2026 "Prompt Augmentation Scales
#   up GRPO Training on Mathematical Reasoning") and makes inference
#   robust to query-format drift.
#
# Determinism contract:
#   - `seed = (session_title, key)` for whole-message templates
#     (per-session) → the *same* template variant for the whole user
#     message in that session, so a single user message stays
#     internally coherent.
#   - `seed = (session_title, key, entry_id)` for repeated entries
#     (per-entry) → consecutive posts/comments/votes inside the same
#     user message can use *different* wrappers, maximising diversity.
#   - 4 cells share `session_title`, so the same problem always draws
#     the same wrappers — template choice is never a confounder in
#     2×2 cell comparisons.

TEMPLATE_POOL: dict[str, list[str]] = {
    # ── Per-session (whole-message scaffolding) ──
    "prompt_intro": [
        "You are {name}, a scientist participating in a scientific discussion on a forum. Read the problem and the existing discussion, then contribute your next analysis.",
        "{name} here, joining a scientific discussion on a forum. Review the problem and the prior thread, then contribute your next analytical step.",
        "As scientist {name}, you are reasoning through a problem alongside others on a forum. Read the problem and existing thread, then contribute.",
        "{name}, a scientist on a forum. Below is the problem and the discussion to date — please add your next step.",
        "Hello {name}. You're a scientist working on this problem with others. Read the context and contribute next.",
        "Welcome, {name}. You are a scientist on a discussion forum. The problem and existing posts are below — add your next contribution.",
        "Scientist {name}, the forum has been active. Read the problem and discussion below, then post your next move.",
        "{name} signing in to a forum-style scientific reasoning session. Review the problem and existing analysis, then contribute.",
        "Acting as scientist {name} on this forum, examine the problem and the discussion record below, then contribute the next step.",
        "Reasoning as {name} on a scientific forum. Read what follows and add your next analytical contribution.",
        "{name}, scientist. Read the problem statement, study the discussion, and add your next analytical move.",
        "You are {name}. The forum thread covers a scientific problem. Review everything below and contribute the next step.",
    ],
    "section_problem": [
        "## Problem",
        "**Problem**",
        "## Question",
        "## The Problem",
        "## Statement",
        "## Problem Statement",
        "**Problem Statement**",
        "### Problem",
        "## Task",
        "**Task**",
        "**Question**",
        "## Question to solve",
    ],
    "section_discussion": [
        "## Discussion",
        "## Forum activity",
        "## Discussion so far",
        "## What others have written",
        "## Prior contributions",
        "**Discussion so far**",
        "### Existing thread",
        "## Thread",
        "## Discussion thread",
        "## Existing posts",
        "## Prior discussion",
        "**Discussion**",
    ],
    "section_next": [
        "## Your Next Contribution",
        "## Your turn",
        "## Continue",
        "## Your next move",
        "## Your reasoning",
        "## Add next step",
        "## Your move",
        "**Your contribution**",
        "## Next step",
        "## Your next analysis",
    ],
    "no_new_content": [
        "No new discussion. Continue.",
        "(No new activity since your last move.)",
        "Nothing new arrived. Continue your reasoning.",
        "No updates yet. Move on.",
        "(No new posts.)",
        "No further activity. Continue.",
        "(Quiet — no new contributions.)",
        "Nothing new since last round. Continue.",
    ],

    # ── Per-entry (each occurrence sampled independently) ──
    "post_header": [
        "[P_{local_id}] by {author}{is_you}",
        "P_{local_id} from {author}{is_you}:",
        "P_{local_id} by {author}{is_you}",
        "Post P_{local_id} ({author}{is_you})",
        "{author}{is_you} — P_{local_id}",
        "{author}{is_you}'s post P_{local_id}:",
        "P_{local_id} ← {author}{is_you}",
        "[P_{local_id}, by {author}{is_you}]",
        "Post {local_id} from {author}{is_you}:",
        "P_{local_id} (by {author}{is_you})",
    ],
    "comment_header": [
        "[C_{local_id}] on P_{local_post_id} by {author}{is_you}",
        "C_{local_id} on P_{local_post_id} ({author}{is_you})",
        "Comment C_{local_id} on P_{local_post_id} from {author}{is_you}:",
        "{author}{is_you} replies to P_{local_post_id} (C_{local_id}):",
        "C_{local_id} (on P_{local_post_id}, by {author}{is_you})",
        "Reply to P_{local_post_id} by {author}{is_you} (C_{local_id}):",
        "[C_{local_id}, on P_{local_post_id}, {author}{is_you}]",
        "C_{local_id} ← {author}{is_you} on P_{local_post_id}",
        "Comment {local_id} on Post {local_post_id} from {author}{is_you}:",
        "P_{local_post_id} comment by {author}{is_you} (C_{local_id})",
    ],
    "vote_event": [
        "[V on {target}] by {author}: {change}, current score of {target}: {cur}",
        "Vote on {target} by {author}: {change} → now {cur}",
        "{author} votes on {target}: {change} (cumulative: {cur})",
        "{target} ← {change} from {author}, total {cur}",
        "Vote: {target} by {author}, {change} (now {cur})",
        "{change} on {target} from {author} — {target} now at {cur}",
        "[Vote] {target} by {author}: {change}; cumulative score {cur}",
        "{author} → {target}: {change} | total: {cur}",
        "Vote event on {target} ({author}): {change}, score {cur}",
        "Score update on {target} by {author}: {change} (now {cur})",
    ],
    "vote_new_text": [
        "{value} vote",
        "voted {value}",
        "cast {value}",
    ],
    "vote_change_text": [
        "changed {prev} → {value}",
        "updated from {prev} to {value}",
        "revised: {prev} → {value}",
    ],
    "anonymous_voter": [
        "Anonymous Scientist",
        "Anonymous reviewer",
        "An anonymous senior",
        "Senior reviewer (anonymous)",
        "Anonymous contributor",
        "Anonymous evaluator",
        "Anonymous senior scientist",
        "Anonymous voter",
        "An unnamed senior",
        "Anonymous",
        "Hidden reviewer",
        "Senior (anonymous)",
    ],
    "you_marker": [
        " [you]",
        " (you)",
        " — you",
        " *self*",
        " (this is you)",
    ],
}

# Whole-message templates seeded by session only; per-entry templates
# additionally seeded by the entry's DB id for finer-grained diversity.
_PER_SESSION_KEYS = frozenset({
    "prompt_intro", "section_problem", "section_discussion",
    "section_next", "no_new_content",
})

# A baseline used by `--no-template-augment` (forces every key to the
# first variant so the output is byte-stable / identical across runs).
_DETERMINISTIC_BASELINE = False


def sample_template(key: str, session_seed: str, entry_seed: str = "") -> str:
    """Deterministically pick a template variant.

    `session_seed` should be `session["title"]` (the problem text); 2×2
    cells running the same problem share this seed, so they always
    draw the same wrappers and template choice is never a confounder
    in cell-vs-cell comparisons.

    `entry_seed` is the DB-global id (post_id / comment_id / vote_id)
    for per-entry keys; the same entry always draws the same template
    no matter how many times the renderer touches it.
    """
    pool = TEMPLATE_POOL[key]
    if _DETERMINISTIC_BASELINE:
        return pool[0]
    if key in _PER_SESSION_KEYS:
        seed = f"template:{key}:{session_seed}"
    else:
        seed = f"template:{key}:{session_seed}:{entry_seed}"
    return random.Random(seed).choice(pool)


# ── Per-session extraction ───────────────────────────────

def _render_update(entries: list[dict], self_name: str,
                   session_seed: str) -> str:
    """Render a batch of timeline entries as a user-turn update.

    Every wrapper string (post header, comment header, vote event,
    `[you]` marker) is sampled from `TEMPLATE_POOL` so the model sees
    ~10 different wrappings of the same semantic content and learns to
    read the information fields rather than memorise wrapper strings.

    Each vote entry must carry a pre-computed ``_cur_score`` reflecting
    the cumulative score *after* this vote was folded — set by the
    caller during the per-actor walk.
    """
    lines: list[str] = []
    for entry in entries:
        if entry["type"] == "post":
            entry_seed = f"P{entry['post_id']}"
            is_you = (sample_template("you_marker", session_seed, entry_seed)
                      if entry["author"] == self_name else "")
            tmpl = sample_template("post_header", session_seed, entry_seed)
            lines.append(tmpl.format(
                local_id=entry["local_id"], author=entry["author"],
                is_you=is_you))
            lines.append(entry["content"].strip())
            lines.append("")
        elif entry["type"] == "comment":
            entry_seed = f"C{entry['comment_id']}"
            is_you = (sample_template("you_marker", session_seed, entry_seed)
                      if entry["author"] == self_name else "")
            tmpl = sample_template("comment_header", session_seed, entry_seed)
            lines.append(tmpl.format(
                local_id=entry["local_id"],
                local_post_id=entry["local_post_id"],
                author=entry["author"], is_you=is_you))
            lines.append(entry["content"].strip())
            lines.append("")
        elif entry["type"] == "vote":
            entry_seed = f"V{entry['vote_id']}"
            tt = "P" if entry["target_type"] == "post" else "C"
            target = f"{tt}_{entry['target_local_id']}"
            value_str = fmt_score(entry["value"])
            cur_str = fmt_score(entry["_cur_score"])
            prev = entry.get("previous_value")
            if prev is not None:
                change_tmpl = sample_template(
                    "vote_change_text", session_seed, entry_seed)
                change = change_tmpl.format(
                    prev=fmt_score(prev), value=value_str)
            else:
                change_tmpl = sample_template(
                    "vote_new_text", session_seed, entry_seed)
                change = change_tmpl.format(value=value_str)
            event_tmpl = sample_template(
                "vote_event", session_seed, entry_seed)
            lines.append(event_tmpl.format(
                author=entry["author"], target=target,
                change=change, cur=cur_str))
            lines.append("")
    return "\n".join(lines).strip()


def _build_actor_view(actor_global: str, posts: list[dict],
                      comments: list[dict], votes: list[dict],
                      actor_coupled: bool, judge_visible: bool
                      ) -> tuple[list[dict], list[dict], list[dict]]:
    """Filter posts/comments/votes to *this* actor's rollout-time view.

    Coupled cells: every post and every comment is visible; the vote
    pool keeps actor votes plus (when ``judge_visible``) judge votes.
    Solo cells: only this actor's own posts are visible, comments do
    not exist in the cell at all, and the vote pool keeps *only* judge
    votes whose target post is one of this actor's own posts.

    Solo-cell solo actors at rollout therefore never see references to
    posts they didn't author themselves — no dangling ``P_5`` for a
    post the actor cannot resolve.  Vote-on-comment is dropped by
    construction (``post_id`` check).
    """
    if actor_coupled:
        v_posts = list(posts)
        v_comments = list(comments)
        v_votes = [v for v in votes
                   if v.get("author_role") == "actor"
                   or (v.get("author_role") == "judge" and judge_visible)]
        return v_posts, v_comments, v_votes

    v_posts = [p for p in posts
               if p["author"] == actor_global and p["author_role"] == "actor"]
    v_comments: list[dict] = []                 # solo: no comment tool
    own_pids = {p["post_id"] for p in v_posts}
    v_votes = []
    if judge_visible:
        v_votes = [v for v in votes
                   if v.get("author_role") == "judge"
                   and v.get("post_id") in own_pids]
    return v_posts, v_comments, v_votes


def extract_session(session: dict, posts: list[dict],
                    comments: list[dict], votes: list[dict],
                    rewards: dict[int, float],
                    advantages: dict[int, float],
                    actor_coupled: bool = True,
                    judge_visible: bool = True) -> list[dict]:
    """Build per-actor trajectory samples for one session.

    Returns one dict per actor.  The view is built independently per
    actor so solo-cell isolation is structural rather than filtered:

      | actor_coupled | judge_visible | this actor's view contains                       |
      |---------------|---------------|---------------------------------------------------|
      | True  (A)     | True          | every post + comment + actor votes + judge votes  |
      | True  (B)     | False         | every post + comment + actor votes                |
      | False (C)     | True          | only own posts + judge votes targeting own posts  |
      | False (D)     | False         | only own posts                                    |

    Local IDs (``P_1, P_2, …``) are assigned over *this view's posts
    only*, so a solo actor sees their own posts numbered consecutively
    from 1 — no gaps that would betray the existence of peers.  Vote
    target IDs resolve against the same per-actor map.

    Vote events appear *only* when this actor's own content references
    voting (``session_uses_vote_language``); vote-language-free actors
    keep a completely score-free training context.

    Each assistant turn has its own reward and advantage; the trainer
    broadcasts these to per-token tensors via segment masks.
    """
    actor_posts = [p for p in posts if p["author_role"] == "actor"]
    if not actor_posts:
        return []

    sid = session["session_id"]
    # Seed the name map by `title` (= problem text) to mirror what the
    # harness used at rollout time — `title` is the stable key shared
    # across 2×2 cells, while `session_id` is per-run and differs.
    name_map = (assign_session_names(session["title"])
                if cfg().get("anonymize_identity") else {})
    session_seed = session["title"]

    min_chars = cfg()["min_content_chars"]
    title = sub_names_in_text(session["title"], name_map)

    # Per-session template draws for the whole-message scaffolding —
    # one consistent set of section headers / intro / fallback for
    # the whole session, so a single user message stays internally
    # coherent.  Per-entry pools (post/comment/vote/anon/you-marker)
    # are sampled inside `_render_update`.
    intro_tmpl = sample_template("prompt_intro", session_seed)
    sec_problem = sample_template("section_problem", session_seed)
    sec_discussion = sample_template("section_discussion", session_seed)
    sec_next = sample_template("section_next", session_seed)
    fallback_no_new = sample_template("no_new_content", session_seed)

    # One actor at a time — each actor's view is fully independent.
    actor_global_names = sorted({p["author"] for p in actor_posts})

    samples: list[dict] = []
    for actor_global in actor_global_names:
        actor_display = name_map.get(actor_global, actor_global)

        # 1) Scope: what this actor saw at rollout time.
        v_posts, v_comments, v_votes = _build_actor_view(
            actor_global, posts, comments, votes,
            actor_coupled, judge_visible)
        if not v_posts:
            continue                                  # actor wrote nothing

        # 2) Per-actor local IDs over this actor's view (deep-copied).
        v_posts, v_comments = assign_local_ids_view(v_posts, v_comments)

        # 3) Vote events appear only when this actor's own content
        #    references voting (kept score-free otherwise).
        include_vote_events = bool(v_votes) and session_uses_vote_language(
            v_posts, v_comments, actor_coupled)

        # 4) Anonymize vote authors per-vote — judge votes draw a
        #    variant from `anonymous_voter`, actor votes carry the
        #    voter's anonymized session name.
        anon_votes = [
            {**v, "author": (
                sample_template("anonymous_voter", session_seed,
                                f"V{v['vote_id']}")
                if v.get("author_role") == "judge"
                else name_map.get(v["author"], v["author"])
            )}
            for v in v_votes
        ]

        # 5) Build this actor's timeline (peer entries already absent
        #    in solo cells; coupled cells include peers naturally).
        timeline = build_timeline(
            v_posts, v_comments, anon_votes,
            include_comments=actor_coupled,
            include_vote_events=include_vote_events,
        )
        for entry in timeline:
            if entry["type"] in ("post", "comment"):
                entry["author"] = name_map.get(
                    entry["author"], entry["author"])
                entry["content"] = sub_names_in_text(
                    entry["content"], name_map)

        # 6) Walk the timeline → user/assistant message pairs.
        messages: list[dict] = []
        turn_rewards: list[float] = []
        turn_advantages: list[float] = []
        turn_post_ids: list[int] = []
        turn_trainable: list[bool] = []

        # Per-actor running cumulative score — votes are folded as the
        # actor walks the timeline so each vote event's "current score"
        # reflects what *this actor* saw in chronological order.
        post_scores: dict[int, int] = {}
        comment_scores: dict[int, int] = {}
        pending_updates: list[dict] = []

        for entry in timeline:
            etype = entry["type"]
            is_self_post = (etype == "post"
                            and entry["author"] == actor_display
                            and entry["author_role"] == "actor")

            # Defensive: own votes are never displayed; in solo cells
            # peer posts shouldn't be in `v_posts` to begin with.
            if etype == "vote" and entry["author"] == actor_display:
                continue
            if etype == "post" and not is_self_post and not actor_coupled:
                continue

            # Vote: fold first, then queue with post-fold cumulative score.
            if etype == "vote":
                fold_payload = {
                    "post_id": entry["target_global_id"]
                               if entry["target_type"] == "post" else None,
                    "comment_id": entry["target_global_id"]
                                  if entry["target_type"] == "comment" else None,
                    "value": entry["value"],
                }
                _fold_vote(post_scores, comment_scores, fold_payload)
                cur = (post_scores[entry["target_global_id"]]
                       if entry["target_type"] == "post"
                       else comment_scores[entry["target_global_id"]])
                pending_updates.append({**entry, "_cur_score": cur})
                continue

            # Self post → always becomes an assistant turn so the chat
            # structure faithfully mirrors what the actor actually saw
            # at rollout time.  Short posts get `is_trainable=False`,
            # which the trainer translates into a zeroed response_mask
            # for that turn — the post stays in the model's input
            # context but contributes nothing to the gradient.
            if is_self_post:
                content = sub_names_in_text(entry["content"].strip(), name_map)
                pid = entry["post_id"]
                is_trainable = len(content) >= min_chars

                if not messages:
                    parts = [
                        intro_tmpl.format(name=actor_display), "",
                        sec_problem, "",
                        title.strip(), "",
                    ]
                    if pending_updates:
                        parts += [
                            sec_discussion, "",
                            _render_update(pending_updates, actor_display,
                                           session_seed), "",
                        ]
                    parts.append(sec_next)
                    messages.append({"role": "user",
                                     "content": "\n".join(parts)})
                else:
                    update_text = _render_update(pending_updates, actor_display,
                                                 session_seed)
                    if not update_text:
                        update_text = fallback_no_new
                    update_text += "\n\n" + sec_next
                    messages.append({"role": "user", "content": update_text})

                messages.append({"role": "assistant", "content": content})
                turn_rewards.append(rewards[pid])
                turn_advantages.append(round(advantages[pid], 4))
                turn_post_ids.append(pid)
                turn_trainable.append(is_trainable)
                pending_updates = []
            else:
                # Peer post or comment — coupled cell only.  Queue for
                # the next user message.
                pending_updates.append(entry)

        # Drop the actor entirely only when they posted nothing at all
        # (length-zero turn list).  When at least one turn exists but
        # none are trainable (all posts < min_chars), the sample stays
        # in the JSONL — the trainer masks it out and the empty-mask
        # filter in `RLDataset` does the final drop.
        if not turn_rewards:
            continue

        samples.append({
            "messages": messages,
            "turn_rewards": turn_rewards,
            "turn_advantages": turn_advantages,
            "turn_post_ids": turn_post_ids,
            "turn_trainable": turn_trainable,
            "session_id": sid,
            "actor": actor_display,
        })
    return samples


# ── Advantage normalization ─────────────────────────────

def _session_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _session_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 1.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return var ** 0.5


def _estimate_position_slope(
        rewards_per_session: dict[str, dict[int, float]]) -> float:
    """OLS estimate of the linear reward-vs-position slope.

    Later posts tend to score higher (they build on prior discussion).
    This slope lets us detrend before advantage normalization so the
    model doesn't learn "post late = good".

    Fits ``r = a + b·t`` by closed-form least squares using *every*
    actor post's ``(t, r)`` pair across all sessions, where ``t`` is
    the post's normalized position [0, 1] in its session.  Returns
    ``b`` (= slope of ``E[r | t]``).

    OLS uses the full distribution of points instead of just the
    early/late midpoint gap, so it tolerates non-uniform session
    lengths and outlier sessions far better than the previous
    mid-split estimator.  Sessions with < 4 actor posts are excluded
    (their normalized t carries little information).
    """
    ts: list[float] = []
    rs: list[float] = []
    for rewards in rewards_per_session.values():
        pids = sorted(rewards.keys())
        n = len(pids)
        if n < 4:
            continue
        denom = n - 1                            # ≥ 3 ⇒ no zero-div
        for k, pid in enumerate(pids):
            ts.append(k / denom)                 # 0, 1/(n-1), …, 1
            rs.append(rewards[pid])
    if len(ts) < 4:
        return 0.0
    n = len(ts)
    mean_t = sum(ts) / n
    mean_r = sum(rs) / n
    cov_tr = sum((t - mean_t) * (r - mean_r) for t, r in zip(ts, rs))
    var_t = sum((t - mean_t) ** 2 for t in ts)
    if var_t < 1e-9:
        return 0.0
    return cov_tr / var_t


def debias_position(rewards_per_session: dict[str, dict[int, float]],
                    positions: dict[int, float]) -> dict[str, dict[int, float]]:
    """Remove linear position trend from rewards.

    r_debiased = r - slope * (t - 0.5)

    where t is the post's normalized position in its session [0, 1] and
    slope is estimated globally from all sessions.  At t=0.5 (session
    midpoint) the correction is zero; early posts are boosted, late posts
    are reduced.
    """
    slope = _estimate_position_slope(rewards_per_session)
    if abs(slope) < 0.01:
        return rewards_per_session
    debiased: dict[str, dict[int, float]] = {}
    for sid, rewards in rewards_per_session.items():
        debiased[sid] = {}
        for pid, r in rewards.items():
            t = positions.get(pid, 0.5)
            debiased[sid][pid] = r - slope * (t - 0.5)
    print(f"  Position debias: slope={slope:.3f} "
          f"(early boost ≈ +{slope * 0.5:.2f}, late penalty ≈ -{slope * 0.5:.2f})")
    return debiased


def compute_advantages(rewards_per_session: dict[str, dict[int, float]]
                       ) -> dict[int, float]:
    """Advantage = (reward − baseline) / scale, controlled by config.

    Knobs in `RL_context/config.json`:

      advantage_baseline:
        0.0                → no centering
        "mean_session"     → classic GRPO centering (per-session mean)
        "mean_global"      → REINFORCE++-style (dataset mean)
        <any number>       → fixed constant

      advantage_scale:
        "session_std"      → per-session std (default; standard GRPO)
        "global_std"       → dataset std (REINFORCE++-style)
        "none" / 1.0       → no rescaling (raw reward as advantage)
        <any number>       → fixed constant

    A near-zero session std (homogeneous session — every post got the
    same reward) is floored at 1.0 so a degenerate session can't blow up
    the gradient.
    """
    c = cfg()
    baseline_opt = c.get("advantage_baseline", 0.0)
    scale_opt = c.get("advantage_scale", "session_std")
    std_floor = 1.0

    all_rewards = [r for rs in rewards_per_session.values() for r in rs.values()]
    mean_global = _session_mean(all_rewards)
    std_global = max(pstdev(all_rewards) if len(all_rewards) > 1 else 1.0,
                     std_floor)

    def _baseline_for(values: list[float]) -> float:
        if isinstance(baseline_opt, str):
            if baseline_opt == "mean_session":
                return _session_mean(values)
            if baseline_opt == "mean_global":
                return mean_global
            try:
                return float(baseline_opt)
            except ValueError:
                return 0.0
        return float(baseline_opt)

    def _scale_for(values: list[float]) -> float:
        if isinstance(scale_opt, str):
            if scale_opt == "session_std":
                return max(_session_std(values), std_floor)
            if scale_opt == "global_std":
                return std_global
            if scale_opt == "none":
                return 1.0
            try:
                return float(scale_opt)
            except ValueError:
                return 1.0
        return float(scale_opt)

    advantages: dict[int, float] = {}
    for rewards in rewards_per_session.values():
        if not rewards:
            continue
        values = list(rewards.values())
        baseline = _baseline_for(values)
        scale = _scale_for(values)
        for pid, r in rewards.items():
            advantages[pid] = (r - baseline) / scale
    return advantages


# ── Main ─────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract RL training data from parliament.db")
    parser.add_argument("--db", required=True, help="Path to parliament.db")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--min-posts", type=int, default=1,
                        help="Skip sessions with fewer actor trajectories")
    parser.add_argument("--no-template-augment", action="store_true",
                        help="Disable template-pool sampling (every key "
                             "uses the first variant — byte-stable output, "
                             "useful for ablation against the augmented run)")
    args = parser.parse_args()

    global _DETERMINISTIC_BASELINE
    _DETERMINISTIC_BASELINE = args.no_template_augment

    if not Path(args.db).exists():
        print(f"Error: {args.db} not found")
        sys.exit(1)

    actor_coupled, judge_visible = detect_cell_flags(args.db)
    cell_label = {(True, True): "A (Parliament)",
                  (True, False): "B (BlindParliament)",
                  (False, True): "C (Solo)",
                  (False, False): "D (BlindSolo)"}[(actor_coupled, judge_visible)]
    print(f"Cell:            {cell_label}")
    print(f"  actor_coupled = {actor_coupled}, judge_visible = {judge_visible}")
    n_variants = sum(len(v) for v in TEMPLATE_POOL.values())
    print(f"Templates:       {'augmented' if not _DETERMINISTIC_BASELINE else 'baseline (first variant only)'}"
          f"  ({n_variants} variants across {len(TEMPLATE_POOL)} keys)")

    conn = connect(args.db)
    session_ids = load_session_ids(conn)
    print(f"Sessions:        {len(session_ids)}")

    # Pass 1: load all sessions, compute per-session rewards + positions.
    cache: dict[str, tuple[dict, list, list, list, dict[int, float]]] = {}
    rewards_per_session: dict[str, dict[int, float]] = {}
    post_positions: dict[int, float] = {}
    for sid in session_ids:
        session, posts, comments, votes = load_session_data(conn, sid)
        actor_posts = [p for p in posts if p["author_role"] == "actor"]
        rewards = compute_post_rewards(actor_posts, votes)
        cache[sid] = (session, posts, comments, votes, rewards)
        rewards_per_session[sid] = rewards
        # Normalized position: 0 = first actor post, 1 = last
        pids = sorted(rewards.keys())
        for k, pid in enumerate(pids):
            post_positions[pid] = k / max(len(pids) - 1, 1)
    conn.close()

    # Position debiasing: remove linear trend (later posts score higher).
    debiased = debias_position(rewards_per_session, post_positions)

    # Advantage normalization on debiased rewards.
    advantages = compute_advantages(debiased)

    # Pass 2: render samples.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total_samples = skipped_sessions = 0
    sessions_with_vote_events = 0
    with open(args.output, "w") as f:
        for i, sid in enumerate(session_ids):
            session, posts, comments, votes, rewards = cache[sid]
            samples = extract_session(session, posts, comments, votes,
                                      rewards, advantages,
                                      actor_coupled=actor_coupled,
                                      judge_visible=judge_visible)
            if len(samples) < args.min_posts:
                skipped_sessions += 1
                continue
            # Count sessions where vote events ended up rendered (purely
            # for the report — drives by `session_uses_vote_language`).
            if session_uses_vote_language(posts, comments, actor_coupled):
                sessions_with_vote_events += 1
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            total_samples += len(samples)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(session_ids)} sessions, "
                      f"{total_samples} samples", flush=True)

    print(f"\nDone:")
    print(f"  Sessions:        {len(session_ids) - skipped_sessions} "
          f"({skipped_sessions} skipped)")
    print(f"  Samples:         {total_samples}")
    print(f"  Vote-event sess: {sessions_with_vote_events} "
          f"(rest had no actor reference to voting)")
    print(f"  Anonymized:      {cfg().get('anonymize_identity', False)}")
    print(f"  Advantage:       baseline={cfg().get('advantage_baseline', 0.0)}, "
          f"scale={cfg().get('advantage_scale', 'session_std')}")
    print(f"  Output:          {args.output}")


if __name__ == "__main__":
    main()
