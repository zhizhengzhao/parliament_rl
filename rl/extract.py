#!/usr/bin/env python3
"""Extract RL training data from parliament.db.

Each actor post becomes one training sample. We reconstruct the actor's
first-person view at the moment of posting:

  user message  = problem + full prior discussion (posts + comments,
                  optionally annotated with cumulative scores)
                  + [you] markers on the actor's own contributions
  assistant msg = the actual post content
  reward        = sum of judge votes on this post (judges only)
  advantage     = (reward - baseline) / scale, both configurable

All content is natural language — no JSON, no tool calls, no agent-specific
formatting. The goal is to train scientific reasoning, not agent behavior.

Identity is anonymized with a per-session draw from a name pool, and score
annotations are kept only in sessions that actually meta-reference
Parliament scoring. Both behaviors are controlled by `RL_context/config.json`.

Usage:
    python -m rl.extract --db data/run/parliament.db --output data/run/train.jsonl
"""

from __future__ import annotations

import argparse
import json
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

TYPE_ORDER = {"post": 0, "comment": 1}

# Detects whether a post's content actually meta-references Parliament scoring.
SCORE_META_RE = re.compile(
    r"(?i)(score\s*of|scored|\bvote[sd]?\b|high[\-]?scoring|"
    r"negative[\-]?scoring|anonymous\s+(?:vote|scientist)|consensus)"
)
SCIENTIST_RE = re.compile(r"\bScientist_(\d+)\b")


# ── Config loading ───────────────────────────────────────

def load_rl_config() -> dict:
    cfg = json.loads((RL_CONTEXT / "config.json").read_text())
    cfg["prompt_intro"] = (RL_CONTEXT / "prompt_intro.txt").read_text().strip()
    # `name_pool` is the single source of truth for session casting and is
    # identical across all 2×2 cells (Parliament_context and Solo_context
    # share it), so we always read Parliament_context here regardless of
    # which cell produced the data.  `assign_session_names` (imported from
    # `harness.prompts`) deterministically reproduces the runtime mapping
    # from `session_id`, independent of `PRL_CONTEXT`.
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
        "u.role AS author_role "
        "FROM votes v JOIN users u ON v.user_id = u.user_id "
        "LEFT JOIN posts p ON v.post_id = p.post_id "
        "LEFT JOIN comments c ON v.comment_id = c.comment_id "
        "LEFT JOIN posts p2 ON c.post_id = p2.post_id "
        "WHERE COALESCE(p.session_id, p2.session_id) = ? "
        "ORDER BY v.vote_id",
        (session_id,)).fetchall()]

    return session, posts, comments, votes


# ── Local IDs & timeline ─────────────────────────────────

def assign_local_ids(posts: list[dict], comments: list[dict]) -> None:
    """Assign session-local sequential IDs (P_1, P_2, …, C_1, C_2, …).

    Order matches an agent's view: posts ordered by post_id (≡ creation
    order via autoincrement), comments by comment_id.
    """
    for i, p in enumerate(posts, 1):
        p["local_id"] = i
    for i, c in enumerate(comments, 1):
        c["local_id"] = i
    post_id_to_local = {p["post_id"]: p["local_id"] for p in posts}
    for c in comments:
        c["local_post_id"] = post_id_to_local.get(c["post_id"], 0)


def build_timeline(posts: list[dict], comments: list[dict]) -> list[dict]:
    """Merge posts and comments into a chronological timeline."""
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
    timeline.sort(key=lambda x: (
        x["created_at"], TYPE_ORDER[x["type"]], x["local_id"]))
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


# ── Score visibility ─────────────────────────────────────

def session_references_scores(posts: list[dict]) -> bool:
    """True if any actor post in this session meta-references parliament scoring."""
    for p in posts:
        if p["author_role"] == "actor" and SCORE_META_RE.search(p["content"]):
            return True
    return False


def should_show_scores(posts: list[dict]) -> bool:
    mode = cfg().get("score_visibility", "auto")
    if mode == "always":
        return True
    if mode == "never":
        return False
    return session_references_scores(posts)


# ── Rendering ────────────────────────────────────────────

def render_user_message(session_title: str, actor_name: str,
                        timeline_before: list[dict],
                        post_scores: dict[int, int],
                        comment_scores: dict[int, int],
                        show_scores: bool) -> str:
    """Render the actor's first-person view as natural language."""
    c = cfg()
    headers = c["section_headers"]
    you = c["you_marker"]
    score_suffix = c["score_suffix"]

    lines = [c["prompt_intro"].format(name=actor_name), "",
             headers["problem"], "", session_title.strip(), "",
             headers["discussion"], ""]

    for entry in timeline_before:
        is_you = you if entry["author"] == actor_name else ""
        if entry["type"] == "post":
            header = c["post_header"].format(
                local_id=entry["local_id"], author=entry["author"],
                is_you=is_you)
            score = post_scores.get(entry["post_id"], 0)
        else:
            header = c["comment_header"].format(
                local_id=entry["local_id"],
                local_post_id=entry["local_post_id"],
                author=entry["author"], is_you=is_you)
            score = comment_scores.get(entry["comment_id"], 0)
        if show_scores:
            header += score_suffix.format(score=fmt_score(score))
        lines.append(header)
        lines.append(entry["content"].strip())
        lines.append("")

    lines.append(headers["next_contribution"])
    return "\n".join(lines)


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


# ── Per-session extraction ───────────────────────────────

def _render_update(entries: list[dict], post_scores: dict[int, int],
                   comment_scores: dict[int, int],
                   show_scores: bool) -> str:
    """Render a batch of timeline entries as a user-turn update."""
    c = cfg()
    you = c["you_marker"]
    score_suffix = c["score_suffix"]
    lines: list[str] = []
    for entry in entries:
        is_you = ""
        if entry["type"] == "post":
            header = c["post_header"].format(
                local_id=entry["local_id"], author=entry["author"],
                is_you=is_you)
            score = post_scores.get(entry["post_id"], 0)
        else:
            header = c["comment_header"].format(
                local_id=entry["local_id"],
                local_post_id=entry["local_post_id"],
                author=entry["author"], is_you=is_you)
            score = comment_scores.get(entry["comment_id"], 0)
        if show_scores:
            header += score_suffix.format(score=fmt_score(score))
        lines.append(header)
        lines.append(entry["content"].strip())
        lines.append("")
    return "\n".join(lines).strip()


def extract_session(session: dict, posts: list[dict],
                    comments: list[dict], votes: list[dict],
                    rewards: dict[int, float],
                    advantages: dict[int, float]) -> list[dict]:
    """Build per-actor trajectory samples for one session.

    Each actor produces one training sample containing a multi-turn chat:
      user: problem + initial context
      assistant: actor's 1st post
      user: updates since last post
      assistant: actor's 2nd post
      ...

    Each assistant turn has its own reward and advantage. The training
    loop handles per-turn advantages via segment masks.

    Returns one dict per actor (not per post).
    """
    actor_posts = [p for p in posts if p["author_role"] == "actor"]
    if not actor_posts:
        return []

    assign_local_ids(posts, comments)
    timeline = build_timeline(posts, comments)

    sid = session["session_id"]
    name_map = assign_session_names(sid) if cfg().get("anonymize_identity") else {}
    show_scores = should_show_scores(posts)

    for entry in timeline:
        entry["author"] = name_map.get(entry["author"], entry["author"])
        entry["content"] = sub_names_in_text(entry["content"], name_map)

    sorted_votes = sorted(votes, key=lambda v: v["created_at"])
    min_chars = cfg()["min_content_chars"]

    actor_names = sorted(set(name_map.get(p["author"], p["author"])
                             for p in actor_posts))
    title = sub_names_in_text(session["title"], name_map)

    samples: list[dict] = []
    for actor_name in actor_names:
        messages: list[dict] = []
        turn_rewards: list[float] = []
        turn_advantages: list[float] = []
        turn_post_ids: list[int] = []

        vp = 0
        post_scores: dict[int, int] = {}
        comment_scores: dict[int, int] = {}
        # Buffer of timeline entries written by *other* agents (or this
        # actor's own short/no-reward posts) since this actor's last
        # qualifying turn. They get rendered into the next user message
        # as the "what happened while you were away" update.
        pending_updates: list[dict] = []

        c = cfg()
        system_intro = c["prompt_intro"].format(name=actor_name)

        for i, entry in enumerate(timeline):
            cutoff = entry["created_at"]
            while vp < len(sorted_votes) and sorted_votes[vp]["created_at"] < cutoff:
                _fold_vote(post_scores, comment_scores, sorted_votes[vp])
                vp += 1

            if entry["type"] == "post" and entry["author"] == actor_name \
                    and entry["author_role"] == "actor":
                content = sub_names_in_text(entry["content"].strip(), name_map)
                if len(content) < min_chars:
                    pending_updates.append(entry)
                    continue

                pid = entry["post_id"]
                if pid not in rewards:
                    pending_updates.append(entry)
                    continue

                if not messages:
                    user_text = (f"{system_intro}\n\n"
                                 f"{c['section_headers']['problem']}\n\n"
                                 f"{title.strip()}\n\n"
                                 f"{c['section_headers']['discussion']}\n\n")
                    if pending_updates:
                        user_text += _render_update(
                            pending_updates, post_scores, comment_scores,
                            show_scores)
                        user_text += "\n\n"
                    user_text += c["section_headers"]["next_contribution"]
                    messages.append({"role": "user", "content": user_text})
                else:
                    update_text = _render_update(
                        pending_updates, post_scores, comment_scores,
                        show_scores)
                    if not update_text:
                        update_text = "No new discussion. Continue."
                    update_text += "\n\n" + c["section_headers"]["next_contribution"]
                    messages.append({"role": "user", "content": update_text})

                messages.append({"role": "assistant", "content": content})
                turn_rewards.append(rewards[pid])
                turn_advantages.append(round(advantages[pid], 4))
                turn_post_ids.append(pid)
                pending_updates = []
            else:
                pending_updates.append(entry)

        if not turn_rewards:
            continue

        samples.append({
            "messages": messages,
            "turn_rewards": turn_rewards,
            "turn_advantages": turn_advantages,
            "turn_post_ids": turn_post_ids,
            "session_id": sid,
            "actor": actor_name,
            "show_scores": show_scores,
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


def _estimate_position_slope(rewards_per_session: dict[str, dict[int, float]],
                             positions: dict[int, float]) -> float:
    """Estimate the linear reward-vs-position slope from all data.

    Later posts tend to score higher (they build on prior discussion).
    This slope lets us detrend before advantage normalization so the
    model doesn't learn "post late = good".

    Returns the slope: E[reward](t=1) - E[reward](t=0), estimated via
    the gap between the first-half and second-half means (midpoints at
    t≈0.25 and t≈0.75, spanning 0.5 of the range).
    """
    early_r, late_r = [], []
    for rewards in rewards_per_session.values():
        pids = sorted(rewards.keys())
        if len(pids) < 4:
            continue
        mid = len(pids) // 2
        for pid in pids[:mid]:
            early_r.append(rewards[pid])
        for pid in pids[mid:]:
            late_r.append(rewards[pid])
    if not early_r or not late_r:
        return 0.0
    gap = _session_mean(late_r) - _session_mean(early_r)
    # gap spans t=0.25→0.75 (0.5 units), so full-range slope = gap / 0.5
    return gap / 0.5


def debias_position(rewards_per_session: dict[str, dict[int, float]],
                    positions: dict[int, float]) -> dict[str, dict[int, float]]:
    """Remove linear position trend from rewards.

    r_debiased = r - slope * (t - 0.5)

    where t is the post's normalized position in its session [0, 1] and
    slope is estimated globally from all sessions.  At t=0.5 (session
    midpoint) the correction is zero; early posts are boosted, late posts
    are reduced.
    """
    slope = _estimate_position_slope(rewards_per_session, positions)
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
        "none" / 1.0       → raw reward-weighted regression
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
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Error: {args.db} not found")
        sys.exit(1)

    conn = connect(args.db)
    session_ids = load_session_ids(conn)
    print(f"Sessions: {len(session_ids)}")

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
    score_kept_sessions = 0
    with open(args.output, "w") as f:
        for i, sid in enumerate(session_ids):
            session, posts, comments, votes, rewards = cache[sid]
            samples = extract_session(session, posts, comments, votes,
                                      rewards, advantages)
            if len(samples) < args.min_posts:
                skipped_sessions += 1
                continue
            if samples and samples[0]["show_scores"]:
                score_kept_sessions += 1
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
    print(f"  Scores kept in:  {score_kept_sessions} sessions "
          f"(rest had no meta-reference)")
    print(f"  Anonymized:      {cfg().get('anonymize_identity', False)}")
    print(f"  Advantage:       baseline={cfg().get('advantage_baseline', 0)}, "
          f"scale={cfg().get('advantage_scale', 'global_std')}")
    print(f"  Output:          {args.output}")


if __name__ == "__main__":
    main()
