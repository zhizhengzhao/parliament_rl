#!/usr/bin/env python3
"""Extract training data from parliament.db.

Reconstructs lightweight scientific discussions from posts and comments,
then builds (context, action, reward, advantage) samples for GRPO training.

Only Scientist posts become training samples. Comments appear in context
but are not trained on. Advantage is computed per-session (group = session).

Usage:
    python -m rl.extract --db data/train_part1_v3_.../parliament.db --output data/train.jsonl
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_session_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT s.session_id FROM sessions s "
        "JOIN posts p ON s.session_id = p.session_id "
        "ORDER BY s.session_id"
    ).fetchall()
    return [r[0] for r in rows]


def load_session_data(conn: sqlite3.Connection, session_id: str) -> dict:
    """Load all data for one session."""
    session = dict(conn.execute(
        "SELECT session_id, title FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone())

    posts = [dict(r) for r in conn.execute(
        "SELECT p.post_id, p.user_id, p.content, p.created_at, "
        "u.name AS author, u.role AS author_role "
        "FROM posts p JOIN users u ON p.user_id = u.user_id "
        "WHERE p.session_id = ? ORDER BY p.post_id",
        (session_id,),
    ).fetchall()]

    comments = [dict(r) for r in conn.execute(
        "SELECT c.comment_id, c.post_id, c.user_id, c.content, "
        "c.created_at, u.name AS author, u.role AS author_role "
        "FROM comments c JOIN users u ON c.user_id = u.user_id "
        "JOIN posts p ON c.post_id = p.post_id "
        "WHERE p.session_id = ? ORDER BY c.comment_id",
        (session_id,),
    ).fetchall()]

    votes = [dict(r) for r in conn.execute(
        "SELECT v.vote_id, v.post_id, v.comment_id, v.value "
        "FROM votes v "
        "LEFT JOIN posts p ON v.post_id = p.post_id "
        "LEFT JOIN comments c ON v.comment_id = c.comment_id "
        "LEFT JOIN posts p2 ON c.post_id = p2.post_id "
        "WHERE COALESCE(p.session_id, p2.session_id) = ?",
        (session_id,),
    ).fetchall()]

    return {"session": session, "posts": posts,
            "comments": comments, "votes": votes}


def assign_local_ids(posts: list[dict], comments: list[dict]) -> None:
    """Assign session-local sequential IDs (P_1, P_2, C_1, C_2...)."""
    for i, p in enumerate(posts, 1):
        p["local_id"] = i
    for i, c in enumerate(comments, 1):
        c["local_id"] = i
    post_id_to_local = {p["post_id"]: p["local_id"] for p in posts}
    for c in comments:
        c["local_post_id"] = post_id_to_local.get(c["post_id"], 0)


def build_timeline(posts: list[dict], comments: list[dict]) -> list[dict]:
    """Merge posts and comments into a single chronological timeline."""
    timeline = []
    for p in posts:
        timeline.append({
            "type": "post",
            "global_id": p["post_id"],
            "local_id": p["local_id"],
            "author": p["author"],
            "author_role": p["author_role"],
            "content": p["content"],
            "created_at": p["created_at"],
        })
    for c in comments:
        timeline.append({
            "type": "comment",
            "global_id": c["comment_id"],
            "local_id": c["local_id"],
            "local_post_id": c["local_post_id"],
            "author": c["author"],
            "author_role": c["author_role"],
            "content": c["content"],
            "created_at": c["created_at"],
        })
    type_order = {"post": 0, "comment": 1}
    timeline.sort(key=lambda x: (x["created_at"],
                                  type_order.get(x["type"], 9),
                                  x["global_id"]))
    return timeline


def format_entry(entry: dict) -> str:
    """Format a single timeline entry as discussion text."""
    if entry["type"] == "post":
        return f'[P_{entry["local_id"]}] {entry["author"]}:\n{entry["content"]}'
    return (f'[C_{entry["local_id"]}] {entry["author"]} '
            f'(on P_{entry["local_post_id"]}):\n{entry["content"]}')


def compute_post_rewards(posts: list[dict], votes: list[dict]) -> dict[int, float]:
    """Compute reward for each post from its votes."""
    rewards: dict[int, float] = {}
    for p in posts:
        rewards[p["post_id"]] = 0.0
    for v in votes:
        pid = v.get("post_id")
        if pid and pid in rewards:
            rewards[pid] += v["value"]
    return rewards


def compute_advantages(rewards: dict[int, float]) -> dict[int, float]:
    """Compute GRPO advantage: per-session normalization."""
    if not rewards:
        return {}
    values = list(rewards.values())
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std < 1e-8:
        return {pid: 0.0 for pid in rewards}
    return {pid: (r - mean) / std for pid, r in rewards.items()}


def extract_session(data: dict) -> list[dict]:
    """Extract training samples from one session."""
    session = data["session"]
    posts = data["posts"]
    comments = data["comments"]
    votes = data["votes"]

    actor_posts = [p for p in posts if p["author_role"] == "actor"]
    if not actor_posts:
        return []

    assign_local_ids(posts, comments)
    timeline = build_timeline(posts, comments)
    rewards = compute_post_rewards(actor_posts, votes)
    advantages = compute_advantages(rewards)

    samples = []
    for i, entry in enumerate(timeline):
        if entry["type"] != "post" or entry["author_role"] != "actor":
            continue

        post_id = entry["global_id"]
        content = entry["content"]
        if not content or len(content.strip()) < 20:
            continue

        context_parts = ["Problem: " + session["title"]]
        for prev in timeline[:i]:
            context_parts.append(format_entry(prev))
        context = "\n\n".join(context_parts)

        samples.append({
            "context": context,
            "action": content,
            "reward": rewards.get(post_id, 0.0),
            "advantage": round(advantages.get(post_id, 0.0), 4),
            "session_id": session["session_id"],
            "post_id": post_id,
            "author": entry["author"],
        })

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Extract GRPO training data from parliament.db")
    parser.add_argument("--db", required=True, help="Path to parliament.db")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--min-posts", type=int, default=3,
                        help="Skip sessions with fewer actor posts")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Error: {args.db} not found")
        sys.exit(1)

    conn = connect(args.db)
    session_ids = load_session_ids(conn)
    print(f"Sessions: {len(session_ids)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    skipped_sessions = 0

    with open(args.output, "w") as f:
        for i, sid in enumerate(session_ids):
            data = load_session_data(conn, sid)
            samples = extract_session(data)

            if len(samples) < args.min_posts:
                skipped_sessions += 1
                continue

            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            total_samples += len(samples)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(session_ids)} sessions, "
                      f"{total_samples} samples", flush=True)

    conn.close()

    print(f"\nDone:")
    print(f"  Sessions: {len(session_ids) - skipped_sessions} "
          f"({skipped_sessions} skipped)")
    print(f"  Samples:  {total_samples}")
    print(f"  Output:   {args.output}")


if __name__ == "__main__":
    main()
