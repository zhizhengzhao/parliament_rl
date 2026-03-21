"""
Science Parliament — context assembly.

Builds the context (messages list) for each scientist each round.
Replaces CAMEL's memory-based context with a clean, controlled assembly:
  - Earlier posts (no new activity) → full or compressed
  - Agent's own action history → from trace table
  - New/updated posts → always full
  - Guidance prompt

Also handles compression when context overflows.
"""

import asyncio
import json
import os
import sqlite3
from datetime import date, timedelta

from config import VLLM_MAX_MODEL_LEN

CONTEXT_SAFETY_RATIO = 0.85
_PARLIAMENT_START_DATE = "2026-03-17"

# Per-agent high-water marks: agent_id → (last_seen_post_id, last_seen_comment_id)
_watermarks: dict[int, tuple[int, int]] = {}

# Compressed content cache, keyed separately for posts and comments.
# Loaded from / saved to compressed_posts.json.
_compressed_posts: dict[int, str] = {}     # post_id → compressed content
_compressed_comments: dict[int, str] = {}  # comment_id → compressed content
_compressed_loaded = False


def reset():
    """Clear all state. Call at the start of each parliament session."""
    global _compressed_loaded
    _watermarks.clear()
    _compressed_posts.clear()
    _compressed_comments.clear()
    _compressed_loaded = False


def _db_path() -> str:
    return os.environ.get("OASIS_DB_PATH", "parliament.db")


def _load_compressed(output_dir: str):
    """Load compressed_posts.json if it exists. Skips if already loaded."""
    global _compressed_loaded
    if _compressed_loaded:
        return
    path = os.path.join(output_dir, "compressed_posts.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.get("posts", {}).items():
            _compressed_posts[int(k)] = v
        for k, v in data.get("comments", {}).items():
            _compressed_comments[int(k)] = v
    _compressed_loaded = True


def _save_compressed(output_dir: str):
    path = os.path.join(output_dir, "compressed_posts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "posts": {str(k): v for k, v in _compressed_posts.items()},
            "comments": {str(k): v for k, v in _compressed_comments.items()},
        }, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Read forum state from database
# ---------------------------------------------------------------------------

def _get_all_posts(db_path: str) -> list[dict]:
    """Fetch all posts with their comments, sorted by post_id (chronological)."""
    conn = sqlite3.connect(db_path, timeout=3)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT p.post_id, p.user_id, u.name AS author, p.content,
               (p.num_likes - p.num_dislikes) AS score
        FROM post p JOIN user u ON p.user_id = u.user_id
        ORDER BY p.post_id ASC
    """)
    posts = [dict(r) for r in c.fetchall()]
    post_map = {p["post_id"]: p for p in posts}

    c.execute("""
        SELECT cm.comment_id, cm.post_id, cm.user_id, u.name AS author, cm.content,
               (cm.num_likes - cm.num_dislikes) AS score
        FROM comment cm JOIN user u ON cm.user_id = u.user_id
        ORDER BY cm.comment_id ASC
    """)
    for r in c.fetchall():
        d = dict(r)
        parent = post_map.get(d["post_id"])
        if parent is not None:
            parent.setdefault("comments", []).append(d)

    conn.close()
    for p in posts:
        p.setdefault("comments", [])
    return posts


def _get_agent_actions(db_path: str, agent_id: int) -> list[str]:
    """Get a concise list of this agent's past actions from the trace table."""
    conn = sqlite3.connect(db_path, timeout=3)
    c = conn.cursor()
    c.execute("""
        SELECT action, info FROM trace
        WHERE user_id = ? AND action != 'refresh'
        ORDER BY rowid ASC
    """, (agent_id,))
    rows = c.fetchall()
    conn.close()

    lines = []
    for action, info_str in rows:
        try:
            info = json.loads(info_str) if info_str else {}
        except (json.JSONDecodeError, TypeError):
            info = {}

        if action == "create_post":
            preview = (info.get("content") or "")[:80]
            pid = info.get("post_id", "?")
            lines.append(f"You posted (post #{pid}): \"{preview}...\"")
        elif action == "create_comment":
            preview = (info.get("content") or "")[:80]
            pid = info.get("post_id", "?")
            lines.append(f"You commented on post #{pid}: \"{preview}...\"")
        elif action == "like_post":
            lines.append(f"You endorsed post #{info.get('post_id', '?')}")
        elif action == "dislike_post":
            lines.append(f"You challenged post #{info.get('post_id', '?')}")
        elif action == "follow":
            lines.append(f"You followed scientist #{info.get('followee_id', '?')}")
        elif action == "search_posts":
            lines.append(f"You searched: \"{info.get('query', '?')}\"")
        elif action in ("like_comment", "dislike_comment"):
            verb = "endorsed" if action == "like_comment" else "challenged"
            lines.append(f"You {verb} comment #{info.get('comment_id', '?')}")
        elif action == "do_nothing":
            lines.append("You passed your turn")

    return lines


# ---------------------------------------------------------------------------
# Date mapping via round_map.json
# ---------------------------------------------------------------------------

def _load_round_map(output_dir: str) -> list[dict] | None:
    path = os.path.join(output_dir, "round_map.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _id_to_date(item_id: int, boundaries: list[dict], key: str) -> str:
    """Map a post_id or comment_id to a date string."""
    start = date.fromisoformat(_PARLIAMENT_START_DATE)
    for entry in boundaries:
        if item_id <= entry[key]:
            return str(start + timedelta(days=entry["round"] - 1))
    if boundaries:
        return str(start + timedelta(days=boundaries[-1]["round"]))
    return _PARLIAMENT_START_DATE


# ---------------------------------------------------------------------------
# Format posts for display
# ---------------------------------------------------------------------------

def _format_post(post: dict, compressed: bool = False, round_map: list[dict] | None = None) -> str:
    """Format a single post with its comments for context display."""
    pid = post["post_id"]
    score = post["score"] or 0
    author = post["author"]

    date_tag = ""
    if round_map:
        date_tag = f" [{_id_to_date(pid, round_map, 'max_post_id')}]"

    uid = post.get("user_id", "?")
    if compressed and pid in _compressed_posts:
        content = _compressed_posts[pid]
        lines = [f"Post #{pid}{date_tag} [score: {score:+d}] by {author} (scientist_id:{uid}) [summarized]"]
    else:
        content = post["content"] or ""
        lines = [f"Post #{pid}{date_tag} [score: {score:+d}] by {author} (scientist_id:{uid})"]
    lines.append(content)

    comments = post.get("comments", [])
    if comments:
        lines.append("  Comments:")
        for cm in comments:
            cid = cm["comment_id"]
            cm_uid = cm.get("user_id", "?")
            cm_date = ""
            if round_map:
                cm_date = f"{_id_to_date(cid, round_map, 'max_comment_id')}, "
            if compressed and cid in _compressed_comments:
                lines.append(f"  [comment_id:{cid}, {cm_date}{cm['score'] or 0:+d}] {cm['author']} (scientist_id:{cm_uid}) [summarized]: {_compressed_comments[cid]}")
            else:
                lines.append(f"  [comment_id:{cid}, {cm_date}{cm['score'] or 0:+d}] {cm['author']} (scientist_id:{cm_uid}): {cm['content'] or ''}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build context — the main function
# ---------------------------------------------------------------------------

def build_context(agent_id: int, agent_name: str, system_content: str) -> list[dict]:
    """Assemble the complete context for one agent, one round.

    Returns a list of OpenAI-format message dicts ready to send to the model.
    """
    db_path = _db_path()
    output_dir = os.environ.get("PARLIAMENT_RUN_DIR", ".")
    _load_compressed(output_dir)
    round_map = _load_round_map(output_dir)

    posts = _get_all_posts(db_path)
    posts = [p for p in posts if not (
        p["post_id"] == 1 and "Parliament is now in session" in (p["content"] or "")
    )]

    prev_post_wm, prev_comment_wm = _watermarks.get(agent_id, (0, 0))

    old_posts = []
    new_posts = []
    for p in posts:
        is_new_post = p["post_id"] > prev_post_wm
        has_new_comments = any(
            cm["comment_id"] > prev_comment_wm for cm in p.get("comments", [])
        )
        if is_new_post or has_new_comments:
            new_posts.append(p)
        else:
            old_posts.append(p)

    all_post_ids = [p["post_id"] for p in posts]
    all_comment_ids = [cm["comment_id"] for p in posts for cm in p.get("comments", [])]
    new_post_wm = max(all_post_ids) if all_post_ids else prev_post_wm
    new_comment_wm = max(all_comment_ids) if all_comment_ids else prev_comment_wm
    _watermarks[agent_id] = (new_post_wm, new_comment_wm)

    actions = _get_agent_actions(db_path, agent_id)

    parts = []

    if old_posts:
        parts.append("EARLIER POSTS (no new activity since you last saw them):")
        for p in old_posts:
            use_compressed = p["post_id"] in _compressed_posts
            parts.append(_format_post(p, compressed=use_compressed, round_map=round_map))
        parts.append("")

    if actions:
        parts.append("YOUR PREVIOUS ACTIONS:")
        for a in actions:
            parts.append(f"  - {a}")
        parts.append("")

    if new_posts:
        label = "NEW OR UPDATED POSTS:" if old_posts else "FORUM POSTS:"
        parts.append(label)
        for p in new_posts:
            parts.append(_format_post(p, compressed=False, round_map=round_map))
        parts.append("")
    elif not old_posts:
        parts.append("No posts yet. You may be the first to contribute.\n")

    try:
        conn = sqlite3.connect(db_path, timeout=3)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM follow WHERE followee_id = ?", (agent_id,))
        n_followers = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM follow WHERE follower_id = ?", (agent_id,))
        n_following = c.fetchone()[0]
        conn.close()
        parts.append(f"{n_followers} colleagues are following your work. "
                     f"You are following {n_following} colleagues.\n")
    except Exception:
        pass

    from prompts import ROUND_GUIDANCE
    parts.append(ROUND_GUIDANCE)

    user_content = "\n".join(parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: chars / 4."""
    return sum(len(m.get("content", "")) for m in messages) // 4


def context_overflows(messages: list[dict]) -> bool:
    return estimate_tokens(messages) > int(VLLM_MAX_MODEL_LEN * CONTEXT_SAFETY_RATIO)


# ---------------------------------------------------------------------------
# Compression — triggered only on context overflow
# ---------------------------------------------------------------------------

from prompts import COMPRESS_SYSTEM, COMPRESS_USER


async def compress_posts(output_dir: str):
    """Compress all uncompressed posts/comments individually via batch HTTP requests."""
    global _compressed_loaded
    import httpx

    db_path = _db_path()
    posts = _get_all_posts(db_path)

    # Collect items to compress: (type, id, content)
    items: list[tuple[str, int, str]] = []
    for p in posts:
        if p["post_id"] == 1 and "Parliament is now in session" in (p["content"] or ""):
            continue
        if p["post_id"] not in _compressed_posts and p.get("content"):
            items.append(("post", p["post_id"], p["content"]))
        for cm in p.get("comments", []):
            if cm["comment_id"] not in _compressed_comments and cm.get("content"):
                items.append(("comment", cm["comment_id"], cm["content"]))

    if not items:
        return

    print(f"  Compressing {len(items)} items...")

    # Use the same vLLM endpoint via direct HTTP (avoid CAMEL overhead for simple calls)
    base_url = os.environ.get("OPENAI_API_BASE_URL", "http://localhost:8000/v1")
    from config import MODEL_NAME

    async with httpx.AsyncClient(timeout=120) as client:
        sem = asyncio.Semaphore(8)

        async def _compress_one(item_type: str, item_id: int, content: str):
            async with sem:
                try:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        json={
                            "model": MODEL_NAME,
                            "messages": [
                                {"role": "system", "content": COMPRESS_SYSTEM},
                                {"role": "user", "content": COMPRESS_USER.format(content=content)},
                            ],
                            "max_tokens": 512,
                        },
                    )
                    data = resp.json()
                    summary = data["choices"][0]["message"]["content"].strip()
                    if summary:
                        if item_type == "post":
                            _compressed_posts[item_id] = summary
                        else:
                            _compressed_comments[item_id] = summary
                except Exception as e:
                    print(f"  Compression failed for {item_type} #{item_id}: {e}")

        tasks = [_compress_one(t, i, c) for t, i, c in items]
        await asyncio.gather(*tasks)

    _save_compressed(output_dir)
    _compressed_loaded = True
    n = len(_compressed_posts) + len(_compressed_comments)
    print(f"  Compressed {n} items → {output_dir}/compressed_posts.json")


# ---------------------------------------------------------------------------
# Rollback — delete all data created after a given point
# ---------------------------------------------------------------------------

def rollback_to(db_path: str, snap: dict):
    """Delete all data created after the snapshot + reset watermarks.

    Args:
        snap: dict returned by session._snapshot() with keys:
              max_post_id, max_comment_id, max_trace_rowid,
              max_follow_rowid, max_like_rowid, max_dislike_rowid,
              max_comment_like_rowid, max_comment_dislike_rowid.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM post WHERE post_id > ?", (snap["max_post_id"],))
    c.execute("DELETE FROM comment WHERE comment_id > ?", (snap["max_comment_id"],))
    c.execute("DELETE FROM trace WHERE rowid > ?", (snap["max_trace_rowid"],))
    c.execute("DELETE FROM 'like' WHERE rowid > ?", (snap["max_like_rowid"],))
    c.execute("DELETE FROM 'dislike' WHERE rowid > ?", (snap["max_dislike_rowid"],))
    c.execute("DELETE FROM comment_like WHERE rowid > ?", (snap["max_comment_like_rowid"],))
    c.execute("DELETE FROM comment_dislike WHERE rowid > ?", (snap["max_comment_dislike_rowid"],))
    c.execute("DELETE FROM follow WHERE rowid > ?", (snap["max_follow_rowid"],))
    c.execute("DELETE FROM rec")

    # Recalculate denormalized counters — the DELETE above removed
    # like/dislike rows but left num_likes/num_dislikes on post/comment stale.
    c.execute("""
        UPDATE post SET
            num_likes = (SELECT COUNT(*) FROM 'like' WHERE 'like'.post_id = post.post_id),
            num_dislikes = (SELECT COUNT(*) FROM dislike WHERE dislike.post_id = post.post_id)
    """)
    c.execute("""
        UPDATE comment SET
            num_likes = (SELECT COUNT(*) FROM comment_like
                         WHERE comment_like.comment_id = comment.comment_id),
            num_dislikes = (SELECT COUNT(*) FROM comment_dislike
                           WHERE comment_dislike.comment_id = comment.comment_id)
    """)

    conn.commit()
    conn.close()

    max_post_id = snap["max_post_id"]
    max_comment_id = snap["max_comment_id"]
    for agent_id in _watermarks:
        _watermarks[agent_id] = (max_post_id, max_comment_id)

    print(f"  Rolled back: posts>{max_post_id}, comments>{max_comment_id}")
