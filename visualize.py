"""
Science Parliament — visualization module.

Reads the SQLite database and writes a self-contained index.html to the
run output directory.  Auto-refreshes every 8 seconds in the browser.

Usage (standalone):
    python visualize.py <db_path> <output_dir>

Usage (from run_parliament.py, already integrated):
    from visualize import generate_html
    generate_html(db_path, output_dir, question=question,
                  current_round=round_num, num_rounds=num_rounds)
"""

import os
import sqlite3
from datetime import datetime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _score_badge(score: int) -> str:
    score = score or 0
    css = "score-pos" if score > 0 else ("score-neg" if score < 0 else "score-zero")
    label = f"+{score}" if score > 0 else str(score)
    return f'<span class="score {css}">{label}</span>'


def _read_db(db_path: str) -> dict:
    """
    Read posts, comments, and stats from the database.
    timeout=3 lets us wait briefly if the parliament is mid-write.
    """
    conn = sqlite3.connect(db_path, timeout=3)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Posts with author names, sorted by score desc then post_id asc
    c.execute("""
        SELECT p.post_id,
               u.name       AS author,
               p.content,
               (p.num_likes - p.num_dislikes) AS score
        FROM post p
        JOIN user u ON p.user_id = u.user_id
        ORDER BY score DESC, p.post_id ASC
    """)
    posts = [dict(r) for r in c.fetchall()]

    # Comments with author names
    c.execute("""
        SELECT cm.comment_id, cm.post_id,
               u.name AS author,
               cm.content,
               (cm.num_likes - cm.num_dislikes) AS score
        FROM comment cm
        JOIN user u ON cm.user_id = u.user_id
        ORDER BY cm.comment_id ASC
    """)
    comments_by_post: dict = {}
    for row in c.fetchall():
        row = dict(row)
        comments_by_post.setdefault(row["post_id"], []).append(row)

    # Summary stats
    c.execute("SELECT COUNT(*) FROM post")
    n_posts = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM comment")
    n_comments = c.fetchone()[0]
    c.execute("""
        SELECT COUNT(*) FROM trace
        WHERE action IN ('like_post','dislike_post','like_comment','dislike_comment')
    """)
    n_votes = c.fetchone()[0]

    conn.close()
    return {
        "posts": posts,
        "comments_by_post": comments_by_post,
        "n_posts": n_posts,
        "n_comments": n_comments,
        "n_votes": n_votes,
    }


def _build_html(data: dict, question: str, current_round: int, num_rounds: int) -> str:
    """Assemble the complete HTML page."""

    # ── Question banner ────────────────────────────────────────────────────
    q_html = ""
    if question:
        q_html = f"""
<div class="question-banner">
  <span class="q-label">Q</span>
  <span class="q-text">{_esc(question)}</span>
</div>"""

    # ── Round progress ─────────────────────────────────────────────────────
    round_label = (
        f"Round {current_round} / {num_rounds}"
        if num_rounds > 0
        else ("Session opening..." if current_round == 0 else f"Round {current_round}")
    )
    pct = int(current_round / num_rounds * 100) if num_rounds > 0 else 0

    # ── Post cards ─────────────────────────────────────────────────────────
    cards_html = ""
    for post in data["posts"]:
        pid = post["post_id"]
        author = _esc(post["author"] or "?")
        avatar = author[0].upper()
        content = _esc(post["content"] or "")

        # Comments for this post
        cmts_html = ""
        for cm in data["comments_by_post"].get(pid, []):
            ca = _esc(cm["author"] or "?")
            cc = _esc(cm["content"] or "")
            cmts_html += f"""
    <div class="comment">
      <div class="comment-meta">
        <span class="comment-author">{ca}</span>
        {_score_badge(cm['score'])}
      </div>
      <div class="comment-body">{cc}</div>
    </div>"""

        cmts_section = (
            f'<div class="comments">{cmts_html}</div>' if cmts_html else ""
        )

        cards_html += f"""
<div class="card">
  <div class="card-header">
    <div class="avatar">{avatar}</div>
    <span class="card-author">{author}</span>
    <span class="card-id">#{pid}</span>
    {_score_badge(post['score'])}
  </div>
  <div class="card-body">{content}</div>
  {cmts_section}
</div>"""

    if not cards_html:
        cards_html = '<div class="empty">The parliament is about to open… ✨</div>'

    now = datetime.now().strftime("%H:%M:%S")

    # ── Full page ──────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Science Parliament</title>
<style>
:root{{--blue:#4cc9f0;--dark:#1a1a2e;--card:#fff;--bg:#f0f2f7;
      --pos:#2e7d32;--pos-bg:#e8f5e9;--neg:#c62828;--neg-bg:#fce4ec;
      --zero:#666;--zero-bg:#f0f0f0}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--bg);color:var(--dark);line-height:1.65;font-size:15px}}

/* ── header ── */
.header{{background:linear-gradient(135deg,var(--dark) 0%,#16213e 100%);
         color:#fff;padding:22px 28px 18px}}
.header h1{{font-size:1.45rem;font-weight:700;letter-spacing:-.3px}}
.header h1 em{{color:var(--blue);font-style:normal}}
.stats{{display:flex;gap:16px;margin-top:10px;flex-wrap:wrap}}
.stat{{background:rgba(255,255,255,.1);border-radius:6px;
       padding:4px 12px;font-size:.82rem}}
.stat strong{{color:var(--blue)}}

/* ── progress ── */
.progress-bar{{height:4px;background:rgba(255,255,255,.15)}}
.progress-fill{{height:100%;background:var(--blue);
                transition:width .6s ease;width:{pct}%}}
.round-label{{text-align:right;font-size:.78rem;color:rgba(255,255,255,.6);
              padding:4px 28px 0}}

/* ── question ── */
.question-banner{{background:#e3f2fd;border-left:4px solid var(--blue);
                  margin:20px 24px;padding:12px 16px;border-radius:0 8px 8px 0;
                  font-size:.93rem}}
.q-label{{background:var(--blue);color:#fff;font-weight:700;font-size:.73rem;
          letter-spacing:1px;padding:2px 7px;border-radius:4px;
          margin-right:10px;vertical-align:middle}}
.q-text{{vertical-align:middle}}

/* ── forum ── */
.forum{{max-width:780px;margin:0 auto;padding:12px 20px 48px}}

/* ── cards ── */
.card{{background:var(--card);border-radius:12px;margin-bottom:14px;
       box-shadow:0 2px 8px rgba(0,0,0,.07);overflow:hidden}}
.card-header{{display:flex;align-items:center;gap:9px;
              padding:13px 16px 10px;border-bottom:1px solid #f0f0f0}}
.avatar{{width:30px;height:30px;border-radius:50%;flex-shrink:0;
         background:linear-gradient(135deg,var(--blue),#7b2d8b);
         color:#fff;font-weight:700;font-size:.88rem;
         display:flex;align-items:center;justify-content:center}}
.card-author{{font-weight:600;font-size:.93rem}}
.card-id{{color:#bbb;font-size:.78rem;margin-left:2px}}
.score{{margin-left:auto;font-weight:700;font-size:.88rem;
        padding:2px 10px;border-radius:20px}}
.score-pos{{background:var(--pos-bg);color:var(--pos)}}
.score-neg{{background:var(--neg-bg);color:var(--neg)}}
.score-zero{{background:var(--zero-bg);color:var(--zero)}}
.card-body{{padding:12px 16px;font-size:.9rem;color:#333;
            white-space:pre-wrap;word-break:break-word}}

/* ── comments ── */
.comments{{background:#fafafa;border-top:1px solid #f0f0f0;
           padding:8px 16px 12px}}
.comment{{padding:7px 0 7px 14px;border-left:3px solid #e0e0e0;
          margin:5px 0;font-size:.85rem}}
.comment-meta{{display:flex;align-items:center;gap:7px;margin-bottom:3px}}
.comment-author{{font-weight:600;color:#555}}
.comment-body{{color:#444;white-space:pre-wrap;word-break:break-word}}

/* ── misc ── */
.empty{{text-align:center;color:#aaa;padding:60px 20px;font-size:1rem}}
.footer{{text-align:center;padding:14px;font-size:.78rem;color:#aaa;
         border-top:1px solid #e8e8e8;background:#fff}}
.dot{{display:inline-block;width:7px;height:7px;border-radius:50%;
      background:#4caf50;margin-right:5px;
      animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
</style>
</head>
<body>

<div class="header">
  <h1>🔬 <em>Science Parliament</em></h1>
  <div class="stats">
    <div class="stat"><strong>{data['n_posts']}</strong> posts</div>
    <div class="stat"><strong>{data['n_comments']}</strong> comments</div>
    <div class="stat"><strong>{data['n_votes']}</strong> votes</div>
  </div>
</div>
<div class="progress-bar"><div class="progress-fill"></div></div>
<div class="round-label">{round_label}</div>

{q_html}

<div class="forum">
{cards_html}
</div>

<div class="footer">
  <span class="dot"></span>Auto-refreshes every 8s&nbsp;·&nbsp;Last updated {now}
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_html(
    db_path: str,
    output_dir: str,
    question: str = None,
    current_round: int = 0,
    num_rounds: int = 0,
) -> None:
    """
    Read the parliament database and write index.html to output_dir.

    This function is designed to be called after each round from
    run_parliament.py.  It is always wrapped in try/except by the caller
    so any failure here is silent and never affects the parliament run.
    """
    if not os.path.exists(db_path):
        return

    data = _read_db(db_path)
    html = _build_html(data, question, current_round, num_rounds)

    out_path = os.path.join(output_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# CLI entry point (for manual use / testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python visualize.py <db_path> <output_dir> [question]")
        sys.exit(1)

    _db = sys.argv[1]
    _out = sys.argv[2]
    _q = sys.argv[3] if len(sys.argv) > 3 else None

    generate_html(_db, _out, question=_q)
    print(f"Generated: {_out}/index.html")
