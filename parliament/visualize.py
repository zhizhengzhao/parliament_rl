"""
Science Parliament — visualization module.

Reads the SQLite database and writes a self-contained index.html with:
  - Forum view: posts sorted by score, with nested comments
  - Scientists view: ranked profiles with activity indicators
  - Network view: follow relationship graph

Page auto-refreshes every 15 seconds by default (override via serve.py --refresh).
"""

import json
import os
import sqlite3
from datetime import datetime

# Distinct muted colours for scientist avatars (up to 26)
_AVATAR_COLORS = [
    "#6366f1", "#8b5cf6", "#a855f7", "#d946ef", "#ec4899",
    "#f43f5e", "#ef4444", "#f97316", "#eab308", "#84cc16",
    "#22c55e", "#14b8a6", "#06b6d4", "#0ea5e9", "#3b82f6",
    "#6366f1", "#8b5cf6", "#a855f7", "#d946ef", "#ec4899",
    "#f43f5e", "#ef4444", "#f97316", "#eab308", "#84cc16",
    "#22c55e",
]


def _esc(text):
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _read_db(db_path):
    conn = sqlite3.connect(db_path, timeout=3)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT user_id, name, bio FROM user ORDER BY user_id")
    users = [dict(r) for r in c.fetchall()]
    user_map = {u["user_id"]: u["name"] for u in users}

    c.execute("""
        SELECT p.post_id, p.user_id, u.name AS author, p.content,
               (p.num_likes - p.num_dislikes) AS score
        FROM post p JOIN user u ON p.user_id = u.user_id
        ORDER BY score DESC, p.post_id ASC
    """)
    posts = [dict(r) for r in c.fetchall()]

    c.execute("""
        SELECT cm.comment_id, cm.post_id, cm.user_id, u.name AS author,
               cm.content, (cm.num_likes - cm.num_dislikes) AS score
        FROM comment cm JOIN user u ON cm.user_id = u.user_id
        ORDER BY cm.comment_id ASC
    """)
    comments = [dict(r) for r in c.fetchall()]
    comments_by_post = {}
    for cm in comments:
        comments_by_post.setdefault(cm["post_id"], []).append(cm)

    c.execute("""
        SELECT f.follower_id, u1.name AS follower_name,
               f.followee_id, u2.name AS followee_name
        FROM follow f
        JOIN user u1 ON f.follower_id = u1.user_id
        JOIN user u2 ON f.followee_id = u2.user_id
    """)
    follows = [dict(r) for r in c.fetchall()]

    c.execute("SELECT COUNT(*) FROM post")
    n_posts = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM comment")
    n_comments = c.fetchone()[0]
    c.execute("""
        SELECT COUNT(*) FROM trace
        WHERE action IN ('like_post','dislike_post','like_comment','dislike_comment')
    """)
    n_votes = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM follow")
    n_follows = c.fetchone()[0]

    conn.close()
    return dict(
        users=users, user_map=user_map, posts=posts, comments=comments,
        comments_by_post=comments_by_post, follows=follows,
        n_posts=n_posts, n_comments=n_comments, n_votes=n_votes,
        n_follows=n_follows,
    )


def _avatar(name, uid):
    c = _AVATAR_COLORS[uid % len(_AVATAR_COLORS)]
    letter = (name or "?")[0].upper()
    return f'<span class="av" style="background:{c}">{letter}</span>'


def _score_badge(score):
    score = score or 0
    if score > 0:
        return f'<span class="badge pos">+{score}</span>'
    elif score < 0:
        return f'<span class="badge neg">{score}</span>'
    return f'<span class="badge zero">0</span>'


def _forum_html(data):
    h = ""
    for post in data["posts"]:
        pid, uid = post["post_id"], post["user_id"]
        author = _esc(post["author"] or "?")
        content = _esc(post["content"] or "")
        cmts = data["comments_by_post"].get(pid, [])
        n_cmts = len(cmts)

        cmts_html = ""
        for cm in cmts:
            ca = _esc(cm["author"] or "?")
            cc = _esc(cm["content"] or "")
            cmts_html += f"""<div class="cmt">
  <div class="cmt-head">{_avatar(cm['author'], cm['user_id'])}
    <a class="name" href="#" onclick="showProfile({cm['user_id']});return false">{ca}</a>
    {_score_badge(cm['score'])}</div>
  <div class="cmt-body">{cc}</div></div>"""

        cmt_section = f'<div class="cmt-section"><div class="cmt-toggle" onclick="this.parentElement.classList.toggle(\'open\')">{n_cmts} comment{"s" if n_cmts!=1 else ""}</div><div class="cmt-list">{cmts_html}</div></div>' if cmts_html else ""

        h += f"""<article class="post">
  <div class="post-head">
    {_avatar(post['author'], uid)}
    <a class="name" href="#" onclick="showProfile({uid});return false">{author}</a>
    <span class="pid">#{pid}</span>
    <div class="post-head-right">{_score_badge(post['score'])}</div>
  </div>
  <div class="post-body"><div class="post-text">{content}</div><div class="post-expand" onclick="togglePost(this)"><span></span></div></div>
  {cmt_section}
</article>"""

    return h or '<div class="empty-msg">The parliament is about to begin...</div>'


def _scientists_html(data):
    posts_by_user = {}
    for p in data["posts"]:
        posts_by_user.setdefault(p["user_id"], []).append(p)
    comments_by_user = {}
    for cm in data["comments"]:
        comments_by_user.setdefault(cm["user_id"], []).append(cm)
    followers_map = {}
    following_map = {}
    for f in data["follows"]:
        following_map.setdefault(f["follower_id"], []).append(f["followee_name"])
        followers_map.setdefault(f["followee_id"], []).append(f["follower_name"])

    entries = []
    for u in data["users"]:
        uid = u["user_id"]
        u_posts = posts_by_user.get(uid, [])
        total_score = sum(p.get("score", 0) or 0 for p in u_posts)
        entries.append((total_score, uid, u, u_posts, comments_by_user.get(uid, []),
                         following_map.get(uid, []), followers_map.get(uid, [])))
    entries.sort(key=lambda x: -x[0])

    h = ""
    for total_score, uid, u, u_posts, u_cmts, u_following, u_followers in entries:
        name = _esc(u["name"] or "?")
        bar_w = min(max(total_score * 4, 0), 200)
        h += f"""<div class="sci" onclick="showProfile({uid})">
  <div class="sci-left">{_avatar(u['name'], uid)}
    <div><div class="sci-name">{name}</div>
      <div class="sci-nums">{len(u_posts)} posts · {len(u_cmts)} comments · {len(u_followers)} followers</div></div></div>
  <div class="sci-right"><div class="sci-bar-wrap"><div class="sci-bar" style="width:{bar_w}px"></div></div>
    {_score_badge(total_score)}</div></div>"""

    return h or '<div class="empty-msg">No scientists yet.</div>'


def _network_html(data):
    if not data["follows"]:
        return '<div class="empty-msg">No follow relationships yet.</div>'
    following_map = {}
    for f in data["follows"]:
        following_map.setdefault(f["follower_name"], []).append(f["followee_name"])

    h = '<div class="net-grid">'
    for name in sorted(following_map):
        targets = following_map[name]
        tags = "".join(f'<span class="net-tag">{_esc(t)}</span>' for t in sorted(targets))
        h += f'<div class="net-row"><div class="net-from">{_esc(name)}</div><div class="net-arrow">→</div><div class="net-to">{tags}</div></div>'
    h += "</div>"
    return h


def _profile_data(data):
    posts_by_user = {}
    for p in data["posts"]:
        posts_by_user.setdefault(p["user_id"], []).append(p)
    comments_by_user = {}
    for cm in data["comments"]:
        comments_by_user.setdefault(cm["user_id"], []).append(cm)
    following_map = {}
    followers_map = {}
    for f in data["follows"]:
        following_map.setdefault(f["follower_id"], []).append(f["followee_name"])
        followers_map.setdefault(f["followee_id"], []).append(f["follower_name"])

    profiles = {}
    for u in data["users"]:
        uid = u["user_id"]
        profiles[uid] = dict(
            name=u["name"], color=_AVATAR_COLORS[uid % len(_AVATAR_COLORS)],
            posts=[dict(id=p["post_id"], text=p["content"] or "", sc=p.get("score",0) or 0) for p in posts_by_user.get(uid,[])],
            comments=[dict(pid=c["post_id"], text=c["content"] or "", sc=c.get("score",0) or 0) for c in comments_by_user.get(uid,[])],
            following=following_map.get(uid,[]), followers=followers_map.get(uid,[]),
        )
    return profiles


def _build_html(data, question, current_round, num_rounds):
    q_html = ""
    if question:
        q_html = f'<div class="q-banner"><span class="q-chip">PROBLEM</span>{_esc(question)}</div>'

    round_label = f"Round {current_round}/{num_rounds}" if num_rounds > 0 else "Opening..."
    pct = int(current_round / num_rounds * 100) if num_rounds > 0 else 0
    now = datetime.now().strftime("%H:%M:%S")

    forum = _forum_html(data)
    scientists = _scientists_html(data)
    network = _network_html(data)
    pjson = json.dumps(_profile_data(data), ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta http-equiv="refresh" content="15">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Science Parliament</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[
    {{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}},
    {{left:'\\\\(',right:'\\\\)',display:false}},{{left:'\\\\[',right:'\\\\]',display:true}}
  ],throwOnError:false}})"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#f4f5f9;--card:#fff;--border:#e8e9f0;--text:#1e1e2e;--muted:#8b8da0;
  --accent:#6366f1;--accent2:#818cf8;--green:#16a34a;--green-bg:#dcfce7;
  --red:#dc2626;--red-bg:#fee2e2;--gray-bg:#f1f2f6}}
body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--text);line-height:1.7;font-size:14px}}
a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}

/* header */
.hdr{{background:linear-gradient(135deg,#1e1b4b 0%,#312e81 100%);color:#fff;padding:24px 32px 20px}}
.hdr h1{{font-size:1.35rem;font-weight:800;letter-spacing:-.5px}}
.hdr h1 span{{color:var(--accent2);font-weight:400}}
.hdr-stats{{display:flex;gap:18px;margin-top:10px;flex-wrap:wrap}}
.hdr-s{{font-size:.8rem;opacity:.75}}
.hdr-s b{{color:var(--accent2);opacity:1}}
.prog{{height:3px;background:rgba(255,255,255,.1)}}
.prog-fill{{height:100%;background:var(--accent2);transition:width .5s;width:{pct}%}}
.round{{text-align:right;font-size:.72rem;color:rgba(255,255,255,.4);padding:3px 32px 0}}

/* question */
.q-banner{{background:#eef2ff;border-left:4px solid var(--accent);margin:16px 28px 0;
  padding:14px 18px;border-radius:0 10px 10px 0;font-size:.88rem;line-height:1.6}}
.q-chip{{background:var(--accent);color:#fff;font-size:.65rem;font-weight:700;
  padding:2px 8px;border-radius:4px;margin-right:10px;letter-spacing:.8px;vertical-align:middle}}

/* tabs */
.tabs{{display:flex;gap:0;margin:18px 28px 0;border-bottom:2px solid var(--border)}}
.tab{{padding:10px 24px;cursor:pointer;font-weight:600;font-size:.85rem;
  color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-2px;transition:.2s}}
.tab:hover{{color:var(--text)}}.tab.on{{color:var(--accent);border-color:var(--accent)}}
.pane{{display:none;max-width:880px;margin:0 auto;padding:14px 24px 60px}}.pane.on{{display:block}}

/* avatar */
.av{{display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;
  border-radius:50%;color:#fff;font-weight:700;font-size:.82rem;flex-shrink:0}}

/* badge */
.badge{{font-weight:700;font-size:.78rem;padding:2px 10px;border-radius:20px;white-space:nowrap}}
.pos{{background:var(--green-bg);color:var(--green)}}.neg{{background:var(--red-bg);color:var(--red)}}
.zero{{background:var(--gray-bg);color:var(--muted)}}

/* posts */
.post{{background:var(--card);border-radius:14px;margin-bottom:14px;
  border:1px solid var(--border);overflow:hidden;transition:box-shadow .2s}}
.post:hover{{box-shadow:0 4px 20px rgba(0,0,0,.06)}}
.post-head{{display:flex;align-items:center;gap:10px;padding:14px 18px 10px}}
.name{{font-weight:700;font-size:.9rem;color:var(--accent);cursor:pointer}}
.pid{{color:var(--muted);font-size:.75rem}}
.post-head-right{{margin-left:auto;display:flex;align-items:center;gap:8px}}
.post-body{{padding:4px 18px 14px;position:relative}}
.post-text{{font-size:.88rem;white-space:pre-wrap;word-break:break-word;
  max-height:200px;overflow:hidden;transition:max-height .3s ease}}
.post-text.expanded{{max-height:none}}
.post-expand{{display:none;position:absolute;bottom:0;left:0;right:0;height:60px;
  background:linear-gradient(transparent,var(--card));cursor:pointer;
  align-items:flex-end;justify-content:center;padding-bottom:6px}}
.post-expand span{{font-size:.75rem;color:var(--accent);font-weight:600;
  background:var(--card);padding:2px 14px;border-radius:12px;
  border:1px solid var(--border)}}
.post-body.needs-expand .post-expand{{display:flex}}
.post-body.needs-expand .post-text.expanded + .post-expand{{
  position:relative;height:auto;background:none;padding:6px 0 0}}
.post-body.needs-expand .post-text.expanded + .post-expand span::before{{content:'Collapse ▲'}}
.post-expand span::before{{content:'Expand ▼'}}

/* comments */
.cmt-section{{border-top:1px solid var(--border)}}
.cmt-toggle{{padding:8px 18px;font-size:.78rem;color:var(--muted);cursor:pointer;
  font-weight:600;user-select:none}}.cmt-toggle:hover{{color:var(--accent)}}
.cmt-list{{display:none;padding:0 18px 12px}}.cmt-section.open .cmt-list{{display:block}}
.cmt{{padding:8px 0 8px 12px;border-left:3px solid var(--border);margin-bottom:6px}}
.cmt-head{{display:flex;align-items:center;gap:8px;margin-bottom:2px}}
.cmt-body{{font-size:.82rem;color:#444;white-space:pre-wrap;word-break:break-word;
  padding-left:40px;max-height:150px;overflow:hidden;position:relative;
  transition:max-height .3s ease;cursor:pointer}}
.cmt-body.expanded{{max-height:none}}
.cmt-body.clipped::after{{content:'▼ click to expand';position:absolute;bottom:0;left:0;right:0;
  height:40px;background:linear-gradient(transparent,var(--card));display:flex;
  align-items:flex-end;justify-content:center;font-size:.7rem;color:var(--accent);
  font-weight:600;padding-bottom:4px}}
.cmt-body.expanded::after{{display:none}}

/* scientists */
.sci{{background:var(--card);border-radius:12px;margin-bottom:10px;padding:16px 20px;
  border:1px solid var(--border);cursor:pointer;display:flex;align-items:center;
  justify-content:space-between;transition:box-shadow .2s}}
.sci:hover{{box-shadow:0 4px 16px rgba(0,0,0,.08)}}
.sci-left{{display:flex;align-items:center;gap:12px}}
.sci-name{{font-weight:700;font-size:.95rem}}
.sci-nums{{font-size:.78rem;color:var(--muted);margin-top:2px}}
.sci-right{{display:flex;align-items:center;gap:12px}}
.sci-bar-wrap{{width:80px;height:6px;background:var(--gray-bg);border-radius:3px;overflow:hidden}}
.sci-bar{{height:100%;background:var(--accent);border-radius:3px;transition:width .3s}}

/* network */
.net-grid{{display:flex;flex-direction:column;gap:8px}}
.net-row{{background:var(--card);border-radius:10px;padding:12px 18px;
  border:1px solid var(--border);display:flex;align-items:center;gap:12px}}
.net-from{{font-weight:700;min-width:90px;font-size:.9rem}}
.net-arrow{{color:var(--muted);font-size:1.1rem}}
.net-to{{display:flex;flex-wrap:wrap;gap:6px}}
.net-tag{{background:#eef2ff;color:var(--accent);padding:3px 12px;border-radius:20px;
  font-size:.78rem;font-weight:600}}

/* modal */
.overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:200;
  justify-content:center;align-items:flex-start;padding-top:50px;overflow-y:auto}}
.overlay.open{{display:flex}}
.modal{{background:#fff;border-radius:18px;width:92%;max-width:680px;padding:32px;
  position:relative;box-shadow:0 20px 60px rgba(0,0,0,.2);margin-bottom:60px}}
.modal-x{{position:absolute;top:16px;right:20px;font-size:1.4rem;cursor:pointer;
  color:var(--muted);background:none;border:none;line-height:1}}.modal-x:hover{{color:var(--text)}}
.modal h2{{font-size:1.15rem;display:flex;align-items:center;gap:10px}}
.sec{{font-weight:700;font-size:.72rem;color:var(--accent);margin-top:20px;margin-bottom:6px;
  text-transform:uppercase;letter-spacing:.6px}}
.chip{{display:inline-block;background:#eef2ff;color:var(--accent);padding:3px 12px;
  border-radius:20px;font-size:.78rem;font-weight:600;margin:3px 4px 3px 0}}
.mcard{{background:var(--bg);border-radius:10px;padding:12px 16px;margin-bottom:8px;font-size:.84rem}}
.mcard-h{{display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:.78rem;color:var(--muted)}}
.mcard-t{{white-space:pre-wrap;word-break:break-word}}
.empty-msg{{text-align:center;color:var(--muted);padding:60px 20px;font-size:.95rem}}

.foot{{text-align:center;padding:16px;font-size:.72rem;color:var(--muted);
  border-top:1px solid var(--border);background:var(--card)}}
.dot{{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--green);
  margin-right:5px;animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.25}}}}
</style></head><body>

<div class="hdr">
  <h1>🔬 Science <span>Parliament</span></h1>
  <div class="hdr-stats">
    <span class="hdr-s"><b>{data['n_posts']}</b> posts</span>
    <span class="hdr-s"><b>{data['n_comments']}</b> comments</span>
    <span class="hdr-s"><b>{data['n_votes']}</b> votes</span>
    <span class="hdr-s"><b>{data['n_follows']}</b> follows</span>
    <span class="hdr-s"><b>{len(data['users'])}</b> scientists</span>
  </div>
</div>
<div class="prog"><div class="prog-fill"></div></div>
<div class="round">{round_label}</div>

{q_html}

<div class="tabs">
  <div class="tab on" onclick="sw('forum',this)">Forum</div>
  <div class="tab" onclick="sw('sci',this)">Scientists</div>
  <div class="tab" onclick="sw('net',this)">Network</div>
</div>

<div id="p-forum" class="pane on">{forum}</div>
<div id="p-sci" class="pane">{scientists}</div>
<div id="p-net" class="pane">{network}</div>

<div class="overlay" id="ov" onclick="if(event.target===this)cl()">
  <div class="modal"><button class="modal-x" onclick="cl()">&times;</button>
    <div id="mc"></div></div></div>

<div class="foot"><span class="dot"></span>Auto-refreshes every 15s · {now}</div>

<script>
const P={pjson};
function sw(id,el){{document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  document.querySelectorAll('.pane').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');document.getElementById('p-'+id).classList.add('on')}}
function e(s){{const d=document.createElement('div');d.textContent=s||'';return d.innerHTML}}
function b(sc){{if(sc>0)return'<span class="badge pos">+'+sc+'</span>';
  if(sc<0)return'<span class="badge neg">'+sc+'</span>';return'<span class="badge zero">0</span>'}}
function showProfile(uid){{const p=P[uid];if(!p)return;
  let h='<h2><span class="av" style="background:'+p.color+'">'+e(p.name)[0]+'</span>'+e(p.name)+'</h2>';
  h+='<div class="sec">Following ('+p.following.length+')</div>';
  h+=p.following.length?p.following.map(n=>'<span class="chip">'+e(n)+'</span>').join(''):'<span style="color:var(--muted);font-size:.84rem">—</span>';
  h+='<div class="sec">Followers ('+p.followers.length+')</div>';
  h+=p.followers.length?p.followers.map(n=>'<span class="chip">'+e(n)+'</span>').join(''):'<span style="color:var(--muted);font-size:.84rem">—</span>';
  h+='<div class="sec">Posts ('+p.posts.length+')</div>';
  p.posts.forEach(x=>{{h+='<div class="mcard"><div class="mcard-h"><span>#'+x.id+'</span>'+b(x.sc)+'</div><div class="mcard-t">'+e(x.text)+'</div></div>'}});
  if(!p.posts.length)h+='<span style="color:var(--muted);font-size:.84rem">No posts yet</span>';
  h+='<div class="sec">Comments ('+p.comments.length+')</div>';
  p.comments.forEach(x=>{{h+='<div class="mcard"><div class="mcard-h"><span>on #'+x.pid+'</span>'+b(x.sc)+'</div><div class="mcard-t">'+e(x.text)+'</div></div>'}});
  if(!p.comments.length)h+='<span style="color:var(--muted);font-size:.84rem">No comments yet</span>';
  document.getElementById('mc').innerHTML=h;document.getElementById('ov').classList.add('open')}}
function cl(){{document.getElementById('ov').classList.remove('open')}}
document.addEventListener('keydown',ev=>{{if(ev.key==='Escape')cl()}});
function togglePost(btn){{const txt=btn.previousElementSibling;
  txt.classList.toggle('expanded')}}
document.addEventListener('DOMContentLoaded',()=>{{
  document.querySelectorAll('.post-text').forEach(el=>{{
    if(el.scrollHeight>210)el.parentElement.classList.add('needs-expand')}});
  document.querySelectorAll('.cmt-body').forEach(el=>{{
    if(el.scrollHeight>160){{el.classList.add('clipped');
      el.onclick=()=>el.classList.toggle('expanded')}}
  }})
}});
</script></body></html>"""


def generate_html(db_path, output_dir, question=None, current_round=0, num_rounds=0):
    if not os.path.exists(db_path):
        return
    data = _read_db(db_path)
    html = _build_html(data, question, current_round, num_rounds)
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <db_path> <output_dir> [question]")
        sys.exit(1)
    generate_html(sys.argv[1], sys.argv[2], question=sys.argv[3] if len(sys.argv) > 3 else None)
    print(f"Generated: {sys.argv[2]}/index.html")
