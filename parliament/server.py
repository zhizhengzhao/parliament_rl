"""Parliament HTTP API server.

Every request is logged to interaction_log with full request/response summaries.
Judge users cannot post or comment (only vote/follow/read).
"""

import argparse
import glob
import os
import uuid
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .auth import get_current_user, require_admin, set_store
from .config import (DATA_DIR, HOST, PORT, MAX_LIST_LIMIT,
                     DEFAULT_LIST_LIMIT, LOG_SUMMARY_MAX_LEN)
from .store import Store


def _resolve_db_path(name: str | None) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    if name == "last":
        dirs = sorted(d for d in glob.glob(os.path.join(DATA_DIR, "*/"))
                       if os.path.isfile(os.path.join(d, "parliament.db")))
        if not dirs:
            name = None
        else:
            return os.path.join(dirs[-1], "parliament.db")
    ts = datetime.now().strftime("%m%d_%H%M%S")
    prefix = f"{name}_{ts}" if name else ts
    run_dir = os.path.join(DATA_DIR, prefix)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "parliament.db")


def create_app(name: str | None = None, port: int = PORT) -> tuple:
    db_path = _resolve_db_path(name)
    print(f"Database: {db_path}")

    app = FastAPI(title="Science Parliament", version="3.0")

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    store = Store(db_path)
    set_store(store)

    _static = os.path.join(os.path.dirname(__file__), "static")

    def _log(user: dict | None, session_id: str | None,
             method: str, endpoint: str,
             req_summary: str, resp_summary: str, code: int = 200):
        store.log_interaction(
            user_id=user["user_id"] if user else None,
            user_name=user["name"] if user else "anonymous",
            user_role=user.get("role", "unknown") if user else "anonymous",
            session_id=session_id,
            method=method, endpoint=endpoint,
            request_summary=req_summary,
            response_summary=resp_summary,
            response_code=code,
        )

    def _truncate(text: str, n: int = LOG_SUMMARY_MAX_LEN) -> str:
        return text[:n] + "..." if len(text) > n else text

    # ── Request models ────────────────────────────────────────

    class SessionCreate(BaseModel):
        title: str
        description: str = ""
        reference_solution: str = ""

    class PostCreate(BaseModel):
        content: str = None
        text: str = None
        body: str = None
        message: str = None
        def get_content(self) -> str:
            return self.content or self.text or self.body or self.message or ""

    class CommentCreate(BaseModel):
        content: str = None
        text: str = None
        body: str = None
        message: str = None
        reply_to: int | None = None
        parent_comment_id: int | None = None
        parent_id: int | None = None
        def get_content(self) -> str:
            return self.content or self.text or self.body or self.message or ""
        def get_reply_to(self) -> int | None:
            return self.reply_to or self.parent_comment_id or self.parent_id

    class VoteRequest(BaseModel):
        value: int = None
        vote: int = None
        score: int = None
        def get_value(self) -> int:
            for v in (self.value, self.vote, self.score):
                if v is not None:
                    return v
            return 0

    class SearchRequest(BaseModel):
        query: str = None
        q: str = None
        search: str = None
        keyword: str = None
        def get_query(self) -> str:
            return self.query or self.q or self.search or self.keyword or ""

    class FollowRequest(BaseModel):
        followee_id: int = None
        user_id: int = None
        target_id: int = None
        def get_followee_id(self) -> int | None:
            return self.followee_id or self.user_id or self.target_id

    class LeaveRequest(BaseModel):
        reason: str = ""

    # ── Static ────────────────────────────────────────────────

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_static, "index.html"))

    # ── Admin ─────────────────────────────────────────────────

    @app.get("/admin/timeline/{session_id}")
    def get_timeline(session_id: str, user: dict = Depends(require_admin)):
        return store.get_timeline(session_id)

    @app.get("/admin/sessions")
    def admin_sessions(user: dict = Depends(require_admin)):
        sessions = store.list_sessions()
        for s in sessions:
            s["stats"] = store.session_stats(s["session_id"])
        return sessions

    @app.get("/admin/sessions/{session_id}")
    def admin_session_detail(session_id: str, user: dict = Depends(require_admin)):
        session = store.get_session_with_solution(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        session["stats"] = store.session_stats(session_id)
        return session

    @app.get("/admin/users")
    def admin_users(user: dict = Depends(require_admin)):
        return store.list_users(include_keys=True)

    @app.get("/admin/info")
    def admin_info(user: dict = Depends(require_admin)):
        return {"db_path": db_path, "run_dir": os.path.dirname(db_path)}

    @app.get("/admin/sessions/{session_id}/posts")
    def admin_posts(session_id: str, sort: str = "time",
                    user: dict = Depends(require_admin)):
        return store.list_posts(session_id, sort=sort, limit=9999)

    # ── Sessions ──────────────────────────────────────────────

    @app.post("/sessions")
    def create_session(req: SessionCreate, user: dict = Depends(require_admin)):
        sid = str(uuid.uuid4())[:8]
        result = store.create_session(
            sid, req.title, req.description, req.reference_solution, user["user_id"])
        _log(user, sid, "POST", "/sessions",
             f"title={_truncate(req.title)}", f"session_id={sid}")
        return result

    @app.get("/sessions")
    def list_sessions(user: dict = Depends(get_current_user)):
        if not user.get("is_admin"):
            raise HTTPException(403,
                "You cannot list sessions. You already know your session ID.")
        sessions = store.list_sessions()
        for s in sessions:
            s["stats"] = store.session_stats(s["session_id"])
        return sessions

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str, user: dict = Depends(get_current_user)):
        if not user.get("is_admin"):
            raise HTTPException(403,
                "You cannot view session details. "
                "Use /sessions/{id}/posts to read the discussion.")
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        session["stats"] = store.session_stats(session_id)
        return session

    # ── Posts ──────────────────────────────────────────────────

    @app.get("/sessions/{session_id}/posts")
    def list_posts(session_id: str, sort: str = "time",
                   limit: int = DEFAULT_LIST_LIMIT,
                   user: dict = Depends(get_current_user)):
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        cap = 15 if user.get("role") == "judge" else MAX_LIST_LIMIT
        limit = min(limit, cap)
        posts = store.list_posts(session_id, sort=sort, limit=limit)
        post_ids = [p["post_id"] for p in posts]
        _log(user, session_id, "GET", f"/sessions/{session_id}/posts",
             f"sort={sort} limit={limit}",
             f"{len(posts)} posts, ids={post_ids}")
        return posts

    @app.get("/sessions/{session_id}/posts/{post_id}")
    def get_post(session_id: str, post_id: int,
                 user: dict = Depends(get_current_user)):
        post = store.get_post(post_id)
        if not post or post["session_id"] != session_id:
            raise HTTPException(404, "Post not found")
        comment_ids = [c["comment_id"] for c in post.get("comments", [])]
        _log(user, session_id, "GET", f"/sessions/{session_id}/posts/{post_id}",
             "", f"post_id={post_id}, {len(comment_ids)} comments")
        return post

    @app.post("/sessions/{session_id}/posts")
    def create_post(session_id: str, req: PostCreate,
                    user: dict = Depends(get_current_user)):
        if user.get("role") == "judge":
            raise HTTPException(403, "Judges cannot post")
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        if session["status"] == "closed":
            raise HTTPException(403, "Session is closed")
        content = req.get_content()
        if not content:
            raise HTTPException(400, "content is required")
        result = store.create_post(session_id, user["user_id"], content)
        _log(user, session_id, "POST", f"/sessions/{session_id}/posts",
             f"content={_truncate(content)}",
             f"post_id={result['post_id']}")
        return result

    # ── Comments ──────────────────────────────────────────────

    @app.post("/sessions/{session_id}/posts/{post_id}/comments")
    def create_comment(session_id: str, post_id: int, req: CommentCreate,
                       user: dict = Depends(get_current_user)):
        if user.get("role") == "judge":
            raise HTTPException(403, "Judges cannot comment")
        session = store.get_session(session_id)
        if session and session["status"] == "closed":
            raise HTTPException(403, "Session is closed")
        post = store.get_post(post_id)
        if not post or post["session_id"] != session_id:
            raise HTTPException(404, "Post not found")
        content = req.get_content()
        if not content:
            raise HTTPException(400, "content is required")
        result = store.create_comment(
            post_id, user["user_id"], content, req.get_reply_to())
        _log(user, session_id, "POST",
             f"/sessions/{session_id}/posts/{post_id}/comments",
             f"content={_truncate(content)}, reply_to={req.get_reply_to()}",
             f"comment_id={result['comment_id']}")
        return result

    # ── Votes ─────────────────────────────────────────────────

    @app.post("/sessions/{session_id}/posts/{post_id}/vote")
    def vote_post(session_id: str, post_id: int, req: VoteRequest,
                  user: dict = Depends(get_current_user)):
        value = req.get_value()
        if value not in (1, -1, 0):
            raise HTTPException(400, "value must be +1, -1, or 0")
        post = store.get_post(post_id)
        if not post or post["session_id"] != session_id:
            raise HTTPException(404, "Post not found")
        if post["user_id"] == user["user_id"]:
            raise HTTPException(403, "Cannot vote on your own post")
        result = store.vote_post(post_id, user["user_id"], value)
        _log(user, session_id, "POST",
             f"/sessions/{session_id}/posts/{post_id}/vote",
             f"value={value}",
             f"new_score={result['new_score']}")
        return result

    @app.post("/sessions/{session_id}/comments/{comment_id}/vote")
    def vote_comment(session_id: str, comment_id: int, req: VoteRequest,
                     user: dict = Depends(get_current_user)):
        value = req.get_value()
        if value not in (1, -1, 0):
            raise HTTPException(400, "value must be +1, -1, or 0")
        comment = store.conn.execute(
            "SELECT c.comment_id, c.user_id, p.session_id FROM comments c "
            "JOIN posts p ON c.post_id = p.post_id "
            "WHERE c.comment_id = ?", (comment_id,)
        ).fetchone()
        if not comment or comment["session_id"] != session_id:
            raise HTTPException(404, "Comment not found in this session")
        if comment["user_id"] == user["user_id"]:
            raise HTTPException(403, "Cannot vote on your own comment")
        result = store.vote_comment(comment_id, user["user_id"], value)
        _log(user, session_id, "POST",
             f"/sessions/{session_id}/comments/{comment_id}/vote",
             f"value={value}",
             f"new_score={result['new_score']}")
        return result

    # ── My state ──────────────────────────────────────────────

    @app.get("/sessions/{session_id}/my-state")
    def my_state(session_id: str, user: dict = Depends(get_current_user)):
        votes = store.get_user_votes(session_id, user["user_id"])
        following = list(store.get_following_ids(user["user_id"]))
        _log(user, session_id, "GET", f"/sessions/{session_id}/my-state",
             "", f"votes={len(votes['posts'])+len(votes['comments'])}, following={len(following)}")
        return {"votes": votes, "following": following}

    # ── Search ────────────────────────────────────────────────

    @app.post("/sessions/{session_id}/search")
    def search_posts(session_id: str, req: SearchRequest,
                     user: dict = Depends(get_current_user)):
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        query = req.get_query()
        if not query:
            raise HTTPException(400, "query is required")
        results = store.search_posts(session_id, query, limit=MAX_LIST_LIMIT)
        _log(user, session_id, "POST", f"/sessions/{session_id}/search",
             f"query={_truncate(query)}",
             f"{len(results)} results")
        return results

    # ── Follow ────────────────────────────────────────────────

    @app.post("/follow")
    def follow_user(req: FollowRequest, user: dict = Depends(get_current_user)):
        fid = req.get_followee_id()
        if not fid:
            raise HTTPException(400, "followee_id is required")
        if fid == user["user_id"]:
            raise HTTPException(400, "Cannot follow yourself")
        target = store.get_user(fid)
        if not target:
            raise HTTPException(404, "User not found")
        result = store.follow(user["user_id"], fid)
        _log(user, None, "POST", "/follow",
             f"followee_id={fid}", f"followed {target['name']}")
        return result

    @app.post("/unfollow")
    def unfollow_user(req: FollowRequest, user: dict = Depends(get_current_user)):
        fid = req.get_followee_id()
        if not fid:
            raise HTTPException(400, "followee_id is required")
        store.unfollow(user["user_id"], fid)
        _log(user, None, "POST", "/unfollow",
             f"followee_id={fid}", "unfollowed")
        return {"ok": True}

    # ── Join / Leave / Activity ──────────────────────────────

    @app.post("/sessions/{session_id}/join")
    def join_session(session_id: str, user: dict = Depends(get_current_user)):
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        result = store.join_session(user["user_id"], session_id)
        _log(user, session_id, "POST", f"/sessions/{session_id}/join",
             "", "joined")
        return result

    @app.post("/sessions/{session_id}/leave")
    def leave_session(session_id: str, req: LeaveRequest,
                      user: dict = Depends(get_current_user)):
        result = store.leave_session(user["user_id"], session_id, req.reason)
        _log(user, session_id, "POST", f"/sessions/{session_id}/leave",
             f"reason={_truncate(req.reason)}" if req.reason else "",
             "left session")
        return result

    @app.get("/sessions/{session_id}/activity")
    def session_activity(session_id: str,
                         user: dict = Depends(get_current_user)):
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        activity = store.session_activity(session_id)
        _log(user, session_id, "GET", f"/sessions/{session_id}/activity",
             "", f"active={activity['active_count']}, last={activity['last_activity']}")
        return activity

    @app.get("/sessions/{session_id}/participants")
    def session_participants(session_id: str,
                             user: dict = Depends(get_current_user)):
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        participants = store.get_session_participants(session_id)
        _log(user, session_id, "GET", f"/sessions/{session_id}/participants",
             "", f"{len(participants)} participants")
        return participants

    # ── Users ─────────────────────────────────────────────────

    @app.get("/users")
    def list_users_endpoint(user: dict = Depends(get_current_user)):
        if not user.get("is_admin"):
            raise HTTPException(403, "You cannot list all users.")
        return store.list_users()

    @app.get("/me")
    def me(user: dict = Depends(get_current_user)):
        result = {
            "user_id": user["user_id"],
            "name": user["name"],
            "role": user.get("role", "actor"),
            "bio": user["bio"],
            "is_admin": user.get("is_admin", False),
        }
        if not user.get("is_admin"):
            result["followers"] = store.get_followers(user["user_id"])
            result["following"] = store.get_following(user["user_id"])
        return result

    return app, store, db_path


def main():
    parser = argparse.ArgumentParser(description="Science Parliament")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (creates data/<name>_<timestamp>/ folder)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--seed", action="store_true",
                        help="Create users (actors + judges)")
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--judges", type=int, default=4)
    args = parser.parse_args()

    app, store, db_path = create_app(args.name, args.port)

    if args.seed:
        from .seed import seed_data
        seed_data(store, args.actors, args.judges)

    import uvicorn
    print(f"\nScience Parliament running at http://localhost:{args.port}")
    print(f"API docs: http://localhost:{args.port}/docs")
    print(f"Database: {db_path}\n")
    uvicorn.run(app, host=HOST, port=args.port)


if __name__ == "__main__":
    main()
