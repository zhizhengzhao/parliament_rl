"""Parliament HTTP API server.

Every request is logged to interaction_log with full request/response summaries.
Judge users cannot post or comment (only vote).
"""

import argparse
import json
import os
import uuid
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .auth import get_current_user, require_admin, set_store
from .config import DATA_DIR, HOST, PORT, LOG_SUMMARY_MAX_LEN
from .store import Store


def _resolve_db_path(name: str | None, db_dir: str | None = None) -> str:
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        return os.path.join(db_dir, "parliament.db")
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    prefix = f"{name}_{ts}" if name else ts
    run_dir = os.path.join(DATA_DIR, prefix)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "parliament.db")


def create_app(name: str | None = None, port: int = PORT,
               db_dir: str | None = None) -> tuple:
    db_path = _resolve_db_path(name, db_dir)
    print(f"Database: {db_path}")

    app = FastAPI(title="Science Parliament", version="3.0")

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    store = Store(db_path)
    set_store(store)

    _static = os.path.join(os.path.dirname(__file__), "static")

    # ── Helpers ────────────────────────────────────────────────

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

    def _sid_from_path(path: str) -> str | None:
        parts = path.strip("/").split("/")
        for i, p in enumerate(parts):
            if p == "sessions" and i + 1 < len(parts):
                return parts[i + 1]
        return None

    async def _body(request: Request) -> dict:
        try:
            return await request.json()
        except Exception:
            pass
        try:
            raw = (await request.body()).decode(errors="replace").strip()
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return {}

    _TEXT_KEYS = ("content", "text", "body", "message", "answer",
                  "response", "solution", "analysis", "submission",
                  "post", "comment", "reply", "description", "data")

    def _get_text(body: dict, *extra_keys) -> str:
        for k in (*_TEXT_KEYS, *extra_keys):
            v = body.get(k)
            if v and isinstance(v, str):
                return v.strip()
        return ""

    def _get_int(body: dict, *keys) -> int | None:
        for k in keys:
            v = body.get(k)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    continue
        return None

    # ── Error-logging middleware ──────────────────────────────

    @app.middleware("http")
    async def log_failed_requests(request: Request, call_next):
        response = await call_next(request)
        if response.status_code < 400:
            return response
        path = request.url.path
        if not (path.startswith("/sessions/") or path in
                ("/me", "/users")):
            return response
        user = None
        try:
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                user = store.get_user_by_key(auth[7:].strip())
        except Exception:
            pass
        store.log_interaction(
            user_id=user["user_id"] if user else None,
            user_name=user["name"] if user else "",
            user_role=user.get("role", "") if user else "",
            session_id=_sid_from_path(path),
            method=request.method, endpoint=path,
            request_summary="",
            response_summary=f"error {response.status_code}",
            response_code=response.status_code,
        )
        return response

    # ── Request models (admin only) ────────────────────────────

    class SessionCreate(BaseModel):
        title: str
        description: str = ""
        reference_solution: str = ""

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
        posts = store.get_all_posts(session_id)
        if sort == "score":
            posts.sort(key=lambda p: p.get("score", 0), reverse=True)
        return posts

    @app.get("/admin/sessions/{session_id}/votes")
    def admin_votes(session_id: str,
                    user: dict = Depends(require_admin)):
        rows = store._fetchall(
            "SELECT v.vote_id, v.user_id, v.post_id, v.comment_id, "
            "v.value, v.previous_value, u.name AS author, u.role "
            "FROM votes v JOIN users u ON v.user_id = u.user_id "
            "LEFT JOIN posts p ON v.post_id = p.post_id "
            "LEFT JOIN comments c ON v.comment_id = c.comment_id "
            "LEFT JOIN posts p2 ON c.post_id = p2.post_id "
            "WHERE COALESCE(p.session_id, p2.session_id) = ? "
            "ORDER BY v.vote_id",
            (session_id,))
        return [dict(r) for r in rows]

    # ── Sessions ──────────────────────────────────────────────

    @app.post("/sessions")
    def create_session(req: SessionCreate, user: dict = Depends(require_admin)):
        sid = str(uuid.uuid4())[:8]
        result = store.create_session(
            sid, req.title, req.description, req.reference_solution, user["user_id"])
        _log(user, sid, "POST", "/sessions",
             f"title={_truncate(req.title)}", f"session_id={sid}")
        return result

    @app.post("/sessions/{session_id}/close")
    def close_session(session_id: str, user: dict = Depends(require_admin)):
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        store.close_session(session_id)
        _log(user, session_id, "POST", f"/sessions/{session_id}/close",
             "", "closed")
        return {"session_id": session_id, "status": "closed"}

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
    async def create_post(session_id: str, request: Request,
                          user: dict = Depends(get_current_user)):
        if user.get("role") == "judge":
            raise HTTPException(403, "Judges cannot post")
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        if session["status"] == "closed":
            raise HTTPException(403, "Session is closed")
        body = await _body(request)
        content = _get_text(body)
        if not content:
            raise HTTPException(400,
                'content is required. Send: {"content": "your text"}')
        result = store.create_post(session_id, user["user_id"], content)
        _log(user, session_id, "POST", f"/sessions/{session_id}/posts",
             f"content={_truncate(content)}",
             f"post_id={result['post_id']}")
        return result

    # ── Comments ──────────────────────────────────────────────

    @app.post("/sessions/{session_id}/posts/{post_id}/comments")
    async def create_comment(session_id: str, post_id: int, request: Request,
                             user: dict = Depends(get_current_user)):
        if user.get("role") == "judge":
            raise HTTPException(403, "Judges cannot comment")
        session = store.get_session(session_id)
        if session and session["status"] == "closed":
            raise HTTPException(403, "Session is closed")
        post = store.get_post(post_id)
        if not post or post["session_id"] != session_id:
            raise HTTPException(404, "Post not found")
        body = await _body(request)
        content = _get_text(body)
        if not content:
            raise HTTPException(400,
                'content is required. Send: {"content": "your text"}')
        result = store.create_comment(post_id, user["user_id"], content)
        _log(user, session_id, "POST",
             f"/sessions/{session_id}/posts/{post_id}/comments",
             f"content={_truncate(content)}",
             f"comment_id={result['comment_id']}")
        return result

    # ── Votes ─────────────────────────────────────────────────

    @app.post("/sessions/{session_id}/posts/{post_id}/vote")
    async def vote_post(session_id: str, post_id: int, request: Request,
                        user: dict = Depends(get_current_user)):
        body = await _body(request)
        value = _get_int(body, "value", "vote", "score", "rating")
        if value not in (1, -1):
            raise HTTPException(400, "value must be +1 or -1")
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
    async def vote_comment(session_id: str, comment_id: int, request: Request,
                           user: dict = Depends(get_current_user)):
        body = await _body(request)
        value = _get_int(body, "value", "vote", "score", "rating")
        if value not in (1, -1):
            raise HTTPException(400, "value must be +1 or -1")
        comment = store._fetchone(
            "SELECT c.comment_id, c.user_id, p.session_id FROM comments c "
            "JOIN posts p ON c.post_id = p.post_id "
            "WHERE c.comment_id = ?", (comment_id,))
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
        mine = store.get_user_content_ids(session_id, user["user_id"])
        _log(user, session_id, "GET", f"/sessions/{session_id}/my-state",
             "", f"votes={len(votes['posts'])+len(votes['comments'])}")
        return {"votes": votes, **mine}

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
    async def leave_session(session_id: str, request: Request,
                            user: dict = Depends(get_current_user)):
        body = await _body(request)
        reason = _get_text(body, "reason") or ""
        result = store.leave_session(user["user_id"], session_id, reason)
        _log(user, session_id, "POST", f"/sessions/{session_id}/leave",
             f"reason={_truncate(reason)}" if reason else "",
             "left session")
        return result

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
        return {
            "user_id": user["user_id"],
            "name": user["name"],
            "role": user.get("role", "actor"),
            "bio": user["bio"],
            "is_admin": user.get("is_admin", False),
        }

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
    parser.add_argument("--db-dir", type=str, default=None,
                        help="Directory for parliament.db (overrides --name)")
    args = parser.parse_args()

    app, store, db_path = create_app(args.name, args.port,
                                      db_dir=args.db_dir)

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
