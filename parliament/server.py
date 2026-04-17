"""Parliament HTTP API server.

Every request is logged to interaction_log with full request/response
summaries. Judges can vote but cannot post or comment.
"""

import argparse
import json
import os
import uuid
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .auth import get_current_user, require_admin, set_store
from .config import DATA_DIR, HOST, LOG_SUMMARY_MAX_LEN, PORT
from .store import Store

_STATIC = os.path.join(os.path.dirname(__file__), "static")
_TEXT_KEYS = ("content", "text", "body", "message", "answer", "response",
              "solution", "analysis", "submission", "post", "comment",
              "reply", "description", "data")


# ── Body parsing helpers (defensive against client variation) ─

async def _read_body(request: Request) -> dict:
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


def _get_text(body: dict, *extra_keys) -> str:
    for k in (*_TEXT_KEYS, *extra_keys):
        v = body.get(k)
        if v and isinstance(v, str):
            return v.strip()
    return ""


def _get_int(body: dict, *keys) -> int | None:
    for k in keys:
        v = body.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return None


def _truncate(text: str, n: int = LOG_SUMMARY_MAX_LEN) -> str:
    return text[:n] + "..." if len(text) > n else text


def _sid_from_path(path: str) -> str | None:
    parts = path.strip("/").split("/")
    for i, p in enumerate(parts):
        if p == "sessions" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _resolve_db_path(name: str | None, db_dir: str | None) -> str:
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        return os.path.join(db_dir, "parliament.db")
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = os.path.join(DATA_DIR, f"{name}_{ts}" if name else ts)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "parliament.db")


# ── Request models ────────────────────────────────────────

class SessionCreate(BaseModel):
    title: str
    description: str = ""
    reference_solution: str = ""


# ── App factory ───────────────────────────────────────────

def create_app(name: str | None = None, port: int = PORT,
               db_dir: str | None = None) -> tuple[FastAPI, Store, str]:
    db_path = _resolve_db_path(name, db_dir)
    print(f"Database: {db_path}")

    app = FastAPI(title="Science Parliament", version="3.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    store = Store(db_path)
    set_store(store)

    def log(user: dict | None, sid: str | None, method: str, endpoint: str,
            req_summary: str, resp_summary: str, code: int = 200) -> None:
        store.log_interaction(
            user_id=user["user_id"] if user else None,
            user_name=user["name"] if user else "anonymous",
            user_role=user.get("role", "unknown") if user else "anonymous",
            session_id=sid, method=method, endpoint=endpoint,
            request_summary=req_summary, response_summary=resp_summary,
            response_code=code,
        )

    async def _do_vote(target_type: str, target_id: int, session_id: str,
                       request: Request, user: dict) -> dict:
        body = await _read_body(request)
        value = _get_int(body, "value", "vote", "score", "rating")
        if not value or abs(value) > 3:
            raise HTTPException(400, "value must be between -3 and +3 (not 0)")

        if target_type == "post":
            target = store.get_post(target_id)
            owner_id = target["user_id"] if target else None
            in_session = bool(target and target["session_id"] == session_id)
            endpoint = f"/sessions/{session_id}/posts/{target_id}/vote"
        else:
            target = store.comment_meta(target_id)
            owner_id = target["user_id"] if target else None
            in_session = bool(target and target["session_id"] == session_id)
            endpoint = f"/sessions/{session_id}/comments/{target_id}/vote"

        if not in_session:
            raise HTTPException(404, f"{target_type.capitalize()} not found")
        if owner_id == user["user_id"]:
            raise HTTPException(403, f"Cannot vote on your own {target_type}")

        result = store.vote(target_type, target_id, user["user_id"], value)
        log(user, session_id, "POST", endpoint,
            f"value={value}", f"new_score={result['new_score']}")
        return result

    # ── Error-logging middleware ──────────────────────────

    @app.middleware("http")
    async def log_failed_requests(request: Request, call_next):
        response = await call_next(request)
        if response.status_code < 400:
            return response
        path = request.url.path
        if not (path.startswith("/sessions/") or path in ("/me", "/users")):
            return response
        user = None
        try:
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                user = store.get_user_by_key(auth[7:].strip())
        except Exception:
            pass
        log(user, _sid_from_path(path), request.method, path,
            "", f"error {response.status_code}", response.status_code)
        return response

    # ── Static ────────────────────────────────────────────

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_STATIC, "index.html"))

    # ── Admin ─────────────────────────────────────────────

    @app.get("/admin/info")
    def admin_info(_: dict = Depends(require_admin)):
        return {"db_path": db_path, "run_dir": os.path.dirname(db_path)}

    @app.get("/admin/sessions")
    def admin_sessions(_: dict = Depends(require_admin)):
        sessions = store.list_sessions()
        for s in sessions:
            s["stats"] = store.session_stats(s["session_id"])
        return sessions

    @app.get("/admin/sessions/{session_id}")
    def admin_session_detail(session_id: str, _: dict = Depends(require_admin)):
        session = store.get_session(session_id, include_solution=True)
        if not session:
            raise HTTPException(404, "Session not found")
        session["stats"] = store.session_stats(session_id)
        return session

    @app.get("/admin/sessions/{session_id}/posts")
    def admin_posts(session_id: str, sort: str = "time",
                    _: dict = Depends(require_admin)):
        posts = store.get_all_posts(session_id)
        if sort == "score":
            posts.sort(key=lambda p: p.get("score", 0), reverse=True)
        return posts

    @app.get("/admin/sessions/{session_id}/votes")
    def admin_votes(session_id: str, _: dict = Depends(require_admin)):
        return store.get_session_votes(session_id)

    @app.get("/admin/timeline/{session_id}")
    def admin_timeline(session_id: str, _: dict = Depends(require_admin)):
        return store.get_timeline(session_id)

    @app.get("/admin/users")
    def admin_users(_: dict = Depends(require_admin)):
        return store.list_users(include_keys=True)

    # ── Sessions ──────────────────────────────────────────

    @app.post("/sessions")
    def create_session(req: SessionCreate, user: dict = Depends(require_admin)):
        sid = str(uuid.uuid4())[:8]
        result = store.create_session(sid, req.title, req.description,
                                      req.reference_solution, user["user_id"])
        log(user, sid, "POST", "/sessions",
            f"title={_truncate(req.title)}", f"session_id={sid}")
        return result

    @app.post("/sessions/{session_id}/close")
    def close_session(session_id: str, user: dict = Depends(require_admin)):
        if not store.get_session(session_id):
            raise HTTPException(404, "Session not found")
        store.close_session(session_id)
        log(user, session_id, "POST", f"/sessions/{session_id}/close",
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

    # ── Posts ─────────────────────────────────────────────

    @app.get("/sessions/{session_id}/posts/{post_id}")
    def get_post(session_id: str, post_id: int,
                 user: dict = Depends(get_current_user)):
        post = store.get_post(post_id)
        if not post or post["session_id"] != session_id:
            raise HTTPException(404, "Post not found")
        log(user, session_id, "GET",
            f"/sessions/{session_id}/posts/{post_id}",
            "", f"post_id={post_id}, {len(post['comments'])} comments")
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
        body = await _read_body(request)
        content = _get_text(body)
        if not content:
            raise HTTPException(400,
                'content is required. Send: {"content": "your text"}')
        result = store.create_post(session_id, user["user_id"], content)
        log(user, session_id, "POST", f"/sessions/{session_id}/posts",
            f"content={_truncate(content)}", f"post_id={result['post_id']}")
        return result

    # ── Comments ──────────────────────────────────────────

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
        body = await _read_body(request)
        content = _get_text(body)
        if not content:
            raise HTTPException(400,
                'content is required. Send: {"content": "your text"}')
        result = store.create_comment(post_id, user["user_id"], content)
        log(user, session_id, "POST",
            f"/sessions/{session_id}/posts/{post_id}/comments",
            f"content={_truncate(content)}", f"comment_id={result['comment_id']}")
        return result

    # ── Votes ─────────────────────────────────────────────

    @app.post("/sessions/{session_id}/posts/{post_id}/vote")
    async def vote_post(session_id: str, post_id: int, request: Request,
                        user: dict = Depends(get_current_user)):
        return await _do_vote("post", post_id, session_id, request, user)

    @app.post("/sessions/{session_id}/comments/{comment_id}/vote")
    async def vote_comment(session_id: str, comment_id: int, request: Request,
                           user: dict = Depends(get_current_user)):
        return await _do_vote("comment", comment_id, session_id, request, user)

    # ── My state ──────────────────────────────────────────

    @app.get("/sessions/{session_id}/my-state")
    def my_state(session_id: str, user: dict = Depends(get_current_user)):
        votes = store.get_user_votes(session_id, user["user_id"])
        mine = store.get_user_content_ids(session_id, user["user_id"])
        log(user, session_id, "GET", f"/sessions/{session_id}/my-state",
            "", f"votes={len(votes['posts']) + len(votes['comments'])}")
        return {"votes": votes, **mine}

    # ── Join / Leave / Participants ───────────────────────

    @app.post("/sessions/{session_id}/join")
    def join_session(session_id: str, user: dict = Depends(get_current_user)):
        if not store.get_session(session_id):
            raise HTTPException(404, "Session not found")
        result = store.join_session(user["user_id"], session_id)
        log(user, session_id, "POST", f"/sessions/{session_id}/join",
            "", "joined")
        return result

    @app.post("/sessions/{session_id}/leave")
    async def leave_session(session_id: str, request: Request,
                            user: dict = Depends(get_current_user)):
        body = await _read_body(request)
        reason = _get_text(body, "reason") or ""
        result = store.leave_session(user["user_id"], session_id, reason)
        log(user, session_id, "POST", f"/sessions/{session_id}/leave",
            f"reason={_truncate(reason)}" if reason else "", "left session")
        return result

    @app.get("/sessions/{session_id}/participants")
    def session_participants(session_id: str,
                             user: dict = Depends(get_current_user)):
        if not store.get_session(session_id):
            raise HTTPException(404, "Session not found")
        participants = store.get_session_participants(session_id)
        log(user, session_id, "GET",
            f"/sessions/{session_id}/participants",
            "", f"{len(participants)} participants")
        return participants

    # ── Users ─────────────────────────────────────────────

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


# ── CLI entry ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Science Parliament")
    parser.add_argument("--name", help="Run name (creates data/<name>_<ts>/)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--seed", action="store_true",
                        help="Create users (actors + judges)")
    parser.add_argument("--actors", type=int, default=3)
    parser.add_argument("--judges", type=int, default=3)
    parser.add_argument("--db-dir", help="Directory for parliament.db "
                        "(overrides --name)")
    args = parser.parse_args()

    app, store, db_path = create_app(args.name, args.port, db_dir=args.db_dir)

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
