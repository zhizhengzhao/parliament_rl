"""Parliament SQLite storage layer.

Tables: users, sessions, posts, comments, votes,
session_participants, interaction_log.
"""

from __future__ import annotations

import secrets
import sqlite3
import threading

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    api_key TEXT UNIQUE NOT NULL,
    role TEXT DEFAULT 'actor',
    bio TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    reference_solution TEXT DEFAULT '',
    created_by INTEGER NOT NULL,
    status TEXT DEFAULT 'open',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (created_by) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS posts (
    post_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (post_id) REFERENCES posts(post_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS votes (
    vote_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    post_id INTEGER,
    comment_id INTEGER,
    value INTEGER NOT NULL,
    previous_value INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, post_id),
    UNIQUE(user_id, comment_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (post_id) REFERENCES posts(post_id),
    FOREIGN KEY (comment_id) REFERENCES comments(comment_id)
);

CREATE TABLE IF NOT EXISTS session_participants (
    user_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    joined_at TEXT DEFAULT (datetime('now')),
    left_at TEXT,
    leave_reason TEXT,
    PRIMARY KEY (user_id, session_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS interaction_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now')),
    user_id INTEGER,
    user_name TEXT,
    user_role TEXT,
    session_id TEXT,
    method TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    request_summary TEXT,
    response_summary TEXT,
    response_code INTEGER DEFAULT 200
);
"""


def _generate_key(name: str) -> str:
    return f"sp_{name.lower()}_{secrets.token_hex(8)}"


class Store:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def close(self):
        self.conn.close()

    def _write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def _fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    # ── Interaction log ───────────────────────────────────────

    def log_interaction(self, user_id: int | None, user_name: str,
                        user_role: str, session_id: str | None,
                        method: str, endpoint: str,
                        request_summary: str, response_summary: str,
                        response_code: int = 200) -> None:
        self._write(
            "INSERT INTO interaction_log "
            "(user_id, user_name, user_role, session_id, method, endpoint, "
            "request_summary, response_summary, response_code) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, user_name, user_role, session_id,
             method, endpoint, request_summary, response_summary, response_code),
        )

    def get_timeline(self, session_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT * FROM interaction_log WHERE session_id = ? "
            "ORDER BY log_id ASC", (session_id,))
        return [dict(r) for r in rows]

    # ── Session participation ────────────────────────────────

    def join_session(self, user_id: int, session_id: str) -> dict:
        self._write(
            "INSERT OR REPLACE INTO session_participants "
            "(user_id, session_id, status, joined_at, left_at, leave_reason) "
            "VALUES (?, ?, 'active', datetime('now'), NULL, NULL)",
            (user_id, session_id),
        )
        return {"user_id": user_id, "session_id": session_id, "status": "active"}

    def leave_session(self, user_id: int, session_id: str,
                      reason: str = "") -> dict:
        self._write(
            "UPDATE session_participants SET status = 'left', "
            "left_at = datetime('now'), leave_reason = ? "
            "WHERE user_id = ? AND session_id = ?",
            (reason, user_id, session_id),
        )
        return {"user_id": user_id, "session_id": session_id,
                "status": "left", "reason": reason}

    def get_session_participants(self, session_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT sp.*, u.name, u.role FROM session_participants sp "
            "JOIN users u ON sp.user_id = u.user_id "
            "WHERE sp.session_id = ? ORDER BY sp.joined_at",
            (session_id,))
        return [dict(r) for r in rows]

    # ── Users ─────────────────────────────────────────────────

    def create_user(self, name: str, role: str = "actor", bio: str = "") -> dict:
        key = _generate_key(name)
        self._write(
            "INSERT INTO users (name, api_key, role, bio) VALUES (?, ?, ?, ?)",
            (name, key, role, bio),
        )
        return self.get_user_by_key(key)

    def get_user_by_key(self, api_key: str) -> dict | None:
        row = self._fetchone(
            "SELECT * FROM users WHERE api_key = ?", (api_key,))
        return dict(row) if row else None

    def get_user(self, user_id: int) -> dict | None:
        row = self._fetchone(
            "SELECT * FROM users WHERE user_id = ?", (user_id,))
        return dict(row) if row else None

    def list_users(self, include_keys: bool = False) -> list[dict]:
        if include_keys:
            rows = self._fetchall(
                "SELECT user_id, name, api_key, role, bio, created_at FROM users")
        else:
            rows = self._fetchall(
                "SELECT user_id, name, role, bio, created_at FROM users")
        return [dict(r) for r in rows]

    # ── Sessions ──────────────────────────────────────────────

    def create_session(self, session_id: str, title: str,
                       description: str, reference_solution: str,
                       user_id: int) -> dict:
        self._write(
            "INSERT INTO sessions "
            "(session_id, title, description, reference_solution, created_by) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, title, description, reference_solution, user_id),
        )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT session_id, title, description, status, created_by, created_at "
            "FROM sessions WHERE session_id = ?", (session_id,))
        return dict(row) if row else None

    def get_session_with_solution(self, session_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return dict(row) if row else None

    def close_session(self, session_id: str) -> None:
        self._write(
            "UPDATE sessions SET status = 'closed' WHERE session_id = ?",
            (session_id,),
        )

    def list_sessions(self) -> list[dict]:
        rows = self._fetchall(
            "SELECT session_id, title, status, created_at "
            "FROM sessions ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    def session_stats(self, session_id: str) -> dict:
        n_posts = self._fetchone(
            "SELECT COUNT(*) FROM posts WHERE session_id = ?",
            (session_id,))[0]
        n_comments = self._fetchone(
            "SELECT COUNT(*) FROM comments c JOIN posts p ON c.post_id = p.post_id "
            "WHERE p.session_id = ?", (session_id,))[0]
        n_votes = self._fetchone(
            "SELECT COUNT(*) FROM votes v "
            "LEFT JOIN posts p ON v.post_id = p.post_id "
            "LEFT JOIN comments c ON v.comment_id = c.comment_id "
            "LEFT JOIN posts p2 ON c.post_id = p2.post_id "
            "WHERE COALESCE(p.session_id, p2.session_id) = ?",
            (session_id,))[0]
        return {"posts": n_posts, "comments": n_comments, "votes": n_votes}

    # ── Posts ─────────────────────────────────────────────────

    def create_post(self, session_id: str, user_id: int,
                    content: str) -> dict:
        cur = self._write(
            "INSERT INTO posts (session_id, user_id, content) VALUES (?, ?, ?)",
            (session_id, user_id, content),
        )
        return self.get_post(cur.lastrowid)

    def get_post(self, post_id: int) -> dict | None:
        row = self._fetchone(
            "SELECT p.*, u.name AS author, u.role AS author_role "
            "FROM posts p JOIN users u ON p.user_id = u.user_id "
            "WHERE p.post_id = ?", (post_id,))
        if not row:
            return None
        post = dict(row)
        post["score"] = self._score("post", post_id)
        post["comment_count"] = self._fetchone(
            "SELECT COUNT(*) FROM comments WHERE post_id = ?",
            (post_id,))[0]
        post["comments"] = self._comments(post_id)
        return post

    def get_all_posts(self, session_id: str) -> list[dict]:
        """Get all posts with comments for admin/harness use."""
        rows = self._fetchall(
            "SELECT p.*, u.name AS author, u.role AS author_role "
            "FROM posts p JOIN users u ON p.user_id = u.user_id "
            "WHERE p.session_id = ? ORDER BY p.post_id",
            (session_id,))
        result = []
        for r in rows:
            p = dict(r)
            p["score"] = self._score("post", p["post_id"])
            p["comments"] = self._comments(p["post_id"])
            p["comment_count"] = len(p["comments"])
            result.append(p)
        return result

    def _score(self, target_type: str, target_id: int) -> int:
        col = "post_id" if target_type == "post" else "comment_id"
        row = self._fetchone(
            f"SELECT COALESCE(SUM(v.value), 0) "
            f"FROM votes v WHERE v.{col} = ?",
            (target_id,))
        return row[0]

    def _comments(self, post_id: int) -> list[dict]:
        rows = self._fetchall(
            "SELECT c.*, u.name AS author, u.role AS author_role "
            "FROM comments c JOIN users u ON c.user_id = u.user_id "
            "WHERE c.post_id = ? ORDER BY c.comment_id ASC",
            (post_id,))
        return [{"score": self._score("comment", r["comment_id"]), **dict(r)}
                for r in rows]

    # ── Comments ──────────────────────────────────────────────

    def create_comment(self, post_id: int, user_id: int,
                       content: str) -> dict:
        cur = self._write(
            "INSERT INTO comments (post_id, user_id, content) "
            "VALUES (?, ?, ?)",
            (post_id, user_id, content),
        )
        row = self._fetchone(
            "SELECT c.*, u.name AS author FROM comments c "
            "JOIN users u ON c.user_id = u.user_id WHERE c.comment_id = ?",
            (cur.lastrowid,))
        cm = dict(row)
        cm["score"] = self._score("comment", cm["comment_id"])
        return cm

    # ── Votes ─────────────────────────────────────────────────

    def vote_post(self, post_id: int, user_id: int, value: int) -> dict:
        with self._lock:
            old = self.conn.execute(
                "SELECT value FROM votes WHERE post_id = ? AND user_id = ?",
                (post_id, user_id),
            ).fetchone()
            old_value = old[0] if old else None
            if old is not None:
                self.conn.execute(
                    "DELETE FROM votes WHERE post_id = ? AND user_id = ?",
                    (post_id, user_id))
            self.conn.execute(
                "INSERT INTO votes (post_id, user_id, value, previous_value) "
                "VALUES (?, ?, ?, ?)",
                (post_id, user_id, value, old_value))
            self.conn.commit()
            new_score = self._score("post", post_id)
        return {"post_id": post_id, "value": value,
                "previous_value": old_value,
                "new_score": new_score}

    def vote_comment(self, comment_id: int, user_id: int, value: int) -> dict:
        with self._lock:
            old = self.conn.execute(
                "SELECT value FROM votes WHERE comment_id = ? AND user_id = ?",
                (comment_id, user_id),
            ).fetchone()
            old_value = old[0] if old else None
            if old is not None:
                self.conn.execute(
                    "DELETE FROM votes WHERE comment_id = ? AND user_id = ?",
                    (comment_id, user_id))
            self.conn.execute(
                "INSERT INTO votes (comment_id, user_id, value, previous_value) "
                "VALUES (?, ?, ?, ?)",
                (comment_id, user_id, value, old_value))
            self.conn.commit()
            new_score = self._score("comment", comment_id)
        return {"comment_id": comment_id, "value": value,
                "previous_value": old_value,
                "new_score": new_score}

    # ── User state ────────────────────────────────────────────

    def get_user_votes(self, session_id: str, user_id: int) -> dict:
        post_votes = {}
        for r in self._fetchall(
            "SELECT v.post_id, v.value FROM votes v "
            "JOIN posts p ON v.post_id = p.post_id "
            "WHERE p.session_id = ? AND v.user_id = ? AND v.post_id IS NOT NULL",
            (session_id, user_id)):
            post_votes[r[0]] = r[1]
        comment_votes = {}
        for r in self._fetchall(
            "SELECT v.comment_id, v.value FROM votes v "
            "JOIN comments c ON v.comment_id = c.comment_id "
            "JOIN posts p ON c.post_id = p.post_id "
            "WHERE p.session_id = ? AND v.user_id = ? AND v.comment_id IS NOT NULL",
            (session_id, user_id)):
            comment_votes[r[0]] = r[1]
        return {"posts": post_votes, "comments": comment_votes}

    def get_user_content_ids(self, session_id: str, user_id: int) -> dict:
        my_posts = [r[0] for r in self._fetchall(
            "SELECT post_id FROM posts WHERE session_id = ? AND user_id = ?",
            (session_id, user_id))]
        my_comments = [r[0] for r in self._fetchall(
            "SELECT c.comment_id FROM comments c "
            "JOIN posts p ON c.post_id = p.post_id "
            "WHERE p.session_id = ? AND c.user_id = ?",
            (session_id, user_id))]
        return {"my_posts": my_posts, "my_comments": my_comments}

