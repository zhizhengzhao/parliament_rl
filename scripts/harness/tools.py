"""Tool definitions and execution for Parliament agents.

Actor tools: python_exec, vote, submit, wait
Judge tools: python_exec, vote

- python_exec: run code, does NOT end round
- vote: cast votes, does NOT end round (actor) / ENDS round (judge)
- submit: post and/or comments, ENDS round (actor only)
- wait: wait for new content, ENDS round (actor only)

Session ID and API key are injected automatically; the LLM never sees them.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import aiohttp

# ── Tool definitions (OpenAI function-calling format) ─────

_PYTHON_EXEC = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Run Python code for calculations or verification. "
            "Returns stdout. Does NOT end your round."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
            },
            "required": ["code"],
        },
    },
}

_ACTOR_VOTE_PARAMS = {
    "type": "object",
    "properties": {
        "votes": {
            "type": "array",
            "description": (
                "Votes on posts (P_xxx) or comments (C_xxx). "
                "+1 correct/advances, -1 error/redundant."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "target_type": {
                        "type": "string", "enum": ["post", "comment"],
                    },
                    "target_id": {
                        "type": "integer",
                        "description": "Number from P_xxx or C_xxx",
                    },
                    "value": {"type": "integer", "enum": [1, -1]},
                },
                "required": ["target_type", "target_id", "value"],
            },
        },
    },
    "required": ["votes"],
}

_JUDGE_VOTE_PARAMS = {
    "type": "object",
    "properties": {
        "votes": {
            "type": "array",
            "description": (
                "Votes on posts (P_xxx) or comments (C_xxx). "
                "Score from -3 (fundamentally wrong) to +3 (excellent progress)."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "target_type": {
                        "type": "string", "enum": ["post", "comment"],
                    },
                    "target_id": {
                        "type": "integer",
                        "description": "Number from P_xxx or C_xxx",
                    },
                    "value": {
                        "type": "integer",
                        "enum": [-3, -2, -1, 1, 2, 3],
                        "description": (
                            "+3 matches solution, +2 substantial progress, "
                            "+1 genuine non-trivial advance, "
                            "-1 trivial/repetitive/minor error, "
                            "-2 misleading, -3 fundamentally wrong"
                        ),
                    },
                },
                "required": ["target_type", "target_id", "value"],
            },
        },
    },
    "required": ["votes"],
}

ACTOR_TOOLS = [
    _PYTHON_EXEC,
    {
        "type": "function",
        "function": {
            "name": "vote",
            "description": (
                "Cast votes on posts or comments. "
                "Does NOT end your round — vote first, then submit or wait."
            ),
            "parameters": _ACTOR_VOTE_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit your contributions for this round: "
                "a new post and/or comments on existing posts. "
                "This ENDS your turn — you will see new content next round."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "post": {
                        "type": "string",
                        "description": (
                            "Your new post content. "
                            "Omit if you only want to comment."
                        ),
                    },
                    "comments": {
                        "type": "array",
                        "description": (
                            "Comments on existing posts. "
                            "Use the post_id number from P_xxx."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "post_id": {
                                    "type": "integer",
                                    "description": "The number from P_xxx",
                                },
                                "content": {"type": "string"},
                            },
                            "required": ["post_id", "content"],
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": (
                "Wait for new posts or comments from other scientists "
                "before contributing. ENDS your turn."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

JUDGE_TOOLS = [
    _PYTHON_EXEC,
    {
        "type": "function",
        "function": {
            "name": "vote",
            "description": (
                "Submit your evaluations: cast votes (-3 to +3) on "
                "posts and comments. This ENDS your turn."
            ),
            "parameters": _JUDGE_VOTE_PARAMS,
        },
    },
]


def get_tools(role: str) -> list[dict]:
    return JUDGE_TOOLS if role == "judge" else ACTOR_TOOLS


# ── Session-level ID mapping ─────────────────────────────

class IdMap:
    """Maps global DB IDs to session-local sequential IDs.

    Shared by all agents in a session so everyone sees the same
    P_1, P_2, P_3... for the same posts.
    """

    def __init__(self):
        self._post_g2l: dict[int, int] = {}
        self._post_l2g: dict[int, int] = {}
        self._comment_g2l: dict[int, int] = {}
        self._comment_l2g: dict[int, int] = {}
        self._next_post: int = 1
        self._next_comment: int = 1

    def map_post(self, global_id: int) -> int:
        if global_id not in self._post_g2l:
            local = self._next_post
            self._next_post += 1
            self._post_g2l[global_id] = local
            self._post_l2g[local] = global_id
        return self._post_g2l[global_id]

    def map_comment(self, global_id: int) -> int:
        if global_id not in self._comment_g2l:
            local = self._next_comment
            self._next_comment += 1
            self._comment_g2l[global_id] = local
            self._comment_l2g[local] = global_id
        return self._comment_g2l[global_id]

    def resolve_post(self, local_id: int) -> int | None:
        return self._post_l2g.get(local_id)

    def resolve_comment(self, local_id: int) -> int | None:
        return self._comment_l2g.get(local_id)

    def localize_content(self, items: list[dict]) -> None:
        """Convert global IDs to session-local IDs in-place."""
        for item in items:
            if item["type"] == "post":
                item["id"] = self.map_post(item["id"])
            elif item["type"] == "comment":
                item["id"] = self.map_comment(item["id"])
                item["post_id"] = self.map_post(item["post_id"])
            elif item["type"] == "vote":
                if item["target_type"] == "post":
                    item["target_id"] = self.map_post(item["target_id"])
                else:
                    item["target_id"] = self.map_comment(item["target_id"])


# ── Tool executor ─────────────────────────────────────────

class ToolExecutor:
    """Executes tool calls against Parliament API."""

    def __init__(self, parliament_url: str, session_id: str,
                 api_key: str, http: aiohttp.ClientSession,
                 id_map: IdMap, role: str = "actor"):
        self.base = parliament_url
        self.sid = session_id
        self.key = api_key
        self.http = http
        self.id_map = id_map
        self._role = role
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._my_posts: set[int] = set()
        self._my_comments: set[int] = set()

    async def _api(self, method: str, path: str,
                   body: dict | None = None) -> dict | list | str:
        url = f"{self.base}{path}"
        kwargs: dict[str, Any] = {"headers": self._headers}
        if body is not None:
            kwargs["json"] = body
        async with self.http.request(method, url, **kwargs) as resp:
            text = await resp.text()
            if resp.status >= 400:
                return f"Error {resp.status}: {text[:300]}"
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

    @staticmethod
    def _to_int(v) -> int | None:
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            try:
                return int(v.strip().lstrip("+"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_str(v) -> str:
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        if v is not None:
            return str(v).strip()
        return ""

    async def execute_votes(self, votes) -> dict:
        """Execute vote calls. Handles JSON-in-string from LLM."""
        results: dict[str, list] = {"votes": [], "errors": [], "skipped": []}

        if isinstance(votes, str):
            try:
                votes = json.loads(votes)
            except (json.JSONDecodeError, ValueError):
                results["errors"].append(f"could not parse votes: {votes[:200]}")
                return results
        if not isinstance(votes, list):
            votes = [votes] if isinstance(votes, dict) else []
        if not votes:
            return results

        for v in votes:
            if not isinstance(v, dict):
                results["errors"].append(f"invalid vote format: {str(v)[:100]}")
                continue

            ttype = v.get("target_type", "post")
            tid = self._to_int(v.get("target_id"))
            value = self._to_int(v.get("value"))

            if tid is None or value is None:
                results["errors"].append(f"missing target_id or value: {v}")
                continue
            max_val = 1 if self._role == "actor" else 3
            if not value or abs(value) > max_val:
                allowed = "±1" if self._role == "actor" else "±1 to ±3"
                results["errors"].append(
                    f"vote value must be {allowed}, got {value}")
                continue

            own = (ttype == "post" and tid in self._my_posts) or \
                  (ttype == "comment" and tid in self._my_comments)
            if own:
                results["skipped"].append(
                    f"Cannot vote on your own {ttype} {ttype[0].upper()}_{tid}")
                continue

            if ttype == "comment":
                global_tid = self.id_map.resolve_comment(tid)
                if global_tid is None:
                    results["errors"].append(f"unknown comment C_{tid}")
                    continue
                path = f"/sessions/{self.sid}/comments/{global_tid}/vote"
            else:
                global_tid = self.id_map.resolve_post(tid)
                if global_tid is None:
                    results["errors"].append(f"unknown post P_{tid}")
                    continue
                path = f"/sessions/{self.sid}/posts/{global_tid}/vote"
            resp = await self._api("POST", path, {"value": value})
            if isinstance(resp, dict):
                results["votes"].append(
                    {"id": tid, "type": ttype, "value": value})
            else:
                results["errors"].append(f"vote {ttype}#{tid}: {resp}")
        return results

    async def execute_submit(self, args: dict) -> dict:
        """Execute submit (post + comments). Handles comment/comments and JSON-in-string."""
        if not isinstance(args, dict):
            return {"post_id": None, "comments": [],
                    "errors": [f"invalid arguments type: {type(args).__name__}"]}
        results: dict = {"post_id": None, "comments": [], "errors": []}

        post_content = self._to_str(args.get("post", ""))
        if post_content:
            resp = await self._api("POST", f"/sessions/{self.sid}/posts",
                                   {"content": post_content})
            if isinstance(resp, dict) and "post_id" in resp:
                local_pid = self.id_map.map_post(resp["post_id"])
                results["post_id"] = local_pid
                self._my_posts.add(local_pid)
            else:
                results["errors"].append(f"post: {resp}")

        comments_raw = (args.get("comments")
                        or args.get("comment")
                        or [])

        if isinstance(comments_raw, str):
            try:
                comments_raw = json.loads(comments_raw)
            except (json.JSONDecodeError, ValueError):
                results["errors"].append(
                    f"comment needs post_id: \"{comments_raw[:80]}\" — "
                    f"use {{\"post_id\": N, \"content\": \"...\"}}")
                comments_raw = []
        if not isinstance(comments_raw, list):
            comments_raw = [comments_raw]

        for cm in comments_raw:
            if not isinstance(cm, dict):
                results["errors"].append(
                    f"invalid comment format: {str(cm)[:100]} — "
                    f"use {{\"post_id\": N, \"content\": \"...\"}}")
                continue

            local_pid = self._to_int(cm.get("post_id"))
            content = self._to_str(cm.get("content", ""))

            if not local_pid:
                results["errors"].append(
                    f"comment missing post_id: {str(cm)[:100]}")
                continue
            if not content:
                results["errors"].append(
                    f"comment on P_{local_pid} has empty content")
                continue

            global_pid = self.id_map.resolve_post(local_pid)
            if global_pid is None:
                results["errors"].append(f"unknown post P_{local_pid}")
                continue

            resp = await self._api(
                "POST", f"/sessions/{self.sid}/posts/{global_pid}/comments",
                {"content": content})
            if isinstance(resp, dict) and "comment_id" in resp:
                local_cid = self.id_map.map_comment(resp["comment_id"])
                results["comments"].append(local_cid)
                self._my_comments.add(local_cid)
            else:
                results["errors"].append(f"comment on P_{local_pid}: {resp}")

        if not results["post_id"] and not results["comments"] and not results["errors"]:
            results["errors"].append(
                "Nothing submitted. Include a post or comments to contribute.")

        return results

    async def join(self) -> dict:
        return await self._api("POST", f"/sessions/{self.sid}/join")

    async def leave(self, reason: str = "") -> dict:
        return await self._api("POST", f"/sessions/{self.sid}/leave",
                               {"reason": reason})

    @staticmethod
    async def python_exec(code: str, timeout: float = 10) -> str:
        if not code:
            return "Error: code is required"
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return "Error: execution timed out (10s limit)"
            output = stdout.decode(errors="replace").strip()
            if proc.returncode != 0:
                return f"Error: {output}" if output else "Error: non-zero exit code"
            return output if output else "(no output)"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
