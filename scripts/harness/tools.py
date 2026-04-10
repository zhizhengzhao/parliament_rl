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

import io
import json
import contextlib
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

_VOTE_PARAMS = {
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
            "parameters": _VOTE_PARAMS,
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
                "Submit your evaluations: cast +1/-1 votes on "
                "posts and comments. This ENDS your turn."
            ),
            "parameters": _VOTE_PARAMS,
        },
    },
]


def get_tools(role: str) -> list[dict]:
    return JUDGE_TOOLS if role == "judge" else ACTOR_TOOLS


# ── Tool executor ─────────────────────────────────────────

class ToolExecutor:
    """Executes tool calls against Parliament API."""

    def __init__(self, parliament_url: str, session_id: str,
                 api_key: str, http: aiohttp.ClientSession):
        self.base = parliament_url
        self.sid = session_id
        self.key = api_key
        self.http = http
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

    async def execute_votes(self, votes: list[dict]) -> dict:
        """Execute vote calls. Used by both actor and judge."""
        results: dict[str, list] = {"votes": [], "errors": [], "skipped": []}
        for v in votes:
            ttype = v.get("target_type", "post")
            tid = v.get("target_id")
            value = v.get("value")
            if tid is None or value is None:
                continue
            own = (ttype == "post" and tid in self._my_posts) or \
                  (ttype == "comment" and tid in self._my_comments)
            if own:
                results["skipped"].append(
                    f"Cannot vote on your own {ttype} {ttype[0].upper()}_{tid}")
                continue
            if ttype == "comment":
                path = f"/sessions/{self.sid}/comments/{tid}/vote"
            else:
                path = f"/sessions/{self.sid}/posts/{tid}/vote"
            resp = await self._api("POST", path, {"value": value})
            if isinstance(resp, dict):
                results["votes"].append(
                    {"id": tid, "type": ttype, "value": value})
            else:
                results["errors"].append(f"vote {ttype}#{tid}: {resp}")
        return results

    async def execute_submit(self, args: dict) -> dict:
        """Execute submit (post + comments). Actor only."""
        results: dict = {"post_id": None, "comments": [], "errors": []}

        post_content = args.get("post", "")
        if post_content:
            resp = await self._api("POST", f"/sessions/{self.sid}/posts",
                                   {"content": post_content})
            if isinstance(resp, dict) and "post_id" in resp:
                results["post_id"] = resp["post_id"]
                self._my_posts.add(resp["post_id"])
            else:
                results["errors"].append(f"post: {resp}")

        for cm in args.get("comments", []):
            pid = cm.get("post_id")
            content = cm.get("content", "")
            if pid and content:
                resp = await self._api(
                    "POST", f"/sessions/{self.sid}/posts/{pid}/comments",
                    {"content": content})
                if isinstance(resp, dict) and "comment_id" in resp:
                    results["comments"].append(resp["comment_id"])
                    self._my_comments.add(resp["comment_id"])
                else:
                    results["errors"].append(f"comment on {pid}: {resp}")

        return results

    async def join(self) -> dict:
        return await self._api("POST", f"/sessions/{self.sid}/join")

    async def leave(self, reason: str = "") -> dict:
        return await self._api("POST", f"/sessions/{self.sid}/leave",
                               {"reason": reason})

    @staticmethod
    def python_exec(code: str) -> str:
        if not code:
            return "Error: code is required"
        buf = io.StringIO()
        restricted_globals = {"__builtins__": __builtins__}
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, restricted_globals)  # noqa: S102
            output = buf.getvalue()
            return output if output else "(no output)"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
