"""Tool definitions and execution for Parliament agents (polling mode).

Actor tools: python_exec, submit, wait
Judge tools: python_exec, submit

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
        "description": "Run Python code for calculations or verification. Returns stdout.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
            },
            "required": ["code"],
        },
    },
}

_SUBMIT_PROPERTIES = {
    "post": {
        "type": "string",
        "description": "Your new post content. Omit if you only want to comment/vote.",
    },
    "comments": {
        "type": "array",
        "description": "Comments on existing posts. Use the post_id number from P_xxx.",
        "items": {
            "type": "object",
            "properties": {
                "post_id": {"type": "integer", "description": "The number from P_xxx"},
                "content": {"type": "string"},
            },
            "required": ["post_id", "content"],
        },
    },
    "votes": {
        "type": "array",
        "description": "Votes on posts (P_xxx) or comments (C_xxx). +1 correct/advances, -1 error/redundant.",
        "items": {
            "type": "object",
            "properties": {
                "target_type": {"type": "string", "enum": ["post", "comment"]},
                "target_id": {"type": "integer", "description": "Number from P_xxx or C_xxx"},
                "value": {"type": "integer", "enum": [1, -1]},
            },
            "required": ["target_type", "target_id", "value"],
        },
    },
}

ACTOR_TOOLS = [
    _PYTHON_EXEC,
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit all your contributions for this round. "
                "Can include: a new post, comments on existing posts, "
                "and votes (+1/-1). "
                "This ends your turn — you will see new content next round."
            ),
            "parameters": {
                "type": "object",
                "properties": _SUBMIT_PROPERTIES,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": (
                "Wait for new posts, comments, or votes from other scientists "
                "before contributing. Use when you want to see others' work "
                "before your next step. Ends your turn."
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
            "name": "submit",
            "description": (
                "Submit your evaluations for this round. "
                "Can include: votes (+1/-1) on posts and comments. "
                "You CANNOT post or comment. "
                "This ends your turn — you will see new content next round."
            ),
            "parameters": {
                "type": "object",
                "properties": {"votes": _SUBMIT_PROPERTIES["votes"]},
            },
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

    async def execute_submit(self, args: dict, role: str) -> dict:
        """Execute a submit tool call. Returns summary of what was done."""
        results = {"post_id": None, "comments": [], "votes": [], "errors": []}

        post_content = args.get("post", "")
        if post_content and role == "actor":
            resp = await self._api("POST", f"/sessions/{self.sid}/posts",
                                   {"content": post_content})
            if isinstance(resp, dict) and "post_id" in resp:
                results["post_id"] = resp["post_id"]
            else:
                results["errors"].append(f"post: {resp}")

        for cm in args.get("comments", []):
            if role != "actor":
                continue
            pid = cm.get("post_id")
            content = cm.get("content", "")
            if pid and content:
                resp = await self._api(
                    "POST", f"/sessions/{self.sid}/posts/{pid}/comments",
                    {"content": content})
                if isinstance(resp, dict) and "comment_id" in resp:
                    results["comments"].append(resp["comment_id"])
                else:
                    results["errors"].append(f"comment on {pid}: {resp}")

        for v in args.get("votes", []):
            ttype = v.get("target_type", "post")
            tid = v.get("target_id")
            value = v.get("value")
            if tid is not None and value is not None:
                if ttype == "comment":
                    path = f"/sessions/{self.sid}/comments/{tid}/vote"
                else:
                    path = f"/sessions/{self.sid}/posts/{tid}/vote"
                resp = await self._api("POST", path, {"value": value})
                if isinstance(resp, dict):
                    results["votes"].append({"id": tid, "type": ttype, "value": value})
                else:
                    results["errors"].append(f"vote {ttype}#{tid}: {resp}")

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
