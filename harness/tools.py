"""Tool definitions and execution for Parliament agents.

Two actor tool sets, switched by `PRL_CONTEXT`:

  Coupled  (Parliament_context, default):
      python_exec, vote, submit(comments, post), wait
      ↳ collaborative — vote on peers, comment, wait for new content
  Independent (Solo_context):
      python_exec, submit(step), leave
      ↳ solo derivation — no peers to react to, exits with `leave`

Judge tools: python_exec, vote (same in both modes).

Round-ending tools (per role):
  Actor (coupled)      → submit, wait
  Actor (independent)  → submit, leave
  Judge                → vote (no event, runner stays asleep)

The independent `submit(step)` is the same wire-level call as the
coupled `submit(post)` — the LLM-facing argument name is renamed for
prompt cleanliness; the internal API and DB schema are unchanged.

Session ID and API key are injected by the executor; the LLM never
sees them and cannot vote outside its session.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import aiohttp

# ── Tool schemas ──────────────────────────────────────────

_PYTHON_EXEC = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": ("Run Python code for calculations or verification. "
                        "Returns stdout. Does NOT end your round."),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
            },
            "required": ["code"],
        },
    },
}


def _vote_schema(allowed_values: list[int], description: str,
                 value_doc: str | None = None) -> dict:
    value_prop = {"type": "integer", "enum": allowed_values}
    if value_doc:
        value_prop["description"] = value_doc
    return {
        "type": "object",
        "properties": {
            "votes": {
                "type": "array",
                "description": description,
                "items": {
                    "type": "object",
                    "properties": {
                        "target_type": {"type": "string",
                                        "enum": ["post", "comment"]},
                        "target_id": {"type": "integer",
                                      "description": "Number from P_xxx or C_xxx"},
                        "value": value_prop,
                    },
                    "required": ["target_type", "target_id", "value"],
                },
            },
        },
        "required": ["votes"],
    }


_ACTOR_VOTE_PARAMS = _vote_schema(
    [1, -1],
    "Votes on posts (P_xxx) or comments (C_xxx). "
    "+1 correct/advances, -1 error/redundant.",
)

_JUDGE_VOTE_PARAMS = _vote_schema(
    [-3, -2, -1, 1, 2, 3],
    "Votes on posts (P_xxx) or comments (C_xxx). "
    "Score from -3 (fundamentally wrong) to +3 (excellent progress).",
    value_doc=("+3 matches solution, +2 substantial progress, "
               "+1 genuine non-trivial advance, "
               "-1 trivial/repetitive/minor error, "
               "-2 misleading, -3 fundamentally wrong"),
)


_COUPLED_ACTOR_TOOLS = [
    _PYTHON_EXEC,
    {"type": "function", "function": {
        "name": "vote",
        "description": ("Cast votes on posts or comments. "
                        "Does NOT end your round — vote first, then submit or wait."),
        "parameters": _ACTOR_VOTE_PARAMS,
    }},
    {"type": "function", "function": {
        "name": "submit",
        "description": ("Submit your contributions for this round: "
                        "comments on existing posts and/or a new post. "
                        "This ENDS your turn — you will see new content next round."),
        "parameters": {
            "type": "object",
            "properties": {
                "comments": {
                    "type": "array",
                    "description": (
                        "Comments on existing posts — quick reactions, "
                        "questions, agreements, or corrections. "
                        "Use the post_id number from P_xxx."),
                    "items": {
                        "type": "object",
                        "properties": {
                            "post_id": {"type": "integer",
                                        "description": "The number from P_xxx"},
                            "content": {"type": "string"},
                        },
                        "required": ["post_id", "content"],
                    },
                },
                "post": {
                    "type": "string",
                    "description": (
                        "A focused, verifiable logical step. "
                        "Reference the discussion (e.g. 'Building on P_3, ...'). "
                        "Omit if you only want to comment."),
                },
            },
        },
    }},
    {"type": "function", "function": {
        "name": "wait",
        "description": ("Wait for new posts or comments from other scientists "
                        "before contributing. ENDS your turn."),
        "parameters": {"type": "object", "properties": {}},
    }},
]

_INDEPENDENT_ACTOR_TOOLS = [
    _PYTHON_EXEC,
    {"type": "function", "function": {
        "name": "submit",
        "description": ("Record one reasoning step in your derivation. "
                        "ENDS your turn — anonymous reviewers may score it."),
        "parameters": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "string",
                    "description": (
                        "A focused, verifiable reasoning step — one logical "
                        "move from a previous claim to a new one. Reference "
                        "earlier steps by index (e.g. 'Building on P_3, ...')."),
                },
            },
            "required": ["step"],
        },
    }},
    {"type": "function", "function": {
        "name": "leave",
        "description": ("Declare your derivation complete and retire from "
                        "the session. ENDS your turn permanently — you will "
                        "not be polled again. Use only when fully certain "
                        "the answer chain is settled."),
        "parameters": {"type": "object", "properties": {}},
    }},
]

JUDGE_TOOLS = [
    _PYTHON_EXEC,
    {"type": "function", "function": {
        "name": "vote",
        "description": ("Submit your evaluations: cast votes (-3 to +3) on "
                        "posts and comments. This ENDS your turn."),
        "parameters": _JUDGE_VOTE_PARAMS,
    }},
]


def get_tools(role: str, coupled: bool = True) -> list[dict]:
    """Return the tool spec list for one role.

    Judge tools are identical across coupled / independent (judges always
    see all posts and score them); only the actor tool set differs.
    """
    if role == "judge":
        return JUDGE_TOOLS
    return _COUPLED_ACTOR_TOOLS if coupled else _INDEPENDENT_ACTOR_TOOLS


# ── Generic value coercion ────────────────────────────────

def to_int(v: Any) -> int | None:
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


def to_str(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if v is not None:
        return str(v).strip()
    return ""


# ── Session-level ID mapping ──────────────────────────────

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
        self._next_post = 1
        self._next_comment = 1

    def _map(self, kind: str, global_id: int) -> int:
        if kind == "post":
            g2l, l2g = self._post_g2l, self._post_l2g
            if global_id not in g2l:
                g2l[global_id] = self._next_post
                l2g[self._next_post] = global_id
                self._next_post += 1
            return g2l[global_id]
        g2l, l2g = self._comment_g2l, self._comment_l2g
        if global_id not in g2l:
            g2l[global_id] = self._next_comment
            l2g[self._next_comment] = global_id
            self._next_comment += 1
        return g2l[global_id]

    def map_post(self, global_id: int) -> int:
        return self._map("post", global_id)

    def map_comment(self, global_id: int) -> int:
        return self._map("comment", global_id)

    def resolve_post(self, local_id: int) -> int | None:
        return self._post_l2g.get(local_id)

    def resolve_comment(self, local_id: int) -> int | None:
        return self._comment_l2g.get(local_id)

    def localize_content(self, items: list[dict]) -> None:
        """Convert global IDs to session-local IDs in-place."""
        for item in items:
            t = item["type"]
            if t == "post":
                item["id"] = self.map_post(item["id"])
            elif t == "comment":
                item["id"] = self.map_comment(item["id"])
                item["post_id"] = self.map_post(item["post_id"])
            elif t == "vote":
                if item["target_type"] == "post":
                    item["target_id"] = self.map_post(item["target_id"])
                else:
                    item["target_id"] = self.map_comment(item["target_id"])


# ── python_exec subprocess helper ─────────────────────────

PYTHON_EXEC_TIMEOUT_S = 10
PYTHON_EXEC_MAX_OUTPUT = 10_000


async def python_exec(code: str, timeout: float = PYTHON_EXEC_TIMEOUT_S) -> str:
    """Run code in an isolated subprocess. Returns combined stdout/stderr."""
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
            return f"Error: execution timed out ({int(timeout)}s limit)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    output = stdout.decode(errors="replace").strip()
    if proc.returncode != 0:
        return f"Error: {output}" if output else "Error: non-zero exit code"
    if not output:
        return "(no output)"
    if len(output) > PYTHON_EXEC_MAX_OUTPUT:
        return (output[:PYTHON_EXEC_MAX_OUTPUT]
                + f"\n\n[OUTPUT TRUNCATED — {len(output):,} chars, "
                f"showing first {PYTHON_EXEC_MAX_OUTPUT:,}. "
                f"Reduce print volume to avoid truncation.]")
    return output


# ── Tool executor (Parliament HTTP client) ────────────────

class ToolExecutor:
    """Executes tool calls against Parliament API.

    Tracks the agent's own posts and comments so self-votes can be
    rejected client-side with explicit feedback, and enforces per-role
    vote value ranges (actor ±1, judge ±1..±3).
    """

    def __init__(self, parliament_url: str, session_id: str, api_key: str,
                 http: aiohttp.ClientSession, id_map: IdMap,
                 role: str = "actor"):
        self.base = parliament_url
        self.sid = session_id
        self.http = http
        self.id_map = id_map
        self._role = role
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._my_posts: set[int] = set()
        self._my_comments: set[int] = set()

    # ── HTTP ──

    async def _api(self, method: str, path: str,
                   body: dict | None = None) -> dict | list | str:
        kwargs: dict[str, Any] = {"headers": self._headers}
        if body is not None:
            kwargs["json"] = body
        async with self.http.request(method, f"{self.base}{path}", **kwargs) as resp:
            text = await resp.text()
            if resp.status >= 400:
                return f"Error {resp.status}: {text[:300]}"
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

    async def join(self) -> dict:
        return await self._api("POST", f"/sessions/{self.sid}/join")

    async def leave(self, reason: str = "") -> dict:
        return await self._api("POST", f"/sessions/{self.sid}/leave",
                               {"reason": reason})

    # ── Vote ──

    @staticmethod
    def _coerce_vote_list(votes: Any, errors: list[str]) -> list[dict]:
        if isinstance(votes, str):
            try:
                votes = json.loads(votes)
            except (json.JSONDecodeError, ValueError):
                errors.append(f"could not parse votes: {votes[:200]}")
                return []
        if isinstance(votes, dict):
            return [votes]
        if isinstance(votes, list):
            return votes
        return []

    async def _vote_one(self, v: dict, results: dict) -> None:
        if not isinstance(v, dict):
            results["errors"].append(f"invalid vote format: {str(v)[:100]}")
            return
        ttype = v.get("target_type", "post")
        tid = to_int(v.get("target_id"))
        value = to_int(v.get("value"))
        if tid is None or value is None:
            results["errors"].append(f"missing target_id or value: {v}")
            return
        max_val = 1 if self._role == "actor" else 3
        if not value or abs(value) > max_val:
            allowed = "±1" if self._role == "actor" else "±1 to ±3"
            results["errors"].append(
                f"vote value must be {allowed}, got {value}")
            return

        owned = ((ttype == "post" and tid in self._my_posts)
                 or (ttype == "comment" and tid in self._my_comments))
        if owned:
            results["skipped"].append(
                f"Cannot vote on your own {ttype} {ttype[0].upper()}_{tid}")
            return

        if ttype == "comment":
            global_tid = self.id_map.resolve_comment(tid)
            if global_tid is None:
                results["errors"].append(f"unknown comment C_{tid}")
                return
            path = f"/sessions/{self.sid}/comments/{global_tid}/vote"
        else:
            global_tid = self.id_map.resolve_post(tid)
            if global_tid is None:
                results["errors"].append(f"unknown post P_{tid}")
                return
            path = f"/sessions/{self.sid}/posts/{global_tid}/vote"

        resp = await self._api("POST", path, {"value": value})
        if isinstance(resp, dict):
            results["votes"].append({"id": tid, "type": ttype, "value": value})
        else:
            results["errors"].append(f"vote {ttype}#{tid}: {resp}")

    async def execute_votes(self, votes: Any) -> dict:
        """Execute vote calls. Tolerates JSON-in-string and dict-instead-of-list."""
        results: dict[str, list] = {"votes": [], "errors": [], "skipped": []}
        vote_list = self._coerce_vote_list(votes, results["errors"])
        for v in vote_list:
            await self._vote_one(v, results)
        return results

    # ── Submit ──

    async def _submit_post(self, content: str, results: dict) -> None:
        resp = await self._api("POST", f"/sessions/{self.sid}/posts",
                               {"content": content})
        if isinstance(resp, dict) and "post_id" in resp:
            local_pid = self.id_map.map_post(resp["post_id"])
            results["post_id"] = local_pid
            self._my_posts.add(local_pid)
        else:
            results["errors"].append(f"post: {resp}")

    async def _submit_one_comment(self, cm: Any, results: dict) -> None:
        if not isinstance(cm, dict):
            results["errors"].append(
                f"invalid comment format: {str(cm)[:100]} — "
                f'use {{"post_id": N, "content": "..."}}')
            return

        local_pid = to_int(cm.get("post_id"))
        content = to_str(cm.get("content", ""))

        if not local_pid:
            results["errors"].append(f"comment missing post_id: {str(cm)[:100]}")
            return
        if not content:
            results["errors"].append(f"comment on P_{local_pid} has empty content")
            return

        global_pid = self.id_map.resolve_post(local_pid)
        if global_pid is None:
            results["errors"].append(f"unknown post P_{local_pid}")
            return

        resp = await self._api(
            "POST", f"/sessions/{self.sid}/posts/{global_pid}/comments",
            {"content": content})
        if isinstance(resp, dict) and "comment_id" in resp:
            local_cid = self.id_map.map_comment(resp["comment_id"])
            results["comments"].append(local_cid)
            self._my_comments.add(local_cid)
        else:
            results["errors"].append(f"comment on P_{local_pid}: {resp}")

    @staticmethod
    def _coerce_comment_list(raw: Any, errors: list[str]) -> list:
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                errors.append(
                    f'comment needs post_id: "{raw[:80]}" — '
                    f'use {{"post_id": N, "content": "..."}}')
                return []
        if isinstance(raw, list):
            return raw
        return [raw] if raw else []

    async def execute_submit(self, args: Any) -> dict:
        """Execute submit (post/step + comments). Tolerates several arg shapes.

        Accepts both `post` (coupled mode) and `step` (independent mode)
        for the new-content field — they are the same wire-level call.
        """
        if not isinstance(args, dict):
            return {"post_id": None, "comments": [],
                    "errors": [f"invalid arguments type: {type(args).__name__}"]}

        results: dict = {"post_id": None, "comments": [], "errors": []}

        # Independent-mode `step` aliases coupled-mode `post`.
        post_content = to_str(args.get("post", "") or args.get("step", ""))
        if post_content:
            await self._submit_post(post_content, results)

        comments_raw = args.get("comments") or args.get("comment") or []
        for cm in self._coerce_comment_list(comments_raw, results["errors"]):
            await self._submit_one_comment(cm, results)

        if not results["post_id"] and not results["comments"] and not results["errors"]:
            results["errors"].append(
                "Nothing submitted. Include a post/step or comments to contribute.")
        return results
