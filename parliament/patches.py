"""
Monkey-patches for OASIS — adapts social media simulator to Science Parliament.
Import this module before creating any agents.

Patches:
  1. SocialAgent.__init__              — multi-step iterations
  2. SocialAgent.perform_action_by_data — skip memory write (Qwen fix)
  3. SocialAgent.perform_action_by_llm  — context.py-based assembly
  4. Platform.refresh                   — merge followed scientists' posts
  5. Platform.search_posts              — clean JSON for display
  6. SocialAgent.perform_interview      — remove twitter prompt
  7. Tool descriptions                  — via tools.apply_tool_descriptions()
  8. Logger redirection                 — route to log/<timestamp>/
"""

import json as _json
import logging
import os
import random
import sqlite3
import time
import traceback as _tb
from datetime import datetime

from camel.messages import BaseMessage
from camel.types import OpenAIBackendRole

from oasis.social_agent.agent import SocialAgent
from oasis.social_platform.database import get_db_path
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType, RecsysType

_log_dir = os.environ.get("PARLIAMENT_LOG_DIR", "./log")
os.makedirs(_log_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_id_to_name() -> dict:
    try:
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        c.execute("SELECT user_id, name FROM user")
        result = {row[0]: row[1] for row in c.fetchall()}
        conn.close()
        return result
    except Exception:
        return {}


def _clean_posts(posts: list, id_to_name: dict) -> list:
    """Clean search results for agent display."""
    clean = []
    for post in posts:
        uid = post.get("user_id")
        author = id_to_name.get(uid, f"Scientist_{uid}")
        clean_comments = []
        for c in post.get("comments", []):
            cuid = c.get("user_id")
            clean_comments.append({
                "comment_id": c["comment_id"],
                "scientist_id": cuid,
                "author": id_to_name.get(cuid, f"Scientist_{cuid}"),
                "content": c.get("content", ""),
                "score": c.get("score", 0),
            })
        clean.append({
            "post_id": post["post_id"],
            "scientist_id": uid,
            "author": author,
            "content": post.get("content", ""),
            "score": post.get("score", 0),
            "comments": clean_comments,
        })
    return clean


class ContextOverflowError(Exception):
    """Raised when context exceeds safety limit, triggering compression."""
    pass


# ---------------------------------------------------------------------------
# 1. SocialAgent.__init__
# ---------------------------------------------------------------------------

_original_init = SocialAgent.__init__


def _patched_init(self, *args, single_iteration=True, **kwargs):
    _original_init(self, *args, **kwargs)
    if not single_iteration:
        from config import MAX_ITERATION
        self.max_iteration = MAX_ITERATION


SocialAgent.__init__ = _patched_init

# ---------------------------------------------------------------------------
# 2. SocialAgent.perform_action_by_data
# ---------------------------------------------------------------------------


_TOOL_ALIASES = {
    "comment": "create_comment",
    "post": "create_post",
    "endorse_post": "like_post",
    "endorse": "like_post",
    "challenge_post": "dislike_post",
    "endorse_comment": "like_comment",
    "challenge_comment": "dislike_comment",
}


async def _patched_perform_action_by_data(self, func_name, *args, **kwargs):
    func_name = func_name.value if isinstance(func_name, ActionType) else func_name
    func_name = _TOOL_ALIASES.get(func_name, func_name)

    if func_name == "follow":
        followee_id = kwargs.get("followee_id") or (args[0] if args else None)
        if followee_id is not None:
            try:
                conn = sqlite3.connect(get_db_path())
                c = conn.cursor()
                c.execute("SELECT user_id FROM user WHERE user_id = ?", (followee_id,))
                valid = c.fetchone() is not None
                conn.close()
            except Exception:
                valid = False
            if not valid:
                return {"success": False, "error": f"scientist_id {followee_id} does not exist"}
            if followee_id == self.social_agent_id:
                return {"success": False, "error": "cannot follow yourself"}

    for tool in self.env.action.get_openai_function_list():
        if tool.func.__name__ == func_name:
            return await tool.func(*args, **kwargs)
    raise ValueError(f"Function {func_name} not found.")


SocialAgent.perform_action_by_data = _patched_perform_action_by_data

# ---------------------------------------------------------------------------
# 3. SocialAgent.perform_action_by_llm
# ---------------------------------------------------------------------------

_agent_log = logging.getLogger("social.agent")
_ALL_SOCIAL_ACTIONS = [action.value for action in ActionType]

# Per-round failure tracking — session.py reads and resets these each round.
round_fail_count = 0
round_agent_count = 0


def reset_round_stats():
    global round_fail_count, round_agent_count
    round_fail_count = 0
    round_agent_count = 0


def log_event(data: dict):
    """Append a JSON event to run_log.jsonl in the current run directory."""
    run_dir = os.environ.get("PARLIAMENT_RUN_DIR", ".")
    path = os.path.join(run_dir, "run_log.jsonl")
    data["timestamp"] = datetime.now().isoformat()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(data, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


async def _patched_perform_action_by_llm(self):
    global round_fail_count, round_agent_count
    from context import build_context, estimate_tokens, context_overflows

    round_agent_count += 1
    t0 = time.time()
    agent_name = getattr(getattr(self, "user_info", None), "name", "?")
    messages = build_context(
        agent_id=self.social_agent_id,
        agent_name=agent_name,
        system_content=self.system_message.content,
    )
    num_tokens = estimate_tokens(messages)

    if context_overflows(messages):
        raise ContextOverflowError(f"~{num_tokens} tokens exceeds limit")

    try:
        self.memory.clear()
    except AttributeError:
        for attr in ("records", "_records"):
            if hasattr(self.memory, attr):
                getattr(self.memory, attr).clear()
                break

    user_msg = BaseMessage.make_user_message(
        role_name="User", content=messages[-1]["content"],
    )

    try:
        _agent_log.info(f"Agent {self.social_agent_id} ({agent_name}) ~{num_tokens} tokens")
        response = await self.astep(user_msg)
        tool_calls = response.info.get("tool_calls", [])
        duration = round(time.time() - t0, 2)

        tc_list = []
        for tc in tool_calls:
            _agent_log.info(f"Agent {self.social_agent_id}: {tc.tool_name} {tc.args}")
            tc_list.append({"name": tc.tool_name, "args": tc.args, "result": str(tc.result)[:200]})

        log_event({
            "event": "agent_done",
            "agent_id": self.social_agent_id,
            "agent_name": agent_name,
            "tokens": num_tokens,
            "duration_s": duration,
            "success": True,
            "tool_calls": tc_list,
            "response_preview": (response.msgs[0].content[:300] if response.msgs else None),
        })

        return response

    except ContextOverflowError:
        raise
    except Exception as e:
        round_fail_count += 1
        duration = round(time.time() - t0, 2)

        log_event({
            "event": "agent_done",
            "agent_id": self.social_agent_id,
            "agent_name": agent_name,
            "tokens": num_tokens,
            "duration_s": duration,
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e)[:500],
            "traceback": "".join(_tb.format_exception(type(e), e, e.__traceback__))[-1000:],
        })

        if "input tokens" in str(e).lower() and "context length" in str(e).lower():
            raise ContextOverflowError(str(e)) from e
        _agent_log.error(f"Agent {self.social_agent_id}: {e}")
        return e


SocialAgent.perform_action_by_llm = _patched_perform_action_by_llm

# ---------------------------------------------------------------------------
# 4. Platform.refresh
# ---------------------------------------------------------------------------


async def _patched_refresh(self, agent_id: int):
    if self.recsys_type == RecsysType.REDDIT:
        current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
    else:
        current_time = self.sandbox_clock.get_time_step()
    try:
        user_id = agent_id
        self.pl_utils._execute_db_command("SELECT post_id FROM rec WHERE user_id = ?", (user_id,))
        post_ids = [r[0] for r in self.db_cursor.fetchall()]

        self.pl_utils._execute_db_command(
            "SELECT post.post_id FROM post "
            "JOIN follow ON post.user_id = follow.followee_id "
            "WHERE follow.follower_id = ?", (user_id,))
        following_ids = [r[0] for r in self.db_cursor.fetchall()]

        pool = list(set(post_ids + following_ids))
        if not pool:
            return {"success": False, "message": "No posts available."}

        selected = random.sample(pool, min(len(pool), self.refresh_rec_post_count))
        ph = ",".join("?" for _ in selected)
        self.pl_utils._execute_db_command(
            f"SELECT post_id, user_id, original_post_id, content, "
            f"quote_content, created_at, num_likes, num_dislikes, "
            f"num_shares FROM post WHERE post_id IN ({ph})", selected)
        results = self.db_cursor.fetchall()
        if not results:
            return {"success": False, "message": "No posts found."}

        posts = self.pl_utils._add_comments_to_posts(results)
        self.pl_utils._record_trace(user_id, ActionType.REFRESH.value, {"posts": posts}, current_time)
        return {"success": True, "posts": posts}
    except Exception as e:
        return {"success": False, "error": str(e)}


Platform.refresh = _patched_refresh

# ---------------------------------------------------------------------------
# 5. Platform.search_posts
# ---------------------------------------------------------------------------

_original_search_posts = Platform.search_posts


async def _patched_search_posts(self, agent_id: int, query: str):
    result = await _original_search_posts(self, agent_id, query)
    if result.get("success") and result.get("posts"):
        result["posts"] = _clean_posts(result["posts"], _get_id_to_name())
    return result


Platform.search_posts = _patched_search_posts

# ---------------------------------------------------------------------------
# 6. SocialAgent.perform_interview
# ---------------------------------------------------------------------------


async def _patched_perform_interview(self, interview_prompt: str):
    user_msg = BaseMessage.make_user_message(
        role_name="User", content="You are a scientist in the Science Parliament.",
    )
    if self.interview_record:
        self.update_memory(message=user_msg, role=OpenAIBackendRole.SYSTEM)

    openai_messages, num_tokens = self.memory.get_context()
    openai_messages = (
        [{"role": self.system_message.role_name,
          "content": self.system_message.content.split("# RESPONSE METHOD")[0]}]
        + openai_messages
        + [{"role": "user", "content": interview_prompt}]
    )
    response = await self._aget_model_response(openai_messages=openai_messages, num_tokens=num_tokens)
    content = response.output_messages[0].content
    if self.interview_record:
        self.update_memory(message=response.output_messages[0], role=OpenAIBackendRole.USER)

    result = await self.env.action.perform_action(
        {"prompt": interview_prompt, "response": content}, ActionType.INTERVIEW.value)
    return {
        "user_id": self.social_agent_id, "prompt": openai_messages,
        "content": content, "success": result.get("success", False),
    }


SocialAgent.perform_interview = _patched_perform_interview

# ---------------------------------------------------------------------------
# 7. Tool descriptions
# ---------------------------------------------------------------------------

from tools import apply_tool_descriptions
apply_tool_descriptions()

# ---------------------------------------------------------------------------
# 8. Logger redirection
# ---------------------------------------------------------------------------


def _redirect_loggers_to_run_dir():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fmt = logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
    for name, fn in {
        "social.agent": f"social.agent-{now}.log",
        "social.twitter": f"social.twitter-{now}.log",
        "oasis.env": f"oasis-{now}.log",
    }.items():
        logger = logging.getLogger(name)
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                logger.removeHandler(h)
        fh = logging.FileHandler(os.path.join(_log_dir, fn), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


_redirect_loggers_to_run_dir()
