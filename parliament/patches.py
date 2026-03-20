"""
Monkey-patches for OASIS to adapt it from a social media simulator
to a Science Parliament.  Import this module before creating any agents.

Patches applied:
  1.  SocialAgent.__init__             — multi-step iterations per round
  2.  SocialAgent.perform_action_by_data — skip memory write (Qwen system-msg fix)
  3.  SocialAgent.perform_action_by_llm  — context.py-based assembly + anomaly log
  4.  SocialAction docstrings            — parliament-style tool descriptions
  5.  Platform.refresh                   — merge followed scientists' posts
  6.  Platform.search_posts              — clean JSON for agent display
  7.  SocialAgent.perform_interview      — remove leftover twitter prompt
  8.  Logger redirection                 — route OASIS logs to log/<timestamp>/
"""

import logging
import os
import random
import sqlite3
from datetime import datetime
from camel.messages import BaseMessage
from camel.types import OpenAIBackendRole

from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.database import get_db_path
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType, RecsysType

_log_dir = os.environ.get("PARLIAMENT_LOG_DIR", "./log")
os.makedirs(_log_dir, exist_ok=True)

def _get_id_to_name() -> dict:
    """Query the database for a user_id → scientist name mapping."""
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, name FROM user")
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return result
    except Exception:
        return {}


def _clean_posts(posts: list, id_to_name: dict) -> list:
    """Clean search results for agent display (used by search_posts only)."""
    clean = []
    for post in posts:
        uid = post.get("user_id")
        author = id_to_name.get(uid, f"Scientist_{uid}")
        clean_comments = []
        for c in post.get("comments", []):
            cuid = c.get("user_id")
            cauthor = id_to_name.get(cuid, f"Scientist_{cuid}")
            clean_comments.append({
                "comment_id": c["comment_id"],
                "scientist_id": cuid,
                "author": cauthor,
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


# ---------------------------------------------------------------------------
# 1. SocialAgent.__init__ — multi-step iterations per round
# ---------------------------------------------------------------------------

_original_init = SocialAgent.__init__


def _patched_init(self, *args, single_iteration=True, **kwargs):
    _original_init(self, *args, **kwargs)
    if not single_iteration:
        from config import MAX_ITERATION
        self.max_iteration = MAX_ITERATION


SocialAgent.__init__ = _patched_init

# ---------------------------------------------------------------------------
# 2. SocialAgent.perform_action_by_data — skip memory write
#
#    ManualActions are platform-level events (e.g. the opening post), not
#    agent decisions.  Writing them to memory with SYSTEM role breaks models
#    that enforce strict system-message ordering (Qwen3.5).
# ---------------------------------------------------------------------------


async def _patched_perform_action_by_data(self, func_name, *args, **kwargs):
    func_name = func_name.value if isinstance(func_name, ActionType) else func_name
    for tool in self.env.action.get_openai_function_list():
        if tool.func.__name__ == func_name:
            return await tool.func(*args, **kwargs)
    raise ValueError(f"Function {func_name} not found in the list.")


SocialAgent.perform_action_by_data = _patched_perform_action_by_data

# ---------------------------------------------------------------------------
# 3. SocialAgent.perform_action_by_llm — parliament user message + anomaly log
# ---------------------------------------------------------------------------
_agent_log = logging.getLogger("social.agent")

_ALL_SOCIAL_ACTIONS = [action.value for action in ActionType]


def _log_anomaly(
    anomaly_type: str,
    agent,
    full_context: list,
    num_tokens: int,
    response,
    error: Exception = None,
) -> None:
    """Append one anomaly record to anomalies.jsonl in the run output directory.

    Each record captures everything needed to reproduce and debug the issue:
    the complete context sent to the model, the model's full response, and
    (for exceptions) the traceback string.

    anomaly_type values:
      "no_tool_calls" — model replied with text but called no tool
      "exception"     — astep() raised an exception
    """
    import json
    import traceback

    run_dir = os.environ.get("PARLIAMENT_RUN_DIR", ".")
    anomaly_path = os.path.join(run_dir, "anomalies.jsonl")

    # Safely serialize an arbitrary value to something JSON-compatible
    def _safe(obj):
        try:
            json.dumps(obj, ensure_ascii=False)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    # Serialize the context (list of message dicts from CAMEL/OpenAI format)
    serializable_context = []
    for msg in full_context:
        if isinstance(msg, dict):
            serializable_context.append({k: _safe(v) for k, v in msg.items()})
        else:
            serializable_context.append(str(msg))

    record = {
        "timestamp": datetime.now().isoformat(),
        "type": anomaly_type,
        "agent_id": agent.social_agent_id,
        "agent_name": getattr(
            getattr(agent, "user_info", None), "name",
            str(agent.social_agent_id)
        ),
        "num_tokens_in_context": num_tokens,
        # Complete input: what was sent to the model this turn
        "full_context": serializable_context,
        # Complete output: what the model returned
        "response_text": None,
        "response_tool_calls": [],
        # Error info (exception type only)
        "error": (
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
            if error is not None else None
        ),
    }

    if response is not None:
        # Model's text content (always present even when tool calls exist)
        try:
            if response.msgs:
                record["response_text"] = response.msgs[0].content
        except Exception:
            pass
        # Tool calls the model issued (may be empty for no_tool_calls case)
        try:
            tcs = response.info.get("tool_calls", [])
            record["response_tool_calls"] = [
                {
                    "tool_name": tc.tool_name,
                    "args": tc.args,
                    "result": str(tc.result),
                }
                for tc in tcs
            ]
        except Exception:
            pass

    try:
        with open(anomaly_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _agent_log.warning(
            f"[ANOMALY:{anomaly_type}] agent={agent.social_agent_id} "
            f"({record['agent_name']}) → {anomaly_path}"
        )
    except Exception as log_error:
        _agent_log.error(f"Failed to write anomaly log: {log_error}")


async def _patched_perform_action_by_llm(self):
    from context import build_context, estimate_tokens, context_overflows

    agent_name = getattr(
        getattr(self, "user_info", None), "name", str(self.social_agent_id)
    )

    # Build context from scratch — no CAMEL memory accumulation
    messages = build_context(
        agent_id=self.social_agent_id,
        agent_name=agent_name,
        system_content=self.system_message.content,
    )

    num_tokens = estimate_tokens(messages)

    if context_overflows(messages):
        _agent_log.warning(
            f"Agent {self.social_agent_id} context overflow "
            f"(~{num_tokens} tokens) — raising to trigger compression"
        )
        raise ContextOverflowError(
            f"Context ~{num_tokens} tokens exceeds safety limit"
        )

    # Clear CAMEL memory so astep only sees our custom context.
    # Without this, CAMEL accumulates all previous rounds' messages
    # (including repeated forum snapshots), causing context explosion.
    try:
        self.memory.clear()
    except AttributeError:
        # Fallback: reset the internal records list directly
        if hasattr(self.memory, 'records'):
            self.memory.records.clear()
        elif hasattr(self.memory, '_records'):
            self.memory._records.clear()

    user_msg = BaseMessage.make_user_message(
        role_name="User", content=messages[-1]["content"],
    )

    try:
        _agent_log.info(
            f"Agent {self.social_agent_id} ({agent_name}) "
            f"observing (~{num_tokens} tokens)"
        )
        response = await self.astep(user_msg)
        tool_calls = response.info.get("tool_calls", [])

        if not tool_calls:
            _log_anomaly("no_tool_calls", self, messages, num_tokens, response)
        else:
            for tc in tool_calls:
                _agent_log.info(
                    f"Agent {self.social_agent_id} action: "
                    f"{tc.tool_name} args={tc.args}"
                )
                if tc.tool_name not in _ALL_SOCIAL_ACTIONS:
                    _agent_log.info(
                        f"Agent {self.social_agent_id} tool result: {tc.result}"
                    )

        return response

    except ContextOverflowError:
        raise
    except Exception as e:
        # If vLLM returned 400 due to context length, re-raise as
        # ContextOverflowError so session.py can trigger compression.
        err_msg = str(e).lower()
        if "input tokens" in err_msg and "context length" in err_msg:
            raise ContextOverflowError(str(e)) from e
        _agent_log.error(f"Agent {self.social_agent_id} exception: {e}")
        _log_anomaly("exception", self, messages, num_tokens, None, error=e)
        return e


class ContextOverflowError(Exception):
    """Raised when context exceeds safety limit, triggering compression."""
    pass


SocialAgent.perform_action_by_llm = _patched_perform_action_by_llm

# ---------------------------------------------------------------------------
# 4. SocialAction docstrings
# ---------------------------------------------------------------------------

SocialAction.create_post.__doc__ = (
    "Publish a new top-level contribution to the forum. Other scientists "
    "will see it, can comment on it, and can endorse or challenge it. "
    "Posts with higher scores (endorsements minus challenges) are shown "
    "more prominently to other scientists.\n\n"
    "Good posts contain: a derivation, a verified calculation, a conjecture "
    "with evidence, a synthesis of multiple threads, or a well-reasoned "
    "correction. Avoid repeating what someone else already posted.\n\n"
    "Args:\n"
    "    content (str): Your contribution.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'post_id': 50}"
)

SocialAction.create_comment.__doc__ = (
    "Reply directly to a specific post. Comments create focused dialogue "
    "under that post \u2014 use them to verify a claim, correct an error, "
    "extend a derivation, ask a targeted question, or connect the post "
    "to another thread.\n\n"
    "Commenting is often more valuable than creating a new post, because "
    "it builds on existing work rather than starting a separate thread.\n\n"
    "Args:\n"
    "    post_id (int): The post to reply to (see 'post_id' in the forum).\n"
    "    content (str): Your reply.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'comment_id': 123}"
)

SocialAction.like_post.__doc__ = (
    "Endorse a post. This increases its score, which affects how "
    "prominently it appears in the forum \u2014 higher-scored contributions "
    "are seen by more scientists. Endorsing strong work helps the "
    "entire parliament find and build on the best ideas.\n\n"
    "Use this when a post contains sound reasoning, a correct calculation, "
    "or a valuable insight that others should see and build upon.\n\n"
    "Args:\n"
    "    post_id (int): The post to endorse (see 'post_id' in the forum).\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'like_id': 123}"
)

SocialAction.dislike_post.__doc__ = (
    "Challenge a post. This decreases its score, making it less "
    "prominent in the forum so fewer scientists spend time on it. "
    "Use this to flag posts with errors, flawed logic, or misleading "
    "claims \u2014 before others waste effort building on a wrong foundation.\n\n"
    "When you challenge a post, consider also commenting to explain "
    "what the error is, so the author and others can learn from it.\n\n"
    "Args:\n"
    "    post_id (int): The post to challenge (see 'post_id' in the forum).\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'dislike_id': 123}"
)

SocialAction.like_comment.__doc__ = (
    "Endorse a comment. This increases its score, signaling to others "
    "that the comment is accurate and valuable.\n\n"
    "Args:\n"
    "    comment_id (int): The comment to endorse "
    "(see 'comment_id' in the forum).\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'comment_like_id': 456}"
)

SocialAction.dislike_comment.__doc__ = (
    "Challenge a comment. This decreases its score, signaling to others "
    "that the comment may contain errors.\n\n"
    "Args:\n"
    "    comment_id (int): The comment to challenge "
    "(see 'comment_id' in the forum).\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'comment_dislike_id': 456}"
)

SocialAction.search_posts.__doc__ = (
    "Search the forum for posts matching a keyword or topic. Use this "
    "before posting to check if someone has already addressed your idea, "
    "or to find earlier work you want to build on or reference.\n\n"
    "Searching first avoids duplication and helps you write comments "
    "that connect to the existing discussion.\n\n"
    "Args:\n"
    "    query (str): A keyword or phrase to search for.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'posts': [...]}"
)

SocialAction.follow.__doc__ = (
    "Follow a scientist. Once you follow someone, their future "
    "contributions will reliably appear in the forum material you "
    "receive each round, regardless of their score. This is useful "
    "when you spot a scientist exploring a promising direction and "
    "you want to track their progress, build on their work, or "
    "verify their claims in later rounds.\n\n"
    "Following is a research strategy \u2014 it ensures you stay "
    "informed about the scientists whose work matters most to "
    "the thread you are pursuing.\n\n"
    "Args:\n"
    "    followee_id (int): The scientist_id of the scientist to follow "
    "(see 'scientist_id' in each forum post or comment).\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'follow_id': 123}"
)

SocialAction.do_nothing.__doc__ = (
    "Explicitly pass your turn this round.\n\n"
    "IMPORTANT: If you do not call any tool at all, your round ends "
    "immediately with no record of your decision. Calling do_nothing "
    "makes your choice to pause explicit. Always call this rather "
    "than staying silent.\n\n"
    "Use this when:\n"
    "- The problem is solved and you have verified there are no gaps.\n"
    "- You have read the forum and genuinely have nothing new to add.\n"
    "- Others are already covering what you would have contributed.\n\n"
    "Returns:\n"
    "    dict: {'success': True}"
)

# ---------------------------------------------------------------------------
# 6. Platform.refresh — merge followed scientists' posts into candidate pool
# ---------------------------------------------------------------------------


async def _patched_refresh(self, agent_id: int):
    if self.recsys_type == RecsysType.REDDIT:
        current_time = self.sandbox_clock.time_transfer(
            datetime.now(), self.start_time
        )
    else:
        current_time = self.sandbox_clock.get_time_step()
    try:
        user_id = agent_id

        # Recommended posts from the recsys
        rec_query = "SELECT post_id FROM rec WHERE user_id = ?"
        self.pl_utils._execute_db_command(rec_query, (user_id,))
        post_ids = [row[0] for row in self.db_cursor.fetchall()]

        # Posts from followed colleagues
        query_following_post = (
            "SELECT post.post_id FROM post "
            "JOIN follow ON post.user_id = follow.followee_id "
            "WHERE follow.follower_id = ?"
        )
        self.pl_utils._execute_db_command(query_following_post, (user_id,))
        following_post_ids = [row[0] for row in self.db_cursor.fetchall()]

        candidate_pool = list(set(post_ids + following_post_ids))

        if not candidate_pool:
            return {"success": False, "message": "No posts available."}

        if len(candidate_pool) >= self.refresh_rec_post_count:
            selected_post_ids = random.sample(candidate_pool, self.refresh_rec_post_count)
        else:
            selected_post_ids = candidate_pool

        placeholders = ", ".join("?" for _ in selected_post_ids)
        post_query = (
            f"SELECT post_id, user_id, original_post_id, content, "
            f"quote_content, created_at, num_likes, num_dislikes, "
            f"num_shares FROM post WHERE post_id IN ({placeholders})"
        )
        self.pl_utils._execute_db_command(post_query, selected_post_ids)
        results = self.db_cursor.fetchall()
        if not results:
            return {"success": False, "message": "No posts found."}

        results_with_comments = self.pl_utils._add_comments_to_posts(results)

        action_info = {"posts": results_with_comments}
        self.pl_utils._record_trace(
            user_id, ActionType.REFRESH.value, action_info, current_time
        )
        return {"success": True, "posts": results_with_comments}
    except Exception as e:
        return {"success": False, "error": str(e)}


Platform.refresh = _patched_refresh

# ---------------------------------------------------------------------------
# 7. Platform.search_posts — clean output for agent display
# ---------------------------------------------------------------------------
_original_search_posts = Platform.search_posts


async def _patched_search_posts(self, agent_id: int, query: str):
    result = await _original_search_posts(self, agent_id, query)
    if result.get("success") and result.get("posts"):
        id_to_name = _get_id_to_name()
        result["posts"] = _clean_posts(result["posts"], id_to_name)
    return result


Platform.search_posts = _patched_search_posts

# ---------------------------------------------------------------------------
# 8. SocialAgent.perform_interview — remove leftover twitter prompt
# ---------------------------------------------------------------------------


async def _patched_perform_interview(self, interview_prompt: str):
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content="You are a scientist in the Science Parliament.",
    )
    if self.interview_record:
        self.update_memory(message=user_msg, role=OpenAIBackendRole.SYSTEM)

    openai_messages, num_tokens = self.memory.get_context()
    openai_messages = (
        [{
            "role": self.system_message.role_name,
            "content": self.system_message.content.split("# RESPONSE METHOD")[0],
        }]
        + openai_messages
        + [{"role": "user", "content": interview_prompt}]
    )
    response = await self._aget_model_response(
        openai_messages=openai_messages, num_tokens=num_tokens
    )
    content = response.output_messages[0].content
    if self.interview_record:
        self.update_memory(
            message=response.output_messages[0], role=OpenAIBackendRole.USER
        )

    interview_data = {"prompt": interview_prompt, "response": content}
    result = await self.env.action.perform_action(
        interview_data, ActionType.INTERVIEW.value
    )
    return {
        "user_id": self.social_agent_id,
        "prompt": openai_messages,
        "content": content,
        "success": result.get("success", False),
    }


SocialAgent.perform_interview = _patched_perform_interview

# ---------------------------------------------------------------------------
# 9. Redirect OASIS loggers to log/<timestamp>/
#
#    OASIS hardcodes FileHandlers to ./log/ at import time.  We close those
#    and re-attach handlers pointing to the timestamped log directory.
#    Called once here (catches social.agent + social.twitter) and once more
#    from run_parliament.py after `import oasis` (catches oasis.env).
# ---------------------------------------------------------------------------
def _redirect_loggers_to_run_dir():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fmt = logging.Formatter(
        "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
    )
    logger_files = {
        "social.agent": f"social.agent-{now}.log",
        "social.twitter": f"social.twitter-{now}.log",
        "oasis.env": f"oasis-{now}.log",
    }
    for logger_name, filename in logger_files.items():
        logger = logging.getLogger(logger_name)
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                logger.removeHandler(h)
        fh = logging.FileHandler(
            os.path.join(_log_dir, filename), encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


_redirect_loggers_to_run_dir()
