"""
Monkey-patches for OASIS to adapt it from a social media simulator
to a Science Parliament. Import this module before creating any agents.

What we patch:
1. SocialAgent.__init__       — support single_iteration=False (max_iteration=5)
2. SocialAgent.perform_action_by_llm — parliament-style user message
3. SocialEnvironment templates — parliament-style environment descriptions
4. SocialAction docstrings     — parliament-style tool descriptions
5. Platform.refresh            — merge followed users' posts into candidate pool
6. SocialAgent.perform_interview — remove "You are a twitter user" leftover
"""

import os
import random
from datetime import datetime
from string import Template

from camel.messages import BaseMessage

os.makedirs("./log", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Patch SocialAgent.__init__ to support single_iteration
# ---------------------------------------------------------------------------
from oasis.social_agent.agent import SocialAgent

_original_init = SocialAgent.__init__


def _patched_init(self, *args, single_iteration=True, **kwargs):
    _original_init(self, *args, **kwargs)
    if not single_iteration:
        from config import MAX_ITERATION
        self.max_iteration = MAX_ITERATION


SocialAgent.__init__ = _patched_init

# ---------------------------------------------------------------------------
# 1b. Patch SocialAgent.perform_action_by_data — skip memory write
#     ManualActions are platform-level events (e.g. opening post), not agent
#     decisions. The agent doesn't need to "remember" them — the results are
#     visible via forum refresh. The original writes with SYSTEM role, which
#     breaks models with strict system-message ordering (e.g. Qwen3.5).
# ---------------------------------------------------------------------------
async def _patched_perform_action_by_data(self, func_name, *args, **kwargs):
    func_name = func_name.value if isinstance(func_name,
                                              ActionType) else func_name
    function_list = self.env.action.get_openai_function_list()
    for i in range(len(function_list)):
        if function_list[i].func.__name__ == func_name:
            func = function_list[i].func
            result = await func(*args, **kwargs)
            return result
    raise ValueError(f"Function {func_name} not found in the list.")


SocialAgent.perform_action_by_data = _patched_perform_action_by_data

# ---------------------------------------------------------------------------
# 2. Patch SocialAgent.perform_action_by_llm — parliament user message
# ---------------------------------------------------------------------------
import logging

_agent_log = logging.getLogger("social.agent")

from oasis.social_platform.typing import ActionType

_ALL_SOCIAL_ACTIONS = [action.value for action in ActionType]


async def _patched_perform_action_by_llm(self):
    env_prompt = await self.env.to_text_prompt()
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=(
            "A new round of the parliament session has begun. "
            "Review what your fellow scientists have posted, then "
            "decide your next action \u2014 post an analysis, comment on "
            "someone's work, endorse or challenge a post, or do "
            "nothing if you have nothing new to add.\n\n"
            f"{env_prompt}"
        ),
    )
    try:
        _agent_log.info(
            f"Agent {self.social_agent_id} observing environment: "
            f"{env_prompt}"
        )
        response = await self.astep(user_msg)
        for tool_call in response.info["tool_calls"]:
            action_name = tool_call.tool_name
            args = tool_call.args
            _agent_log.info(
                f"Agent {self.social_agent_id} performed "
                f"action: {action_name} with args: {args}"
            )
            if action_name not in _ALL_SOCIAL_ACTIONS:
                _agent_log.info(
                    f"Agent {self.social_agent_id} get the result: "
                    f"{tool_call.result}"
                )
        return response
    except Exception as e:
        _agent_log.error(f"Agent {self.social_agent_id} error: {e}")
        return e


SocialAgent.perform_action_by_llm = _patched_perform_action_by_llm

# ---------------------------------------------------------------------------
# 3. Patch SocialEnvironment templates
# ---------------------------------------------------------------------------
from oasis.social_agent.agent_environment import SocialEnvironment

SocialEnvironment.followers_env_template = Template(
    "$num_followers colleagues are following your work."
)
SocialEnvironment.follows_env_template = Template(
    "You are following $num_follows colleagues."
)
SocialEnvironment.posts_env_template = Template(
    "Here are the current contributions from the parliament:\n$posts"
)
SocialEnvironment.env_template = Template(
    "$posts_env\n\n"
    "$followers_env $follows_env\n\n"
    "Based on the above, decide what action would best advance the "
    "parliament's progress on the question."
)

_original_to_text_prompt = SocialEnvironment.to_text_prompt


async def _patched_to_text_prompt(self, **kwargs):
    followers_env = await self.get_followers_env()
    follows_env = await self.get_follows_env()
    posts = await self.action.refresh()
    if posts["success"]:
        import json
        posts_env = json.dumps(posts["posts"], indent=4)
        posts_env = self.posts_env_template.substitute(posts=posts_env)
    else:
        posts_env = (
            "No contributions have been posted yet. "
            "You may be the first to share your analysis."
        )
    return self.env_template.substitute(
        followers_env=followers_env,
        follows_env=follows_env,
        posts_env=posts_env,
    )


SocialEnvironment.to_text_prompt = _patched_to_text_prompt

# ---------------------------------------------------------------------------
# 4. Patch SocialAction docstrings
# ---------------------------------------------------------------------------
from oasis.social_agent.agent_action import SocialAction

SocialAction.create_post.__doc__ = (
    "Publish your analysis, findings, a sub-question, or any contribution "
    "to the parliament forum. Other scientists will be able to see, comment "
    "on, and vote on your post.\n\n"
    "Args:\n"
    "    content (str): Your scientific contribution.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'post_id': 50}"
)

SocialAction.create_comment.__doc__ = (
    "Reply to a specific post \u2014 correct an error, refine a calculation, "
    "ask a clarifying question, or extend the analysis.\n\n"
    "Args:\n"
    "    post_id (int): The ID of the post to reply to.\n"
    "    content (str): Your reply.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'comment_id': 123}"
)

SocialAction.like_post.__doc__ = (
    "Endorse a post \u2014 signal that you find its reasoning sound and its "
    "contribution valuable. Posts with more endorsements become more visible "
    "to other scientists.\n\n"
    "Args:\n"
    "    post_id (int): The ID of the post to endorse.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'like_id': 123}"
)

SocialAction.dislike_post.__doc__ = (
    "Challenge a post \u2014 signal that you believe it contains errors or "
    "flawed reasoning. Posts with more challenges become less visible.\n\n"
    "Args:\n"
    "    post_id (int): The ID of the post to challenge.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'dislike_id': 123}"
)

SocialAction.like_comment.__doc__ = (
    "Endorse a comment \u2014 signal agreement with its content.\n\n"
    "Args:\n"
    "    comment_id (int): The ID of the comment to endorse.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'comment_like_id': 456}"
)

SocialAction.dislike_comment.__doc__ = (
    "Challenge a comment \u2014 signal disagreement with its content.\n\n"
    "Args:\n"
    "    comment_id (int): The ID of the comment to challenge.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'comment_dislike_id': 456}"
)

SocialAction.search_posts.__doc__ = (
    "Search the parliament forum for posts matching a keyword or topic. "
    "Useful when you want to find what others have said about a specific "
    "sub-problem or concept.\n\n"
    "Args:\n"
    "    query (str): A keyword or phrase to search for.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'posts': [...]}"
)

SocialAction.follow.__doc__ = (
    "Follow a colleague \u2014 their future posts will appear in your feed, "
    "so you can track their ongoing contributions.\n\n"
    "Args:\n"
    "    followee_id (int): The user ID of the colleague to follow.\n\n"
    "Returns:\n"
    "    dict: {'success': True, 'follow_id': 123}"
)

SocialAction.do_nothing.__doc__ = (
    "Skip this round. Use this when you have nothing new to contribute "
    "right now \u2014 the discussion is progressing well without your input, "
    "or you need more time to think.\n\n"
    "Note: if you have already finished your work for this round (e.g. "
    "posted your analysis, commented, or used tools), simply stop "
    "without calling any tool \u2014 do NOT call do_nothing at the end. "
    "Only call do_nothing when you decide to take no action at all.\n\n"
    "Returns:\n"
    "    dict: {'success': True}"
)

# ---------------------------------------------------------------------------
# 5. Patch Platform.refresh — merge followed users' posts into candidate pool
# ---------------------------------------------------------------------------
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import RecsysType


async def _patched_refresh(self, agent_id: int):
    if self.recsys_type == RecsysType.REDDIT:
        current_time = self.sandbox_clock.time_transfer(
            datetime.now(), self.start_time
        )
    else:
        current_time = self.sandbox_clock.get_time_step()
    try:
        user_id = agent_id

        rec_query = "SELECT post_id FROM rec WHERE user_id = ?"
        self.pl_utils._execute_db_command(rec_query, (user_id,))
        rec_results = self.db_cursor.fetchall()
        post_ids = [row[0] for row in rec_results]

        query_following_post = (
            "SELECT post.post_id FROM post "
            "JOIN follow ON post.user_id = follow.followee_id "
            "WHERE follow.follower_id = ?"
        )
        self.pl_utils._execute_db_command(query_following_post, (user_id,))
        following_posts = self.db_cursor.fetchall()
        following_post_ids = [row[0] for row in following_posts]

        candidate_pool = list(set(post_ids + following_post_ids))

        if not candidate_pool:
            return {"success": False, "message": "No posts available."}

        if len(candidate_pool) >= self.refresh_rec_post_count:
            selected_post_ids = random.sample(
                candidate_pool, self.refresh_rec_post_count
            )
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
# 6. Patch SocialAgent.perform_interview — remove twitter leftover
# ---------------------------------------------------------------------------
from camel.types import OpenAIBackendRole


async def _patched_perform_interview(self, interview_prompt: str):
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content="You are a scientist in the Science Parliament.",
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
