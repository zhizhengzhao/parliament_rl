"""Science Parliament — tool management.

Handles loading external tool packages and overriding OASIS tool descriptions
to match the parliament context.  Tool sets are controlled via config.TOOL_SETS.
"""

from oasis.social_agent.agent_action import SocialAction


# ---------------------------------------------------------------------------
# Load external tool packages
# ---------------------------------------------------------------------------

def load_tools(tool_sets: list[str] | None = None) -> list:
    """Load tools based on config.  Returns a list of CAMEL FunctionTools."""
    if tool_sets is None:
        from config import TOOL_SETS
        tool_sets = TOOL_SETS

    tools = []
    for name in tool_sets:
        if name == "sympy":
            from camel.toolkits import SymPyToolkit
            tools.extend(SymPyToolkit().get_tools())
        # Future tool sets go here:
        # elif name == "wolfram":
        #     from camel.toolkits import WolframToolkit
        #     tools.extend(WolframToolkit().get_tools())
        # elif name == "code_exec":
        #     ...
    return tools


# ---------------------------------------------------------------------------
# Override OASIS tool descriptions for parliament context
# ---------------------------------------------------------------------------

def apply_tool_descriptions():
    """Replace OASIS default docstrings with parliament-specific ones."""

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
