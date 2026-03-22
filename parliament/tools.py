"""Science Parliament — tool management.

Handles loading external tool packages and overriding OASIS tool descriptions
to match the parliament context.  Tool sets are controlled via config.TOOL_SETS.
"""

import os
import subprocess
import tempfile

from oasis.social_agent.agent_action import SocialAction


# ---------------------------------------------------------------------------
# Python code executor — sandboxed computation for scientists
# ---------------------------------------------------------------------------

_PYTHON_PREAMBLE = """\
import math, cmath, fractions, decimal, itertools, collections, re, json
try:
    import numpy as np
except ImportError:
    pass
try:
    import scipy.constants as const
except ImportError:
    pass
try:
    import sympy
    from sympy import *
except ImportError:
    pass
"""

_PYTHON_TIMEOUT = 30
_PYTHON_MAX_OUTPUT = 4000


def run_python(code: str) -> str:
    """Execute Python code to perform a computation and return the printed
    output. Use this as your personal lab bench — run calculations, verify
    formulas, test edge cases, and produce numerical evidence that
    strengthens or refutes a claim in the parliament.

    A computation that confirms or disproves a claim is the strongest
    possible contribution. If someone posts a formula, you can check it.
    If you derive a result, you can verify it numerically. If you suspect
    an error, you can find a counterexample.

    Available packages (pre-imported):
    - math, cmath — standard math functions
    - numpy (as np) — numerical arrays, linear algebra, FFT
    - scipy.constants (as const) — physical constants and unit conversions
      e.g. const.c, const.hbar, const.eV, const.k, const.m_e, const.N_A
    - sympy — symbolic math (all symbols imported via 'from sympy import *')

    IMPORTANT: You MUST use print() to produce output. Only printed text
    is returned. Variables assigned without printing will not appear.

    Example:
        code = '''
        # Check energy-time uncertainty for a 1ns lifetime
        delta_t = 1e-9
        delta_E = const.hbar / delta_t
        print(f"Energy uncertainty: {delta_E / const.eV:.2e} eV")

        # Verify a matrix eigenvalue claim
        M = np.array([[2, 1], [1, 3]])
        eigenvalues = np.linalg.eigvalsh(M)
        print(f"Eigenvalues: {eigenvalues}")
        '''

    Args:
        code (str): Python code to execute.

    Returns:
        str: The printed output, or an error message if execution fails.
    """
    full_code = _PYTHON_PREAMBLE + "\n" + code

    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="parliament_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(full_code)

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True, text=True,
            timeout=_PYTHON_TIMEOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.returncode != 0 and result.stderr:
            err = result.stderr.strip().splitlines()
            output += "\n[error]\n" + "\n".join(err[-15:])

        if not output.strip():
            output = "(no output — use print() to see results)"

        return output[:_PYTHON_MAX_OUTPUT]

    except subprocess.TimeoutExpired:
        return f"(execution timed out after {_PYTHON_TIMEOUT}s)"
    except Exception as e:
        return f"(execution error: {e})"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Load external tool packages
# ---------------------------------------------------------------------------

def load_tools(tool_sets: list[str] | None = None) -> list:
    """Load tools based on config.  Returns a list of CAMEL FunctionTools."""
    if tool_sets is None:
        from config import TOOL_SETS
        tool_sets = TOOL_SETS

    from camel.toolkits import FunctionTool

    tools = []
    for name in tool_sets:
        if name == "sympy":
            from camel.toolkits import SymPyToolkit
            tools.extend(SymPyToolkit().get_tools())
        elif name == "python":
            tools.append(FunctionTool(run_python))
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
        "Challenge a post you disagree with or found an error in. This "
        "is a constructive scientific action — it lowers the post's score "
        "so the parliament focuses on stronger work instead. Challenging "
        "flawed reasoning is just as important as endorsing correct work; "
        "both help the parliament converge on the right answer.\n\n"
        "Use this when a post contains errors, unjustified assumptions, "
        "or a wrong conclusion. Pair it with a comment explaining the "
        "flaw so others can verify your objection.\n\n"
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
        "Challenge a comment you believe is incorrect. This signals to "
        "other scientists that the comment may contain errors, helping "
        "the parliament avoid building on flawed reasoning.\n\n"
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
