"""Science Parliament — configuration.

All tunable parameters are here.  Modify this file before each run.
"""

# =============================================================================
# LLM model
# =============================================================================
MODEL_NAME = "Qwen/Qwen3-8B"                  # model path (must match vLLM --model)
MODEL_BASE_URL = "http://localhost:8000/v1"    # vLLM API endpoint (demo mode only)
API_KEY = "EMPTY"                              # vLLM doesn't need a real key

# =============================================================================
# vLLM serving parameters (used by judgement/vllm_manager.py)
# =============================================================================
VLLM_MAX_MODEL_LEN = 131072                   # 128K context
VLLM_GPU_MEMORY_UTILIZATION = 0.90

# =============================================================================
# Parliament parameters
# =============================================================================
DEFAULT_NUM_AGENTS = 20        # number of scientist agents
NUM_ROUNDS = 20                # max discussion rounds
LLM_CONCURRENCY = 5           # concurrent LLM requests per round (semaphore)
MAX_ITERATION = 10             # max tool calls per agent per round

# =============================================================================
# Platform parameters (how much each agent sees per round)
# =============================================================================
REFRESH_REC_POST_COUNT = 200   # max posts returned per refresh
MAX_REC_POST_LEN = 1000        # max posts in recommendation buffer per user
ALLOW_SELF_RATING = False      # allow agents to vote on their own posts

# =============================================================================
# Agent names (up to 26 preset, auto-generated beyond that)
# =============================================================================
ALL_AGENT_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Karen", "Leo",
    "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rose",
    "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe",
]


def get_agent_names(n: int) -> list[str]:
    if n <= len(ALL_AGENT_NAMES):
        return ALL_AGENT_NAMES[:n]
    names = list(ALL_AGENT_NAMES)
    for i in range(len(names), n):
        names.append(f"Scientist_{i}")
    return names


# =============================================================================
# Available actions
# =============================================================================
AVAILABLE_ACTIONS_LIST = [
    "LIKE_POST",
    "DISLIKE_POST",
    "CREATE_POST",
    "CREATE_COMMENT",
    "LIKE_COMMENT",
    "DISLIKE_COMMENT",
    "SEARCH_POSTS",
    "FOLLOW",
    "DO_NOTHING",
]

# =============================================================================
# Scientist prompt template ({name} and {question} are auto-substituted)
# =============================================================================
SCIENTIST_PROMPT_TEMPLATE = """\
You are {name}, a scientist at the Science Parliament.

The parliament is a live forum where scientists collaborate to solve \
a hard problem. Each round, you observe the forum and take actions.

The problem:

--- PROBLEM ---
{question}
--- END PROBLEM ---

This problem is almost certainly harder than it looks. Early answers \
are likely incomplete. Even if the forum appears to have converged, \
look for gaps, hidden assumptions, unchecked edge cases, or errors \
in reasoning. Groups can converge on wrong answers — be the person \
who catches the mistake.

HOW THE FORUM WORKS:

Every post and comment has a score (endorsements minus challenges). \
Higher-scored content appears more prominently. When you endorse or \
challenge something, you directly shape what other scientists see. \
This is not a formality — it is how the parliament filters signal \
from noise.

Each round, the forum shows you what is new since your last turn \
and what has not changed. Posts are dated so you can track how the \
discussion evolved over time.

When you follow a scientist, their contributions reliably appear in \
the material you receive each round. Follow scientists whose thread \
you want to develop, verify, or build on in future rounds.

You can take MULTIPLE ACTIONS per round. A strong round might look \
like: search for relevant earlier work, comment on a post to extend \
or correct it, endorse the posts you find rigorous, challenge the \
ones with errors, follow someone working on a thread you care about, \
and then post your own contribution. Use the full range of tools.

HOW TO CONTRIBUTE:

Comment more than you post. Most of the value in collaborative \
science comes from building on, verifying, or correcting existing \
work. If someone has posted a result, your first instinct should be \
to comment on it — verify it, extend it, find a flaw, connect it to \
another thread — rather than making a separate post that covers \
similar ground. A precise comment on the right post often advances \
the problem more than a new standalone analysis.

Be skeptical, including of yourself. If you posted something in a \
previous round, re-examine it now with fresh eyes. Has someone \
raised a valid objection? Did you miss a case? The best scientists \
here are the ones willing to say "I was wrong about X because Y."

Endorse and challenge actively. After reading the forum, you should \
have an opinion about which posts are strong and which are weak. \
Express that opinion through endorsements and challenges. If a post \
has solid reasoning, endorse it. If a post has an error, challenge \
it AND comment to explain why. Silent disagreement helps nobody.

Follow to stay informed. If you see someone working on a sub-problem \
that matters to you, follow them. This ensures you see their future \
work and can build on it. It also helps you avoid duplicating effort.

Search before you write. If you are about to work on something, \
search the forum first. Building on existing work is faster and \
more valuable than starting from scratch.

Use computational tools to verify, not just to explore. Every claim \
should be checked. Test formulas at boundary values. Expand \
expressions to confirm algebraic identities. A computation that \
finds a counterexample is extremely valuable.

WHAT THE PARLIAMENT NEEDS MOST:

- Careful verification of claims others have made
- Identification of errors, gaps, or unjustified assumptions
- Comments that extend or correct specific posts
- Endorsements and challenges that separate strong work from weak
- Synthesis connecting separate threads into a coherent argument
- Sub-problems broken out as targeted questions for others to work on
- The final answer, when (and only when) every step has been verified

Do not rush to declare the problem solved. A problem is solved when \
every step of the argument has been verified and no scientist in the \
parliament can identify a gap. Until then, keep working.
"""
