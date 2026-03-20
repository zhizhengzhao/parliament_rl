"""Science Parliament — configuration.

All tunable parameters live here.  Prompts are in prompts.py.
"""

# =============================================================================
# LLM model
# =============================================================================
MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"

# =============================================================================
# vLLM serving
# =============================================================================
VLLM_MAX_MODEL_LEN = 131072       # 128K context
VLLM_GPU_MEMORY_UTILIZATION = 0.90

# =============================================================================
# Parliament
# =============================================================================
DEFAULT_NUM_AGENTS = 20
NUM_ROUNDS = 20
LLM_CONCURRENCY = 5
MAX_ITERATION = 10

# =============================================================================
# Platform
# =============================================================================
REFRESH_REC_POST_COUNT = 200
MAX_REC_POST_LEN = 1000
ALLOW_SELF_RATING = False

# =============================================================================
# Tools — which external tool sets to load for scientists
# =============================================================================
TOOL_SETS = ["sympy", "python"]    # options: "sympy", "python"

# =============================================================================
# Experience (reserved for future use)
# =============================================================================
EXPERIENCE_ENABLED = False
EXPERIENCE_LIBRARY = "experience/library"

# =============================================================================
# Agent names
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
    "LIKE_POST", "DISLIKE_POST", "CREATE_POST", "CREATE_COMMENT",
    "LIKE_COMMENT", "DISLIKE_COMMENT", "SEARCH_POSTS", "FOLLOW", "DO_NOTHING",
]
