"""Science Parliament configuration (local model version).

All tunable parameters are here. Modify this file before each run.

Local model setup:
  vllm serve Qwen/Qwen3.5-9B \\
    --port 8000 \\
    --tensor-parallel-size 1 \\
    --max-model-len 262144 \\
    --reasoning-parser qwen3 \\
    --enable-auto-tool-choice \\
    --tool-call-parser qwen3_coder
"""

# =============================================================================
# LLM 模型配置（本地 vLLM 服务）
# =============================================================================
MODEL_NAME = "Qwen/Qwen3.5-9B"
MODEL_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"

# =============================================================================
# 议会参数
# =============================================================================
DEFAULT_NUM_AGENTS = 5         # 科学家数量
NUM_ROUNDS = 3                 # 讨论轮数
LLM_CONCURRENCY = 10          # LLM API 最大并发请求数
MAX_ITERATION = 5              # 每个 agent 每轮最多执行几步工具调用

# =============================================================================
# 平台参数（控制 agent 每轮能看到多少内容）
# =============================================================================
REFRESH_REC_POST_COUNT = 50    # 每次 refresh 返回的帖子数上限（原值 5，太小会漏看）
MAX_REC_POST_LEN = 200         # 推荐系统缓冲区中每用户最多存多少帖子
ALLOW_SELF_RATING = False      # 是否允许 agent 给自己的帖子点赞/踩

# =============================================================================
# 输出
# =============================================================================
OUTPUT_DIR = "output"

# =============================================================================
# Agent 名字（最多 26 个预设名，超出则自动生成 Scientist_N）
# =============================================================================
ALL_AGENT_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Karen", "Leo",
    "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rose",
    "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe",
]


def get_agent_names(n: int) -> list[str]:
    """Return the first n agent names. If n > 26, generate extra names."""
    if n <= len(ALL_AGENT_NAMES):
        return ALL_AGENT_NAMES[:n]
    names = list(ALL_AGENT_NAMES)
    i = len(names)
    while len(names) < n:
        names.append(f"Scientist_{i}")
        i += 1
    return names


# =============================================================================
# Agent 可用动作
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
# Prompt 模板（{name} 和 {question} 会被自动替换）
# =============================================================================
SCIENTIST_PROMPT_TEMPLATE = """\
You are {name}, a scientist and member of the Science Parliament.

The following question has been assigned to your parliament for \
collaborative resolution:

--- QUESTION ---
{question}
--- END QUESTION ---

You will work with other scientists on a shared forum to solve this. \
You do NOT need to solve the entire problem alone. Focus on what you \
can contribute right now:

- Identify and solve a key sub-problem that moves things forward.
- Propose a sub-question or a related question worth investigating.
- Point out a flaw or gap in someone else's reasoning.
- Verify or refine a calculation posted by another scientist.
- Synthesize insights from multiple posts into a clearer picture.
- Or do your own deep analysis if you see a path to the answer.

You have access to computational tools (e.g. SymPy for symbolic math). \
Use them when you need to verify calculations, solve equations, or \
check formulas.

After observing what others have posted, decide what action best pushes \
the problem toward a solution. You may:
- Create a post with your analysis, a sub-question, or a new angle.
- Comment on someone's post to correct, refine, or extend their work.
- Like a post with sound reasoning. Dislike a post with clear errors.
- Follow a scientist whose work you find valuable.
- Search for relevant posts.
- Do nothing if you have nothing new to add right now.

Remember: you are part of a team. The goal is collective progress. \
Every contribution should add new value — do not repeat what others \
have already said.
"""
