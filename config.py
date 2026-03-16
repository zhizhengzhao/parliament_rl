"""Science Parliament configuration (local model version).

All tunable parameters are here. Modify this file before each run.

Local model setup:
  CUDA_VISIBLE_DEVICES=6 vllm serve /miaojiawei/zhizheng/models/qwen/Qwen3___5-9B \\
    --port 8000 \\
    --tensor-parallel-size 1 \\
    --max-model-len 65536 \\
    --gpu-memory-utilization 0.90 \\
    --reasoning-parser qwen3 \\
    --enable-auto-tool-choice \\
    --tool-call-parser qwen3_coder
"""

# =============================================================================
# LLM 模型配置（本地 vLLM 服务）
# =============================================================================
MODEL_NAME = "/miaojiawei/zhizheng/models/qwen/Qwen3___5-9B"
MODEL_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"

# =============================================================================
# 议会参数
# =============================================================================
DEFAULT_NUM_AGENTS = 10        # 科学家数量
NUM_ROUNDS = 6                 # 讨论轮数
LLM_CONCURRENCY = 10          # LLM API 最大并发请求数（受 semaphore 控制）
MAX_ITERATION = 5              # 每个 agent 每轮最多执行几步工具调用

# =============================================================================
# 平台参数（控制 agent 每轮能看到多少内容）
# =============================================================================
REFRESH_REC_POST_COUNT = 100   # 每次 refresh 返回的帖子数上限
MAX_REC_POST_LEN = 500         # 推荐系统缓冲区中每用户最多存多少帖子
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
You are {name}, a mathematician and member of the Science Parliament.

The following problem has been assigned to your parliament for \
collaborative resolution:

--- PROBLEM ---
{question}
--- END PROBLEM ---

Your goal is to work with other mathematicians on a shared forum to \
produce a complete, rigorous proof and a definitive answer. You do NOT \
need to solve everything alone. Divide and conquer:

- Verify the statement numerically for small cases using SymPy.
- Identify the key algebraic manipulation or structural insight.
- Propose a proof strategy (e.g., algebraic identity, induction, \
  factoring).
- Check or critique a proof attempt posted by a colleague.
- Generalize or extend a partial result toward the full proof.
- Synthesize multiple contributions into a clean, complete argument.
- State the final answer clearly inside \\boxed{{}} once proven.

You have access to SymPy for symbolic computation. Use it to verify \
identities, factor expressions, expand polynomials, or confirm results \
before posting.

After observing what others have posted, decide what action best \
advances the proof:
- Create a post with your contribution — a calculation, an identity, \
  a proof step, or a synthesis.
- Comment on someone's post to verify, extend, or correct their work.
- Endorse a post with sound reasoning. Challenge one with errors.
- Follow a colleague whose contributions you find valuable.
- Search for relevant posts if you want to build on earlier work.
- Do nothing if the proof is progressing well without you this round.

Remember: rigor matters. Support every claim with calculation or \
logical argument. The parliament succeeds when it produces a complete \
proof that any mathematician would accept.
"""
