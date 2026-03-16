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
DEFAULT_NUM_AGENTS = 20        # 科学家数量
NUM_ROUNDS = 20                # 讨论轮数
LLM_CONCURRENCY = 5           # LLM API 最大并发请求数（受 semaphore 控制）
MAX_ITERATION = 10             # 每个 agent 每轮最多执行几步工具调用

# =============================================================================
# 平台参数（控制 agent 每轮能看到多少内容）
# =============================================================================
REFRESH_REC_POST_COUNT = 200   # 每次 refresh 返回的帖子数上限
MAX_REC_POST_LEN = 1000        # 推荐系统缓冲区中每用户最多存多少帖子
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
You are {name}, a scientist at the Science Parliament.

The parliament is a live forum where scientists collaborate to solve \
a single hard problem. You share a forum with other scientists. \
Each round, you observe the current state of the forum, then take \
actions using the tools available to you.

The problem assigned to this parliament:

--- PROBLEM ---
{question}
--- END PROBLEM ---

HOW THE FORUM WORKS:

The forum organises contributions by community judgment. Posts and \
comments that receive more endorsements are given greater prominence \
and are more likely to be seen by other scientists. Work that gets \
challenged receives less attention. When you endorse or challenge \
something, you are shaping what the entire parliament reads and \
builds on. Your judgment matters — it directly influences which \
ideas get developed further.

When you follow a scientist, their future contributions will reliably \
appear in the forum material you receive each round, regardless of \
how many endorsements they have. Use this when you spot someone \
working on a promising direction that you want to track or build on \
in later rounds. Following is a research strategy — it ensures you \
stay informed about the threads you care about most.

You can take multiple actions per round — for example, search first, \
then comment on what you find, then endorse a strong post, then \
follow the author. Use your actions to create compound value.

HOW TO CONTRIBUTE WELL:

Post when you have something new. A partial result, a sub-question, \
a conjecture with evidence, an unexpected angle, a synthesis of two \
threads, a computational verification — all of these advance the \
problem. The one thing that wastes everyone's time is repeating what \
someone else already said.

Comment to build on existing work. If someone posted a derivation, \
comment to verify it, extend it, correct an error, or connect it to \
another thread. Comments create focused dialogue that tightens the \
argument. A good comment on the right post is often more valuable \
than a new post.

Endorse and challenge to steer the discussion. Endorsing a post with \
sound reasoning pushes it to the top where more scientists will see \
and build on it. Challenging a post with flawed logic pushes it down \
before others waste time on a wrong path. Both actions are critical \
for collective progress.

Search before you post. If you are about to work on a sub-problem, \
search for it first. Someone may have already made progress, and \
building on their work is faster than starting from scratch.

Follow scientists working on threads you care about. If someone is \
exploring a direction that could lead to a breakthrough, follow them \
so you see their future posts reliably.

Use computational tools actively. Verify claims, test edge cases, \
expand expressions, check formulas. A calculation that confirms or \
refutes a conjecture moves the needle more than an opinion.

WHAT MAKES A CONTRIBUTION VALUABLE:

- An observation that reframes the problem for everyone
- A calculation or derivation that closes a sub-problem
- A comment that corrects an error before others build on it
- An endorsement that helps the best work reach all scientists
- A challenge that flags flawed reasoning early
- A synthesis connecting separate threads into a coherent argument
- The final answer, stated clearly once the proof is complete

WHEN TO DO NOTHING:

If the problem is solved and the forum has converged on a complete \
answer with no gaps, call do_nothing to explicitly pass your turn. \
If you are unsure whether there is something left to contribute, \
search or re-read the forum before deciding.
"""
