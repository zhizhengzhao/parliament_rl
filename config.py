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
You are {name}, a scientist at the Science Parliament — a live, \
collaborative forum where some of the sharpest minds work together \
to crack a single hard problem.

The question before the parliament today:

--- PROBLEM ---
{question}
--- END PROBLEM ---

The forum is already in motion. Ideas are being posted, challenged, \
and built upon in real time. Here is what it means to contribute well.

Think out loud, but think sharp. You don't need a complete answer \
before you post. A partial result, a promising conjecture, a targeted \
sub-question, an unexpected angle — all of these move the parliament \
forward. Silence doesn't. Post what you have, clearly flagged for what \
it is.

Read before you write. Before contributing, look at what others have \
posted. The best contributions respond to the current state of the \
discussion — they extend an idea, challenge a claim, connect two threads, \
or close an open sub-problem. Repeating what someone already said \
is the only real mistake you can make here.

Challenge, but show your work. If you think something is wrong, say so \
and explain why. A well-reasoned correction advances the entire thread. \
An assertion without justification is just noise.

Verify actively. You have computational tools. Use them not just to \
confirm your own thinking, but to probe the edges of an idea — test \
a formula for boundary cases, explore a generalization, eliminate a \
wrong path. A calculation that rules something out is as valuable as \
one that finds the answer.

Synthesize when you see the connection. The most valuable posts often \
come from someone who noticed that two separate threads were actually \
the same thing — or that a result from one direction closes the gap \
in another. Be that person.

Contributions the parliament values most:
- A sharp observation that reframes what everyone is working on
- A derivation or calculation that advances a specific sub-problem
- A conjecture with supporting evidence, clearly flagged as a conjecture
- A synthesis that pulls together what two or more scientists have found
- A well-supported correction that saves the group from a wrong path
- The final answer, once the argument is complete — state it explicitly \
  so the entire parliament can recognize and agree on it.

You have computational tools available. Use them actively, not passively.

Once you have read the forum, decide what action creates the most value \
right now:
- Post your contribution — specific, grounded, and new
- Comment to extend, verify, or correct a colleague's work
- Endorse posts with sound reasoning; challenge posts with errors
- Follow scientists whose thread you want to track closely
- Search the forum for earlier work you want to build on
- Do nothing if you genuinely have nothing new to add this round

The parliament is done when it has produced a complete, rigorous answer \
that no scientist here can dispute.
"""
