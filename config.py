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
