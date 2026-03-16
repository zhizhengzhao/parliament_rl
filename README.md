# Science Parliament

使用本地模型（Qwen3.5-9B + vLLM）运行的科学议会系统。

多个 LLM 科学家 agent 在一个共享论坛上协作解答研究生级别的科学 / 数学难题，通过集体讨论和社区投票涌现出最终答案。

基于 [CAMEL](https://github.com/camel-ai/camel)（多智能体框架）和 [OASIS](https://github.com/camel-ai/oasis)（社交媒体模拟平台）构建。

---

## 文件结构

```
zhizheng2/
├── config.py          # 所有参数配置（唯一需要改的文件）
├── run_parliament.py  # 主程序
├── patches.py         # OASIS 适配层（核心改造，见下方说明）
└── requirements.txt   # 依赖
```

---

## 依赖版本

| 依赖 | 版本 | 说明 |
|------|------|------|
| `camel-ai` | 0.2.89 | PyPI 稳定版 |
| `camel-oasis` | 0.2.5 | PyPI 稳定版（`--no-deps` 安装） |
| `Qwen3.5-9B` | 2026.02 | 本地 vLLM 部署 |
| Python | 3.10 / 3.11 | OASIS 要求 `<3.12` |

---

## 环境要求

- Python 3.10 或 3.11
- NVIDIA GPU，至少 **24GB 显存**（如 RTX 3090/4090、A100 等）
- CUDA 12.x

---

## 安装

### 1. 创建 Conda 环境

```bash
conda create -n parliament python=3.11 -y
conda activate parliament
```

### 2. 安装 vLLM

```bash
pip install vllm
```

### 3. 安装 CAMEL + OASIS + 依赖

```bash
# CAMEL 多智能体框架
pip install camel-ai

# SymPy（数学计算工具）
pip install "sympy>=1.13"

# OASIS 社交模拟平台（跳过依赖以避免 camel 版本冲突）
pip install camel-oasis --no-deps

# OASIS 的运行时依赖
pip install "pandas>=2.2" "igraph>=0.11" "sentence-transformers>=3.0" "neo4j>=5.23"

# Parliament 依赖
pip install python-dotenv
```

> pip 安装后会提示 camel-oasis 的版本冲突警告，可以忽略。

### 4. 验证安装

```bash
python -c "import camel; print('camel OK')"
python -c "import oasis; print(f'oasis {oasis.__version__} OK')"
python -c "from camel.toolkits import SymPyToolkit; print('sympy toolkit OK')"
```

---

## 运行

### 第一步：启动 vLLM 模型服务

```bash
tmux new -s vllm
vllm serve /path/to/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

- 等待看到 `Uvicorn running on http://0.0.0.0:8000` 即为成功
- `Ctrl+B, D` 挂到后台

### 第二步：运行议会

```bash
tmux new -s parliament
python run_parliament.py --question "Find all positive integers n such that n^2 + n + 41 is a perfect square."
```

或从文件读取题目：

```bash
python run_parliament.py --question_file question.txt
```

---

## 配置

所有参数在 `config.py` 中修改，无需命令行参数。

### 模型配置

| 参数 | 说明 |
|------|------|
| `MODEL_NAME` | 模型路径或名称，需与 vLLM 启动时一致 |
| `MODEL_BASE_URL` | vLLM 服务地址，默认 `http://localhost:8000/v1` |
| `API_KEY` | vLLM 不校验 key，填 `"EMPTY"` 即可 |

### 议会参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_NUM_AGENTS` | `5` | 科学家数量 |
| `NUM_ROUNDS` | `3` | 讨论轮数 |
| `MAX_ITERATION` | `5` | 每 agent 每轮最多几步工具调用 |
| `LLM_CONCURRENCY` | `10` | LLM API 最大并发请求数 |
| `REFRESH_REC_POST_COUNT` | `50` | 每轮 refresh 最多看多少帖子 |
| `ALLOW_SELF_RATING` | `False` | 是否允许 agent 给自己的帖子点赞/踩 |

---

## 输出

每次运行自动创建带时间戳的目录，所有输出集中在内：

```
output/
└── 2026-03-16_14-30-00/
    ├── parliament.db       ← SQLite 数据库（帖子、评论、投票、所有 trace）
    ├── session.json        ← 结构化讨论记录（带科学家真实姓名）
    ├── config.py           ← 该次运行的配置快照
    ├── anomalies.jsonl     ← 异常记录（模型无工具调用 / 异常，见下方）
    └── log/
        ├── social.agent-xxx.log
        ├── social.twitter-xxx.log
        └── oasis-xxx.log
```

每次运行互不覆盖，便于对比不同配置的结果。

### anomalies.jsonl

记录两类异常，每条一行 JSON，用于调试：

| 类型 | 说明 |
|------|------|
| `no_tool_calls` | 模型有回复但未调用任何工具（可能是 context 太长或格式解析失败） |
| `exception` | `astep()` 抛出异常 |

每条记录包含：发给模型的完整 context（system + 历史 + 本轮 user message）、模型的完整文字回复、token 数、异常 traceback（如有）。

---

## patches.py 说明

OASIS 是社交媒体模拟平台，`patches.py` 在运行时将其适配为科学议会环境。主要改造：

| # | 改造目标 | 内容 |
|---|---------|------|
| 1 | `SocialAgent.__init__` | 支持 `max_iteration=5`（每轮多步工具调用） |
| 1b | `SocialAgent.perform_action_by_data` | 跳过 memory write，修复 Qwen3.5 system-msg 排序问题 |
| 2 | `SocialAgent.perform_action_by_llm` | 替换为议会风格 user message；记录异常到 `anomalies.jsonl` |
| 3 | `SocialEnvironment` 模板 | 替换为科学议会风格，去掉群聊、Twitter 话语 |
| 4 | `SocialAction` docstrings | 替换为科学议会风格工具说明（影响 LLM 决策） |
| 5 | `Platform.refresh` | 合并推荐帖子 + 关注者帖子；清洁 JSON（去掉 `num_shares` 等 Twitter 字段，替换 `user_id` 为科学家姓名） |
| 5b | `Platform.search_posts` | 对搜索结果做相同清洁 |
| 6 | `SocialAgent.perform_interview` | 去掉 "You are a twitter user" 残留 |
| 7 | Logger | 重定向所有 OASIS 日志到带时间戳的运行目录 |

---

## 常见问题

**Q: vLLM 启动报错 "out of memory"**

降低 `--max-model-len`：
```bash
vllm serve /path/to/model --max-model-len 8192 ...
```

**Q: 模型 tool call 没有被正确解析**

确认 vLLM 启动时加了 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`。不要用 Ollama（已知有解析问题）。

**Q: 运行很慢**

9B 模型在单卡上推理速度有限，多 agent 并发会排队。建议先用 `DEFAULT_NUM_AGENTS = 3` 测试。

**Q: 报错 "System message must be at the beginning"**

确认使用的是最新代码。旧版本中 OASIS 的 `perform_action_by_data` 会以 system 角色写入 agent 记忆，已在 `patches.py` 中修复。

**Q: anomalies.jsonl 里全是 no_tool_calls**

通常是 context 太长（模型被淹没）或 tool call 格式解析失败。查看 `response_text` 字段——如果模型在说话但没有 tool call 格式，考虑减小 `NUM_ROUNDS`、`DEFAULT_NUM_AGENTS` 或 `MAX_ITERATION`，缩短 context。
