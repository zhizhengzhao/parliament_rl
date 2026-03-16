# Science Parliament

使用本地模型（Qwen3.5-9B + vLLM）运行的科学议会系统。

多个 LLM 科学家 agent 在一个共享论坛上协作解答研究生级别的科学 / 数学难题，通过集体讨论和社区投票涌现出最终答案。

基于 [CAMEL](https://github.com/camel-ai/camel)（多智能体框架）和 [OASIS](https://github.com/camel-ai/oasis)（社交媒体模拟平台）构建，并对两者的源码做了深度适配（见 `patches.py`）。

---

## 文件结构

```
zhizheng2/
├── config.py          # 所有参数配置（唯一需要改的文件）
├── run_parliament.py  # 主程序
├── patches.py         # OASIS 适配层（核心改造，见下方说明）
├── visualize.py       # 可视化：读 SQLite → 生成 index.html
├── serve.py           # 可视化服务器：HTTP + ngrok 公网隧道
└── requirements.txt   # 依赖（版本敏感，勿随意升级）
```

---

## 依赖版本（版本敏感）

> ⚠️ 本项目对 `camel-ai` 和 `camel-oasis` 的内部 API 有依赖，升级这两个包可能导致 `patches.py` 失效。**请使用下列固定版本。**

| 包 | 版本 | 说明 |
|----|------|------|
| `camel-ai` | `0.2.89` | 多智能体框架，`patches.py` 对其内部方法打补丁 |
| `camel-oasis` | `0.2.5` | 社交平台基础设施，同上 |
| `sympy` | `>=1.13` | SymPy 数学工具 |
| `pandas` | `>=2.2` | OASIS 依赖 |
| `igraph` | `>=0.11` | OASIS agent 关系图 |
| `sentence-transformers` | `>=3.0` | OASIS 推荐系统 |
| `neo4j` | `>=5.23` | OASIS 图数据库支持 |
| `python-dotenv` | 任意 | `.env` 文件支持 |
| `pyngrok` | 任意 | 可视化公网隧道（可选） |
| Python | `3.10 / 3.11` | OASIS 要求 `<3.12` |

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

### 3. 安装所有依赖（严格版本）

```bash
# CAMEL 多智能体框架（固定版本）
pip install camel-ai==0.2.89

# SymPy 数学工具
pip install "sympy>=1.13"

# OASIS 社交模拟平台（固定版本，跳过依赖冲突检查）
pip install camel-oasis==0.2.5 --no-deps

# OASIS 运行时依赖
pip install "pandas>=2.2" "igraph>=0.11" "sentence-transformers>=3.0" "neo4j>=5.23"

# 其他
pip install python-dotenv

# 可视化公网隧道（可选，不装也能跑）
pip install pyngrok
```

> pip 安装后会提示 camel-oasis 的版本冲突警告，可以忽略——`--no-deps` 跳过了版本锁定，实际运行不受影响。

### 4. 验证安装

```bash
python -c "import camel; print('camel OK')"
python -c "import oasis; print(f'oasis {oasis.__version__} OK')"
python -c "from camel.toolkits import SymPyToolkit; print('sympy toolkit OK')"
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
| `DEFAULT_NUM_AGENTS` | `5` | 科学家数量（建议 demo 用 3） |
| `NUM_ROUNDS` | `3` | 讨论轮数 |
| `MAX_ITERATION` | `5` | 每 agent 每轮最多几步工具调用 |
| `LLM_CONCURRENCY` | `10` | LLM API 最大并发请求数 |
| `REFRESH_REC_POST_COUNT` | `50` | 每轮 refresh 最多看多少帖子 |
| `ALLOW_SELF_RATING` | `False` | 是否允许 agent 给自己的帖子点赞/踩 |

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

等待看到 `Uvicorn running on http://0.0.0.0:8000` 后按 `Ctrl+B, D` 挂到后台。

### 第二步：运行议会

```bash
tmux new -s parliament
python run_parliament.py --question "Find all positive integers n such that n^2 + n + 41 is a perfect square."
```

或从文件读题：

```bash
python run_parliament.py --question_file question.txt
```

---

## 可视化（可选）

议会跑步过程中，每轮结束后自动在输出目录生成 `index.html`。
用 `serve.py` 可以把它暴露为公网可访问的页面（需要 ngrok token）。

### 启动可视化服务器

在另一个终端（与议会并行运行）：

```bash
# 先跑几轮让 index.html 生成，然后：
python serve.py \
  --output_dir output/2026-03-16_14-30-00/ \
  --token YOUR_NGROK_TOKEN
```

打印出的公网地址（`https://xxxx.ngrok-free.app/index.html`）发给任何人，浏览器打开即可，每 8 秒自动刷新。

**不用 ngrok，只看本地：**

```bash
python serve.py --output_dir output/2026-03-16_14-30-00/ --no-ngrok
# 然后 SSH 隧道：ssh -L 18888:localhost:18888 user@server
# 本地浏览器：http://localhost:18888/index.html
```

**不用 serve.py，跑完后直接看：**

HTML 是自包含的静态文件，跑完后直接下载到本地双击打开即可（不会再刷新）。

---

## 输出目录结构

每次运行自动创建带时间戳的目录，所有输出集中在内，互不覆盖：

```
output/
└── 2026-03-16_14-30-00/
    ├── parliament.db       ← SQLite 数据库（帖子、评论、投票、所有 trace）
    ├── session.json        ← 结构化讨论记录（带科学家真实姓名）
    ├── config.py           ← 该次运行的配置快照
    ├── index.html          ← 可视化页面（每轮覆盖更新）
    ├── anomalies.jsonl     ← 异常记录（见下方说明）
    └── log/
        ├── social.agent-xxx.log
        ├── social.twitter-xxx.log
        └── oasis-xxx.log
```

### anomalies.jsonl — 异常记录

当 agent 出现以下情况时自动写入，每条一行 JSON，用于调试：

| 类型 | 触发条件 |
|------|----------|
| `no_tool_calls` | 模型有回复但未调用任何工具（常见：context 过长、格式解析失败） |
| `exception` | `astep()` 抛出异常 |

每条记录包含：
- 发给模型的**完整 context**（system + 历史对话 + 本轮 user message）
- 模型的**完整文字回复**
- context 的 **token 数**（排查 context 过长问题）
- **完整 traceback**（exception 类型）

---

## patches.py 说明

`patches.py` 在程序启动时（`import patches`）立刻对 OASIS 和 CAMEL 进行适配，将社交媒体模拟器改造为科学议会环境。所有补丁在任何 agent 或平台创建之前生效。

| # | 改造目标 | 内容 |
|---|---------|------|
| 1 | `SocialAgent.__init__` | 新增 `single_iteration` 参数，支持 `max_iteration=5`（每轮多步工具调用） |
| 1b | `SocialAgent.perform_action_by_data` | 跳过 memory write，修复 Qwen3.5 strict system-message 排序问题 |
| 2 | `SocialAgent.perform_action_by_llm` | 替换为议会风格 user message；无工具调用或异常时写入 `anomalies.jsonl` |
| 3 | `SocialEnvironment` 模板 | 替换为科学议会风格，移除群聊、Twitter/Reddit 话语 |
| 4 | `SocialAction` docstrings | 替换为科学议会风格工具说明（直接影响 LLM 工具选择决策） |
| 5 | `Platform.refresh` | 合并推荐帖子 + 关注者帖子；清洁 JSON（移除 `num_shares`、`num_reports` 等 Twitter 字段，`user_id` → 科学家姓名） |
| 5b | `Platform.search_posts` | 对搜索结果做相同清洁，确保 agent 始终看到干净的科学家姓名 |
| 6 | `SocialAgent.perform_interview` | 移除 "You are a twitter user" 残留 |
| 7 | OASIS Logger | 重定向所有 OASIS 日志到带时间戳的运行目录，防止污染工作目录 |

---

## 常见问题

**Q: vLLM 启动报错 "out of memory"**

降低 `--max-model-len`：
```bash
vllm serve /path/to/model --max-model-len 8192 ...
```

**Q: 模型 tool call 没有被正确解析，anomalies.jsonl 里全是 no_tool_calls**

1. 确认 vLLM 启动时加了 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
2. 查看 `anomalies.jsonl` 里的 `response_text` 字段——如果模型在说话但没有 tool call 格式，考虑减小 `NUM_ROUNDS`、`DEFAULT_NUM_AGENTS` 或 `MAX_ITERATION` 缩短 context
3. 不要用 Ollama（已知有 tool call 解析问题）

**Q: 运行很慢**

9B 模型在单卡上推理速度有限，多 agent 并发会排队。先用 `DEFAULT_NUM_AGENTS = 3` 测试。

**Q: 报错 "System message must be at the beginning"**

确认使用最新代码。旧版本 OASIS 的 `perform_action_by_data` 以 system 角色写入 agent 记忆，已在 `patches.py` patch 1b 中修复。

**Q: index.html 没有生成**

可视化在每轮之后才生成，Round 0（开幕帖）结束后不会有 HTML。等第一轮 `[Round 1/N]` 完成后再查看。
