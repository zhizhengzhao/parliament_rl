# Science Parliament (Local Model)

使用本地模型（Qwen3.5-9B + vLLM）运行的科学议会系统。

与 API 版本 (`science_parliament/`) 代码完全一致，仅 `config.py` 中的模型配置不同。

## 环境要求

- Python 3.10 或 3.11
- NVIDIA GPU，至少 **24GB 显存**（如 RTX 3090/4090、A100 等）
- CUDA 12.x

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
# CAMEL（多智能体框架）
pip install camel-ai

# SymPy（数学计算工具）
pip install "sympy>=1.13"

# OASIS（社交模拟平台，跳过依赖以避免 camel 版本冲突）
pip install camel-oasis --no-deps

# OASIS 的运行时依赖
pip install "pandas>=2.2" "igraph>=0.11" "sentence-transformers>=3.0" "neo4j>=5.23"

# Parliament 依赖
pip install python-dotenv
```

> pip 安装后会提示 camel-oasis 的版本冲突警告，可以忽略——我们用 `--no-deps` 跳过了 oasis 的旧版本锁定，实际运行不受影响。

### 4. 验证安装

```bash
python -c "import camel; print('camel OK')"
python -c "import oasis; print(f'oasis {oasis.__version__} OK')"
python -c "from camel.toolkits import SymPyToolkit; print('sympy toolkit OK')"
```

三行都输出 OK 即可。

## 运行

### 第一步：启动 vLLM 模型服务

```bash
tmux new -s vllm
vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

- 首次运行会自动从 Hugging Face 下载模型（约 18GB）
- 等待看到 `Uvicorn running on http://0.0.0.0:8000` 即为成功
- `Ctrl+B, D` 挂到后台

### 第二步：运行议会

```bash
tmux new -s parliament
cd zhizheng2   # 或你的项目目录
python run_parliament.py --question "Find all positive integers n such that n^2 + n + 41 is a perfect square."
```

### 批量运行

```bash
python run_batch.py --data_path questions.csv --max_examples 10
```

## 配置

所有参数在 `config.py` 中修改，无需命令行参数。

### 模型配置

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `"Qwen/Qwen3.5-9B"` | 模型名，需与 vLLM 启动时一致 |
| `MODEL_BASE_URL` | `"http://localhost:8000/v1"` | vLLM 服务地址 |
| `API_KEY` | `"EMPTY"` | vLLM 不校验 key，填 EMPTY 即可 |

### 议会参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_NUM_AGENTS` | `5` | 科学家数量 |
| `NUM_ROUNDS` | `3` | 讨论轮数 |
| `MAX_ITERATION` | `5` | 每 agent 每轮最多几步工具调用 |
| `REFRESH_REC_POST_COUNT` | `50` | 每次 refresh 返回的帖子数上限 |
| `ALLOW_SELF_RATING` | `False` | 是否允许 agent 给自己点赞/踩 |

## 输出

运行后在 `output/` 目录生成：

- `parliament.db` — SQLite 数据库，包含帖子、评论、投票记录
- `session.json` — 结构化讨论记录

## 答案提取

🚧 **施工中** — `extract_answer.py` 尚在完善中，暂不建议直接使用。

## 常见问题

**Q: vLLM 启动报错 "out of memory"**

降低 `--max-model-len`：
```bash
vllm serve Qwen/Qwen3.5-9B --max-model-len 8192 ...
```

**Q: 模型 tool call 没有被正确解析**

确认 vLLM 启动时加了 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`。不要用 Ollama（已知有解析问题）。

**Q: 运行很慢**

9B 模型在单卡上推理速度有限，多 agent 并发会排队。建议先用 `DEFAULT_NUM_AGENTS = 3` 测试。
