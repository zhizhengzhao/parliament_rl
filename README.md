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
cd benchmarks
```

### 2. 安装 CAMEL 和 OASIS（如果还没装）

```bash
pip install -e ./camel
pip install "sympy>=1.13"
pip install "pandas>=2.2" "igraph>=0.11" "sentence-transformers>=3.0" "neo4j>=5.23"
pip install -e ./oasis --no-deps
pip install python-dotenv
```

### 3. 安装 vLLM

```bash
pip install vllm --torch-backend=auto
```

> 如果遇到 PyTorch 版本问题，参考 [vLLM 官方文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)。

### 4. 下载 Qwen3.5-9B

模型会在首次启动 vLLM 时自动从 Hugging Face 下载。如果服务器无法访问 Hugging Face，可以提前手动下载：

```bash
# 方式一：使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.5-9B

# 方式二：使用 modelscope（国内推荐）
pip install modelscope
modelscope download Qwen/Qwen3.5-9B
```

## 运行

### 第一步：启动 vLLM 服务

在一个终端中启动模型服务：

```bash
vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

参数说明：
- `--tensor-parallel-size 1`：单 GPU。多 GPU 可改为 2/4
- `--max-model-len 32768`：最大上下文长度。显存充足可调大，如 65536
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder`：**必须加**，否则 tool calling 不生效

等待看到类似 `Uvicorn running on http://0.0.0.0:8000` 的输出，说明服务启动成功。

可以快速验证：

```bash
curl http://localhost:8000/v1/models
```

应该返回包含 `Qwen/Qwen3.5-9B` 的 JSON。

### 第二步：运行议会

在另一个终端中：

```bash
cd science_parliament_local
python run_parliament.py --question "Find all positive integers n such that n^2 + n + 41 is a perfect square."
```

### 批量运行

```bash
python run_batch.py --data_path questions.csv --max_examples 10
```

## 配置

所有参数在 `config.py` 中修改，无需命令行参数。

### 模型配置（已针对本地设置）

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

## 显存与性能参考

| 配置 | 预估显存 | 说明 |
|------|----------|------|
| `--max-model-len 8192` | ~18GB | 短上下文，适合显存紧张 |
| `--max-model-len 32768` | ~22GB | 推荐，够用 |
| `--max-model-len 65536` | ~28GB | 需要 A100 40GB+ |

每轮每 agent 的输入 token 约 5000-15000（取决于帖子数量），建议 `max-model-len` 至少 16384。

## 与 API 版本的区别

| | API 版本 (`science_parliament/`) | 本地版本 (`science_parliament_local/`) |
|---|---|---|
| 模型 | gpt-4o-mini（远程） | Qwen3.5-9B（本地 vLLM） |
| 费用 | 按 token 计费 | 免费（电费除外） |
| 速度 | 取决于网络和 API 延迟 | 取决于 GPU 算力 |
| Tool calling | OpenAI 服务端处理 | vLLM `--tool-call-parser` 处理 |
| 代码 | 完全一致 | 完全一致 |
| 区别 | 仅 config.py 不同 | 仅 config.py 不同 |

## 常见问题

**Q: vLLM 启动报错 "out of memory"**

降低 `--max-model-len`，或使用量化：
```bash
vllm serve Qwen/Qwen3.5-9B --max-model-len 8192 --quantization awq
```

**Q: 模型输出 tool call 但没有被正确解析**

确认 vLLM 启动时加了 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`。不要使用 Ollama（已知有解析问题）。

**Q: 运行很慢**

9B 模型在单张消费级 GPU 上推理速度有限，26 个 agent 并发会排队。建议先用 `DEFAULT_NUM_AGENTS = 3` 测试。
