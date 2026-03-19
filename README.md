# Science Parliament

多个 LLM 科学家在论坛上协作解答难题，通过讨论、验证、投票涌现出答案，最后由 Judge 综合给出最终回答。

基于 [CAMEL](https://github.com/camel-ai/camel) + [OASIS](https://github.com/camel-ai/oasis)。

---

## 项目结构

```
parliament/              # 核心议会系统
├── config.py            # 所有可调参数（模型、agent 数量、轮次等）
├── patches.py           # OASIS monkey-patches（适配科学论坛场景）
├── session.py           # 核心逻辑：init() / create_model() / run_session()
├── run_parliament.py    # CLI 入口：demo 模式跑单题
├── visualize.py         # 自动生成 HTML 可视化
└── serve.py             # HTTP 服务器（通过 SSH 隧道访问）

judgement/               # Judge + Benchmark
├── judge.py             # Judge：读取论坛 → 综合最终答案
├── run_benchmark.py     # 多卡并行跑数据集
└── launch_vllm.sh       # 一键启动多卡 vLLM

benchmark/               # 数据集
├── gpqa_diamond.csv     # GPQA Diamond（198 题）
└── open_ended/          # 非选择题（预留）
```

---

## Quick Start

### 1. 环境安装

```bash
conda create -n parliament python=3.11 -y && conda activate parliament

pip install vllm
pip install camel-ai==0.2.89
pip install "sympy>=1.13"
pip install camel-oasis==0.2.5 --no-deps
pip install "pandas>=2.2" "igraph>=0.11" "sentence-transformers>=3.0" "neo4j>=5.23"
pip install python-dotenv
```

> **版本锁定**：`camel-ai==0.2.89` 和 `camel-oasis==0.2.5` 不可升级，`patches.py` 依赖其内部 API。

### 2. 启动 vLLM

```bash
# 单卡（demo 或调试）
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/your/model \
  --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

# 多卡（benchmark，每张卡一个 vLLM 实例）
cd judgement
bash launch_vllm.sh /path/to/your/model 8    # 8 GPUs → ports 8000–8007
```

### 3. 修改配置

编辑 `parliament/config.py`，至少改一下模型名称：

```python
MODEL_NAME = "/path/to/your/model"    # 和 vllm serve 的参数一致
```

### 4. 运行

**Demo（跑单题看效果）：**

```bash
cd parliament
python run_parliament.py --question "Prove that n(n+1)(n+2)(n+3)+1 is always a perfect square."
```

**Benchmark（跑 GPQA）：**

```bash
cd judgement
python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv                     # 全部 198 题，8 卡
python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0 --limit 5   # 单卡跑 5 题
python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0,2,4,6       # 指定 GPU
```

---

## 可控参数

### `parliament/config.py`（议会参数）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `"Qwen/Qwen3-8B"` | 模型路径，需和 vLLM 启动时一致 |
| `MODEL_BASE_URL` | `"http://localhost:8000/v1"` | vLLM API 地址（demo 模式使用） |
| `DEFAULT_NUM_AGENTS` | `20` | 科学家数量 |
| `NUM_ROUNDS` | `20` | 最大讨论轮数（早停可能提前结束） |
| `MAX_ITERATION` | `10` | 每 agent 每轮最多几步工具调用 |
| `LLM_CONCURRENCY` | `5` | 每轮内并发请求数 |

### `run_benchmark.py` 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | 数据集路径（CSV 或 JSONL） |
| `--gpus` | `0-7` | GPU 编号：`0,1,2,3` 或 `0-7` 或 `0`（单卡） |
| `--limit` | 全部 | 只跑前 N 题 |
| `--name` | 文件名 | benchmark 名称（影响输出目录名） |

### `launch_vllm.sh` 参数

```bash
bash launch_vllm.sh <MODEL_PATH> [NUM_GPUS]
# 默认 8 卡。每张卡 port = 8000 + gpu_id。
```

### 可视化（通过 SSH 隧道）

```bash
# 服务器上：
cd parliament
python serve.py --output_dir ../output/<timestamp>/

# 本地机器上（新开终端）：
ssh -p 8795 -L 18888:localhost:18888 root@your-server-ip

# 浏览器打开：http://localhost:18888/index.html
```

| 参数 | 说明 |
|------|------|
| `--port` | HTTP 端口（默认 18888） |
| `--refresh N` | 自动刷新间隔（秒），0 = 关闭 |

---

## 输出结构

**Demo 模式：**
```
output/<timestamp>/
├── parliament.db         # SQLite 数据库（帖子、评论、投票）
├── session.json          # 讨论记录
├── round_map.json        # 轮次 ↔ post/comment ID 映射
├── index.html            # 可视化页面
├── anomalies.jsonl       # 异常记录（调试用）
└── config.py             # 本次运行参数快照
```

**Benchmark 模式：**
```
output/gpqa_diamond/<timestamp>/
├── 0/                    # 第 0 题
│   ├── parliament.db
│   ├── session.json
│   ├── round_map.json
│   ├── index.html
│   └── judge_response.json
├── 1/
│   └── ...
├── results.jsonl         # 全部结果（答案、得分、轮次）
└── summary.json          # 准确率汇总
```

---

## Pipeline

```
Question → Parliament (20 scientists × N rounds) → Judge → ANSWER
```

1. **Parliament**：科学家在论坛讨论（发帖、评论、投票、关注、搜索、SymPy 计算）
2. **早停**：连续 2 轮无人改变论坛内容 → 自动提前结束
3. **Judge**：一位资深科学家旁听全程，阅读带日期和得分的讨论记录，给出最终答案

---

## FAQ

**vLLM 报 `"auto" tool choice requires...`**
→ 确认启动 vLLM 时加了 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`

**GPU 显存不够**
→ 降低 `--gpu-memory-utilization` 或 `--max-model-len`

**想先快速验证流程**
→ 改 `config.py` 里 `DEFAULT_NUM_AGENTS = 3`，`NUM_ROUNDS = 3`，单卡跑一题

**停掉所有 vLLM 实例**
→ `pkill -f 'vllm serve'`
