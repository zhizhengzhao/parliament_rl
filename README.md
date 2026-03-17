# Science Parliament

多个 LLM 科学家在共享论坛上协作解答难题，通过集体讨论和社区投票涌现出答案。由独立的 Judge 综合论坛共识给出最终回答。

基于 [CAMEL](https://github.com/camel-ai/camel) + [OASIS](https://github.com/camel-ai/oasis)，用 `patches.py` 深度适配为科学议会环境。

---

## 项目结构

```
parliament/              # 核心议会系统
├── config.py            # 所有可调参数
├── patches.py           # OASIS monkey-patches
├── session.py           # 核心逻辑：init() / create_model() / run_session()
├── run_parliament.py    # CLI 入口：demo 模式跑单题
├── visualize.py         # 自动生成 HTML 可视化
└── serve.py             # HTTP 服务器 + ngrok 隧道

judgement/               # Judge + Benchmark
├── judge.py             # Judge：读取论坛记录 → 综合最终答案
└── run_benchmark.py     # 批量跑数据集（GPQA 等）

benchmark/               # 数据集
├── gpqa_diamond.csv     # GPQA Diamond（198 题，选择题）
└── open_ended/          # 非选择题（预留）
```

所有输出（`output/`、`log/`）始终生成在项目根目录下，不受 `cd` 位置影响。

---

## 安装

> ⚠️ `camel-ai` 和 `camel-oasis` 版本敏感，`patches.py` 依赖其内部 API，**勿随意升级**。

```bash
conda create -n parliament python=3.11 -y && conda activate parliament
pip install vllm
pip install camel-ai==0.2.89
pip install "sympy>=1.13"
pip install camel-oasis==0.2.5 --no-deps
pip install "pandas>=2.2" "igraph>=0.11" "sentence-transformers>=3.0" "neo4j>=5.23"
pip install python-dotenv pyngrok
```

---

## 运行

### 1. 启动 vLLM

```bash
CUDA_VISIBLE_DEVICES=6 vllm serve /path/to/Qwen3.5-9B \
  --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

### 2a. Demo 模式（跑单题）

```bash
cd parliament
python run_parliament.py --question "Prove that n(n+1)(n+2)(n+3)+1 is always a perfect square."
```

### 2b. Benchmark 模式（跑 GPQA）

```bash
cd judgement
python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --limit 10
```

---

## 输出结构

**Demo 模式：**
```
output/<timestamp>/
├── parliament.db       session.json       index.html       anomalies.jsonl       config.py
```

**Benchmark 模式：**
```
output/gpqa_diamond/<timestamp>/
├── 0/                  # 第 0 题
│   ├── parliament.db   session.json   index.html   judge_response.json
├── 1/                  # 第 1 题
│   └── ...
├── results.jsonl       # 每题：答案、是否正确、轮次
└── summary.json        # 准确率
```

---

## 配置（`parliament/config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_NUM_AGENTS` | `20` | 科学家数量 |
| `NUM_ROUNDS` | `20` | 最大讨论轮数 |
| `MAX_ITERATION` | `10` | 每 agent 每轮最多几步工具调用 |
| `LLM_CONCURRENCY` | `5` | 并发数 |

**早停**：连续 2 轮无内容变更（无发帖/评论/投票） → 自动提前结束。

---

## Pipeline

```
Question → Parliament (20 scientists × N rounds) → Judge → ANSWER
```

1. **Parliament**：科学家在论坛讨论（发帖、评论、投票、关注、搜索）
2. **早停检测**：连续 2 轮无人改变论坛内容 → 提前结束
3. **Judge**：资深科学家旁听全程，阅读按得分排序的完整讨论，给出最终答案
