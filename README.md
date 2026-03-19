# Science Parliament

多个 LLM 科学家在论坛上协作解答难题，通过讨论、验证、投票涌现出答案，最后由 Judge 综合给出最终回答。

基于 [CAMEL](https://github.com/camel-ai/camel) + [OASIS](https://github.com/camel-ai/oasis)。

---

## 项目结构

```
parliament/              # 核心议会系统
├── config.py            # 所有可调参数（模型、agent、轮次、vLLM）
├── patches.py           # OASIS monkey-patches
├── session.py           # 核心逻辑：init() / create_model() / run_session()
├── run_parliament.py    # CLI：demo 模式跑单题
├── visualize.py         # 生成单题 HTML 可视化
└── serve.py             # HTTP 服务器（SSH 隧道访问）

judgement/               # Judge + Benchmark
├── judge.py             # Judge：读论坛 → 综合最终答案
├── run_benchmark.py     # 一键跑 benchmark（自动启动 vLLM）
├── vllm_manager.py      # vLLM 生命周期管理
└── benchmark_viz.py     # 生成 benchmark 总览 HTML

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

### 2. 修改配置

编辑 `parliament/config.py`：

```python
MODEL_NAME = "/path/to/your/model"    # 模型路径
```

### 3. 运行

**Demo（跑单题）：**

```bash
# 先手动启动 vLLM
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/model \
  --port 8000 --max-model-len 131072 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

# 另一个终端
cd parliament
python run_parliament.py --question "Prove that n(n+1)(n+2)(n+3)+1 is always a perfect square."
```

**Benchmark（一键跑 GPQA）：**

```bash
cd judgement
python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0,1,2   # 3 卡
python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0       # 单卡
python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv                # 全部 8 卡
```

自动完成：启动 vLLM → 跑 parliament + judge → 生成结果页面 → 启动 HTTP 服务器。

---

## 可控参数

### `parliament/config.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `"Qwen/Qwen3-8B"` | 模型路径 |
| `DEFAULT_NUM_AGENTS` | `20` | 科学家数量 |
| `NUM_ROUNDS` | `20` | 最大讨论轮数 |
| `MAX_ITERATION` | `10` | 每 agent 每轮最多工具调用次数 |
| `LLM_CONCURRENCY` | `5` | 每轮并发请求数 |
| `VLLM_MAX_MODEL_LEN` | `131072` | vLLM context 长度（128K） |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | vLLM GPU 显存占用比例 |

### `run_benchmark.py` 命令行

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | 数据集路径 |
| `--gpus` | `0-7` | GPU 编号：`0`、`0,1,2`、`0-7` |
| `--limit` | 全部 | 只跑前 N 题 |
| `--port` | `18888` | 结果页面 HTTP 端口 |

---

## 输出

**Demo：** `output/<timestamp>/`

**Benchmark：** `output/<bench_name>/<timestamp>/`

```
output/gpqa_diamond/2026-03-19_12-00-00/
├── index.html            # 总览页面（准确率 + 每题结果表格）
├── results.jsonl         # 每题详细记录
├── summary.json          # 准确率汇总
├── 0/                    # 第 0 题
│   ├── parliament.db     # 议会数据库
│   ├── session.json      # 讨论记录
│   ├── index.html        # 单题可视化（点击总览表格行跳转）
│   └── judge_response.json
├── 1/
└── ...
```

---

## Pipeline

```
Question → Parliament (N scientists × M rounds) → Judge → ANSWER
```

1. **Parliament**：科学家讨论（发帖、评论、投票、关注、搜索、SymPy）
2. **早停**：连续 2 轮无内容变更 → 提前结束
3. **Judge**：旁听全程，阅读带日期和得分的讨论记录，输出最终答案

---

## FAQ

| 问题 | 解决 |
|------|------|
| context 超长报错 | 增大 `VLLM_MAX_MODEL_LEN` 或减少 `DEFAULT_NUM_AGENTS` |
| agent 全部 no_tool_calls | 确认 vLLM 启动时加了 `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| 想快速验证流程 | `config.py` 里设 `DEFAULT_NUM_AGENTS = 3`，`NUM_ROUNDS = 3`，单卡跑一题 |
| 停掉所有 vLLM | `pkill -f 'vllm serve'` |
