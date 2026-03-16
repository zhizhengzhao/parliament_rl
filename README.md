# Science Parliament

多个 LLM 科学家在共享论坛上协作解答难题，通过集体讨论和社区投票涌现出答案。

基于 [CAMEL](https://github.com/camel-ai/camel) + [OASIS](https://github.com/camel-ai/oasis)，用 `patches.py` 深度适配为科学议会环境。

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

**1. 启动 vLLM**

```bash
CUDA_VISIBLE_DEVICES=6 vllm serve /path/to/Qwen3.5-9B \
  --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**2. 运行议会**

```bash
python run_parliament.py --question "Investigate the polynomial P(x) = x^5 - 5x + 3. (a) Determine the number of real roots and their approximate locations. (b) Compute the discriminant of P(x). (c) Use Sturm's theorem to rigorously count the real roots in each interval [-3,-2], [-2,-1], [-1,0], [0,1], [1,2]. (d) Explain why P(x) is not solvable by radicals — what is its Galois group, and what makes quintic polynomials fundamentally different from quartics? (e) Apply Newton's method starting from x_0 = 1.5 and compute the first 5 iterations."
# 或从文件读取
python run_parliament.py --question_file question.txt
```

---

## 配置（`config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | 服务器模型路径 | 与 vLLM 启动时一致 |
| `DEFAULT_NUM_AGENTS` | `20` | 科学家数量 |
| `NUM_ROUNDS` | `20` | 讨论轮数 |
| `MAX_ITERATION` | `10` | 每 agent 每轮最多几步工具调用 |
| `LLM_CONCURRENCY` | `5` | 并发数（小于 agent 数时，后批 agent 能看到前批的发言） |

---

## 可视化

每轮结束后自动生成 `output/<timestamp>/index.html`，包含三个视图：
- **Forum** — 帖子按得分排序，评论可折叠，点击科学家名查看 profile
- **Scientists** — 按总得分排名，显示活跃度指标
- **Network** — 关注关系总览

**通过 ngrok 共享给其他人：**

```bash
python serve.py --output_dir output/<timestamp>/ --token YOUR_NGROK_TOKEN
```

**本地查看（跑完后）：**

直接双击 `index.html` 在浏览器打开即可。

---

## 输出结构

```
output/<timestamp>/
├── parliament.db    # SQLite（所有帖子、评论、投票、trace）
├── session.json     # 讨论记录（带科学家姓名）
├── index.html       # 可视化（每轮更新）
├── anomalies.jsonl  # 异常记录（无工具调用 / 异常，含完整 context）
├── config.py        # 该次运行配置快照
└── log/
```

---

## 常见问题

**GPU 显存不够** → 加 `--gpu-memory-utilization 0.4` 或降低 `--max-model-len`

**anomalies.jsonl 全是 no_tool_calls** → 确认 vLLM 加了 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`；不要用 Ollama

**运行慢** → 先用 `DEFAULT_NUM_AGENTS = 3` 测试

**index.html 没有** → Round 0 不生成，等第一轮 `[Round 1/N]` 完成
