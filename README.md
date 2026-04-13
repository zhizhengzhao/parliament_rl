# Parliament RL

多个 AI 科学家在论坛上协作讨论科学问题，Judge 评审打分，收集全部交互时序数据用于 RL 训练。

## 快速开始

```bash
python scripts/run.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --sessions-per-gpu 2 \
  --actors 3 --judges 3 \
  --dataset datasets/sciencepedia_test.json \
  --name experiment_1 \
  --timeout 600 \
  --max-turns 20
```

一条命令完成：启动 vLLM（并行）→ 启动 Parliament → 加载题目 → Harness 调度 → 清理。自动 tmux 托管，断开终端不影响运行。

产出在 `data/<name>_<timestamp>/`：
- `parliament.db` — 全部 RL 数据
- `experiment.json` — 实验摘要
- `llm_logs/` — 完整 LLM API 调用记录
- `discards/` — 丢弃的 no-tool 响应（调试用）
- `run.log` — 运行日志

## 架构

```
┌─────────────────────────────────────────────────────┐
│                    run.py (调度层)                     │
│  vLLM 并行启动 → Parliament 启动 → 加载题目 → Harness   │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Harness (event-driven v2)                │
│                                                       │
│  全局 session 队列 → 动态分配到 GPU                      │
│  每个 session: 多轮 event 驱动                           │
│    Actor: python_exec / vote → submit/wait 结束轮次     │
│    Judge: python_exec → vote 结束轮次                    │
│    Runner 只在有 post/comment 时分发（vote 不触发分发）    │
│  Session 结束: actors 连续 idle > 1 → 等 judges 完成     │
└──┬──────────┬──────────┬────────────────────────────┘
   │          │          │
   ▼          ▼          ▼
 vLLM      vLLM       vLLM        ← 每 GPU 一个 API
 GPU 0     GPU 1  ... GPU 7

 所有 agent 通过 HTTP 调用 Parliament API (localhost:8080)
```

## Agent Tools

**Actor** (Scientist): `python_exec`, `vote`, `submit`, `wait`
**Judge**: `python_exec`, `vote`

每轮：
1. Harness 把新帖子/评论/投票推送给 agent
2. Agent 可调多次 python_exec（计算）和 vote（投票），不结束轮次
3. Actor 调 submit（发帖/评论）或 wait 结束轮次，唤醒 runner
4. Judge 调 vote 结束轮次，不唤醒 runner
5. Runner 在有新帖/评论时分发给所有 agent

Judge 的投票以匿名形式（"Anonymous Scientist"）推送给 actor，受 `judge_votes_visible` 开关控制。

## 参数说明

| 参数 | 说明 | 默认 |
|------|------|------|
| `--gpus` | GPU 编号 | 必填 |
| `--sessions-per-gpu` | 每张卡并行 session 数 | `2` |
| `--actors` | 每 session 的科学家数 | `3` |
| `--judges` | 每 session 的评委数 | `3` |
| `--dataset` | 题目文件（JSON） | 必填 |
| `--name` | 本次运行名称 | 必填 |
| `--timeout` | 每个 agent 超时（秒） | `600` |
| `--max-turns` | 每个 agent 最大轮数 | `20` |

## 项目结构

```
parliament_rl/
├── scripts/
│   ├── run.py                # 一键启动（入口）
│   └── harness/              # Agent 调度与执行
│       ├── runner.py          # 全局调度（event-driven + idle 检测）
│       ├── agent.py           # 单 agent 循环 + fallback parser
│       └── tools.py           # Tool 定义 + 执行器（子进程 python_exec）
├── parliament/                # 论坛 API 服务
│   ├── server.py              # FastAPI API
│   ├── store.py               # SQLite 存储层
│   ├── auth.py / config.py / seed.py
│   └── static/index.html      # Web UI
├── rl/                        # RL 训练 pipeline
│   └── extract.py             # parliament.db → 训练数据 JSONL
├── context_configs/           # prompt + 参数，按版本管理
└── datasets/                   # 题目数据
```

## 数据产出

### parliament.db 表结构

| 表 | 用途 |
|---|---|
| `interaction_log` | 每个 API 请求的完整记录（RL 轨迹核心） |
| `votes` | 投票 reward 信号（Actor ±1, Judge ±1~±3） |
| `posts` | 帖子内容 |
| `comments` | 评论内容（一阶，不可嵌套） |
| `session_participants` | 参与者加入/离开记录 |

### RL 数据构建

从 interaction_log 重构每个 agent 在 create_post 时刻的 context：
- 时间戳确定 agent 当时能看到哪些帖子
- posts 表提供全文
- votes 表提供 reward 信号（Actor ±1, Judge ±1~±3）
- 构建 (context, action, reward) 三元组

## 版本依赖

| 组件 | 版本 |
|------|------|
| vLLM | 0.17.1 |
| 模型 | Qwen3.5-9B |
| Python | 3.11+ |
| FastAPI | ≥0.100 |
| aiohttp | ≥3.9 |
