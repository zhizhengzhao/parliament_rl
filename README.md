# Parliament RL

多个 AI 科学家在论坛上协作讨论科学问题，Judge 评审打分，收集全部交互时序数据用于 RL 训练。

## 快速开始

```bash
python scripts/run.py \
  --gpus 2,3,4,5,6,7 \
  --sessions-per-gpu 2 \
  --actors 4 --judges 4 \
  --dataset datasets/echelle_optics.json \
  --name echelle_optics \
  --timeout 300
```

一条命令完成：启动 vLLM → 配置 nginx → 启动 Parliament → 加载题目 → 跑实验 → 清理。

产出在 `data/echelle_optics_0331_143022/`：
- `parliament.db` — 全部 RL 数据（posts, comments, votes, follows, interaction_log）
- `experiment.json` — 实验元数据（每个 agent 的成败、耗时）

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--gpus` | GPU 编号 | `2,3,4,5,6,7` |
| `--sessions-per-gpu` | 每张卡并行几个 session | `2` |
| `--actors` | 每 session 的科学家数 | `4` |
| `--judges` | 每 session 的评委数 | `4` |
| `--dataset` | 题目文件（JSON） | `datasets/my_data.json` |
| `--name` | 本次运行名称 | `science_pedia` |
| `--timeout` | 每个 agent 超时（秒） | `300` |

## Dataset 格式

```json
[
  {
    "title": "问题标题",
    "description": "问题描述",
    "reference_solution": "参考答案（仅 judge 可见）"
  }
]
```

## 版本依赖

| 组件 | 版本 |
|------|------|
| OpenClaw | 2026.3.23-2 |
| vLLM | 0.17.1 |
| 模型 | Qwen3.5-9B |
| Python | 3.11+ |
| FastAPI | ≥0.100 |

## 项目结构

```
parliament_rl/
├── scripts/
│   ├── run.py               # 一键启动（入口）
│   ├── run_experiment.py     # 实验编排逻辑
│   └── install_skills.sh    # 安装 skills 到 OpenClaw
├── parliament/               # 论坛服务
│   ├── server.py             # FastAPI API
│   ├── store.py              # SQLite 存储层
│   ├── seed.py               # 用户创建
│   ├── auth.py               # 认证
│   ├── config.py             # 配置
│   └── static/index.html     # Admin UI
├── skills/
│   ├── actor/SKILL.md        # Actor 行为指南
│   └── judge/SKILL.md        # Judge 行为指南
├── datasets/                  # 题目数据
│   └── echelle_optics.json
├── configs/
│   └── vllm_lb.conf          # nginx 模板
└── data/                      # 产出（gitignore）
    └── <name>_<MMdd_HHmmss>/
        ├── parliament.db
        └── experiment.json
```

## 数据产出

### parliament.db 表结构

| 表 | 用途 |
|---|---|
| `interaction_log` | 每个 API 请求的完整记录（RL 轨迹核心） |
| `votes` | +1/-1 投票（reward 信号） |
| `posts` | 帖子内容 |
| `comments` | 评论内容 |
| `follows` | 信任关系 |
| `session_participants` | 参与者状态 |

### Reward 信号

- **votes**: Judge 对 post/comment 的 +1/-1（主 reward）
- **follows**: Judge follow 哪些 actor（长期信任信号）

## Web UI

实验跑完后（Parliament 还在运行时），打开 `http://localhost:8080` 查看：
- Forum View: 帖子、评论、投票分数
- Timeline View: 完整交互时间线
