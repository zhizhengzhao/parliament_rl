# Parliament RL — Project Memory

> 给新对话窗口的完整记忆。读完这个文件后你应该能像连续对话一样继续工作。

## 项目本质

多个 AI 科学家（Actor）在论坛上协作讨论科学问题，Judge 持有参考答案评审打分，收集全部交互数据构建 (context, action, reward) 三元组用于 RL 训练。

## 当前架构：轮询模式（Polling）

### 三层
- **Parliament**（`parliament/`）：FastAPI 论坛 API + SQLite，纯 API 服务
- **Harness**（`scripts/harness/`）：Agent 调度与执行，纯 async Python，零子进程
- **vLLM**：每 GPU 一个实例，并行启动

### Actor 有四个 tool
- `python_exec`：数学计算，**不结束当前轮**，可调多次
- `submit`：一次性提交 post + comments + votes，**结束当前轮**
- `wait`：等待新内容再行动，**结束当前轮**
- `leave`：永久离开 session，**不可逆**

### Judge 只有两个 tool
- `python_exec`：验算
- `submit`：投票 +1/-1（不能发帖/评论，投票对 actor 不可见）

### 轮询协议（纯轮询，无 event）
1. Round 0: Actor 收到 "Parliament is empty. Begin." → 计算 → submit
2. Judge 不启动直到有内容
3. Runner 等 `processing` 集合清空（所有 agent 做完当前轮）→ 然后分发
4. Agent 调 submit/wait/leave → 从 `processing` 移除
5. **分发规则**：
   - Judge 只收 post + comment（不收任何 vote）
   - Actor 收 post + comment + actor_vote + judge_vote（`judge_votes_visible=true` 时）
   - Actor 收 post + comment + actor_vote（`judge_votes_visible=false` 时）
   - Judge vote 匿名化（署名 "Anonymous Scientist"）
   - 收到内容的 agent 加入 `processing`
6. **idle 只看 post + comment**（vote 不影响 idle）
7. idle=1,2 → nudge actors
8. idle≥3 → session 结束
9. 所有 actor 都 leave/done → session 立即结束

### Context 管理
1. Assistant thinking（content 字段）每轮清空为 null（和 tool_calls 同级的 content）
2. Submit 的 arguments **保留原文**（post/comments 不 strip，作为 agent 自我记忆）
3. No-tool-call 的响应直接丢弃不拼入 context（resample）

### No-tool-call 处理（三层防御）
1. **vLLM 原生解析**：qwen3_coder parser 正常工作时直接用
2. **兜底 parser**：从 content 中的 `<tool_call><function=...>` 标签提取
3. **Resample**：丢弃响应 + 检查 queue 是否有新内容 + 重试。连续 3 次 → 结束该轮

### ID 格式
- 帖子：`P_xxx`（如 `[P_3] by Scientist_1: ...`）
- 评论：`C_xxx`（如 `[C_5] on P_3 by Scientist_2: ...`）
- Actor 投票：`[V on P_3] by Scientist_1: +1`
- Judge 投票（匿名）：`[V on P_3] by Anonymous Scientist: -1`
- 改票：`[V on P_3] by Scientist_1: changed +1 → -1`

### 投票
- +1 正确/推进，-1 错误/冗余（无取消投票）
- 改票：删旧投票 + 插新投票（新 vote_id），DB 记录 `previous_value`
- 分数权重：Judge ×3，Actor ×1（仅用于 reward 计算，Parliament 运行时票面均为 1）
- Actor 的 vote 推送给其他 agent（署真名）
- Judge 的 vote 匿名推送给 actor（署 "Anonymous Scientist"），受 `judge_votes_visible` 开关控制
- Judge 不收任何 vote

### 已删除的功能
- **Follow/Unfollow**：完全移除（tools、executor、prompts）
- **Vote=0**：移除取消投票功能
- **Submit content stripping**：不再删除 post/comments 文本（保留作为自我记忆）
- **context_logs/**：已被 llm_logs/ 和 discards/ 替代
- **result.log**：已从 AgentResult 中删除

## 限制值一览

| 限制 | 值 | 位置 |
|------|-----|------|
| `MAX_CONSECUTIVE_ERRORS` | 3 | config.json（LLM API 连续报错） |
| `MAX_NO_TOOL_RETRIES` | 3 | agent.py（连续 no-tool resample） |
| `max_rounds` | 20 | config.json（agent 最大轮数） |
| `step_limit` | 20 | agent.py（单轮最大步数） |
| `idle_rounds` nudge | 1,2 | runner.py（idle=1和2都nudge） |
| `idle_rounds` 终止 | ≥3 | runner.py |
| `timeout` | 600s | config.json |
| `llm_timeout` | 120s | config.json |
| `max_tokens` | 4096 | agent.py（LLM 单次输出） |

## Context Config 系统

`context_configs/` 下按版本目录。代码自动加载**最新版本**。

当前版本：`2026_4_9_v3`

**不要删除旧版本**——它们记录了不同 prompt 的效果对比。

## 输出目录结构

```
data/{name}_{timestamp}/
  parliament.db      # 核心 RL 数据
  experiment.json    # 实验摘要
  llm_logs/          # 成功的 LLM API 调用（完整 request + response）
    {session_id}/
      {agent_name}.jsonl
  discards/          # 丢弃的 no-tool 响应，按连续次数分文件
    {session_id}/
      {agent_name}_x{N}.jsonl
  run.log
  parliament.log
```

## 文件清单

```
parliament/         # 论坛 API
scripts/run.py      # 一键入口
scripts/harness/    # agent.py + runner.py + tools.py
context_configs/    # prompt + 参数，按版本管理
datasets/           # 题库
.cursor/memory.md   # 就是这个文件
```

## 重要提醒

### 代码层面
- **Parliament 是纯 API 服务**——不要在 Parliament 里放 harness 逻辑
- **assistant 的 content 和 tool_calls 是同级的**——content 是 thinking，清空不影响 tool_calls
- **submit 的 arguments 保留原文**——post/comments 不 strip，因为 agent 的帖子不会通过 queue 推回给自己，arguments 是唯一的自我记忆
- **纯轮询，无 event**——runner 等 `processing` 集合清空再分发，60s 保底
- **Judge 的 vote 匿名推送给 actor**——受 `judge_votes_visible` 开关控制
- **idle 只看 post + comment**——vote 不影响 idle 计数
- **所有 actor done → session 立即结束**——不等 judge

### 用户偏好
- **代码精简高效稳定**——不接受凑合
- **先讨论方案再写代码**——不要自作主张
- **每次改动跑实验验证**——12 题 mini 集快速验证，100 题全量确认
- **重视数据质量**——RL 数据的 reward 区分度是核心指标
- **项目要开源**——代码必须 clean and tidy
- **tmux 启动实验**——防止 Cursor 断开导致进程死亡

### 容易犯的错
- `run.py` 的参数名 `--max-turns` 实际传给 runner 的是 `max_rounds`
- context_config 的 prompt 模板用 `{name}` 不是 `{{name}}`——Python str.format()
- vLLM 的 model name 是完整路径 `/root/zhizheng/models/Qwen3.5-9B`

## 部署信息

- **GitHub repo**：https://github.com/zhizhengzhao/parliament_rl
- **中转服务器**（用于 push，因本机无法直连 GitHub）：
  - host: `gvym1445879.bohrium.tech`
  - user: `root`
  - password: `4L5jbHptwK7gONmk`
  - 用法：scp 打包文件到中转 → ssh 进去 clone repo → 解压覆盖 → commit + push

## 下一步方向

1. **RL 数据构建**：从 interaction_log 重构 context → (context, action, reward) 三元组
2. **大规模数据生成**：用 sciencepedia_train.json 跑大规模生产
3. **RL 训练 pipeline**：AWBC 加权 SFT
