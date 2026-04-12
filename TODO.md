# TODO — Parliament RL

## 已完成
- [x] Parliament server（API + SQLite + interaction_log + session close）
- [x] Judge 权限限制（不能 post/comment，只能 vote）
- [x] Web UI（论坛视图 + 时间线视图 + Score 排序）
- [x] 一键启动脚本 `run.py`（vLLM 并行 + Parliament + Harness + tmux 托管）
- [x] 数据按 `data/<name>_<timestamp>/` 组织
- [x] Dataset 从 JSON 文件加载 + 切分工具
- [x] Fallback tool call parser + no-tool resample
- [x] LLM 调用全记录（llm_logs/ + discards/）
- [x] Vote 独立 tool（actor 不结束轮，judge 结束轮不唤醒 runner）
- [x] Vote 分发（actor 实名，judge 匿名，受开关控制）
- [x] Idle 检测（只看 post/comment，actor_processing + judge_processing 分离）
- [x] 子进程 python_exec（10s 超时，不阻塞事件循环）
- [x] 自投票客户端过滤（ToolExecutor 跟踪 my_posts/my_comments）
- [x] force_close TCP 连接（消除 keep-alive ConnectionReset）
- [x] 参数格式兜底（comment/comments、JSON-in-string、submit 内 votes 提取）
- [x] 所有错误有 feedback（零静默丢弃）
- [x] Prompt 优化（秘书模型、鼓励讨论、反对 vote→wait）

## RL Pipeline
- [ ] Event export: interaction_log → 统一事件流
- [ ] State reconstruction: 重建每个 agent 的 observation
- [ ] Reward aggregation: vote → reward（Judge ×3 权重）
- [ ] AWBC 加权 SFT 数据生成
- [ ] 训练 pipeline

## 实验设计
- [ ] Planted Expert: 注入答案的 agent 引导讨论
- [ ] 对比实验: Parliament RL vs 纯 SFT
- [ ] 多模型实验: 不同基座模型混合讨论
