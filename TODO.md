# TODO — Parliament RL

## 已完成
- [x] Parliament server（API + SQLite + interaction_log）
- [x] Judge 权限限制（不能 post/comment，只能 vote）
- [x] Web UI（论坛视图 + 时间线视图）
- [x] Harness 轮询模式（submit/wait/leave + 动态 GPU 负载均衡）
- [x] 一键启动脚本 `run.py`（vLLM 并行 + Parliament + Harness）
- [x] 数据按 `data/<name>_<timestamp>/` 组织
- [x] Dataset 从 JSON 文件加载
- [x] Fallback tool call parser + no-tool resample
- [x] LLM 调用全记录（llm_logs/ + discards/）
- [x] Vote 分发（actor 实名，judge 匿名，受开关控制）
- [x] Idle 检测修正（全 actor 完成才计数）

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
