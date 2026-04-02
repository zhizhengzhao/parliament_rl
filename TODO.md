# TODO — Parliament RL

## 已完成
- [x] Parliament server（API + SQLite + interaction_log）
- [x] Judge 权限限制（不能 post/comment，只能 vote/follow）
- [x] Web UI（论坛视图 + 时间线视图）
- [x] Actor/Judge skill 文件
- [x] Agent slot 隔离（消除 OpenClaw 文件锁竞争）
- [x] 每 session 前后清理 OpenClaw，实验后彻底还原
- [x] 一键启动脚本 `run.py`（vLLM + nginx + Parliament + 实验）
- [x] vLLM 逐个启动，全部成功才继续
- [x] 数据按 `data/<name>_<timestamp>/` 组织
- [x] Dataset 从 JSON 文件加载

## RL Pipeline
- [ ] Event export: interaction_log → 统一事件流
- [ ] State reconstruction: 重建每个 agent 的 observation
- [ ] Reward aggregation: vote/follow → reward
- [ ] AWBC 加权 SFT 数据生成
- [ ] 训练 pipeline

## 实验设计
- [ ] Planted Expert: 注入答案的 agent 引导讨论
- [ ] 对比实验: Parliament RL vs 纯 SFT
- [ ] 多模型实验: 不同基座模型混合讨论
