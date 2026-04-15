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
- [x] Persona pool 随机采样（42 变体，3 扇区 × 8/6，每 session 随机分配）
- [x] Actor 投票范围代码强制（±1，防 LLM 幻觉绕过 schema）
- [x] Judge 评审标准收紧（无实质推进 = -1，"no neutral score"）
- [x] Actor 高分信号感知 + 严格结束条件（100% 确定才结束）
- [x] 默认 3+3（从 4 actors + 4 judges 缩减）
- [x] 删除 agent timeout（只保留 max_rounds for actor + session_end）
- [x] Judge 无 max_rounds 限制（只由 session_end 退出）
- [x] Queue drain（积攒多个分发 → 合并为一轮处理）
- [x] LLM 分层超时（connect=10s 检测故障 + total=300s 兜底）
- [x] 失败轮次静默跳过（不杀 agent，下轮重试）
- [x] 统一 processing.discard + set_event（Actor 离开 processing 即 set_event）
- [x] 内容全局排序（posts → comments → votes）
- [x] extract.py 轻量级数据重建 + 稳定排序

## RL Pipeline
- [x] extract.py: parliament.db → (context, action, reward, advantage) JSONL
- [ ] GRPO 训练脚本
- [ ] 评估 pipeline

## 实验设计
- [ ] Planted Expert: 注入答案的 agent 引导讨论
- [ ] 对比实验: Parliament RL vs 纯 SFT
- [ ] 多模型实验: 不同基座模型混合讨论
