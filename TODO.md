# TODO

## Done

### Parliament & harness (data generation)
- FastAPI + SQLite server, per-request interaction log, Judge role-level permissions
- Web UI (forum + timeline, score sort, KaTeX math)
- One-click pipeline (`scripts/run.py`): cleanup → parallel vLLM → Parliament → harness → cleanup
- Event-driven runner with split actor/judge processing sets, idle detection, queue drain
- Four actor tools (`python_exec` / `vote` / `submit` / `wait`) and two judge tools; vote range enforced server- and client-side
- Defensive parameter coercion (comment ↔ comments, JSON-in-string, embedded votes), fallback tool-call parser, no-tool resample, all with explicit model-facing feedback
- Subprocess-isolated `python_exec` (10 s timeout, 10 k-char truncation)
- `force_close` TCP to eliminate keep-alive races during long LLM calls
- Per-session deterministic anonymization: 406-name pool + 100-persona scientist + 60-persona judge pools, single-sample sort-without-replacement guarantees no intra-session collision
- Judge vote anonymization (`judge_votes_visible` toggle, default on)

### RL pipeline
- `rl/extract.py`: parliament.db → training JSONL
  - natural-language context with score-annotated headers
  - session-seeded name mapping (single source of truth via `harness.prompts.assign_session_names`)
  - automatic score-visibility detection (kept only in sessions whose content meta-references scoring)
  - configurable advantage: `advantage_baseline` × `advantage_scale` combinations (session/global/constant)
  - judge-only reward (actor votes visible as context signal, don't bias reward)
  - streaming vote fold, `O((P+C+V)·log V)`
- `rl/train.py`: single-file FSDP2 offline RL trainer
  - PPO ratio + clip (default on) or RWR (with `--no-use-ratio`)
  - KL k3 estimator in fp32 with log-ratio clamp (BF16-stable)
  - grad clip, advantage clip, weight decay, cosine + warmup schedule
  - sharded checkpoint save/load, `--keep-last N` rotation
  - clean distributed shutdown (avoids NCCL teardown hang)
- `rl/export.py`: sharded FSDP → merged HF folder (vLLM-loadable for next iteration)
- `scripts/iterate.py`: multi-shard sample → train → export → repeat loop with state-file resume

## Next
- Run the full 4-shard iterative training (`scripts/iterate.py --shards part1..4`, ~54 h on 8 × A100)
- Evaluate each iteration's policy on the held-out `sciencepedia_test.json` (100 problems)
- Direction 2 exploration: `--judges 0` runs (pure peer-review, no ground-truth anchor)
- Optional: install `flash-linear-attention` + `causal-conv1d` to accelerate Qwen3.5's hybrid-attention layers (~2–3× speedup, currently using torch fallback)
