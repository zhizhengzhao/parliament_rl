# Ablation surface — every experimental knob in one place

A grouped reference of every flag and config field that varies a
documented experimental dimension. Each item points at where the
flag lives, what its valid values are, and what it controls.

## 1. Cell selection (the 2×2 axis)

See [`04_2x2_design.md`](04_2x2_design.md) for the full design.

| Cell | Env vars | Cell-specific config |
|---|---|---|
| A — Parliament      | `PRL_CONTEXT=Parliament` (default), `PRL_JUDGE_VOTES_VISIBLE=1` (default) | `context_configs/Parliament_context/` |
| B — BlindParliament | `PRL_CONTEXT=Parliament`, `PRL_JUDGE_VOTES_VISIBLE=0`                        | `context_configs/Parliament_context/` |
| C — Solo            | `PRL_CONTEXT=Solo`,       `PRL_JUDGE_VOTES_VISIBLE=1` (default)             | `context_configs/Solo_context/`       |
| D — BlindSolo       | `PRL_CONTEXT=Solo`,       `PRL_JUDGE_VOTES_VISIBLE=0`                        | `context_configs/Solo_context/`       |

`PRL_JUDGE_VOTES_VISIBLE` accepts truthy values: anything except
`""`, `0`, `false`, `no`, `off` (case-insensitive).

## 2. Cohort + cardinality

| Knob | Where | Effect |
|---|---|---|
| `--actors N`           | `scripts/run.py`, `scripts/iterate.py` | Number of Scientist agents per session |
| `--judges M`           | (same)                                  | Number of Judge agents per session |
| `--judges 0`           | (same)                                  | **Direction 2 (peer-review-only)** — drops the offline reward channel; needs an alternative reward extractor (not implemented in `rl/extract.py` yet) |
| `agent.max_rounds`     | `context_configs/shared/config.json`    | Actor round cap (judges run unbounded) |
| `agent.step_limit`     | (same)                                  | Max LLM calls *within* a single round |
| `agent.max_no_tool_retries` | (same)                             | Tolerated no-tool responses per round |
| `agent.max_consecutive_errors` | (same)                          | LLM API error budget before retiring the round |
| `agent.llm_timeout_s`  | (same)                                  | Per-LLM-call timeout |
| `agent.max_tokens`     | (same)                                  | Completion token cap per LLM response |

## 3. Identity & persona

| Knob | Where | Effect |
|---|---|---|
| `name_pool`                   | `context_configs/shared/config.json` | Pool from which per-session actor / judge cast names are drawn (deterministic, seeded by `session.title`) |
| `persona_pools.scientist`     | (same) | 100 cognitive-style persona snippets; 3 sampled per session without replacement |
| `persona_pools.judge`         | (same) | 60 evaluative-style snippets; 3 sampled per session without replacement |
| `anonymize_identity`          | `context_configs/RL_context/config.json` | When false, training-context headers show raw `Scientist_N` instead of pool names (debug) |

## 4. Training data extraction (`rl/extract.py`)

| Knob | Default | Effect |
|---|---|---|
| `--strip-vote-events`       | **on**   | Remove all vote-event lines from training context; reward still flows through `turn_advantages`. Use `--no-strip-vote-events` for the legacy heuristic (insert vote events when the actor's content references voting). |
| `--no-template-augment`     | off      | Force every wrapper template to its first variant (byte-stable; ablation against the augmented run). 107 variants across 12 keys live in `rl/extract.py:TEMPLATE_POOL`. |
| `advantage_baseline`        | `mean_session` | One of `0.0`, `"mean_session"` (classic GRPO), `"mean_global"` (REINFORCE++-style), or any number. Lives in `RL_context/config.json`. |
| `advantage_scale`           | `session_std`  | One of `"session_std"` (per-session std, GRPO default), `"global_std"`, `"none"`, or any number. |
| `min_content_chars`         | 20       | Posts shorter than this become non-trainable (still in context, mask=0). |

## 5. Training (`rl/train.py`)

| Knob | Default | Effect |
|---|---|---|
| `--ppo-epochs N`             | 2     | SGD passes through `train.jsonl` per iter (alias `--num-epochs` kept for legacy resume). |
| `--use-ppo-clip`             | on    | PPO ratio + asymmetric clip surrogate. `--no-use-ppo-clip` degenerates to RWR. |
| `--clip-ratio-low`           | 0.2   | Lower trust-region bound = `1 - clip_ratio_low`. |
| `--clip-ratio-high`          | 0.25  | Upper trust-region bound = `1 + clip_ratio_high` (asymmetric, DAPO-style). |
| `--beta-kl`                  | 0.005 | KL anchor to frozen base; `0` disables (DAPO-style reference-free). |
| `--advantage-clip`           | 0.0   | `|A|` cap; 0 = off (default). Mainline projects (TRL/OpenRLHF/Verl/DeepSeek-R1/DAPO) don't explicitly clip. |
| `--max-seq-len`              | 8192  | Truncate over-length samples at the nearest user-turn edge (0 = keep full). |
| `--lora-r`                   | 64    | LoRA rank — main-experiment default. |
| `--lora-alpha`               | 128   | LoRA scaling — paired with rank. |
| `--lora-target-modules`      | `q_proj,v_proj,o_proj` | Empty string = `"all-linear"` (covers attention + MLP + linear-attention proj; much higher KL drift risk). |
| `--save-every`               | 0     | 0 = save once at the end of training (one ckpt per iter under `iterate.py`). |
| `--keep-last`                | 1     | Rotate to keep this many `step_*` checkpoints (0 = all). |

`scripts/iterate.py --train-extra "..."` forwards arbitrary flags
into every iter's `rl.train` invocation, e.g.
`"--ppo-epochs 2 --beta-kl 0 --clip-ratio-high 0.28"` for a
DAPO-style reference-free run.

## 6. Evaluation

See [`05_frame_eval.md`](05_frame_eval.md) for in-frame evaluation
specifics.

| Knob | Where | Effect |
|---|---|---|
| `--secretary <model>`        | `eval/frame.py`             | Fixed extractor model (default `$PRL_MODEL_PATH`). Always the original base — never a trained policy. |
| `--secretary-thinking`       | `eval/frame.py`             | Enable Qwen3.5 thinking mode for the secretary (off by default; mechanical extraction needs no CoT). |
| `--include-comments`         | `eval/frame.py`             | Pass actor comments to the secretary in coupled cells (auto-inferred per cell by default). |
| `--enable-thinking`          | `eval/gpqa.py`, `eval/sciencepedia_mc.py` | Qwen3.5 thinking mode for zero-shot CoT eval (on by default). |
| `--n`                        | (same)                      | Samples per question; >1 + temperature>0 → majority vote. |

## 7. Pool draw + schedule (`scripts/iterate.py`)

| Knob | Effect |
|---|---|
| `--pool a.json,b.json,…`    | One or more JSON files concatenated in-order to form the question bank. |
| `--total-questions N`        | Drawn from `--pool` once at launch via `random.Random(--seed).sample(pool, N)`. |
| `--sampling-batch-size B`    | Questions per rollout round (= one iter). |
| `--total-epochs E`           | Cycle the drawn schedule `E` times. `total_iters = E × (N // B)`. |
| `--seed S`                   | Same seed across cells ⇒ byte-identical schedule, removing question selection as a confounder. |
| `--initial-model PATH`       | Starting policy *and* fixed KL-anchor target. Defaults to `$PRL_MODEL_PATH`. |
| `--no-eval`                  | Skip per-iter GPQA evaluation (saves ~10 min/iter). |

## 8. Hardware orchestration

| Knob | Where | Effect |
|---|---|---|
| `--gpus 0,1,…`               | `scripts/run.py`, `scripts/iterate.py`, `eval/frame.py` | GPUs to use; one vLLM (TP=1) per GPU |
| `--sessions-per-gpu`         | (same) | Concurrent sessions per GPU slot. Default 2; **for ≥ 100-question rollouts try 4** — KV cache budget per GPU is ~80 GB, observed usage is < 10 GB at 2 sess/GPU, so we are far from saturation and more concurrent sessions amortise the long-tail wait |
| `--in-tmux`                  | (internal)                   | Skip the tmux self-relaunch (used when already orchestrating) |
| `PRL_PYTHON`                 | env                          | Python binary for vLLM / accelerate child processes (default = `sys.executable`) |
| `PRL_ACCELERATE`             | env                          | accelerate binary (default = `which accelerate`) |
| `PRL_MODEL_PATH`             | env                          | Local model directory the base config points at (default = `Qwen/Qwen3.5-9B`) |

`scripts/_common.py:FORWARDED_ENV` lists every env var that is
propagated through tmux child shells.  In particular `PRL_CONTEXT`
and `PRL_JUDGE_VOTES_VISIBLE` are forwarded — without that, every
2×2 cell silently degrades into cell A.
