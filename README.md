# Parliament RL

A multi-agent scientific forum + offline RL pipeline for training
LLM scientific reasoning. Several LLM scientists collaborate on one
problem per session — posting, commenting, voting on each other's
work — while LLM judges holding the reference solution cast
anonymous votes.

Those judge votes serve **two independent functions**:

1. **Offline reward** — the per-post reward fed into RL training
   (always on; without it there is no reward channel).
2. **Online steering** — when made visible to the Scientists
   *during* the session, judge votes reshape the discussion in
   real time (toggleable via `PRL_JUDGE_VOTES_VISIBLE`).

The resulting per-actor multi-turn trajectories become RL training
samples whose contexts are themselves shaped by whichever
rollout-context settings were active.

📖 **Start here**: [`docs/01_overview.md`](docs/01_overview.md)
(motivation + dual judge role) →
[`docs/02_parliament.md`](docs/02_parliament.md) (data-generation
architecture) →
[`docs/03_rl.md`](docs/03_rl.md) (RL pipeline details) →
[`docs/04_2x2_design.md`](docs/04_2x2_design.md) (2×2 ablation cells) →
[`docs/05_frame_eval.md`](docs/05_frame_eval.md) (in-frame evaluation) →
[`docs/06_knobs.md`](docs/06_knobs.md) (every experimental flag).

## Install

```bash
pip install -e .              # core stack
pip install -e ".[fast]"      # + fused attention kernels (~2-3x vLLM speedup)
pip install -e ".[dev]"       # + pytest
```

After install, every CLI is on `$PATH`:

| command | purpose |
|---|---|
| `prl-server`         | Parliament uvicorn API |
| `prl-run`            | one-shot data collection (vLLM HTTP fleet + Parliament + harness) |
| `prl-iterate`        | outer training loop (one merge+reload cycle per iter) |
| `prl-extract`        | `parliament.db` → `train.jsonl` |
| `prl-train`          | DDP + LoRA + PPO clip + KL anchor; outputs PEFT adapter |
| `prl-export`         | merge LoRA → base, write vLLM-loadable HF folder |
| `prl-eval-gpqa`      | GPQA Diamond zero-shot CoT |
| `prl-eval-mc`        | Sciencepedia held-out MC100 zero-shot |
| `prl-eval-frame`     | in-frame eval with secretary (see `docs/05_frame_eval.md`) |
| `prl-compare-policies` | multi-policy reward comparison on one shard |
| `prl-sample-dataset` | depth-5 category-uniform train/test split |

(`python -m parliament.server`, `python scripts/iterate.py`, etc.
also work — the entry-point names are just shorter aliases.)

Tests:

```bash
pytest                        # 89 pure-Python tests, ~10 s
```

## Quick start — iterative training

```bash
prl-iterate \
  --name nrun_v1 \
  --pool datasets/sciencepedia_train_part1.json,\
datasets/sciencepedia_train_part2.json,\
datasets/sciencepedia_train_part3.json,\
datasets/sciencepedia_train_part4.json \
  --total-questions 1000 --sampling-batch-size 200 \
  --total-epochs 2 --seed 42 \
  --gpus 0,1,2,3,4,5,6,7 --sessions-per-gpu 2 \
  --train-extra "--ppo-epochs 2 --clip-ratio-high 0.25 --beta-kl 0.005"
```

`prl-iterate` self-launches into a `parliament-iterate` tmux
session, survives SSH disconnect, and auto-resumes on re-invocation
via `data/<run>/state.json`. LoRA defaults (`r=64, α=128`) and the
reward formula are already the main-experiment values; `--train-extra`
only carries the per-iter PPO/KL knobs.

Three nested loops (verl-aligned, see [`docs/00_naming.md`](docs/00_naming.md)):

| level | flag | what it controls |
|---|---|---|
| outer  | `--total-epochs N`      | cycle the drawn schedule `N` times |
| middle | one per sampling round  | sample `B` questions → train |
| inner  | `--train-extra "--ppo-epochs N"` | SGD passes through `train.jsonl` |

All four 2×2 cells share the **same draw + schedule** via a common
`--seed`, removing question selection as a confounder.

Each iter does:

```
                              ┌── one iter ──┐
ensure_vllm(model=actor_N)  ─→  rollout (vLLM HTTP fleet, no LoRA)
                                   ↓
                              extract → train.jsonl
                                   ↓
                              stop_vllm  (free 80 GB / GPU for trainer)
                                   ↓
                              accelerate launch -m rl.train (DDP × N)
                                   ↓
                              rl.export: merge LoRA → run_dir/merged/
                                   ↓
                              actor_(N+1) := run_dir/merged
```

vLLM serves the previous iter's *merged* checkpoint as a plain HF
model (no `--enable-lora`), so cudagraph + prefix-cache + full KV
budget are all on and rollout speed equals the base model.  At the
end of each iter the LoRA adapter is merged back into the base via
`rl.export` and the next iter restarts vLLM from the new merged
folder.  Per-iter framework overhead (kill + train + merge +
restart) is ~120 s; the rest is real work.

## Project structure

```
parliament_rl/
├── parliament/              FastAPI server + SQLite store
├── harness/                 async agent runtime (HTTP client of the server)
├── rl/                      extract, train, export
├── eval/
│   ├── gpqa.py              GPQA Diamond zero-shot CoT
│   ├── sciencepedia_mc.py   held-out boxed-letter MC zero-shot
│   ├── secretary.py         cell-agnostic answer extractor (pure logic)
│   ├── frame.py             in-frame eval (rollout in own cell → secretary → score)
│   ├── frame_sweep.sh       4-cell sweep of frame.py
│   └── gpqa_sweep.sh        base + every iter merged through GPQA
├── scripts/                 CLI orchestrators (run, iterate, sample_dataset, …)
├── context_configs/
│   ├── shared/              name_pool, persona_pools, agent limits, judge prompt
│   │                        (single source of truth for all 4 cells)
│   ├── Parliament_context/  cells A/B: cell-specific actor prompt + override
│   ├── Solo_context/        cells C/D: cell-specific actor prompt + override
│   └── RL_context/          extract-side knobs (advantage normaliser etc.)
├── tests/                   pure-Python unit tests (no GPU/no vLLM/no model)
├── datasets/                Sciencepedia problems
├── docs/                    design docs (00 naming → 06 knobs)
└── data/                    run outputs (gitignored)
```

**Parliament** = `parliament/` + `harness/` + `context_configs/{shared, Parliament_context, Solo_context}/`
**RL**         = `rl/` + `context_configs/RL_context/`

## Datasets

| File | Purpose |
|---|---|
| `datasets/sciencepedia_test.json`              | 100 graduate-level problems, smoke size, disjoint from train |
| `datasets/sciencepedia_train_part{1..4}.json`  | 4 × 1 026 = 4 104 problems, depth-5-uniform sample |
| `datasets/sciencepedia_heldout_mc100.json`     | 100 boxed-letter MC, disjoint from both train & test |

Pass all four `train_part*.json` to `prl-iterate --pool`
(comma-separated) to get the full 4 104-question pool the main
experiment draws from.

Build new splits with `prl-sample-dataset`.

## Environment variables

Host-specific paths are read from env vars so the code runs unchanged
on any machine; defaults work out of the box from inside the target
Python env.

| variable | default | purpose |
|---|---|---|
| `PRL_PYTHON`     | `sys.executable`     | Python (or wrapper script) for accelerate trainer subprocess |
| `PRL_ACCELERATE` | `which accelerate`   | accelerate binary for `prl-train` |
| `PRL_MODEL_PATH` | `Qwen/Qwen3.5-9B`    | local model directory (vLLM needs a full path) |
| `PRL_CONTEXT`    | `Parliament`         | 2×2 cell context: `Parliament` (coupled) / `Solo` (independent) |
| `PRL_JUDGE_VOTES_VISIBLE` | `1`         | 2×2 cell visibility axis: `1` shows judge votes to actors, `0` hides |

Every flag and config field is catalogued in
[`docs/06_knobs.md`](docs/06_knobs.md).

## Versions (validated stack)

Python 3.11 · PyTorch 2.10 (CUDA 12.8) · vLLM 0.17.1 · Qwen3.5-9B ·
transformers ≥ 5.5 · accelerate ≥ 1.13 · peft ≥ 0.19 · FastAPI ≥ 0.135 ·
aiohttp ≥ 3.13. See `pyproject.toml` for the full pinned set.

## Run output layout

```
data/<name>_<timestamp>/
├── parliament.db          all posts/comments/votes + per-request log
├── experiment.json        run summary (per-agent stats, exit reasons)
├── llm_logs/<sid>/        every LLM call, grouped by session
├── discards/              no-tool streaks (debug)
├── train.jsonl            per-actor trajectories (after rl.extract)
└── ckpt/
    ├── config.json        trainer config snapshot
    ├── metrics.jsonl      per-step training metrics
    ├── ckpt/
    │   ├── config.json        trainer config snapshot
    │   ├── metrics.jsonl      per-step training metrics
    │   └── step_K/
    │       ├── adapter/       PEFT LoRA folder (input to rl.export)
    │       ├── optimizer.pt
    │       ├── scheduler.pt
    │       └── meta.json
    └── merged/                vLLM-loadable HF folder (after rl.export).
                               This is what the next iter's vLLM loads
                               as `--model`.  ~19 GB; pruned at every
                               iter except total-epoch boundaries.
```

`prl-iterate` additionally writes `state.json` (resume pointer),
`iterate.log`, `manifest.json`, and `shards/ep{E}.round{R}.json` at
the run root, plus a small archival copy under `backups/<run>/`.
