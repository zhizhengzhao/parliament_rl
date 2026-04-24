# Parliament RL

An experimental environment for decentralized multi-agent reasoning
and the offline-RL pipeline built on top of it.

Several LLM scientists collaborate on one scientific problem per
session, posting, commenting, and voting on each other's work. Judges
hold reference solutions and cast anonymous votes that double as (a)
per-post reward and (b) an online steering signal that reshapes the
discussion toward correct directions. The resulting per-post
`(context, action, reward)` triples are lifted into RL training data.

See [`docs/01_overview.md`](docs/01_overview.md) for the research
motivation and the two-fold role of the Judge. See
[`docs/02_parliament.md`](docs/02_parliament.md) for the data-
generation architecture and [`docs/03_rl.md`](docs/03_rl.md) for the
RL side.

## Quick start

One command for iterative training (launches in tmux, survives SSH
disconnect, auto-resumes on re-invocation):

```bash
python scripts/iterate.py \
  --name nrun_v1 \
  --pool datasets/sciencepedia_train_part1.json,\
datasets/sciencepedia_train_part2.json,\
datasets/sciencepedia_train_part3.json,\
datasets/sciencepedia_train_part4.json \
  --total-questions 1000 --sampling-batch-size 200 \
  --total-epochs 2 --seed 42 \
  --train-extra "--ppo-epochs 2 --clip-ratio-high 0.25 --beta-kl 0.005" \
  --gpus 0,1,2,3,4,5,6,7
```

Three nested loops (verl-aligned, see [`docs/00_naming.md`](docs/00_naming.md)):

| level | name | knob | example |
|---|---|---|---|
| outer | **total_epoch** | `--total-epochs N` | cycle the drawn schedule `N` times |
| middle | **iter** | one per sampling round | sample `B` questions → train → export |
| inner | **ppo_epoch** | `--train-extra "--ppo-epochs N"` | SGD passes through `train.jsonl` (PPO clip applies) |

All 2×2 cells share the **same draw + schedule** via a common
`--seed`: `random.sample(pool, total_questions)` cached on launch,
then split into `rounds = total_questions / sampling_batch_size`
per-round batches.

Each iter does

```
vLLM + Parliament + harness  →  parliament.db
         rl/extract            →  train.jsonl
         rl/train              →  PPO ratio + clip + KL anchor → step_K
         rl/export             →  merged HF folder (next iter's policy)
         backup                →  backups/<run>_<ts>/ep{E}.round{R}/
```

For a single-shot data collection without training, use
`scripts/run.py` directly; see the "Scripts" table below.

## Project structure

The project splits into **Parliament** (data generation) and **RL**
(data consumption):

```
parliament_rl/
├── parliament/              # FastAPI server + SQLite store
├── harness/                 # async agent runtime (LLM client of the server)
├── rl/                      # extract, train, export
├── eval/                    # benchmarks (GPQA Diamond, extensible)
├── scripts/
│   ├── run.py               # single run (vLLM + Parliament + harness)
│   ├── iterate.py           # iterative training loop over a deterministic pool draw
│   └── sample_dataset.py    # split a full dataset into train/test
├── context_configs/
│   ├── Parliament_context/  # agent-facing prompts + persona + name pool
│   └── RL_context/          # RL-context rendering knobs
├── datasets/                # Sciencepedia problems (test + 4×train_part)
├── docs/                    # design docs (01_overview, 02_parliament, 03_rl)
└── data/                    # run outputs (gitignored)
```

**Parliament** = `parliament/` + `harness/` + `Parliament_context/`
**RL**         = `rl/` + `RL_context/`

## Scripts

| script | purpose | typical wall time |
|---|---|---|
| `scripts/run.py` | one-shot data collection (cleanup → vLLM → Parliament → harness) | ~6-9 h for 1 026 questions on 8 GPU |
| `scripts/iterate.py` | sample → extract → train → export → repeat over the drawn schedule × `total_epochs` | ~1.5 h per iter at `sampling_batch_size=200` |
| `scripts/sample_dataset.py` | split a full dataset by depth-5 category | seconds |
| `python -m rl.extract` | `parliament.db` → `train.jsonl` (per-actor trajectories) | ~1 min per iter |
| `python -m rl.train` (via `accelerate launch`) | DDP + LoRA + PPO clip + KL anchor | ~5 min per ppo_epoch for a 200-q iter on 8 GPU |
| `python -m rl.export` | LoRA → merged HF directory (vLLM-loadable) | ~3 min |
| `python -m eval.gpqa` | GPQA Diamond zero-shot CoT accuracy (thinking mode) | ~15 min/model on 1 × A100 |
| `python -m eval.sciencepedia_mc` | held-out 100 multiple-choice questions (disjoint from train) | ~10 min/model |
| `eval/gpqa_sweep.sh` | sweep base + every iter's merged policy through GPQA | ~1 h for 5 models |

All scripts are resumable where that makes sense (iterate via
`state.json`; train via `--resume ckpt/step_K`).

## Ablation surface

Most experimental knobs are already exposed in `context_configs/` or
as CLI flags:

- **2×2 ablation cells** (coupling × judge-visibility, see
  [`docs/04_2x2_design.md`](docs/04_2x2_design.md)):
  - A — Parliament       (default: coupled, visible)
  - B — BlindParliament  (`PRL_JUDGE_VOTES_VISIBLE=0`)
  - C — Solo             (`PRL_CONTEXT=Solo`)
  - D — BlindSolo        (both env vars)
- **Agent cardinality**: `--actors N --judges M`.
- **Remove the judge entirely**: `--judges 0` exercises Direction 2
  (pure peer-review voting, no reference-solution anchor).
- **Persona / name pools**: edit `Parliament_context/config.json` —
  session-level deterministic resampling guarantees no within-session
  collisions.
- **Identity anonymization**: `anonymize_identity: true/false` in
  `RL_context/config.json` — if false, training headers show the raw
  `Scientist_N` instead of names (useful for debugging).
- **Template augmentation**: every training-side wrapper string
  (section header, post header, vote event line, anonymous voter
  label, etc.) is sampled from a 107-string pool in
  `rl/extract.py:TEMPLATE_POOL`. `python -m rl.extract
  --no-template-augment` falls back to a single fixed wrapper for
  ablation against the augmented version.
- **Vote events** appear in training context only when the session's
  actor-side content references voting (`session_uses_vote_language`),
  and only carry vote sources visible to the actor at rollout
  (filtered by cell flags read from `experiment.json`).
- **Advantage shape**: `advantage_baseline` ∈ {0, `mean_session`,
  `mean_global`, any number}, `advantage_scale` ∈ {`session_std`,
  `global_std`, `none`, any number} in `RL_context/config.json`.
- **PPO clip**: `rl/train.py --clip-ratio-low 0.2 --clip-ratio-high 0.25`
  sets the asymmetric trust-region bounds; `--no-use-ppo-clip`
  degenerates to vanilla RWR for ablation.
- **KL anchor**: `--beta-kl 0` drops the base-model anchor entirely
  (DAPO-style reference-free); `--beta-kl 0.005` (default) is the
  DeepSeek-R1-final / DAPO-adjacent middle ground.
- **Other loss knobs**: `--advantage-clip 0` (default; off — main-line
  projects don't explicitly clip) or any positive cap;
  `--max-seq-len 8192` truncates over-length trajectories at the
  nearest user-turn edge.
- **Per-iter overrides**: `scripts/iterate.py --train-extra "..."`
  forwards any flag into every iter's `rl.train` invocation. Common
  uses: `"--ppo-epochs 2 --beta-kl 0.005 --clip-ratio-high 0.25
  --lora-r 64 --lora-alpha 128"`. `--num-epochs` is kept as a
  deprecated alias for `--ppo-epochs`.

## Dataset

`datasets/sciencepedia_test.json` — 100 graduate-level problems with
reference solutions (smoke size, disjoint from train).

`datasets/sciencepedia_train_part{1..4}.json` — 4 × 1 026 = 4 104
problems, sampled uniformly over depth-5 Sciencepedia categories by
`scripts/sample_dataset.py`.  Pass all four to `iterate.py --pool`
(comma-separated) and they are concatenated into a single 4 104-question
pool from which `--total-questions` are deterministically drawn.

`datasets/sciencepedia_heldout_mc100.json` — 100 multiple-choice
problems (boxed letter answer) held out from both train and test.
Pre-disjoint evaluation set for `eval/sciencepedia_mc.py`.

## Environment variables

Host-specific paths are read from environment variables so the code
runs unchanged on any machine.  Defaults work out of the box when
you run from inside the target Python env.

| variable | default | purpose |
|---|---|---|
| `PRL_PYTHON` | `sys.executable` | Python (or wrapper script) for vLLM / accelerate child processes |
| `PRL_ACCELERATE` | `which accelerate` | accelerate binary for `rl.train` / `rl.export` |
| `PRL_MODEL_PATH` | `Qwen/Qwen3.5-9B` | local model directory (vLLM needs a full path) |

If your tmux server was started in a different env, point `PRL_PYTHON`
at a wrapper that activates the right conda env before exec'ing python.
All scripts forward these vars into tmux child shells automatically.

## Versions

| component | version |
|---|---|
| vLLM | 0.17.1 |
| base model | Qwen3.5-9B |
| Python | 3.11+ |
| PyTorch | 2.10 (CUDA 12.8) |
| transformers | ≥ 5.5 |
| accelerate | ≥ 1.13 |
| FastAPI | ≥ 0.100 |
| aiohttp | ≥ 3.9 |

## Data output shape

After a session run, `data/<name>_<timestamp>/` contains:

```
parliament.db        # all posts, comments, votes, and per-request interaction log
experiment.json      # run summary (per-agent stats, exit reasons, tokens, durations)
llm_logs/            # every LLM API call, grouped by session
  <sid>/<agent>.jsonl
discards/            # no-tool streaks (debug)
run.log              # stdout of run.py / iterate.py
parliament.log       # FastAPI uvicorn log
train.jsonl          # per-actor trajectories (after rl.extract)
ckpt/step_K/         # LoRA adapter + optimizer/scheduler state
merged/              # single-file HF folder (after rl.export; vLLM-loadable)
```

`iterate.py` additionally writes `state.json` (resume pointer) and
`iterate.log` at the run root.
