# Naming — three nested loops, verl-aligned

Parliament RL has three nested loops.  We use the same names as
[verl](https://github.com/volcengine/verl) so the vocabulary transfers
to anyone already familiar with the standard RLHF infra.

```
┌─ total_epoch (outer)     ── pass through the drawn question schedule
│  ┌─ iter (middle)        ── one rollout round's sample → train → export
│  │  ┌─ ppo_epoch (inner) ── pass through that iter's train.jsonl
│  │  └────────────────────────
│  └────────────────────────────
└──────────────────────────────────
```

| level | our flag | verl equivalent | OpenRLHF equivalent | what it does |
|---|---|---|---|---|
| outer | `iterate.py --total-epochs N` | `trainer.total_epochs` | `--num_episodes` | Run the drawn schedule `N` times.  Each repetition starts from the previous repetition's final `merged/`, so the second pass over a batch already learns from a stronger policy. |
| middle | one per rollout round | `trainer.step` (approx) | (same) | One sample → extract → train → export on a `sampling_batch_size`-sized batch of questions.  Stored under `data/<name>_iter{NN}_<ts>/`. |
| inner | `rl/train.py --ppo-epochs N` | `actor.ppo_epochs` | `--max_epochs` | Number of SGD passes through the same `train.jsonl`.  With `--use-ppo-clip`, the PPO ratio+clip surrogate keeps each pass inside a trust region around the iter-start π_old.  Legacy alias `--num-epochs`. |

`rounds_per_epoch = total_questions / sampling_batch_size`, so
`total_iters = total_epochs × rounds_per_epoch`.

## Supporting concepts (not verl terms, but necessary here)

| name | flag | what it does |
|---|---|---|
| **pool** | `iterate.py --pool a.json,b.json,...` | Full question bank (one or more JSON files concatenated in-order). |
| **draw** | `--total-questions N --seed S` | Deterministic `rng(S).sample(pool, N)` at launch.  The same seed across all 2×2 cells guarantees every cell sees the same questions at the same positions in the schedule. |
| **sampling_batch_size** | `--sampling-batch-size B` | Questions per rollout round.  Larger ⇒ fewer iters and less vLLM-startup overhead; smaller ⇒ policy updates more often.  200 is our experimental sweet spot. |
| **training_batch_size** | `per_device_batch_size × grad_accum × n_gpus` | Mini-batch for the PPO SGD update (global); independent of the sampling batch. |

## Examples

**Default 4-cell main experiment** (400 q × 3 total_epochs × 2 ppo_epochs = 6 iters):

```bash
python scripts/iterate.py \
    --name main_A \
    --pool datasets/sciencepedia_train.json \
    --total-questions 400 \
    --sampling-batch-size 200 \
    --total-epochs 3 \
    --seed 42 \
    --train-extra "--ppo-epochs 2 --clip-ratio-high 0.25 --beta-kl 0.005" \
    --gpus 0,1,2,3,4,5,6,7 --sessions-per-gpu 4
# schedule = 400 q / 200 = 2 rounds per epoch × 3 epochs = 6 iters
# ep1.round1,2: policy evolves π_0 → π_2
# ep2.round1,2: same 2 batches starting from π_2, ending at π_4
# ep3.round1,2: same 2 batches starting from π_4, ending at π_6
```

**Light smoke test** (50 q × 1 epoch × 1 ppo_epoch = 5 iters):

```bash
python scripts/iterate.py \
    --name smoke \
    --pool datasets/sciencepedia_test.json \
    --total-questions 50 --sampling-batch-size 10 \
    --total-epochs 1 --seed 42 \
    --train-extra "--ppo-epochs 1" \
    --gpus 0,1,2,3,4,5,6,7
```

**Reference-free DAPO style** (no KL anchor, asymmetric clip only):

```bash
python scripts/iterate.py ... \
    --train-extra "--ppo-epochs 2 --clip-ratio-high 0.28 --beta-kl 0"
```

## Fair 2×2 comparison

All four cells share the **exact same** draw + schedule by using the
same `--seed` and `--pool`.  The only flags that differ are the two
env-var knobs documented in `docs/04_2x2_design.md`
(`PRL_CONTEXT` and `PRL_JUDGE_VOTES_VISIBLE`).  Nothing else —
hyperparameters, training algo, KL anchor, clip — changes.

## History

This project used to be shard-driven: `--shards p1.json,p2.json,...`
with one iter per shard, and `total_epochs` meant "repeat the shard
list".  The new design replaces that with a **pool + seed + draw**
scheme because:

* The shard boundaries were an accident of the dataset layout, not a
  research decision — they made it harder to change question counts.
* A shared deterministic draw is the cleanest way to guarantee
  apples-to-apples comparison across 2×2 cells.

The CLI flag `--num-epochs` on `rl/train.py` still works as a
deprecated alias for `--ppo-epochs`; the `--shards` flag on
`iterate.py` has been removed.

## Where it shows up in the code

* `scripts/iterate.py` — `--total-epochs`, `--total-questions`,
  `--sampling-batch-size`, `--seed`, `--pool`; writes `manifest.json`
  and `backups/<name>_<ts>/`; `state.json.history[i]` is tagged with
  `iter`, `total_epoch`, `round`.
* `rl/train.py` — `--ppo-epochs`, `--use-ppo-clip`, `--clip-ratio-low`,
  `--clip-ratio-high`, `--beta-kl`; `metrics.jsonl` rows include
  `ppo_epoch`, `clipfrac`, `approx_kl_old`; checkpoint `meta.json`
  saves `ppo_epoch`.
