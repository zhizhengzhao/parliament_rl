# Naming — three nested loops, verl-aligned

Parliament RL has three nested loops.  We use the same names as
[verl](https://github.com/volcengine/verl) so the vocabulary transfers
to anyone already familiar with the standard RLHF infra.

```
┌─ total_epoch (outer)  ───────── pass through the full shard list
│  ┌─ iter (middle)  ───────── one shard's sample → train → export
│  │  ┌─ ppo_epoch (inner) ──── pass through that iter's train.jsonl
│  │  └────────────────────────
│  └────────────────────────────
└──────────────────────────────────
```

| level | our flag | verl equivalent | OpenRLHF equivalent | what it does |
|---|---|---|---|---|
| outer | `iterate.py --total-epochs N` | `trainer.total_epochs` | `--num_episodes` | Run the shard list `N` times.  Each repetition's first iter starts from the previous repetition's last `merged/` (so the second pass over a shard already learns from the trained policy). |
| middle | (one per shard, no flag) | (no equivalent — verl is single-dataset) | (same) | A single sample → extract → train → export cycle on one shard.  Stored under `data/<name>_iter{NN}_<ts>/`. |
| inner | `rl/train.py --ppo-epochs N` | `actor.ppo_epochs` | `--max_epochs` | Number of passes through the same `train.jsonl` per iter.  We also accept the legacy alias `--num-epochs` so in-flight runs can resume. |

## Examples

**Default** — 4 shards × 1 epoch × 2 ppo_epochs:

```bash
python scripts/iterate.py --shards p1,p2,p3,p4
# ⇒ 4 iters total, train.py loops train.jsonl 2× per iter
```

**Two epochs** — 4 shards × 2 epochs × 2 ppo_epochs = 8 iters:

```bash
python scripts/iterate.py --shards p1,p2,p3,p4 \
       --total-epochs 2 --train-extra "--ppo-epochs 2"
# ep1.iter1: p1 → π_1
# ep1.iter2: p2 → π_2
# ep1.iter3: p3 → π_3
# ep1.iter4: p4 → π_4
# ep2.iter5: p1 (with π_4) → π_5    ← second-pass over p1 already strong
# ...
```

**Single ppo epoch** (lighter training, faster iters):

```bash
python scripts/iterate.py --shards p1,p2,p3,p4 \
       --train-extra "--ppo-epochs 1"
```

## What used to be wrong

Before this rename, `rl/train.py` exposed `--num-epochs`, which
collided with the colloquial use of "epoch" (= one pass through the
dataset).  Two different things were both called "epoch":

* the inner training pass (`for epoch in range(num_epochs)` in `train.py`)
* the outer ReST-style pass through all shards

Now the inner one is unambiguously **`ppo_epoch`** (verl term) and the
outer one is **`total_epoch`** (also verl term).  `iter` keeps its
original meaning — one shard's sample+train cycle.

The CLI flag `--num-epochs` still works as a deprecated alias for
`--ppo-epochs`, so any in-flight `iterate.py` runs continue to resume
without surprises.

## Where it shows up

* `scripts/iterate.py` — `--total-epochs` (outer), `state.json` history
  entries tagged with `total_epoch`.
* `rl/train.py` — `--ppo-epochs`, `metrics.jsonl` rows tagged with
  `ppo_epoch`, checkpoint `meta.json` saves `ppo_epoch`.
* `data/<run>/state.json` history is keyed by `iter`; each entry has
  `total_epoch` so you can group "all ep1 runs" vs "all ep2 runs".
