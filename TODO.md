# TODO

Live list of work that is **not yet implemented in `main`**. Anything
"done" lives in [`README.md`](README.md) + [`docs/`](docs/) and is no
longer tracked here.

## Open

### Infra
- [ ] Per-iter held-out eval at the end of `iterate.py` (currently
      `--no-eval` skips it; the GPQA eval at iter boundaries works
      but doubles wall time on small shards — needs a cheaper eval
      hook, perhaps `eval/sciencepedia_mc.py` on a 50-question subset).
- [ ] Wandb / TensorBoard hook in `rl/train.py` (today: `metrics.jsonl`
      only). Single optional dependency; gated by `--wandb` flag.
- [ ] Compare resuming-from-merged vs. resuming-from-LoRA-adapter
      across `total_epochs` (current: always merges, so the second
      pass loses optimizer state that a kept LoRA could restore).

### Experiments to run
- [ ] Held-out GPQA Diamond + Sciencepedia-MC sweep on the existing
      `mid200_S{1,2,3}` final checkpoints (compares the three
      hyper-parameter cells head-to-head against base).
- [ ] **2×2 ablation** smoke check on each cell using `mid40` data
      (Parliament / BlindParliament / Solo / BlindSolo); see
      [`docs/04_2x2_design.md`](docs/04_2x2_design.md) for the four
      launch commands.
- [ ] Direction 2 (peer-review-only): `--judges 0` runs as the
      vote-only baseline; needs a separate reward-extraction path
      (current `extract.py` requires judge votes).
