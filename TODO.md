# TODO

Live list of work that is **not yet implemented in `main`**. Anything
"done" lives in [`README.md`](README.md) + [`docs/`](docs/) and is no
longer tracked here.

## Open

### Experiments to run
- [ ] **Main 2×2 ablation** on the 1000-question pool draw
      (Parliament / BlindParliament / Solo / BlindSolo) — see
      [`docs/04_2x2_design.md`](docs/04_2x2_design.md) for the launch
      template.  Pre-flight: mini40 smoke on each cell first.
- [ ] **Held-out eval sweep**: GPQA Diamond + Sciencepedia-MC100 on
      every iter's `merged/` checkpoint after the main run completes.
- [ ] **Template-augmentation ablation**: re-extract one cell with
      `--no-template-augment` and compare the resulting GPQA delta
      against the augmented run on the same cell.

### Infra polish (low priority)
- [ ] Per-iter held-out eval inside `iterate.py` (cheaper than the
      current full GPQA Diamond; consider a 50-q
      `eval/sciencepedia_mc.py` subset gated by `--cheap-eval`).
- [ ] Wandb / TensorBoard hook in `rl/train.py` (today: `metrics.jsonl`
      only).  Single optional dependency; gated by `--wandb` flag.
- [ ] Direction 2 (peer-review-only): `--judges 0` runs as the
      vote-only baseline; needs a separate reward-extraction path
      (current `extract.py` requires judge votes).
