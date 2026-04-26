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
- [ ] **Held-out eval sweep** on every iter's `merged/`:
      - zero-shot: `eval/gpqa_sweep.sh` + `eval/sciencepedia_mc.py`;
      - **in-frame**: `eval/frame_sweep.sh` (see
        [`docs/05_frame_eval.md`](docs/05_frame_eval.md)) — compares
        the four cells under each policy's own training conditions
        with one fixed secretary.
- [ ] **Vote-form ablation**: re-extract one cell with
      `--no-strip-vote-events` (legacy heuristic) and measure the
      delta vs the default (stripped) extract on the same rollout.
      Tests how much per-cell signal lives in the vote-event form
      vs the underlying trajectory quality.
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
- [ ] Free-form scoring in `eval/frame.py` is conservative (string
      equivalence after light normalisation). For free-form
      Sciencepedia answers (formulas, numbers), an optional sympy
      backend would close the under-counting gap on numerically
      equivalent expressions.
