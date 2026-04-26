# Frame evaluation — assessing each cell's policy in its own frame

## Why a third evaluation track

Parliament RL ships two *zero-shot* benchmarks:

* `eval/gpqa.py` — GPQA Diamond, 198 four-choice questions.
* `eval/sciencepedia_mc.py` — 100 held-out boxed-letter MC, disjoint
  from train and test.

Both feed the trained policy a single self-contained question and
ask for a CoT answer. This is fine for measuring *general* science
ability, but it has two structural weaknesses for our 2×2 ablation:

1. **Train ↔ eval distribution gap is the same for every cell** — so
   any per-cell win is partly inherited noise of this single mismatch.
2. **The gap is not the same for every cell.** Solo (C/D) policies
   were trained almost in zero-shot form already (`submit(step)`
   alone, no peer context); coupled (A/B) policies were trained as
   forum participants. Forcing every cell onto the zero-shot form
   gives solo cells a transfer-style advantage unrelated to the
   actual quality of the policy.

`eval/frame.py` adds a third, complementary track. Each cell's
policy is run through a session **in its own training frame** (same
coupling, same judge visibility) on the held-out test set. A
**fixed cell-agnostic secretary agent** then collapses each
session's discussion into a single `\boxed{...}` answer, which is
scored against the dataset's gold.

Trade-off: it is more expensive (it actually rolls a Parliament/Solo
session per question) but it measures what each policy is *actually
trained to do*: produce useful reasoning under the conditions it
saw at training time.

## Pipeline

For one (cell, policy):

```
1. collect      scripts/run.py --in-tmux  (cell env vars set)
                vLLM × N(policy)  →  Parliament  →  harness
                ──────────────────────────────────────────────
                data/<name>_<ts>/parliament.db
                  ├ sessions × N_test_questions
                  └ each session: posts/comments/votes
                  cleanup: vLLM + Parliament torn down

2. read         load_sessions_from_db()  →  per-session list of
                actor posts (+ comments in coupled cells); judge
                content is excluded.

3. secretary    secretary vLLM in-process  (model = $PRL_MODEL_PATH
                                            by default — fixed!)
                For each session:
                  build_chat_messages(problem, posts) →
                  vLLM.generate (one batch over all sessions) →
                  parse_letter / parse_boxed → answer

4. score        compare against dataset[title].gold_letter
                (or .answer for free-form, conservative
                normalize_freeform equivalence)

5. write        report.json with per-session records + per-category
                accuracy + overall accuracy
```

## The secretary

Lives in `eval/secretary.py` (pure logic, no IO, unit-testable).

**Model**: a fixed reference model — by default `$PRL_MODEL_PATH`
(the original base, e.g. Qwen3.5-9B). Using one of the trained
policies as secretary would let cell-specific style bias the
extraction; the whole point of frame eval is that the *extractor*
is invariant across cells.

**Prompt**: a fixed system prompt instructing the secretary to read
the discussion and output exactly one `\boxed{...}` answer. No
chain-of-thought (`enable_thinking=False` by default — the
secretary's job is mechanical extraction, not reasoning).

**Inputs**: problem text + ordered actor posts (+ optional comments
in coupled cells). Judge identities, judge votes, peer scores, and
agent-runtime metadata are all excluded.

**Outputs**: a single `\boxed{...}` per session.

* Multiple-choice (`gold_letter` present): parse `\boxed{A-D}`.
* Free-form (`answer` field): parse the inner text of the last
  `\boxed{...}`, compare via `normalize_freeform` (whitespace +
  surrounding-LaTeX trim, lowercase). Conservative on purpose;
  better to under-claim accuracy than to over-claim with brittle
  CAS rules.

## Comments policy

Coupled cells (A, B) have actor↔actor comments; solo cells (C, D)
do not. By default `frame.py` infers `--include-comments` from the
cell:

* A, B → include actor comments (the discussion includes them);
* C, D → no comments to include (none exist in the DB anyway).

`--no-include-comments` forces post-only for all cells, useful as a
cross-cell uniformity ablation.

## Dataset compatibility

`eval/frame.py --dataset` accepts the same JSON shape used by
`scripts/run.py`:

* must be a list of dicts with `problem` (used as session title) and
  either:
  * `gold_letter` (`A`/`B`/`C`/`D`) → MC scoring path;
  * `answer` (boxed or raw) → free-form scoring path.

The dataset row's `category` (depth-5 `/`-joined string) is used for
the per-category accuracy breakdown (first two components are kept,
e.g. `physics/graduate`).

The Sciencepedia held-out 100-question MC set
(`datasets/sciencepedia_heldout_mc100.json`) is the recommended
default frame-eval target — its boxed-letter answers map cleanly to
the secretary's `\boxed{X}` output.

## Hardware schedule

vLLM cannot host two independent model loads on the same GPU pool
simultaneously, so `frame.py` uses a **sequential** schedule on a
single shared pool:

```
phase 1 (collect)            phase 3 (secretary)
   policy vLLM × N GPU   →     base vLLM × N GPU
   Parliament running           (same GPU pool)
   harness driving
   ↓ scripts/run.py teardown
   all GPUs released
```

`frame.py` waits for `scripts/run.py` to finish (which calls
`stop_vllm()` + `stop_parliament()` on its way out) before loading
the secretary in-process. Use `--secretary-tp` to set the secretary
tensor-parallel size (1 by default, since one A100 already comfortably
hosts the 9B base).

## Sweeping all 4 cells

`eval/frame_sweep.sh` chains four `python -m eval.frame` invocations
(one per cell) and prints + saves the comparison table:

```bash
bash eval/frame_sweep.sh \
    --policies A:data/mainA_iter10/merged,B:.../merged,C:.../merged,D:.../merged \
    --dataset  datasets/sciencepedia_heldout_mc100.json \
    --output   data/frame_eval/main_run/ \
    --secretary $PRL_MODEL_PATH \
    --gpus 0,1,2,3,4,5,6,7
```

Output:

```
data/frame_eval/main_run/
├── cell_A.json              # per-cell report (records + accuracy)
├── cell_B.json
├── cell_C.json
├── cell_D.json
└── summary.json             # comparison table
```

Each `cell_*.json` carries:

* the resolved `run_dir` (so re-extraction with a different
  secretary can use `--reuse-run-dir <dir>`),
* `accuracy`, `n_correct`, `n_scored`, `n_total`,
* `per_category` + `per_category_total`,
* full `records` (one per session) with the raw secretary response
  + parsed answer + gold + correctness.

## Smoke test

10-question, 1-actor, 1-judge, 5-round mini-frame-eval on a single
GPU (~5 min/cell):

```bash
python -m eval.frame \
    --cell A \
    --policy $PRL_MODEL_PATH \
    --dataset datasets/sciencepedia_heldout_mc100.json \
    --secretary $PRL_MODEL_PATH \
    --output /tmp/smoke_frame.json \
    --gpus 0 --sessions-per-gpu 1 \
    --actors 1 --judges 1 \
    --max-turns 5 --max-questions 10 \
    --secretary-tp 1
```

`/tmp/smoke_frame.json:accuracy` should be > 0 (not particularly
meaningful at this size; this is a wiring check).

## What this measure does and does not tell you

Frame eval is a *complement* to zero-shot eval, not a replacement:

* **Frame eval** answers: "given that this policy is going to be
  deployed in setting X (coupling × visibility), how well does it
  produce useful reasoning in setting X?"
* **Zero-shot eval** answers: "how well does this policy generalise
  outside any training-time scaffold?"

A cell that wins frame eval but loses zero-shot eval has learned to
exploit its training scaffold; a cell that wins zero-shot but loses
frame eval has internalised reasoning more than scaffold use. Both
signals are interesting; the 2×2 is interpretable along *both*
axes.
