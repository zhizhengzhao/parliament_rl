# RL_context

Knobs that shape the training data produced by `rl/extract.py`.

## Files

| File | Purpose |
|---|---|
| `config.json` | The few knobs that aren't templates: `min_content_chars`, `anonymize_identity`, `advantage_baseline`, `advantage_scale`. |

All wrapping templates live as a multi-variant pool in
`rl/extract.py:TEMPLATE_POOL` (~107 unique strings across 12 keys —
see that module's docstring for the full list).

The actor name pool that Parliament uses at generation time lives in
`../Parliament_context/config.json`. Extract reads that pool as the
single source of truth for the session → name mapping, so the headers
it renders always match what the LLM actually saw during the Parliament
run.

## Template augmentation (extract-side)

Why: with a single fixed wrapper, the model learns a brittle binding
between literal strings ("[V on P_3] by Anonymous Scientist:") and
their semantics.  Diverse wrappers force the model to read the
information fields rather than memorise the wrapper, which both
reduces overfitting (Khan et al. 2026 "Prompt Augmentation Scales up
GRPO Training on Mathematical Reasoning") and makes inference robust
to query-format drift.

**What we augment**: section headers, prompt intros, post headers,
comment headers, vote-event lines, anonymous voter labels, `[you]`
self markers, "no new content" fallbacks.

**What stays fixed**: information fields ({local_id}, {author},
{value}, {cur}, …) — only the surrounding text varies.  The model
must read the data to reason; the wrapper is decorative.

**Determinism contract**:

- Whole-message templates (`prompt_intro`, `section_*`,
  `no_new_content`) are seeded by `session.title` only — one
  consistent variant per session, so a single user message stays
  internally coherent.
- Per-entry templates (`post_header`, `comment_header`, `vote_event`,
  `anonymous_voter`, `you_marker`, etc.) are seeded by
  `(session.title, entry_id)` — different posts/comments/votes inside
  the same user message draw different wrappers, maximising diversity.
- The 2×2 cells share `session.title`, so the same problem always
  draws the same wrappers — template choice is never a confounder in
  cell comparisons.

**Disabling**: `python -m rl.extract --no-template-augment` forces
every key to its first variant (byte-stable output, useful for
ablation against an augmented run).

## Vote events (default: stripped)

`rl/extract.py` accepts `--strip-vote-events / --no-strip-vote-events`
(default **ON**).

When stripped (the recommended main-experiment setting), no vote
events ever enter the rendered training context.  The reward
signal is delivered to the trainer purely via `turn_advantages`
per assistant turn; the input context contains only posts (plus
comments in coupled cells).  This:

* removes the spurious "read the vote field to predict reward"
  shortcut (the field is absent at zero-shot inference time);
* makes the training-context *form* of A vs B nearly identical,
  so the visibility ablation tests *trajectory quality* rather
  than *form*;
* aligns training-input form with evaluation-input form
  (GPQA/MC100 inputs carry no vote events).

When `--no-strip-vote-events` (legacy / fairness ablation), vote
events are inserted only when the session's actor-side content
references voting (`session_uses_vote_language` heuristic), and
the surviving vote pool is filtered by cell:

- A (Parliament):       keep all visible votes (actor + judge)
- B (BlindParliament):  drop judge votes; only actor mutual votes survive
- C (Solo):             keep judge votes (own posts only); no actor votes (no vote tool)
- D (BlindSolo):        no votes at all

The cumulative `current score` on each vote event is re-computed
from the visible-vote pool, so judges never leak through the score
field in cells that hide them.

## Identity anonymization (`anonymize_identity`)

When `true` (default), extract applies the same `session.title` →
name mapping that harness used, so the anonymized cast appears
consistently in the training headers. When `false`, headers show the
raw DB names (`Scientist_1`, `Scientist_2`, …) — useful for debugging.

## Advantage normalization

The full pipeline is two steps — first remove the position trend
("later posts score higher" is a session-mechanic, not a quality
signal), then normalize per session:

```
1. position debias       r' = r - slope * (t - 0.5)        # extract.py:debias_position
2. session normalize     A  = (r' - baseline) / scale       # extract.py:compute_advantages
```

`slope` is the closed-form OLS estimate from every actor post's
`(t, r)` pair across all sessions with ≥4 actor posts. `t` is the
post's normalized position in its session [0, 1]. At the midpoint
the correction is zero; early posts are boosted, late posts are
reduced.

| Key | Options | Default | Semantics |
|---|---|---|---|
| `advantage_baseline` | `0.0`, `"mean_session"`, `"mean_global"`, or any number | `"mean_session"` | Constant to subtract before scaling. Default `"mean_session"` is classic GRPO group centering: each post's advantage is signed by whether it beat the rest of *its own* session, regardless of the session's absolute reward level. |
| `advantage_scale` | `"session_std"`, `"global_std"`, `"none"`, or any number | `"session_std"` | Per-session std (standard GRPO), dataset std (REINFORCE++-style), or a fixed constant. |

**Safety floor**: session std is floored at `1.0` so a degenerate
session (all posts got identical reward) can't produce infinite
advantages. With rewards in ±10 (3 judges × ±3) this keeps `|A|`
bounded below ~10 even for the noisiest case.

`rl/train.py:RLDataset` accepts an optional further `--advantage-clip`
clamp; defaults to **0 (disabled)**, matching mainline projects (TRL,
OpenRLHF, Verl, DeepSeek-R1, DAPO) which rely on zscore normalization
+ PPO ratio clipping for gradient stability rather than a second
`|A|` cap.  An explicit clamp can squash legitimate high-reward
outliers (mid200 measurement: 3.82% of turns had `|A|>2`, all genuine
high-quality posts whose signal we want learned at full strength).

### Defaults explained

- `baseline = "mean_session"`, `scale = "session_std"` — classic GRPO.
  Each session is centered and scaled by its own statistics, so a
  session full of strong posts and a session full of weak posts both
  contribute well-balanced gradients. Combined with **position debias**
  this yields `mean(A) ≈ 0`, `|A|̄ ≈ 0.82` and a 50/50 positive-vs-
  negative split on Sciencepedia (mid200 measurement, 12 iters across
  three hyper-parameter cells).
- We previously ran with `baseline = 0` (preserving the absolute reward
  sign), but the GRPO baseline gave a much cleaner training signal —
  the absolute-reward sign was dominated by judge-strictness drift
  across sessions and washed out the within-session learning signal.
- The 2×2 ablation cells share these defaults; only the rollout-side
  context (coupling and judge visibility) differs across cells.
