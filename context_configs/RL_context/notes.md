# RL_context

Knobs that shape the training data produced by `rl/extract.py`.

## Files

| File | Purpose |
|---|---|
| `prompt_intro.txt` | Opening line of each user message; `{name}` is the actor's anonymized name. |
| `config.json` | All other knobs (templates, score visibility, advantage normalization). |

The actor name pool that Parliament uses at generation time lives in
`../Parliament_context/config.json`. Extract reads that pool as the
single source of truth for the session → name mapping, so the headers
it renders always match what the LLM actually saw during the Parliament
run.

## Templates

`post_header` and `comment_header` are formatted with `{local_id}`,
`{author}`, `{is_you}` (and `{local_post_id}` for comments).
`score_suffix` is appended only when scores are visible (per the rules
below).

## Identity anonymization (`anonymize_identity`)

When `true` (default), extract applies the same `session_id` → name
mapping that harness used, so the anonymized cast appears consistently
in the training headers. When `false`, headers show the raw DB names
(`Scientist_1`, `Scientist_2`, …) — useful for debugging against the
DB.

## Score visibility (`score_visibility`)

Three modes:

- `auto` (default) — scores are kept only in sessions whose actor posts
  meta-reference Parliament scoring (e.g. "vote consensus", "+9 for
  P_5", "Anonymous Scientist"). Sessions where the discussion never
  mentions scores have headers without `(score: ±N)`, eliminating the
  train-vs-inference distribution shift.
- `always` — scores always shown.
- `never` — scores always hidden.

## Advantage normalization

The full pipeline is two steps — first remove the position trend
("later posts score higher" is a session-mechanic, not a quality
signal), then normalize per session:

```
1. position debias       r' = r - slope * (t - 0.5)        # extract.py:debias_position
2. session normalize     A  = (r' - baseline) / scale       # extract.py:compute_advantages
```

`slope` is fitted globally from all sessions (early-half vs late-half
mean reward gap, mapped to a 0→1 span). `t` is the post's normalized
position in its session [0, 1]. At the midpoint the correction is
zero; early posts are boosted, late posts are reduced.

| Key | Options | Default | Semantics |
|---|---|---|---|
| `advantage_baseline` | `0.0`, `"mean_session"`, `"mean_global"`, or any number | `"mean_session"` | Constant to subtract before scaling. Default `"mean_session"` is classic GRPO group centering: each post's advantage is signed by whether it beat the rest of *its own* session, regardless of the session's absolute reward level. |
| `advantage_scale` | `"session_std"`, `"global_std"`, `"none"`, or any number | `"session_std"` | Per-session std (standard GRPO), dataset std (REINFORCE++-style), or a fixed constant. |

**Safety floor**: session std is floored at `1.0` so a degenerate
session (all posts got identical reward) can't produce infinite
advantages. With rewards in ±10, this keeps `|A|` bounded below 10 even
for the noisiest case. `extract.py` then applies `advantage_clip = 2.0`
inside the trainer (`rl/train.py:RLDataset`).

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
