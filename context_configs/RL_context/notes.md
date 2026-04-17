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

```
A_i = (r_i − baseline) / scale
```

| Key | Options | Default | Semantics |
|---|---|---|---|
| `advantage_baseline` | `0.0`, `"mean_session"`, `"mean_global"`, or any number | `0.0` | Constant to subtract before scaling. Default 0 preserves the absolute-reward sign (any judge-positive post keeps positive A). `"mean_session"` is classic GRPO group centering. |
| `advantage_scale` | `"session_std"`, `"global_std"`, `"none"`, or any number | `"session_std"` | Per-session std (standard GRPO), dataset std (REINFORCE++-style), or a fixed constant. |

**Safety floor**: session std is floored at `1.0` so a degenerate
session (all posts got identical reward) can't produce infinite
advantages. With rewards in ±10, this keeps `|A|` bounded below 10 even
for the noisiest case.

### Defaults explained

- `baseline = 0`, `scale = session_std` — every actor post is compared
  against the scale of variation within its own session, but the
  absolute-reward sign is preserved. A post judged `+6` always has
  positive A regardless of whether the session's mean was `+9` (usually
  a net-good session) or `-1` (unusually bad). This matches the
  intuition "positive judge signal is worth learning from, negative
  signal worth avoiding" while still normalizing the gradient magnitude
  session-by-session.
