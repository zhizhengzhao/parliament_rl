# Parliament_context

What the LLM agents see at runtime. Edits here propagate the next time
the harness boots; nothing in this directory is generated.

## Files

| File | Purpose |
|---|---|
| `actor_prompt.txt` | System prompt for Scientists. `{name}`, `{session_title}`, `{persona}` substituted at startup (name is session-anonymized). |
| `judge_prompt.txt` | System prompt for Judges. Same substitutions plus `{reference_solution}`. |
| `config.json` | `name_pool`, `persona_pools`, agent limits, `judge_votes_visible`. |
| `actor_skill.md`, `judge_skill.md` | Concise reference cards mirroring the prompts (for humans). |

## `config.json` keys

- **`name_pool`** — flat list of ~400 given names (diverse cultural
  origins, mostly 1–2 BPE tokens in Qwen). Each session draws a cast
  via a single `rng.sample(pool, 2N)` seeded by `session_id`: first
  `N` go to `Scientist_1..N`, next `N` to `Judge_1..N`, so no two
  agents in a session share a name regardless of role.
- **`persona_pools.scientist`** — 100 single-line cognitive-style
  descriptions ("I am an axiomatic builder", "I hunt for
  counterexamples", etc.). Each session draws 3 without replacement.
- **`persona_pools.judge`** — 60 judge-specific descriptions ("I focus
  on dimensional consistency", "I demand special-case verification",
  etc.). Independent pool from scientist, so no role collision is
  possible by construction.
- **`agent.max_rounds`** — actor round cap (judges run unbounded until
  session ends).
- **`agent.max_consecutive_errors`** — LLM API retry budget before
  skipping a round.
- **`agent.llm_timeout_s`** — total LLM call timeout.
- **`judge_votes_visible`** — when true, actors receive judge votes
  (anonymized as "Anonymous Scientist"); when false, judge votes are
  withheld entirely. Useful for ablating the judge's online-steering
  effect vs its reward-only effect.

The DB keeps stable `Scientist_1..3`, `Judge_1..3` user names for
debuggability. Name anonymization is purely presentational — applied
before the LLM sees anything, and reapplied at extract time with the
same `session_id` seed so training data is consistent.
