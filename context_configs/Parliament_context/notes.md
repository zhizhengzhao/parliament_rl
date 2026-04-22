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
- **`actor_context_coupled`** — when true (this cell), actors see
  every other actor's posts/comments/votes during the session
  (collaborative). When false (`Solo_context`), each actor only sees
  its own history plus optionally judge votes. Drives the **coupling**
  axis of the 2×2 ablation.
- **`judge_votes_visible`** — when true (this cell), actors receive
  judge votes (anonymized as "Anonymous Scientist"); when false,
  judge votes are withheld entirely. Drives the **judge visibility**
  axis. Override at launch with `PRL_JUDGE_VOTES_VISIBLE=0` so a
  single context dir covers both halves of the axis without an
  extra config file.

The 2×2 cells:

  A Parliament       coupled=true  + visible=true   (this dir)
  B BlindParliament  coupled=true  + visible=false  (this dir + env)
  C Solo             coupled=false + visible=true   (Solo_context)
  D BlindSolo        coupled=false + visible=false  (Solo_context + env)

The DB keeps stable `Scientist_1..3`, `Judge_1..3` user names for
debuggability. Name anonymization is purely presentational — applied
before the LLM sees anything, and reapplied at extract time with the
same `session_id` seed so training data is consistent.
