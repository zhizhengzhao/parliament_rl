# Parliament_context (cells A & B)

What the LLM agents see at runtime in the **coupled-actor** cells.
Edits here propagate the next time the harness boots; nothing in this
directory is generated.

## Files

| File | Purpose |
|---|---|
| `actor_prompt.txt` | System prompt for Scientists. `{name}`, `{session_title}`, `{persona}` substituted at startup (name is session-anonymized). |
| `actor_skill.md` | Concise reference card mirroring the prompt (for humans). |
| `config.json` | Cell-specific overrides only — currently `actor_context_coupled: true`. Everything else (agent limits, name pool, persona pools, judge prompt, judge_votes_visible default) is inherited from `../shared/`. |
| `notes.md` | This file. |

The shared resources used by every cell are loaded from
`../shared/`:

* `shared/config.json` — `name_pool`, `persona_pools`, `agent.*`,
  `judge_votes_visible` (default true).
* `shared/judge_prompt.txt` — system prompt for Judges (identical across cells).
* `shared/judge_skill.md` — judge reference card.

`harness/prompts.py:load_context_config` loads the shared config,
then overlays this cell's config on top — so any field defined here
takes precedence.

## 2×2 cells served by this directory

  A Parliament       coupled=true  + visible=true   (this dir)
  B BlindParliament  coupled=true  + visible=false  (this dir + `PRL_JUDGE_VOTES_VISIBLE=0`)

Cells C / D are served by `../Solo_context/`.

## Anonymization

Session-level: `Scientist_1..N` and `Judge_1..N` (the DB-stable user
names) are mapped to per-question deterministic draws from
`shared/config.json:name_pool` via `assign_session_names(title)`,
seeded by the problem title so all four cells running the same
question see the same cast.  Reapplied at extract time with the same
seed for byte-identical training headers.
