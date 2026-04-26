# Solo_context (cells C & D)

What the LLM agents see at runtime in the **independent-actor** cells.
Each actor reasons alone: no peer posts, no peer comments, no actor
votes — only their own chain of `submit(step)` calls plus optionally
judge votes on their own steps.

## Files

| File | Purpose |
|---|---|
| `actor_prompt.txt` | System prompt for Scientists in Solo mode. Same name/persona/title substitutions as the coupled prompt; describes a **solo** workflow with `submit(step)` and `leave`. |
| `actor_skill.md` | Concise reference card mirroring the prompt (for humans). |
| `config.json` | Cell-specific overrides only — currently `actor_context_coupled: false`. Everything else inherited from `../shared/`. |
| `notes.md` | This file. |

The judge prompt (identical to the one used in cells A/B) and all
shared pools / agent limits are loaded from `../shared/`. See
`../Parliament_context/notes.md` for the shared-resource layout.

## Differences vs `Parliament_context`

* `actor_context_coupled = false` ⇒ runner does not push other
  actors' posts/comments/votes into this actor's queue. The actor's
  user turns contain only judge votes (if visible) or stay empty
  (with cell-specific nudges).
* Actor tool set is `{python_exec, submit(step), leave}` — `vote`
  and `comment` are removed (no targets exist), `wait` is replaced
  by `leave` (no peers to wait for).
* Actor prompt mirrors `../Parliament_context/actor_prompt.txt`
  section by section to keep the comparison fair (same "one step per
  submit", same scoring narrative, same Round-0 instruction).

## 2×2 cells served by this directory

  C Solo       coupled=false + visible=true   (this dir)
  D BlindSolo  coupled=false + visible=false  (this dir + `PRL_JUDGE_VOTES_VISIBLE=0`)
