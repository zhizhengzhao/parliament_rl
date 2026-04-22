# Solo_context

The **independent** half of the 2×2 ablation. Each actor reasons alone:
no peer posts, no peer comments, no actor votes — only their own
chain of `submit(step)` calls plus optionally judge votes on their own
steps.

Selected at launch via `PRL_CONTEXT=Solo`. The judge cohort, scoring
rubric, and DB schema are identical to `Parliament_context` — only
the actor's view of the session and the actor's tool set differ.

## Files

| File | Purpose |
|---|---|
| `actor_prompt.txt` | System prompt for Scientists in Solo mode. Same name/persona/title substitutions; describes a **solo** workflow with `submit(step)` and `leave`. |
| `judge_prompt.txt` | Identical to `Parliament_context/judge_prompt.txt` — judges always see all posts and score them. |
| `config.json` | `actor_context_coupled: false`, `judge_votes_visible: true`, otherwise identical to `Parliament_context/config.json` (same name/persona pools, agent limits). |
| `actor_skill.md`, `judge_skill.md` | Concise reference cards for humans. |

## Differences vs `Parliament_context`

* `actor_context_coupled = false` ⇒ runner does not push other actors'
  posts/comments/votes into this actor's queue. The actor's user
  turns contain only judge votes (if visible) or stay empty.
* Actor tool set is `{python_exec, submit(step), leave}` — `vote`
  and `comment` are removed (no targets exist), `wait` is replaced
  by `leave` (no peers to wait for).
* Actor prompt mirrors `Parliament_context/actor_prompt.txt` section
  by section to keep the comparison fair (same "one step per submit",
  same scoring narrative, same Round-0 instruction).

## 2×2 cells

| cell | `PRL_CONTEXT` | `PRL_JUDGE_VOTES_VISIBLE` |
|---|---|---|
| A Parliament | `Parliament` | (default true) |
| B BlindParliament | `Parliament` | `0` |
| C Solo | `Solo` | (default true) |
| D BlindSolo | `Solo` | `0` |
