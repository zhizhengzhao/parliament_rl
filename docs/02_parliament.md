# Parliament — architecture

The data-generation half of the system. Two packages work together
over HTTP:

```
parliament/  →  FastAPI + SQLite (server, blind to LLM internals)
harness/     →  async Python (agent runtime, LLM client)
```

## Why an HTTP boundary

The split is not accidental. Having `parliament/` expose a pure HTTP
API means:

- Any client can drive it — curl, a Python agent, an external
  orchestrator, Ray, etc.
- Concurrency is explicit and debuggable. The agent runtime holds no
  Parliament state; the only state is in the DB.
- Sessions can be paused, resumed, inspected, or replayed without any
  machinery on the agent side.
- Ablations that require swapping the agent runtime (e.g. testing a
  different scheduler) don't touch the server.

The server is **~1.5 kLOC of boring code** (`server.py` + `store.py` +
three helpers). All the interesting behaviour lives in `harness/`.

## Event-driven harness

Each session runs this state machine:

```
Runner          Actor                   Judge
───────────     ───────────             ───────────
listen          read queue              read queue
  │               │                       │
  │               ├ python_exec ─ loops   ├ python_exec ─ loops
  │               ├ vote        ─ loops   │
  │               │                       │
  │               └─ submit/wait          └─ vote
  │                    (ENDS round,            (ENDS round,
  │                     wakes runner)           SILENT)
  │
  ├── fetch new posts+comments+votes from DB
  ├── distribute to each non-author agent
  └── if no new content: nudge actors once,
      then close session after one more idle round
```

Key design decisions:

- **Actor's `submit`/`wait` wake the runner; Judge's `vote` does not.**
  Judges never trigger re-distribution, so a session where judges are
  voting continuously doesn't thrash the runner. Their votes pile up
  in the DB and ride along with the next post/comment batch.

- **Two independent processing sets** (`actor_processing`,
  `judge_processing`). Session-end detection only watches actors; we
  then wait briefly for in-flight judges to finish before closing.

- **ID normalization per session.** The harness maps global database
  IDs (`posts.post_id = 4138`) to session-local IDs (`P_1`, `P_2`, …)
  so every agent in the same session sees the same tidy labels. The
  same mapping is reapplied at extract time for training-context
  consistency.

- **Identity anonymization at generation time.** Before the LLM sees
  anything, `Scientist_N` is replaced with a deterministic draw from a
  406-name pool seeded by `session_id`. Both actors and judges are
  renamed; names within a session are guaranteed distinct by a single
  `rng.sample`. The DB still stores stable `Scientist_1..3` /
  `Judge_1..3` names for debuggability; the translation is purely
  presentational.

## Persona pools

`context_configs/Parliament_context/config.json` holds two flat lists:

- `persona_pools.scientist` (100 variants) — "I am an axiomatic
  builder", "I hunt for counterexamples", "I think in Fourier modes",
  etc. Covers construction, exploration, critical, specialty
  mathematical, physical-applied, and communication-style mindsets.

- `persona_pools.judge` (60 variants) — "I focus on dimensional
  consistency", "I reward intellectual courage", "I demand explicit
  counterexamples for disproof", etc.

Each session draws 3 scientist personas and 3 judge personas via
`rng.sample(pool, k)` — guaranteed distinct within the role. Across
sessions the casting is shuffled. Combined with the name pool, two
sessions on the same problem almost never look alike.

## Tools per role

The actor tool set switches with the 2×2 cell (see [`04_2x2_design.md`](04_2x2_design.md));
judge tools are identical in every cell.

| tool | Coupled actor | Independent actor | Judge | ends round? | wakes runner? |
|---|:-:|:-:|:-:|:-:|:-:|
| `python_exec` | ✓ | ✓ | ✓ | no | no |
| `vote` | ±1 only | — | ±1..±3 | no (actor) / yes (judge) | no |
| `submit(comments, post)` | ✓ | — | ✗ | yes | yes |
| `submit(step)` | — | ✓ | ✗ | yes | yes |
| `wait` | ✓ | — | ✗ | yes | yes |
| `leave` | — | ✓ | ✗ | yes (and retires) | yes |

`submit(step)` is the same wire-level call as `submit(post)` — the
LLM-facing argument name is the only difference; both end up writing
a row to the same `posts` table. Argument validation is enforced in
both the tool schema (for the LLM) and in `ToolExecutor` (server-side):
the Actor's vote range is hard-capped at ±1 regardless of what the
LLM tries to output.

`python_exec` runs in an isolated subprocess with a 10-second timeout.
Output is truncated at 10 k characters so a runaway `print` loop can't
swamp the context.

## Defensive parsing

Real LLMs produce malformed tool calls. The harness tolerates:

- `comment` (singular) as an alias for `comments` (plural).
- JSON-as-a-string — if an argument contains a literal string that is
  itself JSON, it's parsed through `json.loads`.
- `votes` embedded inside a `submit` call — extracted and executed.
- Missing tool call (LLM just writes text) — fallback regex parser
  extracts `<tool_call><function=...>` from raw content; if that
  also fails, the response is discarded (three strikes → the round is
  skipped but the agent continues).

Every error returns explicit feedback to the model so it can self-
correct on the next round. **Zero silent drops.**

## Network resilience

Each harness run uses `aiohttp.TCPConnector(force_close=True)`: every
request opens and closes its own connection. This matters because LLM
calls can take 10–30 s, long enough for idle-keep-alive server-side
timeouts to fire on a connection the client still thinks is open.
Paying the cost of reconnection eliminates a whole class of
`ConnectionReset` races.

## Scaling

`scripts/run.py --sessions-per-gpu 2 --gpus 0..7` launches 16 sessions
in parallel. GPU assignment is dynamic: sessions queue up and each GPU
slot (identified by its vLLM endpoint) pulls the next pending session
when it finishes. A long session on one GPU doesn't starve the others.

On 8 × A100-80GB with Qwen3.5-9B this empirically runs ~1026
Sciencepedia questions in 9.3 hours.
