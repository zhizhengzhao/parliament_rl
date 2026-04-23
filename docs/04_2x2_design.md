# 2×2 ablation — coupling × judge visibility

The Parliament RL pipeline supports four ablation cells along two
orthogonal axes:

| | judge votes **visible** to actors | judge votes **hidden** from actors |
|---|---|---|
| **coupled** actor context | A — Parliament | B — BlindParliament |
| **independent** actor context | C — Solo | D — BlindSolo |

The judge cohort, scoring rubric, reward formula, advantage
calculation, training loss, and DB schema are **identical** across all
four cells. Only the actor side differs:

* Coupling axis: does each actor see other actors' posts/comments/votes
  during the session, or only its own derivation?
* Visibility axis: does the actor receive judge votes (anonymised) on
  any posts during the session, or only after training as a reward?

## Selecting a cell

Two environment variables control the cell. Defaults give cell A
(Parliament).

The rest of the command is **identical across all four cells** — same
`--pool`, `--total-questions`, `--sampling-batch-size`, `--seed`,
`--total-epochs`, `--train-extra` — so every cell sees exactly the
same deterministic question schedule and the same training
hyperparameters.  The only flags that vary are the two env vars:

```bash
COMMON='--pool datasets/sciencepedia_train_part1.json,\
datasets/sciencepedia_train_part2.json,\
datasets/sciencepedia_train_part3.json,\
datasets/sciencepedia_train_part4.json \
  --total-questions 1000 --sampling-batch-size 200 \
  --total-epochs 2 --seed 42 \
  --train-extra "--ppo-epochs 2 --clip-ratio-high 0.25 --beta-kl 0.005 \
                 --lora-r 64 --lora-alpha 128" \
  --gpus 0,1,2,3,4,5,6,7'

# A — Parliament (defaults: coupled, judge-visible)
python scripts/iterate.py --name mainA $COMMON

# B — BlindParliament (coupled, judge-hidden)
PRL_JUDGE_VOTES_VISIBLE=0 python scripts/iterate.py --name mainB $COMMON

# C — Solo (independent, judge-visible)
PRL_CONTEXT=Solo python scripts/iterate.py --name mainC $COMMON

# D — BlindSolo (independent, judge-hidden)
PRL_CONTEXT=Solo PRL_JUDGE_VOTES_VISIBLE=0 \
    python scripts/iterate.py --name mainD $COMMON
```

Both env vars propagate through `iterate.py → scripts/run.py → harness/`
(via `scripts/_common.FORWARDED_ENV`) so no extra CLI plumbing needed.

`PRL_JUDGE_VOTES_VISIBLE` accepted truthy values: anything except
`""`, `0`, `false`, `no`, `off` (case-insensitive).

## Implementation

### Coupling (`actor_context_coupled`)

`harness/runner.py:run_session` reads the flag from the loaded
`config.json`. When false, the actor distribution payload is empty
(plus optional judge votes). The runner still fetches all posts and
distributes them to judges so scoring proceeds normally; it just
withholds them from the other actors.

```python
if actor_context_coupled:
    actor_payload = posts + comments + actor_votes
else:
    actor_payload = []
if judge_votes_visible:
    actor_payload += [anonymised_judge_votes]
```

### Visibility (`judge_votes_visible`)

Already supported pre-2×2 via the same config flag. The env
override `PRL_JUDGE_VOTES_VISIBLE` (resolved in
`runner._resolve_judge_visibility`) lets you flip it without editing
`config.json` so the same context dir covers both halves of the axis.

### Tool sets (actor only)

`harness/tools.get_tools(role, coupled=...)` returns:

| | python_exec | vote | comment | submit | wait | leave |
|---|---|---|---|---|---|---|
| Coupled actor (A/B) | ✓ | ✓ | ✓ (in submit) | `submit(comments, post)` | ✓ | ✗ |
| Independent actor (C/D) | ✓ | ✗ | ✗ | `submit(step)` | ✗ | ✓ |
| Judge (all cells) | ✓ | ✓ | — | — | — | — |

Independent `submit(step)` is the same wire-level call as coupled
`submit(post)`; the LLM-facing argument name is renamed for prompt
cleanliness; `ToolExecutor.execute_submit` accepts both keys.
`leave` is a client-only tool — it sets `result.exit_reason="left"`
and the agent main loop breaks. No DB write, no extra endpoint.

### Naming layer

| coupled vocab | independent vocab |
|---|---|
| post | step |
| submit a comment + a post | submit a step |
| wait for new content | leave the session |

The DB still stores everything as `post` rows, so `rl/extract.py` and
`rl/train.py` are completely unchanged across cells. The training
sample format (`messages`, `turn_rewards`, `turn_advantages`) is
isomorphic by construction.

## Fairness checklist

| dimension | how identical |
|---|---|
| **question schedule** | same `--pool` + `--total-questions` + `--sampling-batch-size` + `--seed` → literally the same batches at the same positions |
| number of actors | same `--actors` |
| number of judges | same `--judges` |
| max rounds | same `--max-turns`, same `agent.max_rounds` |
| token budget | same `MAX_TOKENS` (2048) |
| reward source | same judge cohort, same scoring rubric |
| advantage formula | `rl/extract.py` unchanged (mean_session + position debias) |
| training loss | `rl/train.py` unchanged (PPO clip + KL-to-base) |
| PPO hyperparams | same `--ppo-epochs`, `--clip-ratio-low/high`, `--beta-kl` |
| trajectory format | per-actor multi-turn chat in all cells |
| KL anchor | base model fixed across iters in every cell |

The only intended difference is the **actor's information access**.

## Where things differ at runtime

* **Cell A (Parliament)**: actor sees `posts + comments + actor_votes
  + judge_votes (anonymised)` on its queue between rounds.
* **Cell B (BlindParliament)**: same as A minus judge_votes — actor
  knows other scientists are voting (`vote` tool exists), but the
  anonymous senior-scientist scoring channel is silent.
* **Cell C (Solo)**: actor's queue contains only judge_votes on its
  own steps. No tool to vote or comment.
* **Cell D (BlindSolo)**: actor's queue is empty between rounds. The
  user turns become "No new feedback. Continue your derivation: …"
  nudges. `extract.py` plants a `"No new discussion. Continue."`
  placeholder if no real timeline content exists.

## Minimal smoke check

Run a tiny end-to-end iter (50 q / 10 per round / 1 epoch / 1 ppo_epoch)
with one cell's flags to sanity-check the full pipeline:

```bash
PRL_CONTEXT=Solo  PRL_JUDGE_VOTES_VISIBLE=0  \
python scripts/iterate.py \
    --name smoke_D \
    --pool datasets/sciencepedia_test.json \
    --total-questions 50 --sampling-batch-size 10 \
    --total-epochs 1 --seed 42 \
    --train-extra "--ppo-epochs 1" \
    --gpus 0,1,2,3,4,5,6,7
```

Expect:
* tmux `parliament-iterate` running, vLLM up on 8 GPUs
* `harness` log shows tool list `[python_exec, submit, leave]` for
  each actor (independent Solo cell)
* `parliament.db` accumulates posts (DB schema unchanged across cells)
* `ckpt/metrics.jsonl` records `clipfrac ≈ 0` and `approx_kl_old ≈ 0`
  for `ppo_epoch=1` (ratio ≈ 1, no trust-region action yet)
* `backups/smoke_D_<ts>/ep1.round{1..5}/` populated with metrics +
  train.jsonl per iter
