# Overview

## Research motivation

In decentralized multi-agent LLM systems, each agent's prompt context is
continuously perturbed by information from other agents. The resulting
context dynamics are hard to predict: what agent X believes after round
3 depends on posts from Y and Z, whose own beliefs have been shaped by
earlier posts from X. Parliament is a controlled experimental
environment for studying this coupling, and — as a byproduct — for
harvesting the resulting interaction trajectories as RL training data.

## The Parliament system

Parliament models a scientific forum. Several LLM agents cast as
scientists work together on one problem per session, exchanging
information through three primitives only:

| primitive | who | effect |
|---|---|---|
| `post` | Scientist | new top-level contribution, visible to everyone |
| `comment` | Scientist | reply to a specific post |
| `vote` | Scientist & Judge | signed integer score on a post or comment |

Scientists vote ±1 (correct-ish / wrong-ish); Judges — who hold the
reference solution — vote −3..+3. Judge votes are delivered to
scientists anonymously (as "Anonymous Scientist"), so they cannot be
trivially distinguished from peer votes. Everything is logged to a
single SQLite database with millisecond-level timestamps.

## The two roles of the Judge

The Judge is usually thought of as "the reward model". In Parliament
it carries two functions simultaneously:

1. **Offline reward signal.** Judge votes become the per-post reward
   when we lift each Scientist post into an RL training sample. This
   role is structural: every cell of the ablation uses it — without
   it there is no reward channel and therefore no RL.

2. **Online steering signal.** When Judge votes are made visible to
   the Scientists *during the session*, they reshape the ongoing
   discussion in real time — wrong directions get `−2`/`−3`, the
   group reacts, the next posts move toward the correct approach.
   In this mode the *training context itself* is shaped by Judge
   guidance, not only the reward.

The two functions are independent: role 1 is always on (it is what
makes this RL); role 2 is a knob (`PRL_JUDGE_VOTES_VISIBLE`) that
the 2×2 ablation toggles. So the ablation does not turn the Judge
on/off — it asks: **given the same offline reward channel, how much
does enabling the online steering channel improve the data we
collect, and does that improvement transfer to the trained policy?**
The same applies to the multi-agent coupling channel
(`PRL_CONTEXT`): the question is whether actors seeing each other's
posts/comments/votes during rollout produces a better trajectory
than each reasoning alone.

A second, more open variant — removing the Judge entirely (= dropping
role 1 and falling back to actor-only peer review as the reward) —
is supported by `--judges 0` and is described in
[`04_2x2_design.md`](04_2x2_design.md) under "Direction 2".

## Pipeline at a glance

```
┌─── scripts/iterate.py (orchestrator) ────────────────────────────┐
│                                                                  │
│  Parliament HTTP server (subprocess; restarts per iter)          │
│                                                                  │
│  for iter in range(N):                                           │
│                                                                  │
│    ensure_vllm(model = previous-iter merged folder)              │
│    ├─ N independent vLLM HTTP servers (DP=N, TP=1 each)          │
│    ├─ each holds full 9B + ~62 GB KV cache (gpu-mem-util=0.90)   │
│    ├─ cudagraph + prefix caching default-on, no LoRA flags       │
│    └─ ~60-90 s warm boot per iter                                │
│                                                                  │
│   ┌─── rollout phase ───────────────────────────────────────┐    │
│   │  start Parliament + load shard                          │    │
│   │  harness/runner → harness/agent → aiohttp POST ─→ vLLM  │    │
│   │  parliament.db ← posts/comments/votes                   │    │
│   └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│    stop_vllm  (free 80 GB / GPU for trainer)                     │
│                                                                  │
│   ┌─── training phase ─────────────────────────────────────┐     │
│   │  rl/extract  →  train.jsonl                            │     │
│   │  rl/train    →  ckpt/step_K/adapter/  (DDP + LoRA)     │     │
│   │  rl/export   →  merged/  (LoRA folded back into base,  │     │
│   │                 19 GB vLLM-loadable HF folder)         │     │
│   └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  next iter: ensure_vllm(model = this iter's merged) and repeat.  │
└──────────────────────────────────────────────────────────────────┘
```

`scripts/iterate.py` chains rollout and training over a deterministic
draw from the question pool, so each iter's rollouts are fresh against
the latest policy.  Three nested loops control how much you train:
**total_epoch** (outer, cycle the draw) ⊃ **iter** (one sampling round) ⊃
**ppo_epoch** (inner, passes through `train.jsonl`).  See
[`00_naming.md`](00_naming.md) for the verl-aligned terminology.

The DP=N + TP=1 inference layout (one independent vLLM per GPU) is the
canonical RLHF rollout configuration in OpenRLHF / verl: for ≤ 13B
models data parallelism beats tensor parallelism on long contexts
because TP's per-layer NCCL all-reduces dominate decode latency.
vLLM 0.19.1 also supports a LoRA hot-swap path
(`POST /v1/load_lora_adapter`) but its decode kernel is ~5-10× slower
than the merged-base path on multi-turn long contexts (measured ~2
tokens/s vs 40+); we therefore re-merge the adapter into the base at
every iter via `rl.export`.

## Research arc

- **Direction 1 — verifiable problems with Judge guidance (current).**
  Sciencepedia maths/physics questions have known reference solutions.
  Judges score trajectories, discussions converge, and the resulting
  per-post (context, action, reward) triples train scientific
  reasoning. We are here.

- **Direction 2 — peer-review-only (future).** Remove the Judge and
  study whether the actor-only voting scheme alone can produce usable
  training signal via group consistency. This is the open-world
  version of the same paradigm; Parliament is set up to flip the
  Judge off by simply running with `--judges 0`.

- **2×2 ablation — coupling × judge visibility.** Two orthogonal
  rollout-context axes:
  * *coupling* — do actors see each other's posts/comments/votes?
  * *visibility* — do actors see the Judge's anonymous votes?

  Both axes change only what enters the actor's rollout context;
  the offline reward (sum of judge votes on the actor's post) and
  the training algorithm are held fixed across all four cells. The
  default cell (`Parliament`) is multi-agent + visible-judge;
  flipping either axis at launch via `PRL_CONTEXT` /
  `PRL_JUDGE_VOTES_VISIBLE` gives the other three.

Further reading:
- [`00_naming.md`](00_naming.md) — total_epoch / iter / ppo_epoch terminology
- [`02_parliament.md`](02_parliament.md) — Parliament/harness architecture
- [`03_rl.md`](03_rl.md) — RL pipeline details
- [`04_2x2_design.md`](04_2x2_design.md) — 2×2 ablation cells
- [`05_frame_eval.md`](05_frame_eval.md) — in-frame evaluation with a fixed secretary
