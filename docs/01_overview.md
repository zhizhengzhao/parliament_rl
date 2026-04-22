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

The Judge is usually thought of as "the reward model". But in
Parliament the Judge does **two** things, and both matter:

1. **Reward signal.** Judge votes become the reward when we lift each
   Scientist post into an RL training example.

2. **Online data-quality steering.** Because the Judge's anonymous
   votes are visible to Scientists during the session, they redirect
   the ongoing discussion — wrong directions get `−2`/`−3`, the group
   reacts, the next posts move toward the correct approach. This means
   that even the CONTEXT (not just the reward) of every training
   sample is shaped by the Judge's guidance. The collected trajectory
   is closer to the manifold of good scientific discussion than if
   rollouts were produced without a Judge.

In short: **the Judge is reward + controller at the same time**, and
Parliament was specifically designed so that these two roles interact
naturally in one loop.

## Pipeline at a glance

```
┌─── Parliament (data generation) ────────────────────────────┐
│                                                             │
│  scripts/run.py  →  vLLM + Parliament + harness →           │
│                     parliament.db (posts, comments, votes)  │
│                                                             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─── RL (data consumption) ───────────────────────────────────┐
│                                                             │
│  rl/extract.py   →  train.jsonl  (per-actor trajectories)   │
│  rl/train.py     →  DDP + LoRA + RWR + KL → ckpt/           │
│  rl/export.py    →  merged HF folder (next iter's policy)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

`scripts/iterate.py` chains these two halves together over a sequence
of dataset shards, so each iter's rollouts are fresh against the
latest policy.  Three nested loops control how much you train:
**total_epoch** (outer, full shard list) ⊃ **iter** (one shard) ⊃
**ppo_epoch** (inner, passes through `train.jsonl`).  See
[`00_naming.md`](00_naming.md) for the verl-aligned terminology.

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

- **2×2 ablation — coupling × judge visibility.** Two orthogonal axes
  (do actors see each other's posts? do they see judge votes?) yield
  four cells that isolate the contributions of multi-agent coupling
  and online judge steering. The default cell (`Parliament`) is
  multi-agent + visible-judge; flipping either axis at launch via
  `PRL_CONTEXT` / `PRL_JUDGE_VOTES_VISIBLE` gives the other three.

Further reading:
- [`02_parliament.md`](02_parliament.md) — Parliament/harness architecture
- [`03_rl.md`](03_rl.md) — RL pipeline details
- [`00_naming.md`](00_naming.md) — total_epoch / iter / ppo_epoch terminology
- [`04_2x2_design.md`](04_2x2_design.md) — 2×2 ablation cells
