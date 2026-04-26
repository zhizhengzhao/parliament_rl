"""Top-level CLIs that orchestrate Parliament + the RL pipeline.

Architecture: ``scripts.iterate`` runs a per-iter cycle of
{ vLLM HTTP fleet up → harness rollout → vLLM down → DDP train →
LoRA-merge → vLLM up with merged base }. Parliament server runs
alongside vLLM as the data store for rollout traces.

* ``scripts.iterate`` — outer training loop. Per iter: boot N vLLM
  HTTP servers (one per GPU, DP=N + TP=1, prefix-cached), run
  rollout via ``scripts.run``, kill vLLM to free GPU memory, run
  DDP trainer, call ``rl.export`` to merge LoRA into base, reload
  vLLM from the fresh merged folder for the next iter.
* ``scripts.run`` — single rollout pass: starts vLLM + Parliament,
  runs the harness across all sessions, tears everything down.
* ``scripts.eval_policy_on_shard`` — head-to-head reward comparison
  of multiple policies on the same shard.
* ``scripts.sample_dataset`` — depth-5 category-uniform train/test
  split of a full dataset.
* ``scripts.launch_cell`` — one-click launcher for one 2x2 cell
  (sets ``PRL_CONTEXT`` + ``PRL_JUDGE_VOTES_VISIBLE``, kicks
  iterate.py in tmux, plus a keepalive watcher that takes over
  GPU resources when training finishes).
* ``scripts.gpu_keepalive`` — occupy idle GPUs/CPU/RAM/disk above
  cluster utilization floors (no-op on dedicated hardware).
* ``scripts._common`` — env-var forwarding for tmux self-relaunch
  and a tiny ``Tee`` helper for stdout/log mirroring.
"""
