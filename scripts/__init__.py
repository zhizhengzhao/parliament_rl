"""Top-level CLIs that orchestrate Parliament + the RL pipeline.

Stage 2 architecture: a single ``scripts.iterate`` asyncio process
owns vLLM in-process for the whole training run; Parliament is the
only auxiliary HTTP service and the trainer is a per-iter subprocess.

* ``scripts.iterate`` — outer loop: boots vLLM once, then per iter
  runs rollout (in-process harness) → extract → train (DDP
  subprocess), with LoRA adapter hot-swap between iters and
  ``engine.sleep()/wake_up()`` to share GPUs with the trainer.
* ``scripts.eval_policy_on_shard`` — head-to-head reward comparison
  of multiple policies on the same shard.
* ``scripts.sample_dataset`` — depth-5 category-uniform train/test
  split of a full dataset.
* ``scripts.gpu_keepalive`` — occupy idle GPUs/CPU/RAM/disk above
  cluster utilization floors (no-op on dedicated hardware).
* ``scripts._common`` — env-var forwarding for tmux self-relaunch
  and a tiny ``Tee`` helper for stdout/log mirroring.
"""
