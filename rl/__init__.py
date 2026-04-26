"""Offline RL pipeline — data extraction + training.

Two CLIs, run in order by ``scripts/iterate.py``:

* ``rl.extract`` — ``parliament.db`` → per-actor multi-turn
  trajectory ``train.jsonl``; cell-aware view rebuild + position
  debias + GRPO advantage normalization + template augmentation.
* ``rl.train`` — DDP + LoRA + token-level PPO ratio + asymmetric
  clip + KL anchor to the frozen base. ``π_old`` and ``π_ref`` are
  pre-computed at the start of each iter and frozen across all
  ``ppo_epochs``. Outputs a PEFT adapter folder directly loadable
  by vLLM via ``add_lora`` — no separate merge / export step.

Stage 2 (current): the merged-checkpoint export step (``rl.export``)
was removed because vLLM hot-swaps adapters in-process; merging
the 0.7 GB LoRA into the 19 GB base every iter was pure waste of
disk and CPU time.
"""
