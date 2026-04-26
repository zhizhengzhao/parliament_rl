"""Offline RL pipeline — data extraction + training + export.

Three CLIs, run in order by ``scripts/iterate.py``:

* ``rl.extract`` — ``parliament.db`` → per-actor multi-turn
  trajectory ``train.jsonl``; cell-aware view rebuild + position
  debias + GRPO advantage normalization + template augmentation.
* ``rl.train`` — DDP + LoRA + token-level PPO ratio + asymmetric
  clip + KL anchor to the frozen base. ``π_old`` and ``π_ref`` are
  pre-computed at the start of each iter and frozen across all
  ``ppo_epochs``. Outputs a PEFT adapter folder.
* ``rl.export`` — merge LoRA adapter into the base model; produce
  a full HF folder that the next iter's vLLM loads directly. Hot-
  swap (``/v1/load_lora_adapter``) was tried but vLLM 0.19.1 LoRA
  decode is ~5-10× slower than merged base on long context, so
  merge+reload is the canonical path (see ``scripts/iterate.py``).
"""
