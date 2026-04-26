"""Held-out evaluations for trained / base policies.

* ``eval.gpqa`` — GPQA Diamond (198 four-choice grad-level questions)
  zero-shot CoT under Qwen3.5 thinking mode.
* ``eval.sciencepedia_mc`` — 100 boxed-letter multiple-choice
  problems held out from both train and test partitions.
* ``eval.frame`` — *in-frame* evaluation: each cell's policy is run
  through a Parliament/Solo session on the test set under its own
  cell setting, and a fixed "secretary" agent extracts a single
  final answer from the resulting discussion. Lets us measure
  downstream ability without forcing every policy onto the same
  zero-shot single-turn distribution it never saw at training time.

All three honour ``$PRL_MODEL_PATH`` for the base model location.
"""
