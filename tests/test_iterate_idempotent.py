"""Tests for scripts/iterate.py idempotent-step validators.

The two functions tested (`_train_jsonl_complete`, `_adapter_complete`)
are pure I/O sniffers — they decide whether a previously-written
artifact is structurally complete enough to skip re-running its step.
False-positive (claim complete when it's not) is the dangerous failure
mode (corrupt artifact gets fed to the next step); false-negative
(claim incomplete when it actually is) is harmless (just re-runs).

Stage 2 (in-process vLLM): the legacy ``_merged_complete`` checker
is gone — there's no ``merged/`` folder anymore because vLLM
``add_lora`` accepts the trainer's PEFT adapter directly.  The new
``_adapter_complete`` validates the PEFT adapter folder shape
(``adapter_config.json`` + ``adapter_model.safetensors``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for p in (PROJECT_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from iterate import _train_jsonl_complete, _adapter_complete


# ── _train_jsonl_complete ───────────────────────────────


def _write(p: Path, content: str) -> Path:
    p.write_text(content)
    return p


def test_train_jsonl_complete_valid(tmp_path):
    p = tmp_path / "train.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "x"}],
         "turn_advantages": [0.5], "turn_rewards": [1]},
        {"messages": [{"role": "user", "content": "y"}],
         "turn_advantages": [-0.3], "turn_rewards": [0]},
    ]
    _write(p, "\n".join(json.dumps(r) for r in rows) + "\n")
    assert _train_jsonl_complete(p)


def test_train_jsonl_complete_missing_file(tmp_path):
    assert not _train_jsonl_complete(tmp_path / "nonexistent.jsonl")


def test_train_jsonl_complete_empty_file(tmp_path):
    p = _write(tmp_path / "train.jsonl", "")
    assert not _train_jsonl_complete(p)


def test_train_jsonl_complete_below_min_bytes(tmp_path):
    p = _write(tmp_path / "train.jsonl", "{}\n")
    assert not _train_jsonl_complete(p, min_bytes=100)


def test_train_jsonl_complete_truncated_last_line(tmp_path):
    """Half-written tail = the dangerous case — must NOT be reused."""
    good_row = {"messages": [], "turn_advantages": []}
    text = json.dumps(good_row) + "\n" + json.dumps(good_row)[:-5]  # truncated
    p = _write(tmp_path / "train.jsonl", text + " " * 100)  # padding for size
    assert not _train_jsonl_complete(p)


def test_train_jsonl_complete_missing_required_field(tmp_path):
    """Field omission = schema drift — must NOT be reused."""
    p = _write(tmp_path / "train.jsonl",
               json.dumps({"messages": []}) + "\n" +
               json.dumps({"messages": []}) + "\n" +
               " " * 100)
    assert not _train_jsonl_complete(p)


def test_train_jsonl_complete_garbage_first_line(tmp_path):
    p = _write(tmp_path / "train.jsonl",
               "this is not json\n" +
               json.dumps({"messages": [], "turn_advantages": []}) + "\n" +
               " " * 100)
    assert not _train_jsonl_complete(p)


def test_train_jsonl_complete_single_valid_long_line(tmp_path):
    """One valid line that happens to be > min_bytes also passes."""
    row = {"messages": [{"role": "user", "content": "padding " * 50}],
           "turn_advantages": [0.5]}
    _write(tmp_path / "train.jsonl", json.dumps(row) + "\n")
    assert _train_jsonl_complete(tmp_path / "train.jsonl")


# ── _adapter_complete (PEFT adapter folder for vLLM hot-swap) ──


def _make_adapter(tmp: Path, *,
                  has_config: bool = True,
                  has_weights: bool = True) -> Path:
    """Build a fake PEFT adapter folder.

    Both ``adapter_config.json`` (PEFT metadata) and
    ``adapter_model.safetensors`` (the actual LoRA weights) must be
    present for vLLM ``add_lora`` to accept it.  The shapes don't
    matter for the validator — it's a pure presence check.
    """
    adapter = tmp / "step_42" / "adapter"
    adapter.mkdir(parents=True)
    if has_config:
        (adapter / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 64, "lora_alpha": 128}))
    if has_weights:
        (adapter / "adapter_model.safetensors").write_text("fake weights")
    return adapter


def test_adapter_complete_happy_path(tmp_path):
    adapter = _make_adapter(tmp_path)
    assert _adapter_complete(adapter)


def test_adapter_complete_missing_config(tmp_path):
    adapter = _make_adapter(tmp_path, has_config=False)
    assert not _adapter_complete(adapter)


def test_adapter_complete_missing_weights(tmp_path):
    """Trainer crashed mid-save: config flushed but weights weren't.
    vLLM would die on add_lora; we have to detect this case so the
    iter runs the trainer again instead of pointing vLLM at a corpse.
    """
    adapter = _make_adapter(tmp_path, has_weights=False)
    assert not _adapter_complete(adapter)


def test_adapter_complete_dir_does_not_exist(tmp_path):
    assert not _adapter_complete(tmp_path / "nonexistent_adapter")


def test_adapter_complete_is_a_file(tmp_path):
    """Defensive: someone passed a file path, not a folder."""
    f = tmp_path / "adapter"
    f.write_text("not a folder")
    assert not _adapter_complete(f)
