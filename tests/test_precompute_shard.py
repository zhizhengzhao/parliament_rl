"""Tests for rl/train.py:precompute_log_probs DDP-shard rewrite.

We can't easily spin up real DDP in a unit test.  Instead we test
the **partition + merge logic** directly:

* `_shard_indices(n, world, rank)` — round-robin partition is correct.
* `_compute_local_log_probs(...)` — given a deterministic mock model
  (returns ids[i] as logp), each shard produces the right entries.
* End-to-end equivalence: running the simulated single-rank fast path
  vs running 4 simulated shards then merging produces byte-identical
  ``dataset.samples[i]["old_log_prob"]`` / ``["ref_log_prob"]``.

Skipped unless torch is installed (it's a heavy dep).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

torch = pytest.importorskip("torch")

# Now safe to import the module under test.
from rl.train import (  # noqa: E402
    _shard_indices,
    _compute_local_log_probs,
    _attach_log_probs,
)


# ── _shard_indices ──────────────────────────────────────


def test_shard_indices_partition_disjoint_and_complete():
    n = 25
    world = 8
    seen = []
    for rank in range(world):
        seen.extend(_shard_indices(n, world, rank))
    assert sorted(seen) == list(range(n))


def test_shard_indices_round_robin_balance():
    """No rank gets more than ⌈n/world⌉ samples."""
    n = 25
    world = 8
    cap = -(-n // world)        # ceil(25/8) = 4
    for rank in range(world):
        assert len(_shard_indices(n, world, rank)) <= cap


def test_shard_indices_world_one():
    n = 17
    assert _shard_indices(n, 1, 0) == list(range(n))


def test_shard_indices_world_equals_n():
    n = 5
    for r in range(n):
        assert _shard_indices(n, n, r) == [r]


def test_shard_indices_world_exceeds_n():
    n = 3
    for r in range(5):
        expected = [r] if r < n else []
        assert _shard_indices(n, 5, r) == expected


# ── _compute_local_log_probs + _attach_log_probs ────────


class _MockModel:
    """A fake LM whose per-token log-probs are deterministic functions
    of the input ids.

    Returns ``F.log_softmax(logits)`` where ``logits[..., v] = ids[..., t]``
    for the one-hot vocab dim — so each token's log-prob is computable
    without torch.compile, just from the input.

    We don't actually care about realistic log-probs; we care that
    the shard path produces the **same** tensor as the single-rank
    path on the same input.
    """

    def __init__(self, vocab_size: int = 200):
        self.training = False
        self.vocab_size = vocab_size

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, *, input_ids, attention_mask, use_cache):
        # Build deterministic logits: logits[b, t, v] = sin(ids[b,t] * v / 7)
        # Cheap, deterministic, no zero-vector pathology.
        B, T = input_ids.shape
        v = torch.arange(self.vocab_size, dtype=torch.float32)
        ids_f = input_ids.float().unsqueeze(-1)  # [B, T, 1]
        logits = torch.sin(ids_f * v / 7.0)      # [B, T, V]
        # Pretend this is a standard model output.
        out = type("Out", (), {"logits": logits})()
        return out


class _MockDataset:
    """Container with `.samples` list, each dict has `input_ids`."""

    def __init__(self, n: int, seq_lens: list[int] | None = None):
        seq_lens = seq_lens or [(i % 5) + 4 for i in range(n)]
        self.samples = []
        for i, L in enumerate(seq_lens):
            ids = torch.tensor([(i + 1) * (j + 1) % 200 for j in range(L)],
                               dtype=torch.long)
            self.samples.append({"input_ids": ids})


def _single_rank_compute(model, dataset, device, need_old, need_ref):
    """Reference implementation — non-sharded compute over all indices."""
    return _compute_local_log_probs(
        model, dataset, device,
        list(range(len(dataset.samples))),
        lora_unwrap=None, ref_model=None,
        need_old=need_old, need_ref=need_ref,
    )


def _shard_then_merge(model, dataset, device, world, need_old, need_ref):
    """Simulated multi-rank: sequentially compute each rank's shard,
    then merge — should be byte-identical to single-rank output."""
    merged: dict[int, dict] = {}
    n = len(dataset.samples)
    for rank in range(world):
        idx = _shard_indices(n, world, rank)
        local = _compute_local_log_probs(
            model, dataset, device, idx,
            lora_unwrap=None, ref_model=None,
            need_old=need_old, need_ref=need_ref,
        )
        merged.update(local)
    return merged


def test_local_compute_same_as_single_rank_old_only():
    model = _MockModel()
    ds = _MockDataset(n=20)
    device = torch.device("cpu")

    ref = _single_rank_compute(model, ds, device, need_old=True, need_ref=False)
    shard = _shard_then_merge(model, ds, device, world=4,
                              need_old=True, need_ref=False)
    assert set(ref.keys()) == set(shard.keys()) == set(range(20))
    for idx in ref:
        a = ref[idx]["old"]
        b = shard[idx]["old"]
        assert torch.equal(a, b), f"mismatch at idx={idx}"
        assert "ref" not in ref[idx]
        assert "ref" not in shard[idx]


def test_local_compute_world_sweep():
    """Same equivalence holds for a sweep over world sizes."""
    model = _MockModel()
    ds = _MockDataset(n=17)  # not divisible by anything
    device = torch.device("cpu")
    ref = _single_rank_compute(model, ds, device,
                               need_old=True, need_ref=False)
    for world in (1, 2, 3, 5, 8, 17, 32):
        shard = _shard_then_merge(model, ds, device, world,
                                  need_old=True, need_ref=False)
        assert set(shard.keys()) == set(ref.keys()), f"world={world}"
        for idx in ref:
            assert torch.equal(ref[idx]["old"], shard[idx]["old"]), \
                f"world={world} idx={idx}"


def test_attach_writes_into_dataset():
    ds = _MockDataset(n=4)
    cache = {
        0: {"old": torch.tensor([0.1, 0.2])},
        1: {"old": torch.tensor([0.3, 0.4]),
            "ref": torch.tensor([0.05, 0.06])},
        # idx 2 omitted — left untouched
        3: {"ref": torch.tensor([0.7])},
    }
    _attach_log_probs(ds, cache)
    assert torch.equal(ds.samples[0]["old_log_prob"],
                       torch.tensor([0.1, 0.2]))
    assert "ref_log_prob" not in ds.samples[0]
    assert torch.equal(ds.samples[1]["ref_log_prob"],
                       torch.tensor([0.05, 0.06]))
    assert "old_log_prob" not in ds.samples[2]
    assert torch.equal(ds.samples[3]["ref_log_prob"], torch.tensor([0.7]))
