"""Pure-Python unit tests — no GPU, no vLLM, no model weights required.

Tests cover the cell-agnostic core of the project:

* ``test_secretary``    — answer extraction prompt + parsers + scoring helpers.
* ``test_extract``      — vote-stripping, per-actor view, advantage knobs.
* ``test_context_load`` — shared+cell config layering, dedup invariants.

Run::

    pip install -e ".[dev]"
    pytest

CPU-only, < 1 s end-to-end.
"""
