"""Tests for the context_configs/ shared+cell layering invariants.

Pinned behaviours:

* Both Parliament and Solo cells load successfully and only differ in
  ``actor_context_coupled``.
* Shared-only fields (`name_pool`, `persona_pools`, `agent.*`,
  `judge_votes_visible`) are byte-identical across cells (via the
  shared/config.json single source of truth).
* `judge_prompt` lives in shared/ and both cells return the same string.
* `actor_prompt` is cell-specific and the two cells return different prompts.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _load(ctx: str) -> dict:
    """Force a fresh `harness.prompts` import so PRL_CONTEXT applies."""
    os.environ["PRL_CONTEXT"] = ctx
    import harness.prompts as p
    importlib.reload(p)
    return p.load_context_config()


def test_parliament_cell_loads():
    cfg = _load("Parliament")
    assert cfg["actor_context_coupled"] is True
    assert "name_pool" in cfg and len(cfg["name_pool"]) >= 100
    assert "persona_pools" in cfg
    assert "judge_prompt" in cfg and len(cfg["judge_prompt"]) > 200
    assert "actor_prompt" in cfg and len(cfg["actor_prompt"]) > 200


def test_solo_cell_loads():
    cfg = _load("Solo")
    assert cfg["actor_context_coupled"] is False


def test_shared_fields_identical_across_cells():
    a = _load("Parliament")
    b = _load("Solo")
    # Shared invariants
    assert a["name_pool"] == b["name_pool"]
    assert a["persona_pools"] == b["persona_pools"]
    assert a["agent"] == b["agent"]
    assert a["judge_prompt"] == b["judge_prompt"]
    assert a["judge_votes_visible"] == b["judge_votes_visible"]
    # Cell-specific invariants
    assert a["actor_context_coupled"] != b["actor_context_coupled"]
    assert a["actor_prompt"] != b["actor_prompt"]


def test_no_duplicated_judge_files_in_cell_dirs():
    """The dedup commit removed judge_prompt/judge_skill from cell dirs."""
    for cell in ("Parliament_context", "Solo_context"):
        d = PROJECT_DIR / "context_configs" / cell
        assert not (d / "judge_prompt.txt").exists(), \
            f"{cell}: leftover duplicate judge_prompt.txt"
        assert not (d / "judge_skill.md").exists(), \
            f"{cell}: leftover duplicate judge_skill.md"


def test_shared_dir_has_expected_files():
    d = PROJECT_DIR / "context_configs" / "shared"
    assert (d / "config.json").exists()
    assert (d / "judge_prompt.txt").exists()
    assert (d / "judge_skill.md").exists()


def test_rl_extract_reads_name_pool_from_shared():
    """rl/extract.py:load_rl_config should source name_pool from shared/."""
    from rl.extract import load_rl_config
    cfg = load_rl_config()
    assert "name_pool" in cfg
    assert len(cfg["name_pool"]) >= 100
    # Same byte-identical pool the harness sees
    parl = _load("Parliament")
    assert cfg["name_pool"] == parl["name_pool"]
