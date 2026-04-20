"""Shared utilities for scripts/run.py and scripts/iterate.py."""

from __future__ import annotations

import os
import shlex

# Env vars forwarded into tmux child shells — tmux servers inherit env
# from when they were first started, not from the current caller.
FORWARDED_ENV = (
    "PRL_PYTHON", "PRL_ACCELERATE", "PRL_MODEL_PATH",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
    "PYTORCH_ALLOC_CONF",
    "http_proxy", "https_proxy", "no_proxy",
)


def env_prefix() -> str:
    """Build `KEY=val KEY2=val2 ` string for tmux new-session commands."""
    parts = [f"{k}={shlex.quote(os.environ[k])}"
             for k in FORWARDED_ENV if k in os.environ]
    return " ".join(parts) + " " if parts else ""


class Tee:
    """Write to multiple streams simultaneously (stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()
