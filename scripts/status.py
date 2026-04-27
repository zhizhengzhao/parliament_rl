#!/usr/bin/env python3
"""prl-status — one-line summary of every iterative-training run in
``data/`` (or any directory you point it at).

Reads three small files from each ``data/<name>_<ts>/``:
  * ``manifest.json``  — run config (cell, total iters, started_at)
  * ``state.json``     — current iter + status field
  * ``DONE.txt`` / ``FAILED.txt``  — terminal-state sentinels

Output is a single fixed-width table:

    NAME              STATUS      ITER       WALL   STARTED            FINISHED
    main400e3_A       completed   6/6        445m   16:00:00 04-26     07:25:00 04-27
    main400e3_B       failed@2    1/6        201m   16:00:00 04-26     19:21:00 04-26
    main400e3_C       failed@1    0/6        45m    16:00:00 04-26     16:45:00 04-26
    main400e3_D       completed   6/6        185m   16:00:00 04-26     19:05:00 04-26

Parses nothing else. No GPU access. No imports beyond the std lib.
Designed to be safe to run anytime — including while the experiment
is still running (``status: running``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _parse_failed_txt(path: Path) -> dict:
    """Tiny parser for the ``key: value`` shape of FAILED.txt."""
    info: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return info


def _format_iso(stamp: str | None) -> str:
    if not stamp:
        return "—"
    try:
        # state.json timestamps look like "2026-04-26T07:25:00.123456"
        return datetime.fromisoformat(stamp).strftime("%H:%M:%S %m-%d")
    except (ValueError, TypeError):
        return stamp[:14]


def summarise_run(run_dir: Path) -> dict:
    """Build a single-row summary dict for one run directory.

    Status ladder (highest priority first):
      * DONE.txt exists       → "completed"
      * FAILED.txt exists     → "failed@<iter>"
      * state.json status=…   → that label (running / failed / completed)
      * stale (no run-dir mtime change in > 24 h, no sentinel) → "stale"
      * else                  → "unknown"
    """
    manifest = _read_json(run_dir / "manifest.json") or {}
    state = _read_json(run_dir / "state.json") or {}
    failed_info = _parse_failed_txt(run_dir / "FAILED.txt")
    done_path = run_dir / "DONE.txt"
    failed_path = run_dir / "FAILED.txt"
    now_ts = datetime.now().timestamp()

    total = manifest.get("total_iters") or 0
    completed = state.get("completed") or 0
    name = manifest.get("name") or run_dir.name

    # ``last_started_at`` is set on every iterate.py boot; ``started_at``
    # is the original launch time. For wall-time math while running we
    # prefer last_started_at so a resume after a crash doesn't make wall
    # appear negative; for the human-facing "STARTED" column we still
    # show the original launch time.
    started_iso = manifest.get("started_at")
    last_started_iso = manifest.get("last_started_at") or started_iso

    if done_path.exists():
        status = "completed"
        finished_ts = done_path.stat().st_mtime
    elif failed_path.exists():
        attempt_iter = failed_info.get("failed_iter", "").split("/")[0]
        status = f"failed@{attempt_iter}" if attempt_iter else "failed"
        finished_ts = failed_path.stat().st_mtime
    else:
        explicit_status = state.get("status")
        # If no explicit status and no recent activity, mark stale so
        # ancient half-finished runs don't masquerade as "running".
        run_mtime = run_dir.stat().st_mtime
        idle_hours = (now_ts - run_mtime) / 3600
        if explicit_status:
            status = explicit_status
        elif idle_hours > 24:
            status = "stale"
        else:
            status = "running"
        finished_ts = run_mtime if status != "running" else None

    # Wall time = finished - last_started (or now - last_started for running)
    wall_min = "—"
    if last_started_iso:
        try:
            ref_started_ts = datetime.fromisoformat(
                last_started_iso).timestamp()
            ref_end_ts = finished_ts if finished_ts is not None else now_ts
            mins = (ref_end_ts - ref_started_ts) / 60
            # Negative wall time can only happen on data corruption
            # (e.g. sentinel mtime older than last_started_at because of
            # NTP jump); show "—" rather than nonsense.
            wall_min = f"{mins:.0f}m" if mins >= 0 else "—"
        except (ValueError, TypeError):
            pass

    return {
        "name": name,
        "status": status,
        "iter": f"{completed}/{total}" if total else f"{completed}",
        "wall": wall_min,
        "started": _format_iso(started_iso),
        "finished": _format_iso(
            datetime.fromtimestamp(finished_ts).isoformat()
            if finished_ts is not None else None),
    }


def collect_runs(data_dir: Path) -> list[dict]:
    """Walk ``data/`` looking for run directories.

    A directory is a "run" if it has either ``manifest.json`` or
    ``state.json``.  We deliberately do NOT recurse into
    ``data/<run>/iter*/`` subdirectories — those carry per-iter
    artifacts (parliament.db, ckpt/, …), not a separate run.
    """
    if not data_dir.is_dir():
        return []
    runs = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "manifest.json").exists() or (sub / "state.json").exists():
            runs.append(summarise_run(sub))
    return runs


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "(no runs found)"
    cols = [
        ("NAME",      "name",     22),
        ("STATUS",    "status",   12),
        ("ITER",      "iter",      8),
        ("WALL",      "wall",      8),
        ("STARTED",   "started",  18),
        ("FINISHED",  "finished", 18),
    ]
    out: list[str] = []
    out.append("  ".join(h.ljust(w) for h, _, w in cols))
    for row in rows:
        out.append("  ".join(str(row[k])[:w].ljust(w) for _, k, w in cols))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data-dir", default=None,
        help="Where to look for runs (default: <project>/data)")
    ap.add_argument(
        "--filter", default=None,
        help="Only show rows whose name contains this substring")
    ap.add_argument(
        "--json", action="store_true",
        help="Output JSON list instead of fixed-width table")
    args = ap.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
    else:
        # Project-root-relative — works when run from any CWD inside
        # the project tree.
        here = Path(__file__).resolve()
        data_dir = here.parent.parent / "data"

    rows = collect_runs(data_dir)
    if args.filter:
        rows = [r for r in rows if args.filter in r["name"]]

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(f"Runs under {data_dir}:")
        print(render_table(rows))


if __name__ == "__main__":
    main()
