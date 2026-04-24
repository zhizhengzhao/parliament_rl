#!/usr/bin/env python3
"""Head-to-head policy evaluation on a single shard.

Runs one full Parliament rollout → extract → reward-stats per policy,
all on the same shard, then prints a comparison table. The rollout
pipeline is identical to `scripts/iterate.py` sample + extract, so
the reward/advantage stats are directly comparable to the training-time
`DATA METRICS` output.

Typical use: after `iterate.py` finishes a `--total-epochs 2` run,
compare the epoch-1 final policy (`iter_N/merged`) and the epoch-2
final policy (`iter_2N/merged`) on the first shard to measure how
much the second pass actually moved the policy.

Usage:
    python scripts/eval_policy_on_shard.py \\
        --policies ep1:<path>/iter04_.../merged,ep2:<path>/iter08_.../merged \\
        --shard datasets/mid200_part1.json \\
        --name compare_S2 \\
        --gpus 0,1,2,3,4,5,6,7 \\
        --sessions-per-gpu 2 --actors 3 --judges 3 --max-turns 30 \\
        --keepalive            # on Kuaishou pods: exec gpu_keepalive.py
                               # after the comparison prints, so KML
                               # doesn't reclaim the pod.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the exact same sub-step implementations as iterate.py so the
# numbers are comparable with training-time metrics.jsonl output.
from iterate import (  # noqa: E402
    sample_step, extract_step, metrics_step,
    _find_run_dir,
)

PYTHON_ENV = os.environ.get("PRL_PYTHON", sys.executable)


def _parse_policies(spec: str) -> list[tuple[str, str]]:
    """'tag1:path1,tag2:path2' → [('tag1','path1'), ...]"""
    out: list[tuple[str, str]] = []
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Policy spec '{pair}' must be tag:path")
        tag, path = pair.split(":", 1)
        if not Path(path).is_dir():
            raise ValueError(f"Policy path not found: {path}")
        out.append((tag.strip(), path.strip()))
    if not out:
        raise ValueError("At least one --policies tag:path required")
    return out


def _eval_one(tag: str, policy: str, shard: str, name_prefix: str,
              cfg: dict) -> dict:
    """Rollout + extract one policy on one shard; return metrics dict."""
    run_name = f"{name_prefix}_{tag}"
    print(f"\n{'=' * 72}")
    print(f"[eval] {tag}")
    print(f"  policy: {policy}")
    print(f"  shard:  {shard}")
    print(f"{'=' * 72}")

    # Sample step (reuse if rerun — same idempotent rule as iterate.py).
    existing = _find_run_dir(run_name)
    if existing and (existing / "parliament.db").exists() \
            and (existing / "experiment.json").exists():
        print(f"  [skip] rollout — reusing {existing.name}")
        run_dir = existing
    else:
        run_dir = sample_step(Path(shard), policy, run_name, cfg)

    # Extract (reuse if non-empty).
    train_jsonl = run_dir / "train.jsonl"
    if train_jsonl.exists() and train_jsonl.stat().st_size > 100:
        print(f"  [skip] extract — {train_jsonl.name} exists")
    else:
        train_jsonl = extract_step(run_dir)

    metrics = metrics_step(train_jsonl, run_dir)
    metrics["_run_dir"] = str(run_dir)
    metrics["_policy"] = policy
    return metrics


def _print_comparison(results: dict[str, dict]) -> None:
    """Side-by-side table of the metrics dicts."""
    print(f"\n{'=' * 78}")
    print(f"Comparison — all policies rolled out on the same shard")
    print(f"{'=' * 78}")
    hdr = f"  {'tag':<14} {'n_turns':>8} {'r̄':>8} {'r>0%':>7} {'|A|̄':>7} {'A∈[p10,p90]':>20}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tag, m in results.items():
        if not m:
            print(f"  {tag:<14} (no data)")
            continue
        line = (f"  {tag:<14} {m['n_samples']:>8} "
                f"{m['reward_mean']:>+8.3f} "
                f"{m['reward_pos_pct']:>6.1f}% "
                f"{m['advantage_abs_mean']:>7.3f} "
                f"[{m['advantage_p10']:>+6.2f}, {m['advantage_p90']:>+6.2f}]")
        print(line)
    print(f"{'=' * 78}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policies", required=True,
                   help="Comma-separated 'tag:path' pairs to compare")
    p.add_argument("--shard", required=True, help="Path to shard JSON")
    p.add_argument("--name", required=True,
                   help="Name prefix; per-policy dirs are data/<name>_<tag>_<ts>/")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--sessions-per-gpu", type=int, default=2)
    p.add_argument("--actors", type=int, default=3)
    p.add_argument("--judges", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--keepalive", action="store_true",
                   help="After comparison prints, exec scripts/gpu_keepalive.py "
                        "so the pod stays busy (Kuaishou KML requirement).")
    p.add_argument("--keepalive-args", default="--no-cpu --no-ram --no-disk",
                   help="Flags forwarded to gpu_keepalive.py when --keepalive is set.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policies = _parse_policies(args.policies)
    if not Path(args.shard).exists():
        print(f"FATAL: shard not found: {args.shard}")
        sys.exit(1)

    cfg = {
        "gpus": args.gpus,
        "sessions_per_gpu": args.sessions_per_gpu,
        "actors": args.actors,
        "judges": args.judges,
        "max_turns": args.max_turns,
    }

    t0 = time.time()
    print(f"\n{'=' * 72}")
    print(f"Policy comparison — {args.name}")
    print(f"  shard:       {args.shard}")
    print(f"  N policies:  {len(policies)}")
    for tag, path in policies:
        print(f"    {tag:<14} → {path}")
    print(f"  GPUs:        {args.gpus}")
    print(f"{'=' * 72}\n")

    results: dict[str, dict] = {}
    for tag, policy in policies:
        try:
            results[tag] = _eval_one(tag, policy, args.shard, args.name, cfg)
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] {tag}: {e}")
            results[tag] = {}

    _print_comparison(results)

    out_path = PROJECT_DIR / "data" / f"{args.name}_comparison.json"
    out_path.write_text(json.dumps({
        "name": args.name, "shard": args.shard,
        "policies": {t: p for t, p in policies},
        "results": {t: {k: v for k, v in m.items() if not k.startswith("_")}
                    for t, m in results.items()},
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }, indent=2))
    print(f"Saved: {out_path}")

    if args.keepalive:
        keep_cmd = [PYTHON_ENV, str(SCRIPTS_DIR / "gpu_keepalive.py"),
                    *shlex.split(args.keepalive_args)]
        print(f"\n=== keepalive ==="
              f"\n  exec: {shlex.join(keep_cmd)}\n", flush=True)
        os.execvp(keep_cmd[0], keep_cmd)


if __name__ == "__main__":
    main()
