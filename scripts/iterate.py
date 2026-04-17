#!/usr/bin/env python3
"""One-click iterative RL training.

For each dataset shard, in order:

    1. Launch vLLM from the current policy
    2. Run Parliament + harness (scripts/run.py) → parliament.db
    3. Extract training data (rl/extract.py)     → train.jsonl
    4. Train with FSDP (rl/train.py)              → ckpt/step_K
    5. Export merged HF folder (rl/export.py)     → merged/
    6. Set the merged folder as the next iteration's policy

State is persisted at `data/<name>_<ts>/state.json` so a crashed or
killed run can be resumed by re-invoking the same command — completed
iterations are skipped automatically.

Why shard at all? A single collection of N rollouts is fresh only
against the π that produced it; after training drifts π the remaining
rollouts are stale. Sharding keeps every collection freshly on-policy
for its iteration. The built-in 4×1026 train_part split is the natural
choice: 4 iterations × ≈13.5 h = ≈54 h end-to-end on 8 × A100-80GB.

Usage (typical):
    python scripts/iterate.py \\
        --name nrun_v1 \\
        --shards datasets/sciencepedia_train_part1.json,\\
                 datasets/sciencepedia_train_part2.json,\\
                 datasets/sciencepedia_train_part3.json,\\
                 datasets/sciencepedia_train_part4.json \\
        --gpus 0,1,2,3,4,5,6,7

Resume (same command, same --name, it picks up from state.json):
    python scripts/iterate.py --name nrun_v1 --shards ... --gpus ...

Stop the background tmux job at any time:
    tmux kill-session -t parliament-iterate
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_MODEL = "/root/zhizheng/models/Qwen3.5-9B"
ACCELERATE = "/root/miniconda3/envs/parliament/bin/accelerate"
PYTHON_ENV = "/root/miniconda3/envs/parliament/bin/python"
ITERATE_TMUX = "parliament-iterate"


class Tee:
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


# ── Sub-step runners ─────────────────────────────────────

def run(cmd: list[str], cwd: Path = PROJECT_DIR) -> None:
    """Run subprocess, inherit stdout/stderr, raise on failure."""
    print(f"\n$ {shlex.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def find_run_dir(name_prefix: str) -> Path:
    """Locate the `data/<name>_<timestamp>/` dir that scripts/run.py just created."""
    candidates = sorted((PROJECT_DIR / "data").glob(f"{name_prefix}_*"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"Could not find data dir matching {name_prefix}_*")
    return candidates[-1]


def find_last_checkpoint(ckpt_dir: Path) -> Path:
    steps = sorted(ckpt_dir.glob("step_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    if not steps:
        raise RuntimeError(f"No step_* dirs in {ckpt_dir}")
    return steps[-1]


def sample_step(shard: str, model: str, name: str, cfg: dict) -> Path:
    """Stage 1: scripts/run.py does cleanup → vLLM → Parliament → harness → DB.

    We pass `--in-tmux` so it doesn't wrap itself in a second tmux
    layer; our iterate.py is already running inside one.
    """
    run([
        PYTHON_ENV, "scripts/run.py",
        "--in-tmux",
        "--gpus", cfg["gpus"],
        "--sessions-per-gpu", str(cfg["sessions_per_gpu"]),
        "--actors", str(cfg["actors"]),
        "--judges", str(cfg["judges"]),
        "--dataset", shard,
        "--name", name,
        "--model", model,
        "--max-turns", str(cfg["max_turns"]),
    ])
    return find_run_dir(name)


def extract_step(run_dir: Path) -> Path:
    train_jsonl = run_dir / "train.jsonl"
    run([PYTHON_ENV, "-m", "rl.extract",
         "--db", str(run_dir / "parliament.db"),
         "--output", str(train_jsonl)])
    return train_jsonl


def train_step(train_jsonl: Path, model: str, run_dir: Path,
               num_gpus: int, extra: list[str] | None = None) -> Path:
    ckpt_dir = run_dir / "ckpt"
    cmd = [
        ACCELERATE, "launch",
        "--config_file", "rl/accelerate_fsdp.yaml",
        "--num_processes", str(num_gpus),
        "-m", "rl.train",
        "--data", str(train_jsonl),
        "--output", str(ckpt_dir),
        "--model", model,
        "--ref-model", model,
    ]
    if extra:
        cmd += extra
    run(cmd)
    return ckpt_dir


def export_step(ckpt_dir: Path, run_dir: Path, num_gpus: int) -> Path:
    merged_dir = run_dir / "merged"
    last = find_last_checkpoint(ckpt_dir)
    run([
        ACCELERATE, "launch",
        "--config_file", "rl/accelerate_fsdp.yaml",
        "--num_processes", str(num_gpus),
        "-m", "rl.export",
        "--ckpt", str(last),
        "--output", str(merged_dir),
    ])
    return merged_dir


def prune_sharded_ckpt(ckpt_dir: Path) -> None:
    """Delete step_* directories once the merged export succeeds.

    Saves ~51 GB per iteration; the merged/ folder has everything we
    need for the next iteration's vLLM rollout and as a KL reference.
    """
    for d in ckpt_dir.glob("step_*"):
        shutil.rmtree(d, ignore_errors=True)
    print(f"  Pruned sharded checkpoints in {ckpt_dir}", flush=True)


# ── Iteration loop ───────────────────────────────────────

def run_one_iteration(iter_n: int, total: int, shard: str, model: str,
                      name_prefix: str, cfg: dict,
                      train_extra: list[str]) -> str:
    t0 = time.time()
    name = f"{name_prefix}_iter{iter_n:02d}"
    num_gpus = len(cfg["gpus"].split(","))
    print(f"\n{'=' * 72}")
    print(f"Iteration {iter_n}/{total} — shard={shard}")
    print(f"  Current policy: {model}")
    print(f"  Run name:       {name}")
    print(f"{'=' * 72}")

    run_dir = sample_step(shard, model, name, cfg)
    train_jsonl = extract_step(run_dir)
    ckpt_dir = train_step(train_jsonl, model, run_dir, num_gpus, train_extra)
    merged = export_step(ckpt_dir, run_dir, num_gpus)
    prune_sharded_ckpt(ckpt_dir)

    dur = (time.time() - t0) / 60
    print(f"\n  Iteration {iter_n} done in {dur:.0f} min → {merged}\n")
    return str(merged)


def load_state(state_file: Path, initial_model: str) -> dict:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"completed": 0, "current_model": initial_model, "history": []}


def save_state(state_file: Path, state: dict) -> None:
    state_file.write_text(json.dumps(state, indent=2))


# ── tmux self-launch ─────────────────────────────────────

def relaunch_in_tmux(argv: list[str]) -> None:
    subprocess.run(["tmux", "start-server"], capture_output=True)
    time.sleep(1)
    # Don't kill existing iterate session — user may be resuming; instead
    # attach-to-existing when the resume flag is implicit.
    if subprocess.run(["tmux", "has-session", "-t", ITERATE_TMUX],
                      capture_output=True).returncode == 0:
        print(f"Tmux session '{ITERATE_TMUX}' already running.")
        print(f"  Attach: tmux attach -t {ITERATE_TMUX}")
        print(f"  Or kill: tmux kill-session -t {ITERATE_TMUX}")
        sys.exit(1)

    cmd = f"{shlex.quote(sys.executable)} {shlex.join(argv + ['--in-tmux'])}"
    r = subprocess.run(
        ["tmux", "new-session", "-d", "-s", ITERATE_TMUX, cmd],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FATAL: tmux launch failed: {r.stderr}")
        sys.exit(1)
    print(f"Iterative training launched in tmux '{ITERATE_TMUX}'")
    print(f"  Attach:  tmux attach -t {ITERATE_TMUX}")
    print(f"  Follow:  tail -f data/<name>_*/iterate.log")


# ── Main ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterative sample → train → export loop over dataset shards.")
    p.add_argument("--shards", required=True,
                   help="Comma-separated dataset JSON paths")
    p.add_argument("--name", required=True,
                   help="Run name (creates data/<name>_<ts>/)")
    p.add_argument("--initial-model", default=BASE_MODEL,
                   help="Starting policy. Resume auto-detects from state.json.")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--sessions-per-gpu", type=int, default=2)
    p.add_argument("--actors", type=int, default=3)
    p.add_argument("--judges", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--train-extra", default="",
                   help="Extra flags appended to rl.train, e.g. "
                        "\"--num-epochs 1 --beta-kl 0.01\"")
    p.add_argument("--in-tmux", action="store_true",
                   help="Internal: skip tmux relaunch")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.in_tmux:
        relaunch_in_tmux(sys.argv)
        return

    shards = [s.strip() for s in args.shards.split(",") if s.strip()]
    for s in shards:
        if not Path(s).exists():
            print(f"FATAL: shard not found: {s}")
            sys.exit(1)

    # Find-or-create the top-level run dir. We pick an existing one if
    # the name prefix already has a state.json so resume works.
    existing = sorted((PROJECT_DIR / "data").glob(f"{args.name}_*"),
                      key=lambda p: p.stat().st_mtime)
    existing = [p for p in existing if (p / "state.json").exists()]
    if existing:
        out_dir = existing[-1]
        print(f"Resuming existing run: {out_dir}")
    else:
        out_dir = PROJECT_DIR / "data" / (
            f"{args.name}_{datetime.now().strftime('%m%d_%H%M%S')}")
        out_dir.mkdir(parents=True, exist_ok=True)

    state_file = out_dir / "state.json"
    log_path = out_dir / "iterate.log"
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    cfg = {
        "gpus": args.gpus,
        "sessions_per_gpu": args.sessions_per_gpu,
        "actors": args.actors,
        "judges": args.judges,
        "max_turns": args.max_turns,
    }
    train_extra = shlex.split(args.train_extra) if args.train_extra else []

    state = load_state(state_file, args.initial_model)

    print(f"\n{'=' * 72}")
    print(f"Iterative RL — {args.name}")
    print(f"{'=' * 72}")
    print(f"  Output:         {out_dir}")
    print(f"  Shards:         {len(shards)}")
    print(f"  Starting policy:{state['current_model']}")
    print(f"  Completed:      {state['completed']}/{len(shards)}")
    print(f"  GPUs:           {cfg['gpus']}")
    print(f"  Started:        {datetime.now().isoformat()}")
    print(f"{'=' * 72}\n")

    t_start = time.time()
    for i, shard in enumerate(shards, 1):
        if i <= state["completed"]:
            print(f"Skipping iter {i} (already completed)")
            continue
        merged = run_one_iteration(
            i, len(shards), shard, state["current_model"],
            args.name, cfg, train_extra)
        state = {
            "completed": i,
            "current_model": merged,
            "history": state["history"] + [{
                "iter": i, "shard": shard,
                "merged": merged,
                "timestamp": datetime.now().isoformat(),
            }],
        }
        save_state(state_file, state)

    total_min = (time.time() - t_start) / 60
    print(f"\n{'=' * 72}")
    print(f"All {len(shards)} iterations complete in {total_min:.0f} min.")
    print(f"Final policy: {state['current_model']}")
    print(f"History:      {state_file}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
