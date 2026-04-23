#!/usr/bin/env python3
"""One-click iterative PPO-clip training (ReST-style, GRPO-compatible).

Data design (no more shard files — all cells share a deterministic schedule):

    1.  `--pool` points at a full question bank (e.g. the merged
        sciencepedia_train_*.json, ~4k questions).
    2.  At launch we deterministically draw `total_questions` from the
        pool using `--seed`; the draw is split into rounds of
        `sampling_batch_size` each.  Every 2×2 cell with the same seed
        sees the exact same batches at the exact same positions.
    3.  One `iter` = one rollout round.  `total_epochs` cycles the
        drawn schedule from the top, so the same batches are seen
        again with a stronger policy.

    total_iters  = total_epochs × rounds_per_epoch
    rounds_per_epoch = total_questions / sampling_batch_size

Inside each iter:

    1. scripts/run.py  →  vLLM + Parliament + harness  →  parliament.db
    2. rl.extract      →  train.jsonl (per-actor trajectory)
    3. metrics_step    →  reward/advantage stats printed to stdout
    4. rl.train        →  ckpt/step_K  (PPO clip + KL-anchor; DDP + LoRA)
    5. rl.export       →  merged/      (LoRA merged, vLLM-loadable)
    6. eval.gpqa       →  optional, --no-eval skips
    7. merged/ becomes the next iter's actor policy.  KL anchor (=
       `--initial-model`) is fixed across every iter so drift never
       compounds.

Disk hygiene — one merged per total_epoch: after each iter exports its
~19 GB merged/, the *previous* iter's merged/ is pruned unless it sits
on an outer-epoch boundary (`iter % rounds_per_epoch == 0`).  At any
time the tree holds: every completed epoch's final merged + the
in-flight iter's merged (= ~2 × 19 GB steady state).

Backups — `backups/<run>_<ts>/ep{E}.round{R}/`: each iter's small
artifacts (metrics.json, train.jsonl, training-step metrics,
train_config.json) are copied here the moment the iter finishes, plus
a rolling `state.json` snapshot at the top level.  Large files
(parliament.db, merged/, llm_logs) are NEVER copied — those live in
data/ until pruned.

Resume: re-invoking the same command picks up where state.json left
off.  Every sub-step inside an iter is idempotent (skip if its output
already exists) so partial crashes don't redo expensive work.

Usage:
    python scripts/iterate.py \\
        --name main_A \\
        --pool datasets/sciencepedia_train_part1.json,\\
datasets/sciencepedia_train_part2.json,\\
datasets/sciencepedia_train_part3.json,\\
datasets/sciencepedia_train_part4.json \\
        --total-questions 1000 \\
        --sampling-batch-size 200 \\
        --total-epochs 2 \\
        --seed 42 \\
        --train-extra "--ppo-epochs 2 --clip-ratio-high 0.25 --beta-kl 0.005" \\
        --gpus 0,1,2,3,4,5,6,7

Stop: tmux kill-session -t parliament-iterate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from _common import Tee, env_prefix  # noqa: E402

BASE_MODEL = os.environ.get("PRL_MODEL_PATH", "Qwen/Qwen3.5-9B")
ACCELERATE = os.environ.get(
    "PRL_ACCELERATE", shutil.which("accelerate") or "accelerate")
PYTHON_ENV = os.environ.get("PRL_PYTHON", sys.executable)
ITERATE_TMUX = "parliament-iterate"

# Files per iter that get copied into backups/ for offline analysis.
# Large files (parliament.db, merged/, llm_logs) are intentionally
# excluded — the backup dir is meant to stay small and archival.
BACKUP_FILES = ("metrics.json", "train.jsonl", "experiment.json")


# ── Shell helpers ────────────────────────────────────────

def run(cmd: list[str], cwd: Path = PROJECT_DIR) -> None:
    """Run subprocess, inherit stdout/stderr, raise on non-zero exit."""
    print(f"\n$ {shlex.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _find_run_dir(name_prefix: str) -> Path | None:
    """Most-recent `data/<name_prefix>_<ts>/`, or None."""
    candidates = sorted((PROJECT_DIR / "data").glob(f"{name_prefix}_*"),
                        key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _find_last_ckpt(ckpt_dir: Path) -> Path | None:
    """Highest-numbered `step_*` under ckpt_dir, or None."""
    steps = [int(p.name.split("_", 1)[1])
             for p in ckpt_dir.glob("step_*")
             if p.name.split("_", 1)[1].isdigit()]
    return (ckpt_dir / f"step_{max(steps)}") if steps else None


def _merged_complete(merged: Path) -> bool:
    """True if `merged/` is a valid vLLM-loadable HF folder."""
    return (merged.is_dir()
            and (merged / "config.json").exists()
            and any(merged.glob("*.safetensors"))
            and (merged / "tokenizer_config.json").exists())


# ── Deterministic question schedule ──────────────────────

def build_schedule(pool_paths: list[str], total_questions: int,
                   sampling_batch_size: int, seed: int
                   ) -> list[list[dict]]:
    """Load + concatenate the pool(s), draw `total_questions`
    (seeded), split into batches of `sampling_batch_size` each.

    Multiple pool paths are concatenated in the given order before
    sampling; the seed alone determines which questions end up where,
    so the list order is irrelevant for reproducibility.

    Same seed ⇒ same batches ⇒ fair comparison across the 2×2 cells.
    """
    pool: list[dict] = []
    for p in pool_paths:
        items = json.loads(Path(p).read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError(f"{p}: must be a JSON list, "
                             f"got {type(items).__name__}")
        pool.extend(items)
    if total_questions > len(pool):
        raise ValueError(f"total_questions ({total_questions}) > pool size "
                         f"({len(pool)})")
    rng = random.Random(seed)
    drawn = rng.sample(pool, total_questions)
    return [drawn[i:i + sampling_batch_size]
            for i in range(0, total_questions, sampling_batch_size)]


# ── Sub-step runners ─────────────────────────────────────

def sample_step(shard: Path, model: str, name: str, cfg: dict) -> Path:
    """scripts/run.py: cleanup → vLLM → Parliament → harness → DB."""
    run([
        PYTHON_ENV, "scripts/run.py", "--in-tmux",
        "--gpus", cfg["gpus"],
        "--sessions-per-gpu", str(cfg["sessions_per_gpu"]),
        "--actors", str(cfg["actors"]),
        "--judges", str(cfg["judges"]),
        "--dataset", str(shard),
        "--name", name,
        "--model", model,
        "--max-turns", str(cfg["max_turns"]),
    ])
    run_dir = _find_run_dir(name)
    if run_dir is None:
        raise RuntimeError(f"scripts/run.py finished but data/{name}_* "
                           f"was not created")
    return run_dir


def extract_step(run_dir: Path) -> Path:
    """parliament.db → per-actor trajectory JSONL."""
    train_jsonl = run_dir / "train.jsonl"
    run([PYTHON_ENV, "-m", "rl.extract",
         "--db", str(run_dir / "parliament.db"),
         "--output", str(train_jsonl)])
    return train_jsonl


def train_step(train_jsonl: Path, model: str, run_dir: Path,
               num_gpus: int, extra: list[str]) -> Path:
    """accelerate launch rl.train (DDP + LoRA + PPO clip + KL anchor)."""
    ckpt_dir = run_dir / "ckpt"
    run([
        ACCELERATE, "launch",
        "--config_file", "rl/accelerate_ddp.yaml",
        "--num_processes", str(num_gpus),
        "-m", "rl.train",
        "--data", str(train_jsonl),
        "--output", str(ckpt_dir),
        "--model", model,
        "--ref-model", model,
        *extra,
    ])
    return ckpt_dir


def export_step(ckpt_dir: Path, run_dir: Path, num_gpus: int) -> Path:
    """LoRA merge (single-process) or FSDP gather (legacy accelerate)."""
    merged = run_dir / "merged"
    last = _find_last_ckpt(ckpt_dir)
    if last is None:
        raise RuntimeError(f"No checkpoints in {ckpt_dir}")
    if (last / "adapter").exists():
        run([PYTHON_ENV, "-m", "rl.export",
             "--ckpt", str(last), "--output", str(merged)])
    else:
        run([ACCELERATE, "launch",
             "--config_file", "rl/accelerate_ddp.yaml",
             "--num_processes", str(num_gpus),
             "-m", "rl.export",
             "--ckpt", str(last), "--output", str(merged)])
    return merged


def eval_step(model_path: str, run_dir: Path, eval_gpu: int,
              max_model_len: int = 16384) -> float | None:
    """GPQA Diamond on one GPU; returns accuracy or None on failure."""
    out_path = run_dir / "gpqa_diamond.json"
    cmd = [PYTHON_ENV, "-m", "eval.gpqa",
           "--model", model_path, "--output", str(out_path),
           "--max-model-len", str(max_model_len)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    print(f"\n$ CUDA_VISIBLE_DEVICES={eval_gpu} {shlex.join(cmd)}\n", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)
    if r.returncode != 0:
        print(f"  WARN: eval failed (rc={r.returncode}); continuing loop")
        return None
    try:
        return float(json.loads(out_path.read_text())["accuracy"])
    except Exception as e:
        print(f"  WARN: could not parse {out_path}: {e}")
        return None


# ── Disk hygiene ─────────────────────────────────────────

def prune_sharded_ckpt(ckpt_dir: Path) -> None:
    """Delete step_* directories once the merged export succeeds."""
    for d in ckpt_dir.glob("step_*"):
        shutil.rmtree(d, ignore_errors=True)
    print(f"  Pruned sharded checkpoints in {ckpt_dir}")


def prune_old_merged(name_prefix: str, current_iter: int,
                     rounds_per_epoch: int) -> None:
    """Drop the previous iter's merged/ unless it's an epoch boundary.

    After iter `k` finishes exporting, we delete iter `k-1`'s merged/
    unless `k-1` was the last round of its total_epoch (i.e.
    `(k-1) % rounds_per_epoch == 0`), in which case we keep it as
    an archival snapshot of that epoch's final policy.
    """
    if current_iter <= 1:
        return
    prev = current_iter - 1
    if prev % rounds_per_epoch == 0:
        return  # epoch boundary — preserve
    prev_run = _find_run_dir(f"{name_prefix}_iter{prev:02d}")
    if prev_run is None:
        return
    merged = prev_run / "merged"
    if merged.is_dir():
        shutil.rmtree(merged, ignore_errors=True)
        print(f"  Pruned old merged: {prev_run.name}/merged "
              f"(intermediate iter)")


# ── Metrics & backups ────────────────────────────────────

def metrics_step(train_jsonl: Path, run_dir: Path) -> dict:
    """Summarise reward / advantage distribution across the iter.

    Writes `metrics.json` (machine-readable) and prints a one-liner so
    iter-over-iter drift can be eyeballed via `grep "DATA METRICS"`.
    """
    rewards: list[float] = []
    advs: list[float] = []
    n_turns = 0
    for line in open(train_jsonl):
        s = json.loads(line)
        rewards.extend(float(r) for r in s.get("turn_rewards", []))
        advs.extend(float(a) for a in s.get("turn_advantages", []))
        n_turns += len(s.get("turn_rewards", []))
    if not rewards:
        return {}

    def q(xs: list[float], p: float) -> float:
        return sorted(xs)[int(p * (len(xs) - 1))]

    summary = {
        "n_samples": n_turns,
        "reward_mean": sum(rewards) / len(rewards),
        "reward_pos_pct": 100 * sum(1 for r in rewards if r > 0) / len(rewards),
        "advantage_mean": sum(advs) / len(advs),
        "advantage_p10": q(advs, 0.10),
        "advantage_p90": q(advs, 0.90),
        "advantage_abs_mean": sum(abs(a) for a in advs) / len(advs),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"  DATA METRICS  n={summary['n_samples']}  "
          f"r̄={summary['reward_mean']:+.3f}  "
          f"r>0={summary['reward_pos_pct']:.1f}%  "
          f"|A|̄={summary['advantage_abs_mean']:.3f}  "
          f"A∈[{summary['advantage_p10']:+.2f}, {summary['advantage_p90']:+.2f}]")
    return summary


def backup_iter(backup_dir: Path, run_dir: Path,
                total_epoch: int, round_n: int) -> None:
    """Copy this iter's small artifacts into `backups/<run>/ep{E}.round{R}/`.

    Uses `BACKUP_FILES` for iter-level files and additionally pulls the
    training-side `ckpt/metrics.jsonl` and `ckpt/config.json` for full
    reproducibility. Large files (parliament.db, merged/, llm_logs)
    are intentionally skipped.
    """
    dest = backup_dir / f"ep{total_epoch}.round{round_n}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in BACKUP_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copy(src, dest / name)
    ckpt_metrics = run_dir / "ckpt" / "metrics.jsonl"
    if ckpt_metrics.exists():
        shutil.copy(ckpt_metrics, dest / "train_metrics.jsonl")
    ckpt_cfg = run_dir / "ckpt" / "config.json"
    if ckpt_cfg.exists():
        shutil.copy(ckpt_cfg, dest / "train_config.json")


# ── Iter runner ──────────────────────────────────────────

def run_one_iteration(iter_n: int, total: int, shard_path: Path,
                      actor_model: str, train_anchor_model: str,
                      name_prefix: str, cfg: dict, train_extra: list[str],
                      do_eval: bool, rounds_per_epoch: int
                      ) -> tuple[str, dict]:
    """One iter: rollout → extract → train → export → optional eval.

    Every sub-step is idempotent: if its output artifact already
    exists from a prior crashed run we skip the work and reuse it.

    `actor_model` is served by vLLM for Parliament.
    `train_anchor_model` is the fixed base used as both `--model` and
    `--ref-model`; the KL anchor therefore always points at the original
    checkpoint regardless of how far the policy has drifted.
    """
    t0 = time.time()
    name = f"{name_prefix}_iter{iter_n:02d}"
    num_gpus = len(cfg["gpus"].split(","))
    eval_gpu = int(cfg["gpus"].split(",")[0])
    print(f"\n{'=' * 72}")
    print(f"Iter {iter_n}/{total}  shard={shard_path.name}")
    print(f"  Actor:      {actor_model}")
    print(f"  KL anchor:  {train_anchor_model}")
    print(f"  Output:     data/{name}_*/")
    print(f"{'=' * 72}")

    # 1. Rollout (skip if parliament.db + experiment.json already there)
    existing = _find_run_dir(name)
    if existing and (existing / "parliament.db").exists() \
            and (existing / "experiment.json").exists():
        print(f"  [skip] rollout — reusing {existing.name}")
        run_dir = existing
    else:
        run_dir = sample_step(shard_path, actor_model, name, cfg)

    # 2. Extract (skip if train.jsonl non-empty)
    train_jsonl = run_dir / "train.jsonl"
    if train_jsonl.exists() and train_jsonl.stat().st_size > 100:
        print(f"  [skip] extract — {train_jsonl.name} "
              f"({train_jsonl.stat().st_size // 1024} KB)")
    else:
        train_jsonl = extract_step(run_dir)

    metrics = metrics_step(train_jsonl, run_dir)

    # 3. Train (resume from latest step_* if present)
    ckpt_dir = run_dir / "ckpt"
    extra = list(train_extra)
    last = _find_last_ckpt(ckpt_dir)
    if last is not None:
        print(f"  [resume] train — from {last.name}")
        extra += ["--resume", str(last)]
    train_step(train_jsonl, train_anchor_model, run_dir, num_gpus, extra)

    # 4. Export (skip if merged/ already valid)
    merged = run_dir / "merged"
    if _merged_complete(merged):
        print(f"  [skip] export — merged/ already complete")
    else:
        merged = export_step(ckpt_dir, run_dir, num_gpus)
    prune_sharded_ckpt(ckpt_dir)
    prune_old_merged(name_prefix, iter_n, rounds_per_epoch)

    # 5. Eval (optional)
    acc = eval_step(str(merged), run_dir, eval_gpu) if do_eval else None

    dur = (time.time() - t0) / 60
    print(f"\n  Iter {iter_n} done in {dur:.0f} min → {merged}")
    if acc is not None:
        print(f"  GPQA Diamond: {acc:.4f}\n")
    return str(merged), {"metrics": metrics, "gpqa_acc": acc, "minutes": dur}


# ── State file ──────────────────────────────────────────

def load_state(path: Path, initial_model: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "completed": 0,
        "current_model": initial_model,   # actor for next iter
        "base_model": initial_model,      # fixed KL anchor
        "base_gpqa": None,                # baseline accuracy, filled once
        "history": [],
    }


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2))


# ── tmux self-launch ─────────────────────────────────────

def relaunch_in_tmux(argv: list[str]) -> None:
    subprocess.run(["tmux", "start-server"], capture_output=True)
    time.sleep(1)
    if subprocess.run(["tmux", "has-session", "-t", ITERATE_TMUX],
                      capture_output=True).returncode == 0:
        print(f"Tmux session '{ITERATE_TMUX}' already running.")
        print(f"  Attach: tmux attach -t {ITERATE_TMUX}")
        print(f"  Kill:   tmux kill-session -t {ITERATE_TMUX}")
        sys.exit(1)
    cmd = (f"{env_prefix()}{shlex.quote(sys.executable)} "
           f"{shlex.join(argv + ['--in-tmux'])}")
    r = subprocess.run(
        ["tmux", "new-session", "-d", "-s", ITERATE_TMUX, cmd],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FATAL: tmux launch failed: {r.stderr}")
        sys.exit(1)
    print(f"Iterative training launched in tmux '{ITERATE_TMUX}'")
    print(f"  Attach: tmux attach -t {ITERATE_TMUX}")
    print(f"  Follow: tail -f data/<name>_*/iterate.log")


# ── Main ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterative PPO-clip training over a deterministic "
                    "sample from a shared question pool.")
    p.add_argument("--name", required=True, help="Run name")
    p.add_argument("--pool", required=True,
                   help="Question pool JSON path, or comma-separated list "
                        "of JSONs that will be concatenated (e.g. the 4 "
                        "sciencepedia_train_part*.json files)")
    p.add_argument("--total-questions", type=int, required=True,
                   help="Questions drawn from --pool (single draw at launch)")
    p.add_argument("--sampling-batch-size", type=int, required=True,
                   help="Questions per rollout round (= one iter)")
    p.add_argument("--total-epochs", type=int, default=1,
                   help="Passes over the drawn schedule (verl: total_epochs)")
    p.add_argument("--seed", type=int, default=42,
                   help="Pool sampling seed — same seed ⇒ same schedule "
                        "across all 4 cells for fair ablation.")
    p.add_argument("--initial-model", default=BASE_MODEL,
                   help="Starting policy; also the fixed KL-anchor target.")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--sessions-per-gpu", type=int, default=2)
    p.add_argument("--actors", type=int, default=3)
    p.add_argument("--judges", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--train-extra", default="",
                   help="Extra flags for rl.train, e.g. "
                        "\"--ppo-epochs 2 --clip-ratio-high 0.25 "
                        "--beta-kl 0.005\"")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip per-iter GPQA eval (saves ~10 min/iter)")
    p.add_argument("--in-tmux", action="store_true",
                   help="Internal: skip tmux relaunch")
    return p.parse_args()


def _resolve_out_dir(name: str) -> Path:
    """Find-or-create the top-level run dir, resuming the most recent
    non-iter top-level dir whose prefix matches."""
    existing = sorted(
        (p for p in (PROJECT_DIR / "data").glob(f"{name}_*")
         if "_iter" not in p.name and p.is_dir()),
        key=lambda p: p.stat().st_mtime)
    if existing:
        out = existing[-1]
        print(f"Resuming existing run: {out}")
        return out
    out = PROJECT_DIR / "data" / f"{name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()

    if not args.in_tmux:
        relaunch_in_tmux(sys.argv)
        return

    # Validate inputs
    pool_paths = [p.strip() for p in args.pool.split(",") if p.strip()]
    if not pool_paths:
        print("FATAL: --pool is empty")
        sys.exit(1)
    for p in pool_paths:
        if not Path(p).exists():
            print(f"FATAL: pool not found: {p}")
            sys.exit(1)
    if args.total_questions % args.sampling_batch_size != 0:
        print(f"FATAL: --total-questions ({args.total_questions}) must be "
              f"divisible by --sampling-batch-size ({args.sampling_batch_size})")
        sys.exit(1)
    if args.total_epochs < 1:
        print("FATAL: --total-epochs must be >= 1")
        sys.exit(1)

    rounds_per_epoch = args.total_questions // args.sampling_batch_size
    total_iters = args.total_epochs * rounds_per_epoch

    # Top-level dirs and logging
    out_dir = _resolve_out_dir(args.name)
    state_file = out_dir / "state.json"
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    backup_dir = PROJECT_DIR / "backups" / out_dir.name
    backup_dir.mkdir(parents=True, exist_ok=True)

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

    # Deterministic schedule — identical across all cells with same seed.
    schedule = build_schedule(pool_paths, args.total_questions,
                              args.sampling_batch_size, args.seed)
    assert len(schedule) == rounds_per_epoch

    manifest = {
        "name": args.name,
        "pool": pool_paths,
        "total_questions": args.total_questions,
        "sampling_batch_size": args.sampling_batch_size,
        "total_epochs": args.total_epochs,
        "rounds_per_epoch": rounds_per_epoch,
        "total_iters": total_iters,
        "seed": args.seed,
        "initial_model": args.initial_model,
        "train_extra": args.train_extra,
        "started_at": datetime.now().isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (backup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 72}")
    print(f"Iterative PPO (clip + KL anchor) — {args.name}")
    print(f"{'=' * 72}")
    print(f"  Output:            {out_dir}")
    print(f"  Backups:           {backup_dir}")
    print(f"  Pool:              {', '.join(pool_paths)}")
    print(f"  Draw:              {args.total_questions} q × "
          f"seed={args.seed}")
    print(f"  Sampling batch:    {args.sampling_batch_size}  "
          f"→ {rounds_per_epoch} rounds/epoch")
    print(f"  Total epochs:      {args.total_epochs}")
    print(f"  Total iters:       {total_iters}")
    print(f"  Actor / KL anchor: {state['current_model']} / {state['base_model']}")
    print(f"  Completed:         {state['completed']}/{total_iters}")
    print(f"  Train extra:       {args.train_extra}")
    print(f"  GPUs:              {cfg['gpus']}")
    print(f"  Per-iter eval:     {'off' if args.no_eval else 'GPQA Diamond'}")
    print(f"  Started:           {datetime.now().isoformat()}")
    print(f"{'=' * 72}\n")

    # Baseline GPQA on the original base — runs once before iter 1.
    if state["completed"] == 0 and state["base_gpqa"] is None and not args.no_eval:
        eval_gpu = int(cfg["gpus"].split(",")[0])
        baseline_dir = out_dir / "baseline_eval"
        baseline_dir.mkdir(exist_ok=True)
        print(f"--- Baseline GPQA on {state['base_model']} ---")
        state["base_gpqa"] = eval_step(
            state["base_model"], baseline_dir, eval_gpu)
        save_state(state_file, state)

    t_start = time.time()
    for total_epoch in range(1, args.total_epochs + 1):
        for round_idx in range(rounds_per_epoch):
            iter_n = (total_epoch - 1) * rounds_per_epoch + round_idx + 1
            if iter_n <= state["completed"]:
                print(f"Skipping iter {iter_n} (already completed)")
                continue

            # Materialise this round's shard file on disk (idempotent).
            shard_path = shard_dir / f"ep{total_epoch}.round{round_idx + 1}.json"
            if not shard_path.exists():
                shard_path.write_text(
                    json.dumps(schedule[round_idx], ensure_ascii=False))

            merged, summary = run_one_iteration(
                iter_n, total_iters, shard_path,
                actor_model=state["current_model"],
                train_anchor_model=state["base_model"],
                name_prefix=args.name, cfg=cfg, train_extra=train_extra,
                do_eval=not args.no_eval,
                rounds_per_epoch=rounds_per_epoch)

            run_dir = _find_run_dir(f"{args.name}_iter{iter_n:02d}")
            if run_dir is not None:
                backup_iter(backup_dir, run_dir, total_epoch, round_idx + 1)

            state = {
                **state,
                "completed": iter_n,
                "current_model": merged,
                "history": state["history"] + [{
                    "iter": iter_n,
                    "total_epoch": total_epoch,
                    "round": round_idx + 1,
                    "shard": str(shard_path),
                    "merged": merged,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": summary["metrics"],
                    "gpqa_acc": summary["gpqa_acc"],
                    "minutes": round(summary["minutes"], 1),
                }],
            }
            save_state(state_file, state)
            shutil.copy(state_file, backup_dir / "state.json")

    total_min = (time.time() - t_start) / 60
    print(f"\n{'=' * 72}")
    print(f"All {total_iters} iters done in {total_min:.0f} min "
          f"({args.total_epochs} epoch × {rounds_per_epoch} rounds).")
    print(f"Final policy: {state['current_model']}")
    print(f"Backups:      {backup_dir}")
    if state.get("base_gpqa") is not None:
        print(f"\n  Base GPQA: {state['base_gpqa']:.4f}")
    for h in state["history"]:
        if h.get("gpqa_acc") is not None:
            tag = f"ep{h['total_epoch']}.round{h['round']}"
            print(f"  {tag}: GPQA = {h['gpqa_acc']:.4f}  "
                  f"(|A|̄ = {h['metrics'].get('advantage_abs_mean', 0):.3f})")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
