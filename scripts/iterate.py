#!/usr/bin/env python3
"""One-click iterative RL training (ReST-style).

Naming (verl-aligned, see docs/00_naming.md):
    iter         One sample → train → export cycle on a single shard.
    total_epoch  One full pass through the shard list.
                 Repeating shards across epochs gives the policy a
                 second chance at every shard with a stronger model.
    ppo_epoch    Passes through the same train.jsonl in `rl.train`
                 (forwarded as --train-extra "--ppo-epochs N").

For each iter, in order:

    1. scripts/run.py  →  vLLM + Parliament + harness  →  parliament.db
    2. rl.extract      →  train.jsonl (per-actor trajectory)
    3. metrics_step    →  reward/advantage stats
    4. rl.train        →  ckpt/step_K  (DDP + LoRA + RWR + KL anchor)
    5. rl.export       →  merged/      (LoRA → base, vLLM-loadable)
    6. eval.gpqa       →  gpqa_diamond.json  (optional, --no-eval skips)
    7. merged/ becomes the next iter's actor policy

The base model is the *fixed* KL anchor across all iters (LoRA's
`disable_adapter()` recovers it for free) so drift does not compound.
Each iter starts a fresh LoRA on the merged base.

Resume: re-invoking the same command picks up where things left off.
Each sub-step is idempotent — it skips when its output already exists.

    sample : skip if parliament.db + experiment.json present
    extract: skip if train.jsonl non-empty
    train  : --resume from latest step_*
    export : skip if merged/ has config + weights + tokenizer

Usage:
    python scripts/iterate.py \\
        --name run1 \\
        --shards datasets/part1.json,datasets/part2.json,... \\
        --total-epochs 2 \\
        --train-extra "--ppo-epochs 2" \\
        --gpus 0,1,2,3,4,5,6,7

Stop: tmux kill-session -t parliament-iterate
"""

from __future__ import annotations

import argparse
import json
import os
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
ACCELERATE = os.environ.get("PRL_ACCELERATE", shutil.which("accelerate") or "accelerate")
PYTHON_ENV = os.environ.get("PRL_PYTHON", sys.executable)
ITERATE_TMUX = "parliament-iterate"


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
        "--config_file", "rl/accelerate_ddp.yaml",
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
    """LoRA merge is single-process; full-FT (legacy) needs accelerate."""
    merged_dir = run_dir / "merged"
    last = find_last_checkpoint(ckpt_dir)
    if (last / "adapter").exists():
        run([PYTHON_ENV, "-m", "rl.export",
             "--ckpt", str(last), "--output", str(merged_dir)])
    else:
        run([
            ACCELERATE, "launch",
            "--config_file", "rl/accelerate_ddp.yaml",
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


def metrics_step(train_jsonl: Path, run_dir: Path) -> dict:
    """Summarise reward/advantage distribution; one number per iter to
    eyeball generation-quality drift across iterations.

    Writes `metrics.json` (machine-readable) and prints a one-line table
    so you can `grep "DATA METRICS" iterate.log` to track quality over
    time.
    """
    rewards: list[float] = []
    advs: list[float] = []
    n_turns_total = 0
    for line in open(train_jsonl):
        s = json.loads(line)
        # per-actor trajectory: each sample has lists of per-turn values
        for r in s.get("turn_rewards", []):
            rewards.append(float(r))
        for a in s.get("turn_advantages", []):
            advs.append(float(a))
        n_turns_total += len(s.get("turn_rewards", []))
    if not rewards:
        return {}

    def q(xs: list[float], p: float) -> float:
        return sorted(xs)[int(p * (len(xs) - 1))]

    summary = {
        "n_samples": n_turns_total,
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


def eval_step(model_path: str, run_dir: Path,
              eval_gpu: int, max_model_len: int = 16384) -> float | None:
    """Run GPQA Diamond on a single GPU and return accuracy.

    GPUs from the rollout phase are already free at this point
    (`scripts/run.py` calls `stop_vllm()` on success). We pin to the
    first listed GPU so the other 7 stay idle for the next iteration's
    parliament boot.
    """
    out_path = run_dir / "gpqa_diamond.json"
    cmd = [PYTHON_ENV, "-m", "eval.gpqa",
           "--model", model_path, "--output", str(out_path),
           "--max-model-len", str(max_model_len)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    print(f"\n$ CUDA_VISIBLE_DEVICES={eval_gpu} {shlex.join(cmd)}\n", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)
    if r.returncode != 0:
        print(f"  WARN: eval failed (rc={r.returncode}); continuing iteration loop")
        return None
    try:
        acc = float(json.loads(out_path.read_text())["accuracy"])
        print(f"  GPQA Diamond accuracy: {acc:.4f}")
        return acc
    except Exception as e:
        print(f"  WARN: could not parse {out_path}: {e}")
        return None


# ── Iteration loop ───────────────────────────────────────

def _find_existing_run_dir(name_prefix: str) -> Path | None:
    """Return the most recent data/<name_prefix>_<ts>/ if any."""
    candidates = sorted((PROJECT_DIR / "data").glob(f"{name_prefix}_*"),
                        key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _is_merged_complete(merged_dir: Path) -> bool:
    """A merged HF folder is usable if it has config + at least one
    safetensors shard + tokenizer config."""
    if not merged_dir.is_dir():
        return False
    has_cfg = (merged_dir / "config.json").exists()
    has_weights = any(merged_dir.glob("*.safetensors"))
    has_tok = (merged_dir / "tokenizer_config.json").exists()
    return has_cfg and has_weights and has_tok


def run_one_iteration(iter_n: int, total: int, shard: str,
                      actor_model: str, train_anchor_model: str,
                      name_prefix: str, cfg: dict,
                      train_extra: list[str], do_eval: bool) -> tuple[str, dict]:
    """One full iteration: rollout → extract → train (LoRA) → export → eval.

    Each step is idempotent: if its output already exists from a prior
    crashed run we skip the work and reuse it.  This lets us resume from
    any failure point without re-doing expensive sampling/training.

    `actor_model` is what vLLM serves to Parliament for data collection.
    `train_anchor_model` is the *fixed* base used both as `--model` and
    `--ref-model`; this never changes across iterations so the KL anchor
    stays referenced to the original Qwen3.5-9B checkpoint.
    """
    t0 = time.time()
    name = f"{name_prefix}_iter{iter_n:02d}"
    num_gpus = len(cfg["gpus"].split(","))
    eval_gpu = int(cfg["gpus"].split(",")[0])
    print(f"\n{'=' * 72}")
    print(f"Iteration {iter_n}/{total} — shard={shard}")
    print(f"  Actor (rollout):     {actor_model}")
    print(f"  Train anchor (base): {train_anchor_model}")
    print(f"  Run name:            {name}")
    print(f"{'=' * 72}")

    # Step 1 — Sample (resumable: skip if parliament.db + experiment.json exist).
    existing = _find_existing_run_dir(name)
    if existing and (existing / "parliament.db").exists() \
            and (existing / "experiment.json").exists():
        print(f"  [skip] sample_step — reusing {existing.name}")
        run_dir = existing
    else:
        run_dir = sample_step(shard, actor_model, name, cfg)

    # Step 2 — Extract (resumable: skip if train.jsonl exists and non-empty).
    train_jsonl = run_dir / "train.jsonl"
    if train_jsonl.exists() and train_jsonl.stat().st_size > 100:
        print(f"  [skip] extract_step — {train_jsonl.name} exists "
              f"({train_jsonl.stat().st_size // 1024} KB)")
    else:
        train_jsonl = extract_step(run_dir)

    metrics = metrics_step(train_jsonl, run_dir)

    # Step 3 — Train (resumable: pass --resume to last step_* if any).
    ckpt_dir = run_dir / "ckpt"
    last_step = max(
        (int(p.name.split("_")[1]) for p in ckpt_dir.glob("step_*")
         if p.name.split("_")[1].isdigit()),
        default=0,
    )
    extra = list(train_extra)
    if last_step > 0:
        resume_path = ckpt_dir / f"step_{last_step}"
        print(f"  [resume] train_step — from step_{last_step}")
        extra += ["--resume", str(resume_path)]
    ckpt_dir = train_step(train_jsonl, train_anchor_model, run_dir,
                          num_gpus, extra)

    # Step 4 — Export (resumable: skip if merged/ already valid).
    merged = run_dir / "merged"
    if _is_merged_complete(merged):
        print(f"  [skip] export_step — merged/ already complete")
    else:
        merged = export_step(ckpt_dir, run_dir, num_gpus)
    prune_sharded_ckpt(ckpt_dir)

    acc = eval_step(str(merged), run_dir, eval_gpu) if do_eval else None
    dur = (time.time() - t0) / 60
    print(f"\n  Iteration {iter_n} done in {dur:.0f} min → {merged}")
    if acc is not None:
        print(f"  GPQA Diamond: {acc:.4f}\n")
    return str(merged), {"metrics": metrics, "gpqa_acc": acc, "minutes": dur}


def load_state(state_file: Path, initial_model: str) -> dict:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {
        "completed": 0,
        "current_model": initial_model,         # actor for next iteration
        "base_model": initial_model,            # fixed KL anchor across all iters
        "base_gpqa": None,                      # baseline accuracy, filled iter 1 prep
        "history": [],
    }


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

    cmd = f"{env_prefix()}{shlex.quote(sys.executable)} {shlex.join(argv + ['--in-tmux'])}"
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
    # Outer loop over the full shard list.  verl's `trainer.total_epochs`.
    # `--shards p1,p2,p3,p4 --total-epochs 2` runs 8 iters (each shard
    # twice), each iter starting from the previous merged model so the
    # second pass over `p1` learns from the iter-4 policy.
    p.add_argument("--total-epochs", type=int, default=1,
                   help="How many times to run through the shard list "
                        "(verl: trainer.total_epochs). Default 1.")
    p.add_argument("--train-extra", default="",
                   help="Extra flags appended to rl.train, e.g. "
                        "\"--ppo-epochs 2 --beta-kl 0.01\"")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip per-iteration GPQA eval (saves ~10 min/iter)")
    p.add_argument("--in-tmux", action="store_true",
                   help="Internal: skip tmux relaunch")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.in_tmux:
        relaunch_in_tmux(sys.argv)
        return

    base_shards = [s.strip() for s in args.shards.split(",") if s.strip()]
    for s in base_shards:
        if not Path(s).exists():
            print(f"FATAL: shard not found: {s}")
            sys.exit(1)
    # Outer loop: run the shard list `total_epochs` times.  We just
    # repeat the list — each repetition becomes additional iters whose
    # actor model is the previous iter's merged checkpoint, so the
    # second pass over a shard already learns from the trained policy.
    if args.total_epochs < 1:
        print("FATAL: --total-epochs must be >= 1")
        sys.exit(1)
    shards = base_shards * args.total_epochs

    # Find-or-create the top-level run dir. We pick an existing one if
    # the name prefix already has a state.json so resume works.
    # Resume the most-recent top-level run dir, even if state.json hasn't
    # been written yet (iter 1 may have crashed before the first save).
    # We exclude per-iter dirs (they have "_iter" in the name).
    existing = sorted(
        (p for p in (PROJECT_DIR / "data").glob(f"{args.name}_*")
         if "_iter" not in p.name and p.is_dir()),
        key=lambda p: p.stat().st_mtime)
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
    print(f"Iterative RL (RWR + KL-to-base) — {args.name}")
    print(f"{'=' * 72}")
    print(f"  Output:           {out_dir}")
    print(f"  Shards (base):    {len(base_shards)}")
    print(f"  Total epochs:     {args.total_epochs}  "
          f"(verl: trainer.total_epochs)")
    print(f"  Total iters:      {len(shards)}  (= shards × total_epochs)")
    print(f"  Base / KL anchor: {state['base_model']}")
    print(f"  Next actor:       {state['current_model']}")
    print(f"  Completed:        {state['completed']}/{len(shards)}")
    print(f"  GPUs:             {cfg['gpus']}")
    print(f"  Per-iter eval:    {'off' if args.no_eval else 'GPQA Diamond'}")
    print(f"  Started:          {datetime.now().isoformat()}")
    print(f"{'=' * 72}\n")

    # Baseline GPQA on the original base model — only run once, before
    # the very first iteration. Sets the floor for the iter-vs-acc curve.
    if state["completed"] == 0 and state["base_gpqa"] is None and not args.no_eval:
        eval_gpu = int(cfg["gpus"].split(",")[0])
        baseline_dir = out_dir / "baseline_eval"
        baseline_dir.mkdir(exist_ok=True)
        print(f"\n--- Baseline GPQA on {state['base_model']} ---")
        state["base_gpqa"] = eval_step(state["base_model"], baseline_dir, eval_gpu)
        save_state(state_file, state)

    t_start = time.time()
    for i, shard in enumerate(shards, 1):
        if i <= state["completed"]:
            print(f"Skipping iter {i} (already completed)")
            continue
        # Which outer epoch (1-indexed) does this iter belong to?
        # iter `i` over a base list of length B is in epoch ((i-1)//B + 1).
        total_epoch = (i - 1) // len(base_shards) + 1
        merged, summary = run_one_iteration(
            i, len(shards), shard,
            actor_model=state["current_model"],
            train_anchor_model=state["base_model"],
            name_prefix=args.name, cfg=cfg, train_extra=train_extra,
            do_eval=not args.no_eval)
        state = {
            **state,
            "completed": i,
            "current_model": merged,
            "history": state["history"] + [{
                "iter": i, "total_epoch": total_epoch, "shard": shard,
                "merged": merged,
                "timestamp": datetime.now().isoformat(),
                "metrics": summary["metrics"],
                "gpqa_acc": summary["gpqa_acc"],
                "minutes": round(summary["minutes"], 1),
            }],
        }
        save_state(state_file, state)

    total_min = (time.time() - t_start) / 60
    print(f"\n{'=' * 72}")
    print(f"All {len(shards)} iters ({args.total_epochs} epoch(s) "
          f"× {len(base_shards)} shards) complete in {total_min:.0f} min.")
    print(f"Final policy: {state['current_model']}")
    print(f"History:      {state_file}")
    if state.get("base_gpqa") is not None:
        print(f"\n  Base GPQA: {state['base_gpqa']:.4f}")
    for h in state["history"]:
        if h.get("gpqa_acc") is not None:
            tag = f"ep{h.get('total_epoch', 1)}.iter{h['iter']}"
            print(f"  {tag}: GPQA = {h['gpqa_acc']:.4f}  "
                  f"(data |A|̄ = {h['metrics'].get('advantage_abs_mean', 0):.3f})")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
