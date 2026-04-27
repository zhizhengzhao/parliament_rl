#!/usr/bin/env python3
"""One-click iterative PPO-clip training (ReST-style, GRPO-compatible).

Architecture
------------

    +─── iterate.py main process ─────────────────────────────────────+
    |                                                                  |
    |   for iter in range(N):                                          |
    |                                                                  |
    |     ─ vLLM fleet (DP=N, TP=1 each, 1 process per GPU)            |
    |        ─ started fresh each iter from the *previous iter's       |
    |          merged folder* (or the original base on iter 1)         |
    |        ─ cudagraphs ON, prefix caching ON, gpu-mem-util=0.90     |
    |        ─ NO ``--enable-lora``: vLLM serves the merged model as   |
    |          a plain HF checkpoint, full speed                       |
    |                                                                  |
    |     ─ rollout phase                                              |
    |        a. start Parliament + load shard                          |
    |        b. asyncio.run(harness.run_experiment(model=<merged>…))   |
    |           — agents POST to vLLMs over HTTP                       |
    |        c. stop Parliament                                        |
    |                                                                  |
    |     ─ training phase                                             |
    |        d. ``stop_vllm()``: free ~80 GB / GPU for the trainer     |
    |        e. ``accelerate launch -m rl.train`` (DDP, LoRA)          |
    |           → ckpt/step_K/adapter/ (~110 MB PEFT folder)           |
    |        f. ``rl.export``: merge LoRA into base + patch any        |
    |           visual/mtp weights → run_dir/merged/ (~19 GB)          |
    |        g. (vLLM stays down until the next iter's rollout)        |
    |                                                                  |
    |     ─ next iter: ``ensure_vllm(model=run_dir/merged)`` and       |
    |       repeat from (a)                                            |
    +──────────────────────────────────────────────────────────────────+

Why merge+reload, not LoRA hot-swap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vLLM 0.19.1 supports ``/v1/load_lora_adapter`` for hot-swap, which on
paper would save ~30 s/iter of merge time and the 19 GB merged folder
on disk.  In practice we measured LoRA-mode generation at ~2 tokens/s
on 15-23 K-token contexts (vs 40+ tokens/s on the merged base) and 0%
prefix-cache hit rate (LoRA scope invalidates the shared system-prompt
prefix every request).  At rollout-dominated ~85% of iter wall-time,
the LoRA path costs more than 30 s of merge skip would save.  We keep
the merged-checkpoint pipeline until upstream vLLM closes that gap.

The DP=N + TP=1 inference layout (one independent vLLM per GPU) is the
canonical RLHF rollout configuration in OpenRLHF / verl — for ≤ 13B
models data parallelism beats tensor parallelism on long contexts
because TP's per-layer NCCL all-reduces dominate decode latency.

Disk hygiene — `ckpt/step_K/`: trainer ckpts pruned after merge
succeeds.  Old ``merged/`` folders (~19 GB each) pruned at every iter
except total-epoch boundaries, which are kept as archival snapshots.
Backups under `backups/<run>/` carry only the small metadata
(metrics.json, train.jsonl, train_metrics.jsonl) — large artifacts
stay in ``data/`` until pruned.

Resume: re-invoking the same command picks up where state.json left
off.  Every sub-step inside an iter is idempotent (skip if its output
already exists) so partial crashes don't redo expensive work.

Usage:
    python scripts/iterate.py \\
        --name main_A \\
        --pool datasets/sciencepedia_train.json \\
        --total-questions 400 \\
        --sampling-batch-size 200 \\
        --total-epochs 3 \\
        --seed 42 \\
        --train-extra "--ppo-epochs 2 --clip-ratio-high 0.25 --beta-kl 0.005" \\
        --gpus 0,1,2,3,4,5,6,7

``--pool`` accepts a single JSON or a comma-separated list; all files
are concatenated into one pool, then ``--total-questions`` are drawn
once at iter 0 (``--seed`` controls which) and frozen for the whole
run. To make 4 cells fairly comparable, pass them all the SAME pool +
SAME seed — the only thing that should differ between cells is
``PRL_CONTEXT`` and ``PRL_JUDGE_VOTES_VISIBLE``.

Stop: tmux kill-session -t parliament-iterate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR))

from _common import Tee, env_prefix  # noqa: E402

# Reuse the vLLM lifecycle helpers from scripts/run.py — that file is
# the single source of truth for "how to boot / wait_ready / kill a
# DP=N TP=1 vLLM fleet". Anything iterate.py needs about vLLM goes
# through it.
import scripts.run as run  # noqa: E402

BASE_MODEL = os.environ.get("PRL_MODEL_PATH", "Qwen/Qwen3.5-9B")
ACCELERATE = os.environ.get(
    "PRL_ACCELERATE", shutil.which("accelerate") or "accelerate")
PYTHON_ENV = os.environ.get("PRL_PYTHON", sys.executable)
ITERATE_TMUX = "parliament-iterate"
ADMIN_KEY = run.ADMIN_KEY
PARLIAMENT_PORT = 8080

# Files per iter that get copied into backups/ for offline analysis.
# Large files (parliament.db, ckpt/, llm_logs) are intentionally
# excluded — the backup dir is meant to stay small and archival.
BACKUP_FILES = ("metrics.json", "train.jsonl", "experiment.json")


# ── Shell helpers ────────────────────────────────────────

def shell_run(cmd: list[str], cwd: Path = PROJECT_DIR) -> None:
    """Run subprocess, stream stdout+stderr into our Tee'd sys.stdout
    so they land in iterate.log too, and raise on non-zero exit.

    Plain ``subprocess.run(check=True)`` would inherit fd 0/1/2 from
    the Python interpreter, bypassing our Tee on
    ``sys.stdout/stderr`` that mirrors everything to ``iterate.log``.
    That meant crashes inside ``accelerate launch -m rl.train`` left
    an iterate.log with only the ``$ <cmd>`` line and no traceback at
    all — a debugging black hole.  Piping + line-by-line relay
    restores full capture.
    """
    print(f"\n$ {shlex.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _find_run_dir(name_prefix: str) -> Path | None:
    """Most-recent `data/<name_prefix>_<ts>/`, or None."""
    candidates = sorted((PROJECT_DIR / "data").glob(f"{name_prefix}_*"),
                        key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _find_last_ckpt(ckpt_dir: Path) -> Path | None:
    """Highest-numbered ``step_*`` under ``ckpt_dir``, or None."""
    steps = [int(p.name.split("_", 1)[1])
             for p in ckpt_dir.glob("step_*")
             if p.name.split("_", 1)[1].isdigit()]
    return (ckpt_dir / f"step_{max(steps)}") if steps else None


def _adapter_complete(adapter_dir: Path) -> bool:
    """True iff ``adapter_dir`` is a complete PEFT adapter folder.

    PEFT writes both ``adapter_config.json`` and the weights file
    (``adapter_model.safetensors`` for our LoRA setup).  Used as a
    sanity check that the trainer's checkpoint is intact before we
    feed it to ``rl.export`` for the merge step.
    """
    return (adapter_dir.is_dir()
            and (adapter_dir / "adapter_config.json").exists()
            and (adapter_dir / "adapter_model.safetensors").exists())


def _merged_complete(merged: Path) -> bool:
    """True iff ``merged/`` is a vLLM-loadable HF folder.

    Stricter than "config.json + ANY safetensors + tokenizer": when
    an index file is present (multi-shard model), every shard listed
    in the weight_map must actually exist on disk.  An interrupted
    ``rl.export`` can leave a partial set of shards plus a complete
    index, which would pass a naive check but break vLLM at load time.
    """
    if not (merged.is_dir()
            and (merged / "config.json").exists()
            and (merged / "tokenizer_config.json").exists()
            and any(merged.glob("*.safetensors"))):
        return False
    idx_path = merged / "model.safetensors.index.json"
    if not idx_path.exists():
        return True
    try:
        idx = json.loads(idx_path.read_text())
        declared = set(idx.get("weight_map", {}).values())
    except (OSError, json.JSONDecodeError):
        return False
    for shard in declared:
        if not (merged / shard).exists():
            return False
    return True


def _train_jsonl_complete(path: Path, min_bytes: int = 100) -> bool:
    """True if ``path`` is a non-empty jsonl with structurally valid records.

    Sniffs the first and last line: both must be valid JSON, both must
    carry ``turn_advantages`` and ``messages`` (the two fields
    ``rl/train.py:RLDataset`` actually reads).  Cheap (only two small
    lines parsed regardless of file size).  Catches the case where a
    crashed extract leaves a half-written tail — without this check
    iterate.py would happily reuse it on resume and the trainer would
    crash on the malformed last line.
    """
    try:
        if not path.exists() or path.stat().st_size < min_bytes:
            return False
    except OSError:
        return False
    try:
        with open(path, "rb") as f:
            first_line = f.readline().decode("utf-8")
            if not first_line.strip():
                return False
            f.seek(0, 2)
            file_size = f.tell()
            tail_size = min(file_size, 64 * 1024)
            f.seek(file_size - tail_size, 0)
            tail = f.read().decode("utf-8", errors="replace")
            last_line = next((ln for ln in reversed(tail.splitlines())
                              if ln.strip()), "")
        for ln in (first_line, last_line):
            obj = json.loads(ln)
            if "messages" not in obj or "turn_advantages" not in obj:
                return False
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, StopIteration):
        return False
    return True


# ── Deterministic question schedule ──────────────────────

def build_schedule(pool_paths: list[str], total_questions: int,
                   sampling_batch_size: int, seed: int
                   ) -> list[list[dict]]:
    """Load + concatenate the pool(s), draw `total_questions`
    (seeded), split into batches of `sampling_batch_size` each.

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

def rollout_step(shard_path: Path, model_name: str, run_name: str,
                 cfg: dict, gpu_endpoints: list[str]) -> Path:
    """Start Parliament, load the shard, run harness against current vLLM fleet.

    The vLLM fleet is brought up *before* this function (by the iter
    setup in ``run_one_iteration``), pointing at the merged HF folder
    of the most recent successful train step (or the base model on
    iter 0). This function only manages the Parliament server + the
    dataset shard + the harness; it does not touch vLLM at all.

    ``model_name`` is the HF model id / path that vLLM is currently
    serving — at iter 0 that's the base, at iter k>0 it's
    ``run_dir/iter_(k-1)/merged``.
    """
    run_dir = (PROJECT_DIR / "data"
               / f"{run_name}_{time.strftime('%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Parliament cleanup: the previous iter's parliament.server is
    # already gone (we always stop it at iter end), but a stale port
    # binding can linger in TIME_WAIT for a few seconds — kill_port
    # is idempotent and cheap.
    run.kill_port(PARLIAMENT_PORT)

    parliament_url = f"http://127.0.0.1:{PARLIAMENT_PORT}"
    parliament_proc = run.start_parliament(
        run_name, cfg["actors"], cfg["judges"], PARLIAMENT_PORT,
        log_path=run_dir / "parliament.log", db_dir=str(run_dir))

    try:
        loaded = run.load_dataset(str(shard_path), parliament_url, 0)
        if loaded == 0:
            raise RuntimeError("no questions loaded into Parliament")

        from harness.runner import run_experiment

        rc = asyncio.run(run_experiment(
            parliament_url=parliament_url,
            admin_key=ADMIN_KEY,
            gpu_endpoints=gpu_endpoints,
            sessions_per_gpu=cfg["sessions_per_gpu"],
            num_actors=cfg["actors"],
            num_judges=cfg["judges"],
            model_name=model_name,
            max_rounds=cfg["max_turns"],
            output_path=str(run_dir / "experiment.json"),
        ))
        if rc != 0:
            raise RuntimeError(f"harness.run_experiment returned {rc}")
    finally:
        run.stop_parliament(parliament_proc)

    return run_dir


def extract_step(run_dir: Path) -> Path:
    """parliament.db → per-actor trajectory JSONL (subprocess)."""
    train_jsonl = run_dir / "train.jsonl"
    shell_run([PYTHON_ENV, "-m", "rl.extract",
               "--db", str(run_dir / "parliament.db"),
               "--output", str(train_jsonl)])
    return train_jsonl


def _find_free_port(start: int = 29500, end: int = 29600) -> int:
    """Find a free local TCP port in [start, end) for DDP rendezvous.

    Default ``accelerate launch`` uses ``29500`` — when iter K-1's
    train socket is still in TIME_WAIT or its rendezvous TCPStore has
    not yet released the port, iter K's bind() fails with EADDRINUSE
    and the whole experiment dies.  Picking a fresh port per iter
    sidesteps the race entirely (only the rare case where every port
    in the range is held would still fail, which is observable and
    fixable; TIME_WAIT contention is silent and devastating).
    """
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free DDP port in [{start}, {end}) — clean up "
        f"orphan rl.train processes (pkill -9 -f rl.train) "
        f"and retry"
    )


def _kill_orphan_train_workers() -> None:
    """SIGKILL any leftover ``rl.train`` / ``accelerate launch`` workers.

    DDP failures (NCCL hang, cudaError, EADDRINUSE) often leave the
    elastic-agent main process dead but its 8 children still attached
    to the GPU.  These zombies hold port 29500-29599, hold GPU memory,
    and refuse to die from a normal SIGTERM.  We pkill -9 anything
    matching ``rl.train`` or ``accelerate launch ... rl.train``
    cmdlines as a belt-and-braces cleanup before each train_step.

    Idempotent — does nothing if there are no orphans.
    """
    for pat in ("rl\\.train", "accelerate.*rl\\.train",
                "torch.distributed.elastic"):
        subprocess.run(
            ["pkill", "-9", "-f", pat],
            capture_output=True, check=False,
        )
    # Linux holds DDP ports in TIME_WAIT for ~60-120 s after the
    # listener exits; a brief sleep gives the kernel a chance to
    # actually release them before _find_free_port runs.
    time.sleep(2)


def train_step(train_jsonl: Path, base_model: str, run_dir: Path,
               num_gpus: int, extra: list[str]) -> Path:
    """``accelerate launch rl.train`` (DDP + LoRA + PPO clip + KL anchor).

    Returns the path to the produced ``adapter/`` folder.  The actual
    HF-format ``merged/`` folder we hand to vLLM is produced by
    ``export_step`` running afterwards.

    Port hygiene:
      * Orphan ``rl.train`` workers from a previous iter are killed
        before launch (NCCL hangs leave zombies behind).
      * The DDP rendezvous port is chosen dynamically from
        ``[29500, 29600)`` — default-port reuse triggered an EADDRINUSE
        crash mid-experiment in the 2026-04-26 main run.
    """
    _kill_orphan_train_workers()
    ddp_port = _find_free_port(29500, 29600)
    print(f"  [train] DDP main_process_port = {ddp_port}", flush=True)

    ckpt_dir = run_dir / "ckpt"
    shell_run([
        ACCELERATE, "launch",
        "--config_file", "rl/accelerate_ddp.yaml",
        "--num_processes", str(num_gpus),
        "--main_process_port", str(ddp_port),
        "-m", "rl.train",
        "--data", str(train_jsonl),
        "--output", str(ckpt_dir),
        "--model", base_model,
        "--ref-model", base_model,
        *extra,
    ])
    last = _find_last_ckpt(ckpt_dir)
    if last is None:
        raise RuntimeError(f"No checkpoint found in {ckpt_dir} after train")
    adapter_dir = last / "adapter"
    if not _adapter_complete(adapter_dir):
        raise RuntimeError(
            f"Trainer finished but {adapter_dir} is missing PEFT files "
            f"(adapter_config.json or adapter_model.safetensors)")
    return adapter_dir


def export_step(ckpt_dir: Path, run_dir: Path) -> Path:
    """LoRA merge: ``ckpt/step_K/adapter/`` + base → ``merged/``.

    Single-process — the merge is just ``W += alpha * BA`` for each
    LoRA-adapted layer, plus copying the base's tokenizer / preprocessor
    configs and patching in the visual/mtp weights that
    ``AutoModelForCausalLM`` doesn't load.  ~30 s on an 80 GB A100,
    output is ~19 GB (a vLLM-loadable HF folder).
    """
    merged = run_dir / "merged"
    last = _find_last_ckpt(ckpt_dir)
    if last is None:
        raise RuntimeError(f"No checkpoints in {ckpt_dir}")
    if not (last / "adapter").exists():
        raise RuntimeError(
            f"{last}/adapter missing — merge requires a PEFT-format "
            f"adapter, but only the legacy FSDP layout is present")
    shell_run([PYTHON_ENV, "-m", "rl.export",
               "--ckpt", str(last), "--output", str(merged)])
    return merged


# ── Disk hygiene ─────────────────────────────────────────

def prune_sharded_ckpt(ckpt_dir: Path) -> None:
    """Delete ``step_*/`` directories once the merged export succeeds.

    The trainer's optimizer / scheduler / accelerate state inside
    each ``step_*/`` is ~700 MB and only useful for mid-iter resume.
    Once the iter has produced a valid ``merged/`` folder, the
    sharded checkpoints are no longer needed.
    """
    for d in ckpt_dir.glob("step_*"):
        shutil.rmtree(d, ignore_errors=True)
    print(f"  Pruned sharded checkpoints in {ckpt_dir}")


def prune_old_merged(name_prefix: str, current_iter: int,
                     rounds_per_epoch: int) -> None:
    """Drop the previous iter's ``merged/`` unless it's an epoch boundary.

    After iter ``k`` finishes exporting, we delete iter ``k-1``'s
    ``merged/`` (~19 GB) unless ``k-1`` was the last round of its
    total_epoch (i.e. ``(k-1) % rounds_per_epoch == 0``), in which
    case we keep it as an archival snapshot of that epoch's final
    policy.  At any time the tree holds: every completed epoch's
    final merged + the current iter's merged (~2 × 19 GB steady).

    NOTE: do not delete the *current* iter's ``merged/`` — the next
    iter's vLLM will load it as ``--model``.
    """
    if current_iter <= 1:
        return
    prev = current_iter - 1
    if prev % rounds_per_epoch == 0:
        return
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
          f"A∈[{summary['advantage_p10']:+.2f}, "
          f"{summary['advantage_p90']:+.2f}]")
    return summary


def backup_iter(backup_dir: Path, run_dir: Path,
                total_epoch: int, round_n: int) -> None:
    """Copy this iter's small artifacts into ``backups/<run>/ep{E}.round{R}/``.

    Uses ``BACKUP_FILES`` for iter-level files and additionally pulls
    the training-side ``ckpt/metrics.jsonl`` and ``ckpt/config.json``
    for full reproducibility. Large files (parliament.db, llm_logs)
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
                      name_prefix: str, cfg: dict,
                      train_extra: list[str], rounds_per_epoch: int,
                      gpus: list[int]) -> tuple[str, dict]:
    """One iter: rollout → extract → train → export.

    Returns ``(new_merged_path, summary)``.

    Pre-conditions:
      - No vLLM is currently running (the previous iter's stop_vllm
        was called at the end of train).  This iter starts vLLM
        with ``actor_model`` as the rollout policy.
      - ``actor_model`` is the path to the previous iter's merged
        folder (or the original base on iter 1).
      - ``train_anchor_model`` is the original base path; it stays
        constant across all iters as the KL-anchor reference.

    What this iter does:
      1. Start vLLM fleet with ``actor_model`` as ``--model``
         (no LoRA flags — full speed inference).
      2. Run the rollout (Parliament + harness) against vLLMs.
      3. Stop vLLMs to release all GPU memory for the trainer.
      4. Run the DDP trainer subprocess (~5-7 min on 4-8 A100).
      5. Run rl.export to merge the LoRA into ``train_anchor_model``,
         producing ``run_dir/merged/`` (the next iter's actor).
      6. Prune intermediate state (sharded ckpts + old merged folders).

    Sub-step idempotency: each step skips if its output artifact
    already exists from a prior crashed run.  vLLM is NOT left
    running across iters — the next iter brings it back up with the
    freshly-merged policy.
    """
    t0 = time.time()
    name = f"{name_prefix}_iter{iter_n:02d}"
    num_gpus = len(gpus)
    gpu_ports = [run.gpu_to_port(g) for g in gpus]
    gpu_endpoints = [f"http://127.0.0.1:{p}/v1" for p in gpu_ports]
    print(f"\n{'=' * 72}")
    print(f"Iter {iter_n}/{total}  shard={shard_path.name}")
    print(f"  Actor:      {actor_model}")
    print(f"  KL anchor:  {train_anchor_model}")
    print(f"  Output:     data/{name}_*/")
    print(f"{'=' * 72}")

    # 1. Rollout (skip if parliament.db + experiment.json already there)
    existing = _find_run_dir(name)
    rollout_done = bool(
        existing
        and (existing / "parliament.db").exists()
        and (existing / "experiment.json").exists()
    )
    if rollout_done:
        print(f"  [skip] rollout — reusing {existing.name}")
        run_dir = existing
    else:
        # Bring the vLLM fleet up with this iter's actor as ``--model``.
        # ``ensure_vllm`` is idempotent: if vLLMs are already running
        # (e.g. a manual resume on the same iter) it just polls
        # /v1/models and returns the existing port list.
        run.ensure_vllm(gpus, actor_model, enable_lora=False)
        run_dir = rollout_step(shard_path, actor_model, name, cfg,
                               gpu_endpoints)

    # 2. Extract (skip if train.jsonl is structurally complete)
    train_jsonl = run_dir / "train.jsonl"
    if _train_jsonl_complete(train_jsonl):
        print(f"  [skip] extract — {train_jsonl.name} "
              f"({train_jsonl.stat().st_size // 1024} KB, validated)")
    else:
        if train_jsonl.exists():
            print(f"  [redo] extract — existing {train_jsonl.name} "
                  f"failed validation, regenerating")
        train_jsonl = extract_step(run_dir)

    metrics = metrics_step(train_jsonl, run_dir)

    # 3. Train: stop vLLM to free ~80 GB / GPU for the trainer.
    #    The trainer always uses ``train_anchor_model`` (the original
    #    base) as both ``--model`` and ``--ref-model``, so the KL
    #    anchor stays at the unmoved base regardless of how far the
    #    policy has drifted.
    ckpt_dir = run_dir / "ckpt"
    last = _find_last_ckpt(ckpt_dir)
    if last is not None and _adapter_complete(last / "adapter"):
        print(f"  [skip] train — {last.name}/adapter already valid")
    else:
        print("  [vllm] stopping fleet to free GPU memory for trainer…")
        run.stop_vllm()
        # Block until every GPU is back below 1 GB used — a SIGKILL'd
        # vLLM normally releases memory in ~5 s but can take 30 s+
        # under contention; starting the DDP trainer too early gives
        # a silent OOM at weight load.  Hard 60 s cap then warn.
        if not run.wait_gpu_idle(timeout_s=60):
            print("  [vllm] WARN: some GPUs still > 1 GB after 60 s "
                  "stop — proceeding anyway, may OOM at trainer init",
                  flush=True)
        extra = list(train_extra)
        if last is not None:
            print(f"  [resume] train — from {last.name}")
            extra += ["--resume", str(last)]
        train_step(train_jsonl, train_anchor_model, run_dir, num_gpus, extra)

    # 4. Export: merge LoRA → base, output run_dir/merged/.  This is
    #    what next iter's vLLM will load as ``--model``.  Skip if
    #    a complete merged folder already exists (resume case).
    merged = run_dir / "merged"
    if _merged_complete(merged):
        print(f"  [skip] export — merged/ already complete")
    else:
        merged = export_step(ckpt_dir, run_dir)

    # 5. Disk hygiene: drop sharded ckpts (no longer needed once
    #    merged/ is built) and the previous iter's merged/ unless
    #    it sits on a total-epoch boundary.
    prune_sharded_ckpt(ckpt_dir)
    prune_old_merged(name_prefix, iter_n, rounds_per_epoch)

    dur_min = (time.time() - t0) / 60
    print(f"\n  Iter {iter_n} done in {dur_min:.0f} min → {merged}")
    return str(merged), {"metrics": metrics, "minutes": dur_min}


# ── State file ──────────────────────────────────────────

def load_state(path: Path, initial_model: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "completed": 0,
        "current_model": initial_model,   # actor for the next iter
        "base_model": initial_model,      # fixed KL anchor
        "history": [],
    }


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2))


def write_done_sentinel(out_dir: Path, total_iters: int,
                        wall_minutes: float,
                        final_model: str) -> None:
    """Write ``DONE.txt`` so external watchers (`prl-status`, the
    keepalive watcher in launch_cell.sh, manual ssh) can confirm
    the cell finished without scraping the iterate.log tail.

    Idempotent — overwrites any stale FAILED.txt left from a
    previous attempt that ended up succeeding on retry.
    """
    (out_dir / "FAILED.txt").unlink(missing_ok=True)
    (out_dir / "DONE.txt").write_text(
        f"status: completed\n"
        f"total_iters: {total_iters}\n"
        f"wall_minutes: {wall_minutes:.1f}\n"
        f"final_model: {final_model}\n"
        f"finished_at: {datetime.now().isoformat()}\n"
    )


def write_failed_sentinel(out_dir: Path, iter_n: int, total_iters: int,
                          attempts: int, exception: BaseException) -> None:
    """Write ``FAILED.txt`` with the iter that died + exception type.

    Status sentinels (DONE.txt / FAILED.txt) are intentionally tiny
    plaintext files so a human attaching tmux at 3 AM can ``cat`` them
    without spinning up Python — useful when the issue itself is that
    Python is broken.
    """
    (out_dir / "DONE.txt").unlink(missing_ok=True)
    (out_dir / "FAILED.txt").write_text(
        f"status: failed\n"
        f"failed_iter: {iter_n}/{total_iters}\n"
        f"attempts: {attempts}\n"
        f"exception_type: {type(exception).__name__}\n"
        f"exception_msg: {str(exception)[:500]}\n"
        f"failed_at: {datetime.now().isoformat()}\n"
        f"\n"
        f"Look at iterate.log around this timestamp for full traceback.\n"
    )


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
                        "of JSONs that will be concatenated")
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
                   help="Base model (frozen for the entire run; LoRA on top "
                        "accumulates the policy update). Also doubles as "
                        "the KL anchor inside rl.train.")
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
                   help="Skip per-iter eval. Currently always implied "
                        "(eval/gpqa.py is not wired into the new "
                        "iterate.py loop yet) — accepted for backward "
                        "compatibility with launch scripts that pass it.")
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

    schedule = build_schedule(pool_paths, args.total_questions,
                              args.sampling_batch_size, args.seed)
    assert len(schedule) == rounds_per_epoch

    # Manifest is partially immutable (config) + partially mutable
    # (timestamps).  ``started_at`` is the original launch time of
    # this run dir; ``last_started_at`` is updated every time
    # iterate.py boots, so resumes don't make the wall-time math
    # negative in `prl-status`.  Existing manifests on resume keep
    # ``started_at`` and only refresh ``last_started_at``.
    now_iso = datetime.now().isoformat()
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    manifest.update({
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
        "architecture": "http_dpN_merge_reload",
        "last_started_at": now_iso,
    })
    manifest.setdefault("started_at", now_iso)         # only on first launch
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (backup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 72}")
    print(f"Iterative PPO (clip + KL anchor) — {args.name}")
    print(f"{'=' * 72}")
    print(f"  Output:            {out_dir}")
    print(f"  Backups:           {backup_dir}")
    print(f"  Pool:              {', '.join(pool_paths)}")
    print(f"  Draw:              {args.total_questions} q × seed={args.seed}")
    print(f"  Sampling batch:    {args.sampling_batch_size}  "
          f"→ {rounds_per_epoch} rounds/epoch")
    print(f"  Total epochs:      {args.total_epochs}")
    print(f"  Total iters:       {total_iters}")
    print(f"  Base model:        {args.initial_model}")
    n_gpus = len(cfg['gpus'].split(','))
    print(f"  Base model:        {args.initial_model}")
    print(f"  GPUs:              {cfg['gpus']} (DP={n_gpus}, TP=1 each)")
    print(f"  Sessions/GPU:      {args.sessions_per_gpu}")
    print(f"  Actor:             {state['current_model']}")
    print(f"  KL anchor:         {state['base_model']}")
    print(f"  Completed:         {state['completed']}/{total_iters}")
    print(f"  Train extra:       {args.train_extra}")
    print(f"  Started:           {datetime.now().isoformat()}")
    print(f"{'=' * 72}\n")

    gpus = [int(g) for g in cfg["gpus"].split(",")]

    # Pre-flight: kill anything left over from a previous crashed run
    # (vLLM tmux sessions, port 8080, etc.).  vLLM itself is brought
    # up inside ``run_one_iteration`` per iter using the iter's actor
    # model — we don't keep it alive across iters because each iter
    # must serve a different merged checkpoint as ``--model``.
    run.cleanup_all(gpus, PARLIAMENT_PORT)

    # Per-iter retry budget — protects 6h experiments from being
    # killed by transient KML / NCCL / GPFS hiccups (cell B/C of the
    # 2026-04-26 main run died this way). Idempotency in
    # ``run_one_iteration`` (rollout / extract / train / export each
    # skip if their output already exists) means a retry from the
    # same iter is essentially free of redundant compute.
    max_iter_retries = 1                         # initial + 1 retry = 2 tries

    try:
        t_start = time.time()
        for total_epoch in range(1, args.total_epochs + 1):
            for round_idx in range(rounds_per_epoch):
                iter_n = (total_epoch - 1) * rounds_per_epoch + round_idx + 1
                if iter_n <= state["completed"]:
                    print(f"Skipping iter {iter_n} (already completed)")
                    continue

                shard_path = (shard_dir
                              / f"ep{total_epoch}.round{round_idx + 1}.json")
                if not shard_path.exists():
                    shard_path.write_text(
                        json.dumps(schedule[round_idx], ensure_ascii=False))

                # Retry the whole iter on failure: cleanup → wait → retry.
                # Cleanup tears down anything the failed attempt left
                # behind (vLLM tmux, parliament.server, train workers
                # stuck on GPU, DDP TCPStore in TIME_WAIT) so the next
                # attempt starts from a clean slate.
                last_exc: BaseException | None = None
                for attempt in range(max_iter_retries + 1):
                    try:
                        new_merged, summary = run_one_iteration(
                            iter_n, total_iters, shard_path,
                            actor_model=state["current_model"],
                            train_anchor_model=state["base_model"],
                            name_prefix=args.name, cfg=cfg,
                            train_extra=train_extra,
                            rounds_per_epoch=rounds_per_epoch,
                            gpus=gpus,
                        )
                        last_exc = None
                        break
                    except BaseException as e:
                        last_exc = e
                        print(f"\n[iter {iter_n}] attempt "
                              f"{attempt + 1}/{max_iter_retries + 1} FAILED: "
                              f"{type(e).__name__}: {str(e)[:200]}",
                              flush=True)
                        # Belt-and-braces cleanup before retry.
                        try:
                            run.stop_vllm()
                        except Exception:
                            pass
                        _kill_orphan_train_workers()
                        run.cleanup_all(gpus, PARLIAMENT_PORT)
                        if attempt < max_iter_retries:
                            print(f"[iter {iter_n}] retrying in 10 s...",
                                  flush=True)
                            time.sleep(10)
                if last_exc is not None:
                    write_failed_sentinel(out_dir, iter_n, total_iters,
                                          max_iter_retries + 1, last_exc)
                    raise last_exc                # propagate to outer try/finally

                run_dir = _find_run_dir(f"{args.name}_iter{iter_n:02d}")
                if run_dir is not None:
                    backup_iter(backup_dir, run_dir, total_epoch,
                                round_idx + 1)

                state = {
                    **state,
                    "completed": iter_n,
                    "current_model": new_merged,
                    "status": "running",
                    "history": state["history"] + [{
                        "iter": iter_n,
                        "total_epoch": total_epoch,
                        "round": round_idx + 1,
                        "shard": str(shard_path),
                        "merged": new_merged,
                        "timestamp": datetime.now().isoformat(),
                        "metrics": summary["metrics"],
                        "minutes": round(summary["minutes"], 1),
                    }],
                }
                save_state(state_file, state)
                shutil.copy(state_file, backup_dir / "state.json")

        total_min = (time.time() - t_start) / 60
        # Mark cell complete: status sentinel + state.json status.
        state = {**state, "status": "completed"}
        save_state(state_file, state)
        shutil.copy(state_file, backup_dir / "state.json")
        write_done_sentinel(out_dir, total_iters, total_min,
                            state["current_model"])
        print(f"\n{'=' * 72}")
        print(f"All {total_iters} iters done in {total_min:.0f} min "
              f"({args.total_epochs} epoch × {rounds_per_epoch} rounds).")
        print(f"Final policy: {state['current_model']}")
        print(f"Backups:      {backup_dir}")
        print(f"{'=' * 72}\n")
    except BaseException:
        # Status sentinel was already written by the inner retry block;
        # here we just make sure state.json reflects the failure so
        # `prl-status` shows the right label.
        try:
            state = {**state, "status": "failed"}
            save_state(state_file, state)
            shutil.copy(state_file, backup_dir / "state.json")
        except Exception:
            pass
        raise
    finally:
        print("\n[shutdown] tearing down vLLM fleet…", flush=True)
        run.stop_vllm()


if __name__ == "__main__":
    main()
