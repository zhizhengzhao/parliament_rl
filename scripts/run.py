#!/usr/bin/env python3
"""One-click Parliament RL data collection.

Full lifecycle: cleanup → vLLM (parallel) → Parliament → dataset → harness.
Each GPU runs its own vLLM API. The harness dynamically assigns sessions
to GPUs from a shared queue. On success, Parliament stays up for web UI.

Usage:
    python scripts/run.py \\
        --gpus 0,1,2,3,4,5,6,7 \\
        --sessions-per-gpu 2 \\
        --actors 3 --judges 3 \\
        --dataset datasets/sciencepedia_test.json \\
        --name experiment_1 \\
        --max-turns 30

    python scripts/run.py --stop-vllm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from _common import Tee, env_prefix  # noqa: E402

VLLM_PYTHON = os.environ.get("PRL_PYTHON", sys.executable)
MODEL_PATH = os.environ.get("PRL_MODEL_PATH", "Qwen/Qwen3.5-9B")
ADMIN_KEY = "sp_admin_parliament"
TMUX_PREFIX = "vllm-gpu"
EXPERIMENT_TMUX = "parliament-run"
GPU_IDLE_MEMORY_THRESHOLD_MIB = 2048


# ── HTTP helpers ─────────────────────────────────────────

def http(method: str, url: str, key: str = "", body: dict | None = None):
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()[:500] if e.fp else ""
        print(f"  HTTP {e.code} on {method} {url}: {body_text}")
    except Exception as e:
        print(f"  Request error {method} {url}: {e}")
    return None


def wait_ready(url: str, timeout: int = 600) -> bool:
    """Poll URL until it returns 200 (or timeout). 600 s covers vLLM's
    first-boot `torch.compile` on Qwen3.5 hybrid attention — second boot
    hits the compile cache and returns in < 60 s."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(3)
    return False


# ── Cleanup ──────────────────────────────────────────────

def kill_port(port: int) -> None:
    try:
        out = subprocess.run(["ss", "-tlnp"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if f":{port} " in line and "pid=" in line:
                pid = int(line.split("pid=")[1].split(",")[0])
                os.kill(pid, signal.SIGKILL)
                print(f"  Killed pid {pid} on port {port}")
    except Exception:
        pass


def stop_vllm() -> None:
    out = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"],
                         capture_output=True, text=True).stdout
    killed = 0
    for name in out.strip().split("\n"):
        if name.startswith(TMUX_PREFIX):
            subprocess.run(["tmux", "kill-session", "-t", name],
                           capture_output=True)
            killed += 1
    if killed:
        print(f"  Killed {killed} vLLM tmux session(s)")
    # Belt-and-braces: kill any orphaned vllm processes by name
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"],
                   capture_output=True)
    time.sleep(3)


def cleanup_all(gpus: list[int], port: int) -> None:
    print("[0/3] Cleanup")
    stop_vllm()
    kill_port(port)
    time.sleep(1)

    for attempt in range(20):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout
        busy = []
        for line in out.strip().split("\n"):
            idx_s, mem_s = line.split(",")
            idx, mem = int(idx_s.strip()), int(mem_s.strip())
            if idx in gpus and mem > GPU_IDLE_MEMORY_THRESHOLD_MIB:
                busy.append(idx)
        if not busy:
            break
        if attempt == 5:
            print(f"  Force-killing GPU processes on {busy}")
            for gpu in busy:
                subprocess.run(
                    f"nvidia-smi --query-compute-apps=pid --format=csv,noheader "
                    f"-i {gpu} | xargs -r kill -9",
                    shell=True, capture_output=True)
        time.sleep(3)
    if busy:
        print(f"  FATAL: GPUs {busy} still occupied after 60s cleanup")
        sys.exit(1)
    print("  Cleanup done\n")


# ── vLLM ─────────────────────────────────────────────────

def gpu_to_port(gpu: int) -> int:
    return 7999 + gpu


def ensure_vllm(gpus: list[int], model_path: str = MODEL_PATH) -> list[int]:
    ports: list[int] = []
    for gpu in gpus:
        port = gpu_to_port(gpu)
        ports.append(port)
        session_name = f"{TMUX_PREFIX}{gpu}"
        subprocess.run(["tmux", "kill-session", "-t", session_name],
                       capture_output=True)
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} {VLLM_PYTHON} "
            f"-m vllm.entrypoints.openai.api_server "
            f"--model {model_path} --port {port} "
            f"--max-model-len 32768 --gpu-memory-utilization 0.90 "
            f"--enable-auto-tool-choice --tool-call-parser qwen3_coder "
            f"--dtype auto 2>&1 | tee /tmp/vllm_gpu{gpu}.log"
        )
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd],
                       capture_output=True)
        print(f"  GPU {gpu} → :{port} launching...")

    print(f"  Waiting for {len(gpus)} instances...", flush=True)
    for gpu, port in zip(gpus, ports):
        if not wait_ready(f"http://127.0.0.1:{port}/v1/models"):
            print(f"  GPU {gpu} → :{port} FAILED")
            stop_vllm()
            sys.exit(1)
        print(f"  GPU {gpu} → :{port} ready")
    print(f"  All {len(ports)} vLLM instances ready")
    return ports


# ── Parliament ───────────────────────────────────────────

def start_parliament(name: str, num_actors: int, num_judges: int,
                     port: int, log_path: Path,
                     db_dir: str) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    cmd = [sys.executable, "-m", "parliament.server",
           "--seed", "--actors", str(num_actors), "--judges", str(num_judges),
           "--port", str(port), "--db-dir", db_dir]
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_DIR),
        stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if not wait_ready(f"http://127.0.0.1:{port}/docs", timeout=15):
        proc.poll()
        log_file.flush()
        tail = log_path.read_text()[-1000:]
        print(f"  FATAL: Parliament failed to start (exit={proc.returncode})")
        print(f"  Log tail:\n{tail}")
        sys.exit(1)
    print(f"  Parliament: {log_path.parent}")
    return proc


def stop_parliament(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def load_dataset(dataset_path: str, parliament_url: str,
                 max_questions: int) -> int:
    questions = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if max_questions > 0:
        questions = questions[:max_questions]
    loaded = 0
    for q in questions:
        title = q.get("title") or q.get("problem") or ""
        desc = q.get("description", "")
        solution = q.get("reference_solution") or q.get("solution") or ""
        answer = q.get("answer", "")
        ref = (f"{solution}\n\nFinal answer: {answer}".strip()
               if answer else solution)
        if not title:
            continue
        result = http("POST", f"{parliament_url}/sessions", ADMIN_KEY,
                      {"title": title, "description": desc,
                       "reference_solution": ref})
        if result and result.get("session_id"):
            loaded += 1
    print(f"  Dataset: {loaded}/{len(questions)} questions loaded")
    return loaded


# ── tmux self-launch ─────────────────────────────────────

def relaunch_in_tmux(argv: list[str], name: str) -> None:
    subprocess.run(["tmux", "start-server"], capture_output=True)
    time.sleep(1)
    subprocess.run(["tmux", "kill-session", "-t", EXPERIMENT_TMUX],
                   capture_output=True)
    cmd = f"{env_prefix()}{shlex.quote(sys.executable)} {shlex.join(argv + ['--in-tmux'])}"
    r = subprocess.run(
        ["tmux", "new-session", "-d", "-s", EXPERIMENT_TMUX, cmd],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FATAL: Failed to create tmux session: {r.stderr}")
        sys.exit(1)
    time.sleep(1)
    if subprocess.run(["tmux", "has-session", "-t", EXPERIMENT_TMUX],
                      capture_output=True).returncode != 0:
        print(f"FATAL: tmux session '{EXPERIMENT_TMUX}' not found after creation")
        sys.exit(1)
    print(f"Experiment launched in tmux session '{EXPERIMENT_TMUX}'")
    print(f"  Attach: tmux attach -t {EXPERIMENT_TMUX}")
    print(f"  Logs:   tail -f data/{name}_*/run.log")


# ── Main ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-click Parliament RL data collection")
    p.add_argument("--gpus", help="Comma-separated GPU IDs")
    p.add_argument("--sessions-per-gpu", type=int, default=2)
    p.add_argument("--actors", type=int, default=3)
    p.add_argument("--judges", type=int, default=3)
    p.add_argument("--dataset", help="Path to questions JSON file")
    p.add_argument("--name", help="Run name")
    p.add_argument("--model", default=MODEL_PATH,
                   help="HF model path (defaults to base Qwen3.5-9B)")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--stop-vllm", action="store_true")
    p.add_argument("--in-tmux", action="store_true",
                   help="Internal flag: already running inside tmux")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.stop_vllm:
        stop_vllm()
        return

    for flag in ("gpus", "dataset", "name"):
        if not getattr(args, flag):
            print(f"FATAL: --{flag} is required")
            sys.exit(2)

    if not args.in_tmux:
        relaunch_in_tmux(sys.argv, args.name)
        return

    if not Path(args.dataset).exists():
        print(f"FATAL: Dataset not found: {args.dataset}")
        sys.exit(1)

    gpus = [int(g) for g in args.gpus.split(",")]
    run_dir = PROJECT_DIR / "data" / f"{args.name}_{time.strftime('%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(run_dir / "run.log", "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"\n{'=' * 60}")
    print(f"Parliament RL — {args.name}")
    print(f"{'=' * 60}")
    print(f"  GPUs:           {gpus}")
    print(f"  Sessions/GPU:   {args.sessions_per_gpu}")
    print(f"  Agents/session: {args.actors} actors + {args.judges} judges")
    print(f"  Max rounds:     {args.max_turns} (actor only)")
    print(f"  Dataset:        {args.dataset}")
    print(f"  Model:          {args.model}")
    print(f"  Output:         {run_dir}")
    print()

    cleanup_all(gpus, args.port)

    print("[1/3] vLLM")
    ports = ensure_vllm(gpus, args.model)
    gpu_endpoints = [f"http://127.0.0.1:{p}/v1" for p in ports]
    print()

    print("[2/3] Parliament")
    parliament_url = f"http://127.0.0.1:{args.port}"
    parliament_proc = start_parliament(
        args.name, args.actors, args.judges, args.port,
        log_path=run_dir / "parliament.log", db_dir=str(run_dir))
    if load_dataset(args.dataset, parliament_url, args.max_questions) == 0:
        print("  FATAL: No questions loaded")
        parliament_proc.terminate()
        sys.exit(1)
    print()

    print("[3/3] Harness")
    from harness.runner import run_experiment

    keep_parliament = False
    try:
        rc = asyncio.run(run_experiment(
            parliament_url=parliament_url,
            admin_key=ADMIN_KEY,
            gpu_endpoints=gpu_endpoints,
            sessions_per_gpu=args.sessions_per_gpu,
            num_actors=args.actors,
            num_judges=args.judges,
            model_name=args.model,
            max_rounds=args.max_turns,
            output_path=str(run_dir / "experiment.json"),
        ))
        if rc != 0:
            print(f"\n  FATAL: Experiment failed with code {rc}")
            sys.exit(rc)
        keep_parliament = True
    finally:
        if keep_parliament:
            stop_vllm()
        else:
            stop_parliament(parliament_proc)

    print(f"\n  Experiment finished.")
    print(f"  Web UI:        http://127.0.0.1:{args.port}")
    print(f"  Parliament:    still running")
    print(f"  vLLM:          stopped")
    print(f"  Output:        {run_dir}")


if __name__ == "__main__":
    main()
