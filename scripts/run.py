#!/usr/bin/env python3
"""One-click Parliament RL data collection.

Starts vLLM (if needed), Parliament, loads dataset, runs experiment, cleans up.

Usage:
    python scripts/run.py \
        --gpus 2,3,4,5,6,7 \
        --sessions-per-gpu 2 \
        --actors 4 --judges 4 \
        --dataset datasets/echelle_optics.json \
        --name echelle_optics \
        --timeout 300
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
VLLM_PYTHON = "/root/miniconda3/envs/parliament/bin/python"
MODEL_PATH = "/root/zhizheng/models/Qwen3.5-9B"
VLLM_PID_FILE = Path("/tmp/parliament_vllm.pid")


# ── Helpers ──────────────────────────────────────────────

def http(method: str, url: str, key: str = "", body: dict | None = None):
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except Exception:
        return None


def wait_ready(url: str, label: str, timeout: int = 300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(3)
    print(f"  TIMEOUT waiting for {label} at {url}")
    return False


# ── vLLM ─────────────────────────────────────────────────

def stop_vllm():
    """Kill all vLLM processes we previously started."""
    if not VLLM_PID_FILE.exists():
        print("  No vLLM PID file found.")
        return
    killed = 0
    for line in VLLM_PID_FILE.read_text().strip().split("\n"):
        pid = int(line.strip())
        try:
            os.kill(pid, 9)
            killed += 1
        except ProcessLookupError:
            pass
    VLLM_PID_FILE.unlink(missing_ok=True)
    print(f"  Killed {killed} vLLM process(es). GPUs released.")


def ensure_vllm(gpus: list[int]) -> list[int]:
    """Start vLLM on each GPU if not already running. Returns list of ports.

    Launches sequentially — each instance confirmed ready before starting
    the next — to avoid CPU/IO contention during model weight loading.
    PIDs are recorded to {VLLM_PID_FILE} for later cleanup.
    """
    ports = []
    pids = []
    for gpu in gpus:
        port = 8000 + gpu - 1
        ports.append(port)
        if http("GET", f"http://localhost:{port}/v1/models"):
            print(f"  GPU {gpu} → :{port} ✓ (already running)")
            continue
        print(f"  GPU {gpu} → :{port} starting...", end="", flush=True)
        proc = subprocess.Popen(
            [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
             "--model", MODEL_PATH,
             "--port", str(port),
             "--max-model-len", "262144",
             "--gpu-memory-utilization", "0.92",
             "--enable-auto-tool-choice",
             "--tool-call-parser", "qwen3_coder",
             "--dtype", "auto"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=open(f"/tmp/vllm_gpu{gpu}.log", "w"),
            stderr=subprocess.STDOUT,
        )
        pids.append(proc.pid)
        if not wait_ready(f"http://localhost:{port}/v1/models", f"vLLM:{port}"):
            print(f" FAILED")
            print(f"  Check /tmp/vllm_gpu{gpu}.log for details")
            print(f"  Cleaning up started processes...")
            for p in pids:
                try:
                    os.kill(p, 9)
                except ProcessLookupError:
                    pass
            sys.exit(1)
        print(f" ready")
    if pids:
        existing = VLLM_PID_FILE.read_text().strip().split("\n") if VLLM_PID_FILE.exists() else []
        all_pids = [l for l in existing if l.strip()] + [str(p) for p in pids]
        VLLM_PID_FILE.write_text("\n".join(all_pids) + "\n")
    print(f"  All {len(ports)} vLLM instances ready")
    return ports


# ── nginx ────────────────────────────────────────────────

def configure_nginx(ports: list[int]):
    servers = "\n".join(f"    server 127.0.0.1:{p};" for p in ports)
    conf = f"""upstream vllm_backends {{
{servers}
}}

server {{
    listen 8888;
    client_max_body_size 10m;
    proxy_read_timeout 300s;
    proxy_connect_timeout 10s;

    location / {{
        proxy_pass http://vllm_backends;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }}
}}
"""
    conf_path = "/etc/nginx/sites-enabled/vllm_lb.conf"
    Path(conf_path).write_text(conf)
    subprocess.run(["nginx", "-s", "reload"], capture_output=True)
    print(f"  nginx: {len(ports)} backends on port 8888")


# ── Parliament ───────────────────────────────────────────

def start_parliament(name: str, num_actors: int, num_judges: int,
                     port: int = 8080) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "parliament.server",
         "--name", name, "--seed",
         "--actors", str(num_actors), "--judges", str(num_judges),
         "--port", str(port)],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if not wait_ready(f"http://localhost:{port}/admin/sessions", "Parliament",
                      timeout=10):
        out = proc.stdout.read().decode()[:500] if proc.stdout else ""
        print(f"  FATAL: Parliament failed to start\n{out}")
        sys.exit(1)
    info = http("GET", f"http://localhost:{port}/admin/info", "sp_admin_parliament")
    print(f"  Parliament: {info.get('run_dir', '?')}")
    return proc


def load_dataset(dataset_path: str, parliament_url: str, admin_key: str,
                 max_questions: int = 0) -> int:
    questions = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if max_questions > 0:
        questions = questions[:max_questions]
    loaded = 0
    for q in questions:
        title = q.get("title") or q.get("problem") or ""
        desc = q.get("description", "")
        solution = q.get("reference_solution") or q.get("solution") or ""
        answer = q.get("answer", "")
        ref = f"{solution}\n\nFinal answer: {answer}".strip() if answer else solution
        if not title:
            continue
        result = http("POST", f"{parliament_url}/sessions", admin_key, {
            "title": title,
            "description": desc,
            "reference_solution": ref,
        })
        if result and result.get("session_id"):
            loaded += 1
    print(f"  Dataset: {loaded} questions loaded from {dataset_path} ({len(questions)} available)")
    return loaded


# ── Experiment ───────────────────────────────────────────

def run_experiment(parliament_url: str, model_api: str,
                   num_actors: int, num_judges: int,
                   timeout: int, parallel: bool, openclaw_agent: str):
    cmd = [
        sys.executable, str(PROJECT_DIR / "scripts" / "run_experiment.py"),
        "--parliament-url", parliament_url,
        "--model-api", model_api,
        "--agents", str(num_actors),
        "--judges", str(num_judges),
        "--timeout", str(timeout),
        "--openclaw-agent", openclaw_agent,
        "--skip-preflight",
    ]
    if parallel:
        cmd.append("--parallel-sessions")
    proc = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    return proc.returncode


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="One-click Parliament RL data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpus", required=True,
                        help="Comma-separated GPU IDs (e.g. 2,3,4,5,6,7)")
    parser.add_argument("--sessions-per-gpu", type=int, default=2,
                        help="Parallel forum sessions per GPU (default: 2)")
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--judges", type=int, default=4)
    parser.add_argument("--dataset", required=True,
                        help="Path to questions JSON file")
    parser.add_argument("--name", required=True,
                        help="Run name (e.g. science_pedia)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per agent in seconds (default: 300)")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Max questions to load (0 = all)")
    parser.add_argument("--openclaw-agent", default="parliament-scientist")
    parser.add_argument("--port", type=int, default=8080,
                        help="Parliament port (default: 8080)")
    parser.add_argument("--stop-vllm", action="store_true",
                        help="Kill all vLLM processes started by this script, then exit")
    args = parser.parse_args()

    if args.stop_vllm:
        stop_vllm()
        return

    gpus = [int(g) for g in args.gpus.split(",")]
    agents_per_session = args.actors + args.judges
    total_sessions = len(gpus) * args.sessions_per_gpu
    concurrent_per_gpu = args.sessions_per_gpu * agents_per_session

    print(f"\n{'='*60}")
    print(f"Parliament RL — {args.name}")
    print(f"{'='*60}")
    print(f"  GPUs:          {gpus}")
    print(f"  Sessions/GPU:  {args.sessions_per_gpu}")
    print(f"  Agents/session:{args.actors} actors + {args.judges} judges")
    print(f"  Concurrency:   {concurrent_per_gpu} requests/GPU")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Timeout:       {args.timeout}s")
    print()

    # 1. vLLM
    print("[1/4] vLLM")
    ports = ensure_vllm(gpus)
    print()

    # 2. nginx
    print("[2/4] nginx")
    configure_nginx(ports)
    print()

    # 3. Parliament
    print("[3/4] Parliament")
    parliament_url = f"http://localhost:{args.port}"
    parliament_proc = start_parliament(args.name, args.actors, args.judges, args.port)
    loaded = load_dataset(args.dataset, parliament_url, "sp_admin_parliament",
                          args.max_questions)
    if loaded == 0:
        print("  FATAL: No questions loaded")
        parliament_proc.terminate()
        sys.exit(1)
    print()

    # 4. Experiment
    print("[4/4] Experiment")
    info = http("GET", f"{parliament_url}/admin/info", "sp_admin_parliament")
    run_dir = info.get("run_dir", "data/") if info else "data/"
    parallel = total_sessions > 1
    try:
        run_experiment(
            parliament_url=parliament_url,
            model_api="http://localhost:8888/v1",
            num_actors=args.actors,
            num_judges=args.judges,
            timeout=args.timeout,
            parallel=parallel,
            openclaw_agent=args.openclaw_agent,
        )
    finally:
        parliament_proc.terminate()
        parliament_proc.wait(timeout=5)

    print(f"\n  Parliament stopped. vLLM still running on ports {ports}.")
    print(f"  Results: {run_dir}")


if __name__ == "__main__":
    main()
