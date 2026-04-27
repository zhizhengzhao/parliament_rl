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


def wait_ready(url: str, timeout: int = 1800) -> bool:
    """Poll URL until it returns 200 (or timeout). 1800 s (= 30 min)
    covers two known-slow paths simultaneously:

      1. vLLM first-boot `torch.compile` on Qwen3.5 hybrid attention
         (second boot hits the compile cache and returns in < 60 s).
      2. Loading the 19 GB model from NFS-shared paths
         (e.g. `/ytech_m2v5_hdd/...`) when several pods read the same
         file concurrently — bandwidth contention can stretch a single
         load past 10 min.  600 s (the previous default) was too tight.
    """
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
    """Kill anything listening on `port`, by every available mechanism.

    `ss -tlnp` only prints owner pid when the caller has CAP_NET_ADMIN
    or owns the socket — in containers that often returns rows with no
    `pid=` field, so we miss the kill.  We also try `fuser` (which uses
    /proc), and finally `pkill -9 -f 'parliament.server'` to catch any
    Parliament uvicorn that might have orphaned children.  Running all
    three is harmless if some find nothing.
    """
    killed_any = False
    # 1. ss -tlnp (works when we have permission to see the pid)
    try:
        out = subprocess.run(["ss", "-tlnp"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if f":{port} " in line and "pid=" in line:
                pid = int(line.split("pid=")[1].split(",")[0])
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed_any = True
                    print(f"  Killed pid {pid} on port {port} (via ss)")
                except ProcessLookupError:
                    pass
    except Exception:
        pass
    # 2. fuser -k (uses /proc, often works when ss doesn't)
    try:
        r = subprocess.run(
            ["fuser", "-k", "-9", f"{port}/tcp"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            killed_any = True
            print(f"  Killed listener on port {port} (via fuser)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # 3. pkill any Parliament uvicorn (catches orphan children)
    subprocess.run(
        ["pkill", "-9", "-f", "parliament.server"],
        capture_output=True,
    )
    # Linux often holds the port in TIME_WAIT briefly after the listener
    # exits — wait a moment to give the next bind a clean port.
    time.sleep(2)
    if not killed_any:
        print(f"  No listener detected on port {port}")


def stop_vllm() -> None:
    """Tear down every vLLM instance launched by `ensure_vllm`.

    Five-layer belt-and-braces because vLLM forks a small process
    tree (entrypoint + EngineCore_DP* + Worker_TP* +
    multiprocessing.spawn pool + flashinfer-cubin worker), and only
    the entrypoint cmdline matches "vllm.entrypoints".  An earlier
    3-layer version left orphan workers holding 7999..8006 open;
    the next iter's vLLM bind() then crashed with EADDRINUSE.

    Why each layer is needed (in order of cleanup completeness):
      1. tmux kill-session — drop the controlling terminal so the
         entrypoint receives SIGHUP.
      2. pkill vllm.entrypoints — kill the main API server process.
      3. pkill VLLM::EngineCore  — kill v1's engine core daemon
         (named ``VLLM::EngineCore_DP*`` in ps).
      4. pkill multiprocessing.spawn workers / fork  — kill the
         per-GPU worker pool that mp.spawn forks (these match
         ``--multiprocessing-fork`` in cmdline).
      5. fuser -k each vLLM port — final port-level kill for any
         straggler still holding the listen socket.
    """
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
    # Layers 2-4: pkill by cmdline pattern.  Each pattern targets
    # one part of the vLLM process tree that survives the previous
    # layer.  ``check=False`` so missing patterns are silent (pkill
    # exits 1 when nothing matched).
    for pattern in (
        "vllm.entrypoints",       # main API server
        "VLLM::EngineCore",       # v1 engine core daemon
        "multiprocessing-fork",   # mp.spawn worker children
    ):
        subprocess.run(["pkill", "-9", "-f", pattern],
                       capture_output=True, check=False)
    # Layer 5: fuser-kill every vLLM port — last-resort port-level kill.
    # Wrapped in FileNotFoundError because minimal containers ship
    # without ``psmisc`` (no ``fuser``); we don't want a missing
    # binary to crash the iter.
    for port in range(7999, 8007):  # 7999..8006 inclusive (8 GPUs)
        try:
            subprocess.run(
                ["fuser", "-k", "-9", f"{port}/tcp"],
                capture_output=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    time.sleep(3)


def wait_gpu_idle(timeout_s: int = 60,
                  idle_threshold_mib: int = 1024) -> bool:
    """Poll nvidia-smi until every GPU's used memory drops below
    ``idle_threshold_mib`` (1 GB by default).  Returns True if all
    GPUs go idle within ``timeout_s`` seconds, False otherwise.

    Why we need this on top of ``stop_vllm``: a SIGKILL'd vLLM
    typically releases its 73 GB of GPU memory within ~5 s (CUDA
    runtime tears down the context on process exit), but on rare
    occasions — under heavy NCCL contention or when the kernel is
    swapped — the cleanup can take 30 s+.  Starting a DDP trainer
    while a vLLM is still letting go of memory triggers OOM at
    weight load time.  Calling ``wait_gpu_idle`` before the
    trainer starts converts that silent OOM into a clean wait.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return True              # no nvidia-smi → assume non-GPU host, OK
        busy = [int(line.strip()) for line in out.strip().split("\n")
                if line.strip().isdigit()]
        if all(m < idle_threshold_mib for m in busy):
            return True
        time.sleep(2)
    return False


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


def _vllm_cmd(gpu: int, port: int, model_path: str,
              enable_lora: bool) -> str:
    """Build the shell command line for one vLLM API server instance.

    Pulled out of ``ensure_vllm`` so we can re-use it from the
    per-GPU retry loop without copy-pasting (and risking divergence).

    Layout knobs explained inline.  See ``ensure_vllm`` for the
    cluster-level rationale (sequential launch, retry-on-fail, etc).
    """
    if enable_lora:
        lora_args = "--enable-lora --max-loras 4 --max-lora-rank 64 "
        env_prefix = "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True "
    else:
        lora_args = ""
        env_prefix = ""
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} {env_prefix}{VLLM_PYTHON} "
        f"-m vllm.entrypoints.openai.api_server "
        f"--model {model_path} --port {port} "
        # ``--max-model-len 32768`` controls both KV cache budget AND
        # cudagraph capture window: vLLM 0.19+ removed the separate
        # ``--max-seq-len-to-capture`` flag (PR #25543) and now uses
        # ``max_model_len`` directly.  Our multi-turn sessions hit
        # 15-23K tokens of accumulated context, so 32768 keeps every
        # sequence inside cudagraph mode (else throughput drops 5-10×).
        f"--max-model-len 32768 --gpu-memory-utilization 0.90 "
        # ``prefetch`` mmaps the whole safetensors into RAM up-front,
        # turning many small NFS reads into one big sequential read.
        # vLLM 0.19+ does this automatically on NFS (PR #37673) but
        # NOT on GPFS (auto-detect only matches `nfs`/`nfs4` mounts),
        # so we set it explicitly — it's a free 30-50% cold-start
        # speedup on shared filesystems.
        f"--safetensors-load-strategy prefetch "
        f"--enable-auto-tool-choice --tool-call-parser hermes "
        f"{lora_args}"
        f"--dtype auto 2>&1 | tee /tmp/vllm_gpu{gpu}.log"
    )


def _start_one_vllm(gpu: int, model_path: str,
                    enable_lora: bool) -> int:
    """Start one vLLM tmux session.  Returns its port (idempotent)."""
    port = gpu_to_port(gpu)
    session_name = f"{TMUX_PREFIX}{gpu}"
    subprocess.run(["tmux", "kill-session", "-t", session_name],
                   capture_output=True)
    # Make sure the tmux's vLLM gets a clean port too (the previous
    # vLLM on this GPU may still be in TIME_WAIT briefly).
    kill_port(port)
    cmd = _vllm_cmd(gpu, port, model_path, enable_lora)
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd],
                   capture_output=True)
    return port


def ensure_vllm(gpus: list[int], model_path: str = MODEL_PATH,
                enable_lora: bool = False,
                per_gpu_timeout_s: int = 1800,
                per_gpu_retries: int = 2) -> list[int]:
    """Bring up one vLLM API per GPU in this pod, *sequentially with retry*.

    Layout = sequential within a pod
    --------------------------------
    All 8 vLLM processes in one pod read the same 19 GB Qwen3.5-9B
    safetensors. Launching concurrently makes 8 vLLMs all hit the
    NFS / GPFS at once before the OS page cache fills — every vLLM
    ends up waiting on its own disk traffic and a 4-pod parallel
    launch can blow past the wait-ready timeout.

    Sequential fixes this almost for free: GPU 0 pays the full
    cold cost (~30-90 s on /pfs/ GPFS now that prefetch is on, see
    ``_vllm_cmd``); once its weights are in the kernel page cache,
    GPUs 1-7 hit warm cache + their own torch-compile cache and
    boot in well under a minute each.

    Why retry per GPU
    -----------------
    The 2026-04-26 main run hit a single GPU 1 cold-start hang on
    one pod (vLLM never came up within 1800 s on that one card,
    likely a transient GPU driver / GPFS hiccup).  The old code
    treated this as fatal and tore down the whole experiment — six
    iters of work lost to a 30-min flake.  Following the pattern
    used by vLLM's own ``multi-node-serving.sh`` and Ray actor
    fault-tolerance in OpenRLHF/verl, we now retry the failing
    GPU up to ``per_gpu_retries`` times before giving up.  Each
    retry kills that single card's tmux + frees its port + starts
    a fresh vLLM there, so the rest of the fleet stays alive.

    LoRA-runtime support (``enable_lora=True``, off by default):
    See ``_vllm_cmd``. iterate.py uses the merge+reload pipeline
    (``rl.export`` produces a vLLM-loadable HF folder per iter,
    vLLM is restarted with that folder as ``--model``).
    """
    ports: list[int] = []
    for i, gpu in enumerate(gpus):
        port = gpu_to_port(gpu)
        ports.append(port)
        url = f"http://127.0.0.1:{port}/v1/models"
        kind = "cold" if i == 0 else "warm"

        for attempt in range(per_gpu_retries + 1):  # initial + retries
            attempt_label = (
                f"attempt {attempt + 1}/{per_gpu_retries + 1}"
                if per_gpu_retries > 0 else "single attempt"
            )
            _start_one_vllm(gpu, model_path, enable_lora)
            print(f"  GPU {gpu} → :{port} launching"
                  f" ({kind}, timeout={per_gpu_timeout_s}s, {attempt_label})...",
                  flush=True)

            if wait_ready(url, timeout=per_gpu_timeout_s):
                print(f"  GPU {gpu} → :{port} ready", flush=True)
                break

            print(f"  GPU {gpu} → :{port} FAILED at {attempt_label} "
                  f"(after {per_gpu_timeout_s}s wait)")
            # Tear down just this one card and let the loop retry.
            session_name = f"{TMUX_PREFIX}{gpu}"
            subprocess.run(["tmux", "kill-session", "-t", session_name],
                           capture_output=True)
            kill_port(port)
            if attempt >= per_gpu_retries:
                # Out of retries → fail the whole fleet.
                print(f"  GPU {gpu} exhausted retries; tearing down "
                      f"every other vLLM and aborting.")
                stop_vllm()
                sys.exit(1)
            print(f"  GPU {gpu} retrying in 5 s...", flush=True)
            time.sleep(5)

    print(f"  All {len(ports)} vLLM instances ready"
          f"{' (with LoRA + sleep mode)' if enable_lora else ''}")
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
    finally:
        # Always tear down both Parliament and vLLM at the end of an iter,
        # whether the experiment succeeded or failed.  An earlier version
        # left Parliament running on success "for the web UI" — but in the
        # iterate.py loop the next iter immediately restarts Parliament,
        # and any leftover instance from the previous iter would race the
        # new bind on 0.0.0.0:8080 and crash the next iter (observed in
        # smokeA Iter 3).  Tearing down unconditionally is the only safe
        # contract when the same port is reused across iters.
        stop_parliament(parliament_proc)
        stop_vllm()

    print(f"\n  Experiment finished.")
    print(f"  Parliament:    stopped")
    print(f"  vLLM:          stopped")
    print(f"  Output:        {run_dir}")


if __name__ == "__main__":
    main()
