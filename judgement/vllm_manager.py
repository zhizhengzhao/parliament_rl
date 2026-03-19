"""Manage vLLM serving instances — start, wait, stop."""

import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "parliament"))
from config import MODEL_NAME, VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTILIZATION


def _port_for(gpu_id: int) -> int:
    return 8000 + gpu_id


def _is_ready(port: int) -> bool:
    try:
        r = urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
        return r.status == 200
    except Exception:
        return False


def start(gpu_ids: list[int], model_path: str | None = None):
    """Launch one vLLM instance per GPU.  Skips GPUs whose port is already serving."""
    model = model_path or MODEL_NAME
    pids = []
    for gid in gpu_ids:
        port = _port_for(gid)
        if _is_ready(port):
            print(f"  GPU {gid} (port {port}): already running, skipping")
            continue
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gid} nohup vllm serve {model} "
            f"--port {port} "
            f"--tensor-parallel-size 1 "
            f"--max-model-len {VLLM_MAX_MODEL_LEN} "
            f"--gpu-memory-utilization {VLLM_GPU_MEMORY_UTILIZATION} "
            f"--reasoning-parser qwen3 "
            f"--enable-auto-tool-choice "
            f"--tool-call-parser qwen3_coder "
            f"> vllm_gpu{gid}.log 2>&1 &"
        )
        subprocess.run(cmd, shell=True)
        print(f"  GPU {gid} (port {port}): starting")
    return [_port_for(g) for g in gpu_ids]


def wait_ready(gpu_ids: list[int], timeout: int = 300):
    """Block until all vLLM instances respond, or raise after timeout."""
    ports = [_port_for(g) for g in gpu_ids]
    deadline = time.time() + timeout
    pending = set(ports)
    while pending and time.time() < deadline:
        for port in list(pending):
            if _is_ready(port):
                pending.discard(port)
                print(f"  port {port}: ready")
        if pending:
            time.sleep(3)
    if pending:
        raise RuntimeError(f"vLLM instances not ready after {timeout}s: ports {pending}")
    print(f"  All {len(ports)} vLLM instances ready.")


def stop():
    """Kill all vLLM serve processes."""
    subprocess.run("pkill -f 'vllm serve'", shell=True)
    print("  All vLLM instances stopped.")
