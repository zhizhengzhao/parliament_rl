#!/usr/bin/env python3
"""Resource keepalive — keep idle GPUs / CPU / RAM / disk above target floors.

For shared clusters that enforce minimum hardware utilization (e.g.
Kuaishou KML, which audits GPU + CPU + memory + disk-IO).  Each
resource gets its own subprocess; SIGTERM/SIGINT releases everything
within ~1 s so a real training job can take over instantly.

Defaults are tuned to comfortably pass typical KML thresholds
(~50% across the board) without starving real co-tenants:
    GPUs  : ~65% util, ~75% memory  (FP16 matmul on tensor cores)
    CPU   : ~50% of cores at ~70% util
    RAM   : ~50% of free memory pinned
    Disk  : ~30 MB/s steady write+read on $TMPDIR or --disk-path

Usage
-----
    python scripts/gpu_keepalive.py                       # all defaults
    python scripts/gpu_keepalive.py --gpus 0,1,4,5        # subset of GPUs
    python scripts/gpu_keepalive.py --no-disk --no-cpu    # only GPU+RAM
    python scripts/gpu_keepalive.py --gpu-util 0.4        # gentler GPU
    python scripts/gpu_keepalive.py --disk-path /m2v_intern/.../keep \\
        --disk-mb-s 50

    # background + log (recommended in tmux on the KML pod)
    nohup python scripts/gpu_keepalive.py > /tmp/keep.log 2>&1 &
    disown

Stop
----
    pkill -f gpu_keepalive       # SIGTERM, all resources free in <1 s
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time

import torch


# ── GPU worker ───────────────────────────────────────────

def gpu_worker(gpu_id: int, mem_frac: float, util: float) -> None:
    """Per-GPU loop: allocate three big square FP16 tensors, matmul forever."""
    torch.cuda.set_device(gpu_id)
    free, total = torch.cuda.mem_get_info(gpu_id)
    target = int(total * mem_frac)
    n = int(((target / 6) ** 0.5) // 256 * 256)
    n = max(n, 1024)
    matmul_ms = 10.0
    sleep_ms = max(0.0, matmul_ms * (1.0 - util) / max(util, 0.01))
    print(f"[gpu{gpu_id}] N={n}  mem≈{(3 * n * n * 2) / 1e9:.1f} GB  "
          f"sleep={sleep_ms:.1f} ms", flush=True)

    a = torch.randn(n, n, device=f"cuda:{gpu_id}", dtype=torch.float16)
    b = torch.randn(n, n, device=f"cuda:{gpu_id}", dtype=torch.float16)
    c = torch.empty(n, n, device=f"cuda:{gpu_id}", dtype=torch.float16)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    last_log = time.time()
    iters = 0
    while not stop["flag"]:
        torch.matmul(a, b, out=c)
        if sleep_ms > 0:
            torch.cuda.synchronize(gpu_id)
            time.sleep(sleep_ms / 1000.0)
        iters += 1
        if time.time() - last_log > 60:
            print(f"[gpu{gpu_id}] {iters} matmuls in last 60 s", flush=True)
            iters, last_log = 0, time.time()

    del a, b, c
    torch.cuda.empty_cache()
    print(f"[gpu{gpu_id}] released", flush=True)


# ── CPU worker ───────────────────────────────────────────

def cpu_worker(worker_id: int, util: float) -> None:
    """Pure-Python tight loop with calibrated sleeps to hit `util` per core."""
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    work_ms, sleep_ms = 50.0, 50.0 * (1.0 - util) / max(util, 0.01)
    print(f"[cpu{worker_id}] util≈{int(util * 100)}% "
          f"({work_ms:.0f}ms work / {sleep_ms:.0f}ms sleep)", flush=True)
    while not stop["flag"]:
        t0 = time.perf_counter()
        x = 1.0001
        while (time.perf_counter() - t0) * 1000 < work_ms:
            x *= 1.0000001
            if x > 1e10:
                x = 1.0001
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
    print(f"[cpu{worker_id}] released", flush=True)


# ── RAM worker ───────────────────────────────────────────

def ram_worker(mb: int) -> None:
    """Allocate `mb` MB of RAM and touch every 4 KB page once a minute so
    the kernel keeps it resident (no swap)."""
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    print(f"[ram] allocating {mb} MB…", flush=True)
    buf = bytearray(mb * 1024 * 1024)
    for off in range(0, len(buf), 4096):
        buf[off] = 1
    print(f"[ram] {mb} MB pinned", flush=True)
    while not stop["flag"]:
        time.sleep(60)
        for off in range(0, len(buf), 4096 * 256):
            buf[off] = (buf[off] + 1) & 0xff
    del buf
    print("[ram] released", flush=True)


# ── Disk worker ──────────────────────────────────────────

def disk_worker(path: str, mb_per_s: float, file_mb: int = 256) -> None:
    """Continuously write and re-read a rotating temp file at `mb_per_s`."""
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    os.makedirs(path, exist_ok=True)
    fp = os.path.join(path, "keepalive.bin")
    chunk = 1024 * 1024
    buf = os.urandom(chunk)
    sleep_s = 1.0 / mb_per_s
    print(f"[disk] {path} target ≈ {mb_per_s} MB/s, file {file_mb} MB", flush=True)
    while not stop["flag"]:
        with open(fp, "wb") as f:
            for _ in range(file_mb):
                if stop["flag"]:
                    break
                f.write(buf)
                f.flush()
                os.fsync(f.fileno())
                time.sleep(sleep_s)
        with open(fp, "rb") as f:
            while not stop["flag"] and f.read(chunk):
                time.sleep(sleep_s)
    try:
        os.remove(fp)
    except FileNotFoundError:
        pass
    print("[disk] released", flush=True)


# ── Main ─────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default="",
                   help="Comma-separated GPU ids; default = all visible")
    p.add_argument("--gpu-util", type=float, default=0.65,
                   help="Target GPU utilization 0..1 (default 0.65)")
    p.add_argument("--gpu-mem-frac", type=float, default=0.75,
                   help="Fraction of GPU memory to occupy (default 0.75)")
    p.add_argument("--cpu-util", type=float, default=0.70,
                   help="Per-core CPU utilization (default 0.70)")
    p.add_argument("--cpu-frac", type=float, default=0.5,
                   help="Fraction of CPU cores to keep busy (default 0.5)")
    p.add_argument("--ram-frac", type=float, default=0.5,
                   help="Fraction of *free* RAM to pin (default 0.5)")
    p.add_argument("--disk-path", default="",
                   help="Where to write/read the keepalive file. "
                        "Default = $TMPDIR or /tmp/keepalive")
    p.add_argument("--disk-mb-s", type=float, default=30.0,
                   help="Sustained disk throughput in MB/s (default 30)")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--no-cpu", action="store_true")
    p.add_argument("--no-ram", action="store_true")
    p.add_argument("--no-disk", action="store_true")
    args = p.parse_args()

    procs: list[mp.Process] = []
    ctx = mp.get_context("spawn")

    if not args.no_gpu:
        if args.gpus:
            gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
        else:
            gpus = list(range(torch.cuda.device_count()))
        for g in gpus:
            procs.append(ctx.Process(target=gpu_worker,
                                     args=(g, args.gpu_mem_frac, args.gpu_util),
                                     daemon=False))
        print(f"GPUs   : {gpus}, util≈{int(args.gpu_util * 100)}%, "
              f"mem≈{int(args.gpu_mem_frac * 100)}%", flush=True)

    if not args.no_cpu:
        n_cores = max(1, int(os.cpu_count() * args.cpu_frac))
        for i in range(n_cores):
            procs.append(ctx.Process(target=cpu_worker, args=(i, args.cpu_util),
                                     daemon=False))
        print(f"CPU    : {n_cores}/{os.cpu_count()} cores at "
              f"≈{int(args.cpu_util * 100)}% util", flush=True)

    if not args.no_ram:
        try:
            with open("/proc/meminfo") as f:
                meminfo = {l.split(":")[0]: int(l.split()[1])
                           for l in f if l.split(":")[0] in
                           ("MemTotal", "MemAvailable")}
            free_mb = meminfo["MemAvailable"] // 1024
        except Exception:
            free_mb = 4096
        mb = int(free_mb * args.ram_frac)
        procs.append(ctx.Process(target=ram_worker, args=(mb,), daemon=False))
        print(f"RAM    : {mb} MB ({int(args.ram_frac * 100)}% of free)",
              flush=True)

    if not args.no_disk:
        disk_path = args.disk_path or os.environ.get(
            "TMPDIR", "/tmp/keepalive")
        procs.append(ctx.Process(target=disk_worker,
                                 args=(disk_path, args.disk_mb_s),
                                 daemon=False))
        print(f"Disk   : {disk_path} @ ≈{args.disk_mb_s} MB/s", flush=True)

    if not procs:
        print("Nothing to do (all --no-* set).", file=sys.stderr)
        sys.exit(1)

    for proc in procs:
        proc.start()

    def _shutdown(signum, frame):
        print(f"\nReceived signal {signum}, releasing all resources…",
              flush=True)
        for proc in procs:
            if proc.is_alive():
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    for proc in procs:
        proc.join()
    print("All workers exited.", flush=True)


if __name__ == "__main__":
    main()
