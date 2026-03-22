"""
Science Parliament — Benchmark runner.

One command does everything: start vLLM → run parliament + judge → generate results page.

Usage:
    cd judgement
    python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0,1,2 --limit 6
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import random
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

_parliament_dir = os.path.join(os.path.dirname(__file__), "..", "parliament")
sys.path.insert(0, os.path.abspath(_parliament_dir))

from session import OUTPUT_BASE, LOG_BASE
from dataset import load_dataset, parse_gpu_ids
import vllm_manager
import benchmark_viz


# ---------------------------------------------------------------------------
# Worker process — one per GPU
# ---------------------------------------------------------------------------

def _worker_main(
    worker_id: int, port: int,
    task_queue: mp.Queue,
    bench_dir: str, log_dir: str,
):
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "parliament")
    ))
    from session import init, create_model, run_session
    from judge import run_judge

    init(log_dir=os.path.join(log_dir, f"gpu{worker_id}"))
    model = create_model(base_url=f"http://localhost:{port}/v1")

    while True:
        try:
            idx, q = task_queue.get_nowait()
        except Exception:
            break

        question_text = q["question"]
        choices = q.get("choices")
        ground_truth = q.get("ground_truth")
        run_dir = os.path.join(bench_dir, str(idx))

        print(f"[GPU {worker_id}] Question {idx} starting...")

        rounds_completed = 0
        early_stopped = False
        stop_reason = "unknown"
        db_path = None
        answer = None
        parliament_error = None
        judge_error = None

        try:
            session = asyncio.run(run_session(
                question=question_text, model=model, output_dir=run_dir,
            ))
            db_path = session["db_path"]
            rounds_completed = session["num_rounds_completed"]
            early_stopped = session["early_stopped"]
            stop_reason = session.get("stop_reason", "max_rounds")
        except Exception as e:
            parliament_error = str(e)
            print(f"[GPU {worker_id}] Question {idx} parliament error: {e}")

        if db_path and os.path.exists(db_path):
            try:
                judge_result = asyncio.run(run_judge(
                    db_path=db_path, question=question_text,
                    model=model, choices=choices, output_dir=run_dir,
                ))
                answer = judge_result["answer"]
            except Exception as e:
                judge_error = str(e)
                print(f"[GPU {worker_id}] Question {idx} judge error: {e}")

        is_correct = None
        if ground_truth is not None and answer is not None:
            is_correct = answer.strip("() ").upper() == str(ground_truth).strip("() ").upper()

        record = {
            "index": idx,
            "question": question_text[:500],
            "choices": choices,
            "ground_truth": ground_truth,
            "parliament_answer": answer,
            "is_correct": is_correct,
            "rounds_completed": rounds_completed,
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
            "gpu": worker_id,
            "parliament_error": parliament_error,
            "judge_error": judge_error,
        }

        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        status = "ok" if is_correct else ("WRONG" if is_correct is False else "?")
        print(f"[GPU {worker_id}] Question {idx} [{status}] answer={answer}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_path: str,
    gpu_ids: list[int],
    limit: int | None = None,
    bench_name: str | None = None,
    serve_port: int = 18888,
):
    questions = load_dataset(dataset_path)
    random.shuffle(questions)
    if limit:
        questions = questions[:limit]

    if bench_name is None:
        bench_name = os.path.splitext(os.path.basename(dataset_path))[0]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bench_dir = os.path.join(OUTPUT_BASE, bench_name, timestamp)
    log_dir = os.path.join(LOG_BASE, f"{bench_name}_{timestamp}")
    os.makedirs(bench_dir, exist_ok=True)

    # ── Step 1: Start vLLM ──────────────────────────────────────────────
    print("\n[1/4] Starting vLLM instances...")
    vllm_manager.start(gpu_ids)
    vllm_manager.wait_ready(gpu_ids)

    # ── Step 2: Run parliament + judge ──────────────────────────────────
    print(f"\n[2/4] Running benchmark: {len(questions)} questions, {len(gpu_ids)} GPUs")
    task_queue = mp.Queue()
    for idx, q in enumerate(questions):
        task_queue.put((idx, q))

    workers = []
    for gid in gpu_ids:
        p = mp.Process(
            target=_worker_main,
            args=(gid, 8000 + gid, task_queue, bench_dir, log_dir),
        )
        p.start()
        workers.append(p)
    for p in workers:
        p.join()

    # ── Step 3: Collect results from disk ────────────────────────────────
    print("\n[3/4] Collecting results...")
    records = []
    for idx in range(len(questions)):
        result_path = os.path.join(bench_dir, str(idx), "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                records.append(json.load(f))

    records.sort(key=lambda r: r.get("index", 0))
    correct = sum(1 for r in records if r.get("is_correct") is True)
    total = len(records)
    accuracy = correct / total if total > 0 else 0

    results_path = os.path.join(bench_dir, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "dataset": dataset_path,
        "bench_name": bench_name,
        "gpu_ids": gpu_ids,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "timestamp": timestamp,
    }
    with open(os.path.join(bench_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Step 4: Generate overview HTML + serve ──────────────────────────
    print("\n[4/4] Generating results page...")
    html_path = benchmark_viz.generate(bench_dir, summary)

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"  Accuracy:  {correct}/{total} = {accuracy:.1%}")
    print(f"  Results:   {results_path}")
    print(f"  Dashboard: {html_path}")
    print(f"{'='*70}")

    print(f"\n  Serving results at http://localhost:{serve_port}/")
    print(f"  SSH tunnel: ssh -p 8795 -L {serve_port}:localhost:{serve_port} root@your-server")
    print(f"  Press Ctrl+C to stop.\n")

    os.chdir(bench_dir)
    server = HTTPServer(("", serve_port), type("H", (SimpleHTTPRequestHandler,),
                         {"log_message": lambda *a: None}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Science Parliament Benchmark")
    parser.add_argument("--dataset", required=True, help="Path to dataset (CSV/JSONL)")
    parser.add_argument("--gpus", type=str, default="0-7",
                        help="GPU IDs: '0,1,2' or '0-7' or '0' (default: 0-7)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--name", default=None, help="Benchmark name (default: filename)")
    parser.add_argument("--port", type=int, default=18888, help="HTTP port for results page")
    args = parser.parse_args()
    run_benchmark(
        args.dataset, gpu_ids=parse_gpu_ids(args.gpus),
        limit=args.limit, bench_name=args.name, serve_port=args.port,
    )


if __name__ == "__main__":
    main()
