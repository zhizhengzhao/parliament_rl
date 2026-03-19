"""
Science Parliament — Benchmark runner (multi-GPU).

Each GPU runs its own worker process.  Workers pull questions from a shared
queue — fast GPUs automatically get more work, no long-tail waiting.

Usage:
    cd judgement
    python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv                    # 8 GPUs
    python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0           # single GPU 0
    python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0,2,4,6     # specific GPUs
    python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0-3         # GPUs 0-3

Prerequisites:
    GPU k → vLLM on port (8000 + k).  Use launch_vllm.sh to start them.
"""

import argparse
import asyncio
import csv
import json
import multiprocessing as mp
import os
import random
import sys
from datetime import datetime

_parliament_dir = os.path.join(os.path.dirname(__file__), "..", "parliament")
sys.path.insert(0, os.path.abspath(_parliament_dir))

from session import OUTPUT_BASE, LOG_BASE


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [_normalize(json.loads(l)) for l in f if l.strip()]
    elif ext in (".csv", ".tsv"):
        with open(path, "r", encoding="utf-8") as f:
            return [_normalize(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported format: {ext}")


def _normalize(raw: dict) -> dict:
    question = (
        raw.get("question") or raw.get("Question")
        or raw.get("problem") or raw.get("Problem") or ""
    ).strip()

    choices = raw.get("choices")
    if choices is not None:
        if isinstance(choices, str):
            choices = json.loads(choices)
        return {"question": question, "choices": choices,
                "ground_truth": raw.get("answer") or raw.get("Answer")}

    correct = raw.get("Correct Answer") or raw.get("correct_answer")
    incorrects = [raw.get(k) for k in
                  ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
                  if raw.get(k)]
    if correct and incorrects:
        all_choices = [correct] + incorrects
        random.shuffle(all_choices)
        letter = chr(ord("A") + all_choices.index(correct))
        return {"question": question, "choices": all_choices, "ground_truth": letter}

    return {"question": question, "choices": None,
            "ground_truth": raw.get("answer") or raw.get("Answer")}


# ---------------------------------------------------------------------------
# Worker process — one per GPU, completely independent
# ---------------------------------------------------------------------------

def _worker_main(
    worker_id: int,
    port: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    bench_dir: str,
    log_dir: str,
):
    """Entry point for each worker process.  Runs in its own address space."""

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
        db_path = None
        answer = None

        try:
            session = asyncio.run(run_session(
                question=question_text, model=model, output_dir=run_dir,
            ))
            db_path = session["db_path"]
            rounds_completed = session["num_rounds_completed"]
            early_stopped = session["early_stopped"]
        except Exception as e:
            print(f"[GPU {worker_id}] Question {idx} parliament error: {e}")

        if db_path and os.path.exists(db_path):
            try:
                judge_result = asyncio.run(run_judge(
                    db_path=db_path, question=question_text,
                    model=model, choices=choices, output_dir=run_dir,
                ))
                answer = judge_result["answer"]
            except Exception as e:
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
            "gpu": worker_id,
        }
        result_queue.put(record)

        status = "ok" if is_correct else ("WRONG" if is_correct is False else "?")
        print(f"[GPU {worker_id}] Question {idx} [{status}] answer={answer}")


# ---------------------------------------------------------------------------
# Main process — distribute tasks, collect results
# ---------------------------------------------------------------------------

def _parse_gpu_ids(gpu_ids_str: str) -> list[int]:
    """Parse '0,1,2,3' or '0-7' into a list of GPU IDs."""
    if "-" in gpu_ids_str and "," not in gpu_ids_str:
        start, end = gpu_ids_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in gpu_ids_str.split(",")]


def run_benchmark(
    dataset_path: str,
    gpu_ids: list[int] | None = None,
    limit: int | None = None,
    bench_name: str | None = None,
):
    if gpu_ids is None:
        gpu_ids = list(range(8))

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

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for idx, q in enumerate(questions):
        task_queue.put((idx, q))

    ports = [8000 + gid for gid in gpu_ids]
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {len(questions)} questions, {len(gpu_ids)} GPUs ({gpu_ids})")
    print(f"Dataset:   {dataset_path}")
    print(f"Output:    {bench_dir}")
    print(f"Ports:     {ports}")
    print(f"{'='*70}\n")

    workers = []
    for gid in gpu_ids:
        port = 8000 + gid
        p = mp.Process(
            target=_worker_main,
            args=(gid, port, task_queue, result_queue, bench_dir, log_dir),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # Collect results
    results_path = os.path.join(bench_dir, "results.jsonl")
    correct = 0
    total = 0
    with open(results_path, "w", encoding="utf-8") as f:
        while not result_queue.empty():
            record = result_queue.get()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            if record.get("is_correct") is True:
                correct += 1

    accuracy = correct / total if total > 0 else 0
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

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"  Results:  {results_path}")
    print(f"{'='*70}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Science Parliament Benchmark")
    parser.add_argument("--dataset", required=True, help="Path to dataset (CSV/JSONL)")
    parser.add_argument("--gpus", type=str, default="0-7",
                        help="GPU IDs: '0,1,2,3' or '0-7' or '0' (default: 0-7)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--name", default=None, help="Benchmark name (default: filename)")
    args = parser.parse_args()
    run_benchmark(
        args.dataset, gpu_ids=_parse_gpu_ids(args.gpus),
        limit=args.limit, bench_name=args.name,
    )


if __name__ == "__main__":
    main()
