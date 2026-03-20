"""
Science Parliament — Baseline runner (no tools, single-turn).

Direct model inference as a control group.
One command does everything: start vLLM → batch inference → generate results page.

Usage:
    cd baseline
    python run_baseline.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0
    python run_baseline.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0,1,2 --limit 20
"""

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "parliament"))
sys.path.insert(0, os.path.join(_project_root, "judgement"))

from session import OUTPUT_BASE
from dataset import load_dataset, parse_gpu_ids
from judge import extract_answer
import vllm_manager
import benchmark_viz

BASELINE_SYSTEM_PROMPT = """\
You are a scientist solving a difficult problem. Think step by step.

Select the single best answer from the provided choices.

You MUST end your response with this exact block:

<<<FINAL>>>
(X)
<<<END>>>

where X is the letter of the correct choice (A, B, C, or D). \
Nothing else should appear between the markers.\
"""

BASELINE_SYSTEM_PROMPT_OPEN = """\
You are a scientist solving a difficult problem. Think step by step.

You MUST end your response with this exact block:

<<<FINAL>>>
your final answer here
<<<END>>>

The answer between <<<FINAL>>> and <<<END>>> must be self-contained.\
"""


def _build_prompt(question: str, choices: list[str] | None) -> str:
    parts = [question]
    if choices:
        parts.append("\nCHOICES:")
        for i, ch in enumerate(choices):
            parts.append(f"  ({chr(ord('A') + i)}) {ch}")
    return "\n".join(parts)


async def _run_batch(questions: list[dict], ports: list[int], concurrency: int = 32):
    import httpx
    from config import MODEL_NAME

    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(questions)

    async def _infer(idx: int, q: dict, client: httpx.AsyncClient, port: int):
        async with semaphore:
            choices = q.get("choices")
            system = BASELINE_SYSTEM_PROMPT if choices else BASELINE_SYSTEM_PROMPT_OPEN
            user_msg = _build_prompt(q["question"], choices)

            try:
                resp = await client.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        "model": MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_msg},
                        ],
                        "max_tokens": 4096,
                    },
                    timeout=300,
                )
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                answer = extract_answer(raw_text)
            except Exception as e:
                print(f"  Question {idx} error: {e}")
                raw_text = ""
                answer = None

            gt = q.get("ground_truth")
            is_correct = None
            if gt is not None and answer is not None:
                is_correct = answer.strip("() ").upper() == str(gt).strip("() ").upper()

            results[idx] = {
                "index": idx,
                "question": q["question"][:500],
                "choices": q.get("choices"),
                "ground_truth": gt,
                "parliament_answer": answer,
                "is_correct": is_correct,
                "rounds_completed": 0,
                "early_stopped": False,
                "gpu": port,
            }

            status = "ok" if is_correct else ("WRONG" if is_correct is False else "?")
            print(f"  [{status}] Question {idx}: answer={answer}")

    async with httpx.AsyncClient() as client:
        tasks = []
        for idx, q in enumerate(questions):
            port = ports[idx % len(ports)]
            tasks.append(_infer(idx, q, client, port))
        await asyncio.gather(*tasks)

    return results


def run_baseline(
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
        base = os.path.splitext(os.path.basename(dataset_path))[0]
        bench_name = f"baseline_{base}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bench_dir = os.path.join(OUTPUT_BASE, bench_name, timestamp)
    os.makedirs(bench_dir, exist_ok=True)

    ports = [8000 + g for g in gpu_ids]

    print("\n[1/3] Starting vLLM instances...")
    vllm_manager.start(gpu_ids)
    vllm_manager.wait_ready(gpu_ids)

    print(f"\n[2/3] Running baseline: {len(questions)} questions, {len(gpu_ids)} GPUs")
    results = asyncio.run(_run_batch(questions, ports))

    print("\n[3/3] Generating results...")
    results_path = os.path.join(bench_dir, "results.jsonl")
    correct = total = 0
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            if r is None:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += 1
            if r.get("is_correct") is True:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    summary = {
        "dataset": dataset_path, "bench_name": bench_name,
        "gpu_ids": gpu_ids, "total": total, "correct": correct,
        "accuracy": accuracy, "timestamp": timestamp,
    }
    with open(os.path.join(bench_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    html_path = benchmark_viz.generate(bench_dir, summary)

    print(f"\n{'='*70}")
    print(f"BASELINE COMPLETE")
    print(f"  Accuracy:  {correct}/{total} = {accuracy:.1%}")
    print(f"  Results:   {results_path}")
    print(f"  Dashboard: {html_path}")
    print(f"{'='*70}")

    print(f"\n  Serving at http://localhost:{serve_port}/")
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
    parser = argparse.ArgumentParser(description="Science Parliament Baseline (no tools)")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPU IDs: '0', '0,1,2', '0-7' (default: 0)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--name", default=None, help="Run name")
    parser.add_argument("--port", type=int, default=18888, help="HTTP port for results page")
    args = parser.parse_args()
    run_baseline(
        args.dataset, gpu_ids=parse_gpu_ids(args.gpus),
        limit=args.limit, bench_name=args.name, serve_port=args.port,
    )


if __name__ == "__main__":
    main()
