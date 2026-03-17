"""
Science Parliament — Benchmark runner.

Runs parliament + judge on a dataset of questions and records results.

Usage:
    cd judgement
    python run_benchmark.py --dataset ../benchmark/gpqa_diamond.csv --limit 10
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from datetime import datetime

# Add parliament/ to path for config, session, patches, etc.
_parliament_dir = os.path.join(os.path.dirname(__file__), "..", "parliament")
sys.path.insert(0, os.path.abspath(_parliament_dir))

from session import OUTPUT_BASE, LOG_BASE, init, create_model, run_session
from judge import run_judge


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """Load questions from CSV or JSONL.

    Returns list of dicts with keys: question, choices (list|None), ground_truth.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return _load_jsonl(path)
    elif ext in (".csv", ".tsv"):
        return _load_csv(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_normalize(json.loads(line)))
    return records


def _load_csv(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(_normalize(row))
    return records


def _normalize(raw: dict) -> dict:
    """Map various column names to a standard format."""
    question = (
        raw.get("question") or raw.get("Question")
        or raw.get("problem") or raw.get("Problem") or ""
    ).strip()

    choices = raw.get("choices")
    if choices is not None:
        if isinstance(choices, str):
            choices = json.loads(choices)
        answer_letter = raw.get("answer") or raw.get("Answer")
        return {"question": question, "choices": choices, "ground_truth": answer_letter}

    # GPQA format: Correct Answer + Incorrect Answer 1/2/3
    correct = raw.get("Correct Answer") or raw.get("correct_answer")
    incorrects = [
        raw.get(k) for k in
        ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
        if raw.get(k)
    ]

    if correct and incorrects:
        all_choices = [correct] + incorrects
        random.shuffle(all_choices)
        answer_letter = chr(ord("A") + all_choices.index(correct))
        return {"question": question, "choices": all_choices, "ground_truth": answer_letter}

    return {
        "question": question,
        "choices": None,
        "ground_truth": raw.get("answer") or raw.get("Answer"),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_benchmark(
    dataset_path: str,
    limit: int | None = None,
    bench_name: str | None = None,
):
    questions = load_dataset(dataset_path)
    if limit:
        questions = questions[:limit]

    if bench_name is None:
        bench_name = os.path.splitext(os.path.basename(dataset_path))[0]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bench_dir = os.path.join(OUTPUT_BASE, bench_name, timestamp)
    log_dir = os.path.join(LOG_BASE, f"{bench_name}_{timestamp}")

    os.makedirs(bench_dir, exist_ok=True)

    init(log_dir=log_dir)
    model = create_model()

    results_path = os.path.join(bench_dir, "results.jsonl")
    correct = 0
    total = 0

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {len(questions)} questions from {dataset_path}")
    print(f"Output → {bench_dir}")
    print(f"{'='*70}\n")

    for idx, q in enumerate(questions):
        question_text = q["question"]
        choices = q.get("choices")
        ground_truth = q.get("ground_truth")

        print(f"\n{'─'*70}")
        print(f"Question {idx + 1}/{len(questions)}")
        print(f"{'─'*70}")
        preview = question_text[:200] + ("..." if len(question_text) > 200 else "")
        print(f"  {preview}\n")

        run_dir = os.path.join(bench_dir, str(idx))

        # ── Parliament ──────────────────────────────────────────────────────
        try:
            session = await run_session(
                question=question_text,
                model=model,
                output_dir=run_dir,
            )
            db_path = session["db_path"]
            rounds_completed = session["num_rounds_completed"]
            early_stopped = session["early_stopped"]
        except Exception as e:
            print(f"  [ERROR] Parliament failed: {e}")
            db_path = None
            rounds_completed = 0
            early_stopped = False

        # ── Judge ────────────────────────────────────────────────────────────
        answer = None
        if db_path and os.path.exists(db_path):
            try:
                judge_result = await run_judge(
                    db_path=db_path,
                    question=question_text,
                    model=model,
                    choices=choices,
                    output_dir=run_dir,
                )
                answer = judge_result["answer"]
            except Exception as e:
                print(f"  [ERROR] Judge failed: {e}")

        # ── Record ───────────────────────────────────────────────────────────
        is_correct = None
        if ground_truth is not None and answer is not None:
            norm_a = answer.strip("() ").upper()
            norm_t = str(ground_truth).strip("() ").upper()
            is_correct = norm_a == norm_t
            if is_correct:
                correct += 1
        total += 1

        record = {
            "index": idx,
            "question": question_text[:500],
            "choices": choices,
            "ground_truth": ground_truth,
            "parliament_answer": answer,
            "is_correct": is_correct,
            "rounds_completed": rounds_completed,
            "early_stopped": early_stopped,
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        status = "ok" if is_correct else ("WRONG" if is_correct is False else "?")
        print(f"\n  [{status}] Answer: {answer}")
        if ground_truth:
            print(f"      Truth:  {ground_truth}")
        if total > 0 and is_correct is not None:
            print(f"      Running: {correct}/{total} = {correct/total:.0%}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "dataset": dataset_path,
        "bench_name": bench_name,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "timestamp": timestamp,
    }
    summary_path = os.path.join(bench_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"  Accuracy: {correct}/{total} = {summary['accuracy']:.1%}")
    print(f"  Results:  {results_path}")
    print(f"{'='*70}\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Science Parliament Benchmark")
    parser.add_argument("--dataset", required=True, help="Path to dataset (CSV/JSONL)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--name", default=None, help="Benchmark name (default: filename)")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.dataset, limit=args.limit, bench_name=args.name))


if __name__ == "__main__":
    main()
