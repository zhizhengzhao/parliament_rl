"""Science Parliament — dataset loading utilities.

Shared by run_benchmark.py and run_baseline.py.
Supports CSV, TSV, and JSONL formats.
"""

import csv
import json
import random


def load_dataset(path: str) -> list[dict]:
    """Load and normalize a dataset file into a list of question dicts."""
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [_normalize(json.loads(l)) for l in f if l.strip()]
    elif ext in (".csv", ".tsv"):
        with open(path, "r", encoding="utf-8") as f:
            return [_normalize(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported format: {ext}")


def _normalize(raw: dict) -> dict:
    """Normalize heterogeneous dataset rows into a standard format.

    Returns: {"question": str, "choices": list|None, "ground_truth": str|None}
    """
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


def parse_gpu_ids(s: str) -> list[int]:
    """Parse GPU ID string: '0', '0,1,2', or '0-7'."""
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]
