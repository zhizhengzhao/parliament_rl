#!/usr/bin/env python3
"""Sciencepedia held-out multiple-choice evaluation (Qwen3.5 thinking mode).

Filters `datasets/sciencepedia_test.json` to only multiple-choice questions
(those whose `answer` matches `\\boxed{A|B|C|D}`), runs each through a local
vLLM with `enable_thinking=True`, parses the model's final boxed letter
(or "answer is (X)" fallback), and reports accuracy.

Why filter MC only:
  Sciencepedia has free-form answers (formulas, numbers, lists) that are
  hard to autograde. The boxed-letter subset (~14% of test set) is a
  clean check that the model picks the right option after CoT reasoning.

Usage:
    python -m eval.sciencepedia_mc \\
        --model /path/to/merged \\
        --output data/<run>/scipedia_mc.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 100 multiple-choice questions from sciencepedia_final, held out from
# both train (4106) and test (100). Pre-built for evaluation convenience;
# rebuild with: select rows where answer matches \boxed{[A-D]} from the
# 51944-question full set, exclude train+test ids, sample 100 with seed=42.
DEFAULT_TEST = PROJECT_ROOT / "datasets" / "sciencepedia_heldout_mc100.json"

GOLD_RE = re.compile(r"\\boxed\{\s*([A-D])\s*\}")
PRED_RE = re.compile(
    r"\\boxed\{\s*([A-D])\s*\}|"                # \boxed{X}
    r"(?:answer is|answer:|final answer)\s*\(?\s*([A-D])\s*\)?",
    re.IGNORECASE,
)

PROMPT_TEMPLATE = (
    "{problem}\n\n"
    "Reason step by step, then state your final choice as \\boxed{{X}} "
    "where X is one of A, B, C, D."
)


def load_mc(path: Path) -> list[dict]:
    """Return only the boxed-letter multiple-choice subset.

    Accepts either a pre-filtered file (with `gold_letter` key) or a
    raw sciencepedia file (extracts `\\boxed{X}` from `answer`).
    """
    raw = json.loads(path.read_text())
    out = []
    for d in raw:
        if d.get("gold_letter"):
            out.append(d)
            continue
        m = GOLD_RE.search(d.get("answer", ""))
        if m:
            out.append({**d, "gold_letter": m.group(1)})
    return out


def parse_pred(text: str) -> str | None:
    """Last \\boxed{X} or 'answer is X' wins (model often self-corrects)."""
    matches = PRED_RE.findall(text)
    letters = [a or b for a, b in matches]
    return letters[-1].upper() if letters else None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sciencepedia held-out MC eval (thinking mode).")
    p.add_argument("--model", required=True, help="Local path or HF hub id")
    p.add_argument("--data", default=str(DEFAULT_TEST),
                   help="Path to sciencepedia_test.json")
    p.add_argument("--output", required=True, help="JSON result path")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Thinking mode generates long reasoning blocks.")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Qwen team recommends 0.6 for thinking mode.")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--n", type=int, default=1,
                   help="Samples per question; >1 + temperature>0 → majority.")
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = p.parse_args()

    rows = load_mc(Path(args.data))
    print(f"Loaded {len(rows)} multiple-choice questions from {args.data}")
    if not rows:
        raise SystemExit("No multiple-choice questions found.")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
    )
    tok = llm.get_tokenizer()

    prompts = [PROMPT_TEMPLATE.format(problem=r["problem"]) for r in rows]
    chat_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        for p in prompts
    ]
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, n=args.n,
    )
    outputs = llm.generate(chat_prompts, sampling)

    correct = 0
    by_cat: Counter = Counter()
    by_cat_total: Counter = Counter()
    records = []
    for i, out in enumerate(outputs):
        samples = [o.text for o in out.outputs]
        preds = [parse_pred(t) for t in samples]
        valid = [x for x in preds if x is not None]
        pred = Counter(valid).most_common(1)[0][0] if valid else None
        gold = rows[i]["gold_letter"]
        ok = pred == gold
        correct += int(ok)
        # First two path components: 'biology/graduate' style
        cat = "/".join(rows[i]["category"].split("/")[:2])
        by_cat_total[cat] += 1
        by_cat[cat] += int(ok)
        records.append({
            "id": rows[i]["id"],
            "category": cat,
            "gold": gold,
            "pred": pred,
            "all_preds": preds,
            "is_correct": ok,
            "output_tail": samples[0][-500:],
        })

    acc = correct / len(outputs)
    result = {
        "model": args.model,
        "data": args.data,
        "n_questions": len(outputs),
        "accuracy": acc,
        "per_category_accuracy": {c: by_cat[c] / by_cat_total[c]
                                  for c in sorted(by_cat_total)},
        "per_category_total": dict(by_cat_total),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "enable_thinking": args.enable_thinking,
        "max_tokens": args.max_tokens,
        "n_samples": args.n,
        "records": records,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n=== Sciencepedia MC ===")
    print(f"  Model:    {args.model}")
    print(f"  Accuracy: {acc:.4f} ({correct}/{len(outputs)})")
    for c in sorted(by_cat_total):
        print(f"    {c:40s} {by_cat[c]:2d}/{by_cat_total[c]:<2d}")
    print(f"  Saved:    {args.output}")


if __name__ == "__main__":
    main()
