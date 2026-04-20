#!/usr/bin/env python3
"""GPQA Diamond zero-shot CoT evaluation (Qwen3.5 thinking mode).

Loads `idavidrein/gpqa:gpqa_diamond` (198 questions) via HuggingFace,
runs each through a local vLLM instance with `enable_thinking=True` in
the chat template so the model's full reasoning block is generated,
parses the final `"The answer is (X)"` letter, and reports overall
plus per-domain accuracy.

Shuffling the four choices per question (seeded, reproducible) removes the
position bias that lets base models pick A disproportionately.

Usage:
    python -m eval.gpqa \\
        --model /path/to/merged \\
        --output data/<run>/gpqa_diamond.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from vllm import LLM, SamplingParams


def load_gpqa(path_or_subset: str) -> list[dict]:
    """Load GPQA rows from a local CSV or the HuggingFace gated dataset.

    Local CSV is preferred: the HF `idavidrein/gpqa` dataset is gated and
    requires network + access approval, while the CSV is a 200 KB file
    we can ship with any checkpoint.
    """
    p = Path(path_or_subset)
    if p.is_file() and p.suffix == ".csv":
        with p.open() as f:
            return list(csv.DictReader(f))
    # HF dataset fallback (agent with approved token + internet only).
    import sys
    import os
    _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path = [x for x in sys.path if os.path.abspath(x) != _proj]
    from datasets import load_dataset                # noqa: E402
    sys.path.insert(0, _proj)
    return list(load_dataset("idavidrein/gpqa", path_or_subset, split="train"))

PROMPT_TEMPLATE = (
    "The following is a multiple choice question about {domain}. "
    "Think step by step, then conclude with \"The answer is (X)\" where "
    "X is A, B, C, or D.\n\n"
    "Question: {question}\n\n"
    "(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}"
)

ANSWER_RE = re.compile(r"(?:answer is|answer:)\s*\(?\s*([ABCD])\s*\)?", re.I)


def build_prompt(row: dict, rng: random.Random) -> tuple[str, str, str]:
    """Shuffle the four choices; return (prompt, correct_letter, domain)."""
    choices = [row["Correct Answer"],
               row["Incorrect Answer 1"],
               row["Incorrect Answer 2"],
               row["Incorrect Answer 3"]]
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [choices[i] for i in order]
    correct_letter = "ABCD"[order.index(0)]
    prompt = PROMPT_TEMPLATE.format(
        domain=row["High-level domain"],
        question=row["Question"],
        A=shuffled[0], B=shuffled[1], C=shuffled[2], D=shuffled[3],
    )
    return prompt, correct_letter, row["High-level domain"]


def parse_answer(text: str) -> str | None:
    """Return the *last* ABCD match (the model often self-corrects)."""
    matches = ANSWER_RE.findall(text)
    return matches[-1].upper() if matches else None


def main() -> None:
    p = argparse.ArgumentParser(description="GPQA Diamond zero-shot CoT eval")
    p.add_argument("--model", required=True, help="Local path or HF hub id")
    p.add_argument("--output", required=True, help="JSON result path")
    p.add_argument("--data", default="gpqa_diamond",
                   help="Local CSV path (preferred) or HF subset name "
                        "(gpqa_diamond | gpqa_main | gpqa_extended).")
    p.add_argument("--seed", type=int, default=0,
                   help="Shuffle seed — identical across model versions "
                        "so each model sees the same letter→answer map")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Qwen3.5 thinking mode emits long reasoning blocks; "
                        "8k is the safe default, raise to 16k for hard math.")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Qwen team recommends 0.6 for thinking mode "
                        "(0.0 hurts reasoning diversity).")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Qwen3.5 reasoning mode. --no-enable-thinking disables it.")
    p.add_argument("--n", type=int, default=1,
                   help="Samples per question. >1 + temperature>0 → majority vote.")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = p.parse_args()

    # ── Dataset ────────────────────────────────────────────
    ds = load_gpqa(args.data)
    print(f"Loaded {len(ds)} questions from {args.data}")

    rng = random.Random(args.seed)
    prompts, answers, domains = [], [], []
    for row in ds:
        prompt, letter, domain = build_prompt(row, rng)
        prompts.append(prompt)
        answers.append(letter)
        domains.append(domain)

    # ── vLLM ───────────────────────────────────────────────
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
    )
    tokenizer = llm.get_tokenizer()

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        for p in prompts
    ]

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
    )
    outputs = llm.generate(chat_prompts, sampling)

    # ── Score ──────────────────────────────────────────────
    correct = 0
    per_domain = Counter()
    per_domain_total = Counter()
    records = []
    for i, out in enumerate(outputs):
        samples = [o.text for o in out.outputs]           # len == args.n
        preds = [parse_answer(t) for t in samples]
        # Majority vote across the n samples (ignoring None parses).
        valid = [x for x in preds if x is not None]
        pred = Counter(valid).most_common(1)[0][0] if valid else None
        is_correct = pred == answers[i]
        correct += int(is_correct)
        per_domain_total[domains[i]] += 1
        per_domain[domains[i]] += int(is_correct)
        records.append({
            "question_id": i,
            "domain": domains[i],
            "correct_letter": answers[i],
            "predicted_letter": pred,
            "all_preds": preds,
            "is_correct": is_correct,
            # Keep the tail only — full CoT makes the JSON huge.
            "output_tail": samples[0][-500:],
        })

    acc = correct / len(outputs)
    result = {
        "model": args.model,
        "data": args.data,
        "n_questions": len(outputs),
        "accuracy": acc,
        "per_domain_accuracy": {d: per_domain[d] / per_domain_total[d]
                                for d in sorted(per_domain_total)},
        "per_domain_total": dict(per_domain_total),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "enable_thinking": args.enable_thinking,
        "max_tokens": args.max_tokens,
        "n_samples": args.n,
        "records": records,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n=== GPQA {args.data} ===")
    print(f"  Model:    {args.model}")
    print(f"  Accuracy: {acc:.4f} ({correct}/{len(outputs)})")
    for d in sorted(per_domain_total):
        print(f"    {d:12s} {per_domain[d]:3d}/{per_domain_total[d]:<3d} "
              f"= {per_domain[d]/per_domain_total[d]:.3f}")
    print(f"  Saved:    {args.output}")


if __name__ == "__main__":
    main()
