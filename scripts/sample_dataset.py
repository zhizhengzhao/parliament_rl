#!/usr/bin/env python3
"""Sample train and test sets from a full dataset.

Deterministic — same seed always produces the same split.
Excludes code-answer questions (unsuitable for forum discussion).

Usage:
    python scripts/sample_dataset.py \
        --input <path/to/full_dataset.json> \
        --output-dir datasets \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sample train/test from dataset")
    parser.add_argument("--input", required=True, help="Full dataset JSON")
    parser.add_argument("--output-dir", default="datasets", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--train-divisor", type=int, default=10,
                        help="For each depth-5 category: keep max(1, count//divisor)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(f"Total questions: {len(data)}")

    # Filter out code-answer questions
    clean = [q for q in data if not q.get("answer", "").strip().startswith("```")]
    print(f"After excluding code answers: {len(clean)}")

    # Group by depth-5 category
    groups = defaultdict(list)
    for q in clean:
        key = "/".join(q["category"].split("/")[:5])
        groups[key].append(q)
    print(f"Depth-5 categories: {len(groups)}")

    # Sample train: max(1, count // divisor) per category
    train, remaining = [], []
    for cat, qs in sorted(groups.items()):
        rng.shuffle(qs)
        n = max(1, len(qs) // args.train_divisor)
        train.extend(qs[:n])
        remaining.extend(qs[n:])
    print(f"Train: {len(train)} questions (divisor={args.train_divisor})")

    # Sample test: uniform across depth-5 categories, from remaining pool
    remaining_groups = defaultdict(list)
    for q in remaining:
        key = "/".join(q["category"].split("/")[:5])
        remaining_groups[key].append(q)

    test = []
    cat_keys = sorted(remaining_groups.keys())
    rng.shuffle(cat_keys)
    i = 0
    while len(test) < args.test_size and i < len(cat_keys):
        cat = cat_keys[i % len(cat_keys)]
        qs = remaining_groups[cat]
        if qs:
            test.append(qs.pop(rng.randrange(len(qs))))
        i += 1
    print(f"Test: {len(test)} questions")

    # Verify no overlap
    train_ids = {q["id"] for q in train}
    test_ids = {q["id"] for q in test}
    assert train_ids.isdisjoint(test_ids), "Train/test overlap!"
    print(f"Overlap: 0 (verified)")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "sciencepedia_train.json"
    test_path = out / "sciencepedia_test.json"
    train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False), encoding="utf-8")
    test_path.write_text(json.dumps(test, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved:")
    print(f"  {train_path} ({len(train)} questions)")
    print(f"  {test_path} ({len(test)} questions)")
    print(f"\nReproducible with: --seed {args.seed}")


if __name__ == "__main__":
    main()
