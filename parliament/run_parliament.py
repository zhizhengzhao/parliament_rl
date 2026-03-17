"""
Science Parliament — demo entry point.

Run a single question through the parliament and save results.

Usage:
    cd parliament
    python run_parliament.py --question "Prove that n(n+1)(n+2)(n+3)+1 is a perfect square."
    python run_parliament.py --question_file question.txt
"""

import argparse
import asyncio
import os
import shutil
from datetime import datetime

from session import OUTPUT_BASE, LOG_BASE, init, create_model, run_session


def parse_args():
    parser = argparse.ArgumentParser(description="Science Parliament")
    parser.add_argument(
        "--question", type=str, default=None,
        help="The scientific question to discuss",
    )
    parser.add_argument(
        "--question_file", type=str, default=None,
        help="Path to a text file containing the question",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    question = args.question
    if question is None and args.question_file:
        with open(args.question_file, "r") as f:
            question = f.read().strip()
    if question is None:
        question = input("Enter the scientific question:\n> ")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(OUTPUT_BASE, timestamp)
    log_dir = os.path.join(LOG_BASE, timestamp)

    init(log_dir=log_dir)
    model = create_model()

    # Save a copy of config for reproducibility
    config_src = os.path.join(os.path.dirname(__file__), "config.py")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(config_src, os.path.join(run_dir, "config.py"))

    session = await run_session(
        question=question,
        model=model,
        output_dir=run_dir,
    )

    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
