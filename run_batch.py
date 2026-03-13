"""
Batch runner: run Science Parliament on multiple questions from a CSV file.
Each question gets its own session directory.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from camel.models import ModelFactory
from camel.types import ModelPlatformType

from config import (
    MODEL_NAME,
    MODEL_BASE_URL,
    API_KEY,
    NUM_ROUNDS,
    OUTPUT_DIR,
)
from run_parliament import run_parliament


async def main():
    parser = argparse.ArgumentParser(description="Batch Science Parliament")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to CSV with a 'Question' column")
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    if API_KEY:
        os.environ["OPENAI_API_KEY"] = API_KEY
    if MODEL_BASE_URL:
        os.environ["OPENAI_API_BASE_URL"] = MODEL_BASE_URL

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=MODEL_NAME,
    )

    df = pd.read_csv(args.data_path)
    end_index = min(args.start_index + args.max_examples, len(df))

    batch_dir = os.path.join(OUTPUT_DIR, "batch")
    os.makedirs(batch_dir, exist_ok=True)
    sessions = []

    for i in range(args.start_index, end_index):
        question = df.iloc[i]["Question"]
        q_output = os.path.join(batch_dir, f"q_{i:04d}")
        print(f"\n{'#'*70}")
        print(f"Question {i+1}/{end_index} (index={i})")
        print(f"{'#'*70}")

        try:
            session = await run_parliament(
                question=question,
                model=model,
                output_dir=q_output,
            )
            sessions.append({"index": i, "num_posts": len(session["discussion"])})
        except Exception as e:
            print(f"[ERROR] Question {i}: {e}")
            sessions.append({"index": i, "error": str(e)})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(batch_dir, f"batch_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump({"model": MODEL_NAME, "rounds": NUM_ROUNDS, "sessions": sessions},
                  f, indent=2)
    print(f"\nBatch summary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
