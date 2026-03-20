"""
Science Parliament — Baseline with Tools runner (single agent, multi-turn).

Single agent with sympy + python tools, multi-turn tool-calling.
Same model, same tools as parliament scientists, but no collaboration.

Usage:
    cd baseline_tools
    python run_baseline_tools.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0
    python run_baseline_tools.py --dataset ../benchmark/gpqa_diamond.csv --gpus 0,1,2 --limit 20
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
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

MAX_TOOL_ROUNDS = 10

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a scientist solving a difficult problem. Think step by step.

You have access to computational tools. Use them to verify your reasoning, \
check numerical values, test edge cases, and look up physical constants. \
A calculation that confirms or disproves your reasoning is extremely valuable.

Select the single best answer from the provided choices.

You MUST end your response with this exact block:

<<<FINAL>>>
(X)
<<<END>>>

where X is the letter of the correct choice (A, B, C, or D). \
Nothing else should appear between the markers.\
"""

SYSTEM_PROMPT_OPEN = """\
You are a scientist solving a difficult problem. Think step by step.

You have access to computational tools. Use them to verify your reasoning, \
check numerical values, test edge cases, and look up physical constants. \
A calculation that confirms or disproves your reasoning is extremely valuable.

You MUST end your response with this exact block:

<<<FINAL>>>
your final answer here
<<<END>>>

The answer between <<<FINAL>>> and <<<END>>> must be self-contained.\
"""


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format for vLLM)
# ---------------------------------------------------------------------------

_PYTHON_PREAMBLE = """\
import math, cmath, fractions, decimal, itertools, collections, re, json
try:
    import numpy as np
except ImportError:
    pass
try:
    import scipy.constants as const
except ImportError:
    pass
try:
    import sympy
    from sympy import *
except ImportError:
    pass
"""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code and return the printed output. Use this "
                "to perform numerical calculations, verify formulas, check "
                "edge cases, solve equations, and look up physical constants. "
                "Pre-imported libraries: math, numpy (as np), "
                "scipy.constants (as const), sympy (all symbols via 'from sympy import *'). "
                "Physical constants: const.c, const.hbar, const.eV, const.k, "
                "const.m_e, const.m_p, const.N_A, const.G, const.R, const.alpha. "
                "You MUST use print() to produce output — only printed text is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


def _execute_python(code: str) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="baseline_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_PYTHON_PREAMBLE + "\n" + code)
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.returncode != 0 and result.stderr:
            err = result.stderr.strip().splitlines()
            output += "\n[error]\n" + "\n".join(err[-15:])
        if not output.strip():
            output = "(no output — use print() to see results)"
        return output[:4000]
    except subprocess.TimeoutExpired:
        return "(execution timed out after 30s)"
    except Exception as e:
        return f"(execution error: {e})"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _dispatch_tool(name: str, arguments: dict) -> str:
    if name == "run_python":
        return _execute_python(arguments.get("code", ""))
    return f"(unknown tool: {name})"


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------

def _build_prompt(question: str, choices: list[str] | None) -> str:
    parts = [question]
    if choices:
        parts.append("\nCHOICES:")
        for i, ch in enumerate(choices):
            parts.append(f"  ({chr(ord('A') + i)}) {ch}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Multi-turn tool-calling inference
# ---------------------------------------------------------------------------

async def _infer_one(
    idx: int, q: dict, client, port: int, model_name: str, semaphore,
) -> dict:
    async with semaphore:
        choices = q.get("choices")
        system = SYSTEM_PROMPT if choices else SYSTEM_PROMPT_OPEN
        user_msg = _build_prompt(q["question"], choices)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        tool_rounds = 0
        final_text = ""

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                resp = await client.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "tools": TOOLS_SCHEMA,
                        "max_tokens": 4096,
                    },
                    timeout=300,
                )
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]

                messages.append(msg)

                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    final_text = msg.get("content", "")
                    break

                tool_rounds += 1
                for tc in tool_calls:
                    fn = tc["function"]
                    try:
                        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    result = _dispatch_tool(fn["name"], args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
            else:
                resp = await client.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": 4096,
                    },
                    timeout=300,
                )
                data = resp.json()
                final_text = data["choices"][0]["message"].get("content", "")

        except Exception as e:
            print(f"  Question {idx} error: {e}")
            final_text = ""

        answer = extract_answer(final_text)
        gt = q.get("ground_truth")
        is_correct = None
        if gt is not None and answer is not None:
            is_correct = answer.strip("() ").upper() == str(gt).strip("() ").upper()

        status = "ok" if is_correct else ("WRONG" if is_correct is False else "?")
        print(f"  [{status}] Question {idx} (tools={tool_rounds}): answer={answer}")

        return {
            "index": idx,
            "question": q["question"][:500],
            "choices": q.get("choices"),
            "ground_truth": gt,
            "parliament_answer": answer,
            "is_correct": is_correct,
            "rounds_completed": tool_rounds,
            "early_stopped": False,
            "gpu": port,
        }


async def _run_batch(questions: list[dict], ports: list[int], concurrency: int = 16):
    import httpx
    from config import MODEL_NAME

    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = []
        for idx, q in enumerate(questions):
            port = ports[idx % len(ports)]
            tasks.append(_infer_one(idx, q, client, port, MODEL_NAME, semaphore))
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline_tools(
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
        bench_name = f"baseline_tools_{base}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bench_dir = os.path.join(OUTPUT_BASE, bench_name, timestamp)
    os.makedirs(bench_dir, exist_ok=True)

    ports = [8000 + g for g in gpu_ids]

    print("\n[1/3] Starting vLLM instances...")
    vllm_manager.start(gpu_ids)
    vllm_manager.wait_ready(gpu_ids)

    print(f"\n[2/3] Running baseline+tools: {len(questions)} questions, {len(gpu_ids)} GPUs")
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
    print(f"BASELINE+TOOLS COMPLETE")
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
    parser = argparse.ArgumentParser(description="Science Parliament Baseline with Tools")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPU IDs: '0', '0,1,2', '0-7' (default: 0)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--name", default=None, help="Run name")
    parser.add_argument("--port", type=int, default=18888, help="HTTP port for results page")
    args = parser.parse_args()
    run_baseline_tools(
        args.dataset, gpu_ids=parse_gpu_ids(args.gpus),
        limit=args.limit, bench_name=args.name, serve_port=args.port,
    )


if __name__ == "__main__":
    main()
