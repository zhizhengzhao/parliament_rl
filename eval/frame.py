#!/usr/bin/env python3
"""Frame evaluation — assess each cell's policy *inside its own training frame*.

Motivation
----------
Standard zero-shot benchmarks (`eval.gpqa`, `eval.sciencepedia_mc`)
prompt the trained policy as a *single agent answering one question*.
But the policy was trained as a *participant in a multi-turn forum*
(coupled cells) or a *solo derivator* (independent cells).  Forcing
every cell onto the same zero-shot, single-turn distribution at
evaluation time produces a metric that is double-confounded:

* training-time distribution ≠ evaluation-time distribution, equally
  for every cell — so any per-cell win is partly inherited noise of
  this single mismatch;
* and any cell whose training distribution happens to be closer to
  the zero-shot form (e.g. solo) gets a transfer-style advantage
  that is unrelated to the actual *quality* of the trained policy.

Frame eval addresses both: each cell's policy is run through a
session **in its own cell setting** on the held-out test set,
producing a discussion (coupled cells) or a chain of solo steps
(independent cells).  A fixed cell-agnostic ``secretary`` agent
(:mod:`eval.secretary`) — same model, same prompt, no judge
visibility, no peer scores — then maps each session's discussion
onto a single ``\\boxed{...}`` answer.  Accuracy is computed against
the dataset's gold answer.

Pipeline
--------
For one (cell, policy):

1. **Collect sessions** — invoke ``scripts/run.py`` with the
   policy as the actor model and the cell-specific env vars
   (``PRL_CONTEXT``, ``PRL_JUDGE_VOTES_VISIBLE``).  This produces
   ``data/<name>_<ts>/parliament.db`` containing every session's
   posts/comments/votes.  Returns the run dir.
2. **Extract sessions** — read ``parliament.db``, group actor
   posts (and, in coupled cells, actor comments) per session.
3. **Apply secretary** — load the secretary model in-process via
   vLLM (one batched generation pass over all sessions), parse a
   single boxed answer per session.
4. **Score** — compare against the dataset's ``gold_letter`` (MC)
   or ``answer`` (free-form, conservative string equivalence).
5. **Save** — write a JSON report with overall + per-category
   accuracy and per-session records.

Sweep all four cells with::

    bash eval/frame_sweep.sh \\
        --policies A:<path>,B:<path>,C:<path>,D:<path> \\
        --dataset datasets/sciencepedia_heldout_mc100.json \\
        --secretary $PRL_MODEL_PATH \\
        --output data/frame_eval/run1/

Important notes
---------------
* The secretary should be a **fixed model** across all four cells —
  typically the original base (`$PRL_MODEL_PATH`).  Using one of the
  trained policies as secretary would let cell-specific style bias
  the extraction.
* Frame eval is expensive: ~1–2 h per cell on 8×A100 for a
  100-question test set with 3 actors × 3 judges × max_turns=30.
  Use ``--max-questions`` and ``--max-turns`` to dial down for smoke
  tests.
* GPU usage is *sequential*: ``scripts/run.py`` releases all vLLMs
  at the end of step 1 before this script loads the secretary in
  step 3, so a single GPU pool serves both phases.

.. note::

    The ``--policies`` argument currently expects full HF-format model
    paths.  Stage-2 training (``rl/train.py``) now writes only the
    PEFT LoRA adapter (``ckpt/step_K/adapter/``) — vLLM hot-swaps it
    without merging.  To eval a LoRA-trained policy you currently
    need to either (a) materialise a merged folder ad-hoc, or
    (b) extend ``collect_sessions`` to pass the adapter path through
    ``/v1/load_lora_adapter`` after vLLM startup.  This is tracked
    as a follow-up (we will resolve it together with the LoRA-aware
    secretary loading).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from eval.secretary import (  # noqa: E402
    build_chat_messages, parse_letter, parse_boxed,
    equivalent_freeform, normalize_freeform,
)


# ── Cell → env ────────────────────────────────────────────

CELL_ENV: dict[str, dict[str, str]] = {
    "A": {"PRL_CONTEXT": "Parliament", "PRL_JUDGE_VOTES_VISIBLE": "1"},
    "B": {"PRL_CONTEXT": "Parliament", "PRL_JUDGE_VOTES_VISIBLE": "0"},
    "C": {"PRL_CONTEXT": "Solo",       "PRL_JUDGE_VOTES_VISIBLE": "1"},
    "D": {"PRL_CONTEXT": "Solo",       "PRL_JUDGE_VOTES_VISIBLE": "0"},
}
CELL_LABEL = {"A": "Parliament", "B": "BlindParliament",
              "C": "Solo", "D": "BlindSolo"}


# ── Step 1: collect sessions via scripts/run.py ──────────

def collect_sessions(*, cell: str, policy: str, dataset: str, name: str,
                     gpus: str, sessions_per_gpu: int, actors: int,
                     judges: int, max_turns: int,
                     max_questions: int) -> Path:
    """Drive scripts/run.py end-to-end for one (cell, policy).

    ``--in-tmux`` is passed so run.py skips its own tmux self-relaunch
    (we are already orchestrating) and runs the cleanup → vLLM →
    Parliament → harness → cleanup pipeline directly.
    """
    env = os.environ.copy()
    env.update(CELL_ENV[cell])

    cmd = [
        sys.executable, str(PROJECT_DIR / "scripts" / "run.py"),
        "--in-tmux",
        "--gpus", gpus,
        "--sessions-per-gpu", str(sessions_per_gpu),
        "--actors", str(actors),
        "--judges", str(judges),
        "--dataset", dataset,
        "--name", name,
        "--model", policy,
        "--max-turns", str(max_turns),
    ]
    if max_questions > 0:
        cmd.extend(["--max-questions", str(max_questions)])

    print(f"\n[frame] cell {cell} ({CELL_LABEL[cell]})  policy={policy}")
    print(f"        env  PRL_CONTEXT={env['PRL_CONTEXT']}  "
          f"PRL_JUDGE_VOTES_VISIBLE={env['PRL_JUDGE_VOTES_VISIBLE']}")
    print(f"        cmd  {' '.join(cmd)}\n")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"scripts/run.py failed for cell {cell} (rc={proc.returncode})")

    # Most-recent matching run dir
    candidates = sorted(
        (PROJECT_DIR / "data").glob(f"{name}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"no data/{name}_* directory found after run.py")
    run_dir = candidates[-1]
    print(f"[frame] cell {cell} sessions collected → {run_dir} "
          f"({(time.time() - t0) / 60:.1f} min)\n")
    return run_dir


# ── Step 2: read sessions from parliament.db ─────────────

def load_sessions_from_db(db_path: Path,
                          include_comments: bool) -> list[dict]:
    """Pull each session's actor posts (+ optional comments) from the DB.

    Returns a list of dicts:

        [{ "session_id": str, "title": str,
           "posts": [{ "post_id": int, "author": str,
                       "content": str, "comments": [...] }, ...] }, ...]

    Posts are ordered by ``post_id`` (= chronology).  Judge posts and
    judge comments are excluded; we only feed the secretary the
    *actor* discussion.  In solo cells, comments don't exist anyway,
    so ``include_comments`` is a no-op there.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    sessions: list[dict] = []
    for srow in conn.execute(
        "SELECT session_id, title FROM sessions ORDER BY created_at"
    ):
        sid = srow["session_id"]
        posts = [dict(r) for r in conn.execute(
            "SELECT p.post_id, p.content, p.created_at, u.name AS author "
            "FROM posts p JOIN users u ON p.user_id = u.user_id "
            "WHERE p.session_id = ? AND u.role = 'actor' "
            "ORDER BY p.post_id",
            (sid,),
        )]
        if include_comments:
            for p in posts:
                p["comments"] = [dict(r) for r in conn.execute(
                    "SELECT c.content, u.name AS author "
                    "FROM comments c JOIN users u ON c.user_id = u.user_id "
                    "WHERE c.post_id = ? AND u.role = 'actor' "
                    "ORDER BY c.comment_id",
                    (p["post_id"],),
                )]
        else:
            for p in posts:
                p["comments"] = []
        sessions.append({"session_id": sid, "title": srow["title"], "posts": posts})
    conn.close()
    return sessions


# ── Step 3: secretary inference ──────────────────────────

def run_secretary(secretary_model: str, sessions: list[dict],
                  *, parse_mode: str, include_comments: bool,
                  tensor_parallel_size: int, max_tokens: int,
                  temperature: float, top_p: float,
                  enable_thinking: bool,
                  gpu_memory_utilization: float,
                  max_model_len: int) -> list[dict]:
    """Load secretary vLLM in-process and batch-generate one answer per session.

    Returns a list of dicts:

        [{ "session_id": str, "title": str,
           "n_posts": int,
           "raw_response": str,
           "parsed_answer": str | None }, ...]
    """
    # Defer the heavy import so callers that only want load_sessions_from_db
    # / parsing don't have to install vLLM.
    from vllm import LLM, SamplingParams

    print(f"[frame] secretary loading: {secretary_model}  "
          f"(tp={tensor_parallel_size}, max_len={max_model_len})")
    t0 = time.time()
    llm = LLM(
        model=secretary_model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
    )
    tokenizer = llm.get_tokenizer()
    print(f"[frame] secretary ready ({time.time() - t0:.0f}s)")

    msgs_per_session = [
        build_chat_messages(s["title"], s["posts"], include_comments)
        for s in sessions
    ]
    chat_prompts = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for m in msgs_per_session
    ]
    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
    )
    print(f"[frame] secretary generating on {len(chat_prompts)} sessions...")
    t0 = time.time()
    outputs = llm.generate(chat_prompts, sampling)
    print(f"[frame] secretary done ({time.time() - t0:.0f}s)")

    parser = parse_letter if parse_mode == "letter" else parse_boxed

    records: list[dict] = []
    for s, o in zip(sessions, outputs):
        text = o.outputs[0].text
        records.append({
            "session_id": s["session_id"],
            "title": s["title"],
            "n_posts": len(s["posts"]),
            "raw_response": text,
            "parsed_answer": parser(text),
        })
    return records


# ── Step 4: scoring ──────────────────────────────────────

_BOXED_OUTER_RE = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)


def _strip_outer_boxed(s: str) -> str:
    m = _BOXED_OUTER_RE.search(s)
    return m.group(1).strip() if m else s.strip()


def score_records(records: list[dict],
                  dataset_items_by_title: dict[str, dict],
                  parse_mode: str) -> dict:
    """Compare predictions against dataset gold; return per-cell accuracy.

    For ``parse_mode=='letter'`` the gold is ``gold_letter`` from the
    dataset row (the held-out MC set has it).  For ``'boxed'`` the
    gold is the ``answer`` field, with any outer ``\\boxed{...}``
    stripped before comparison via
    :func:`secretary.equivalent_freeform`.
    """
    correct = 0
    n_scored = 0
    n_no_gold = 0
    n_no_pred = 0
    n_no_match = 0
    by_cat = Counter()
    by_cat_total = Counter()
    scored: list[dict] = []
    for r in records:
        item = dataset_items_by_title.get(r["title"])
        if not item:
            n_no_match += 1
            continue
        if parse_mode == "letter":
            gold = item.get("gold_letter")
            if not gold:
                n_no_gold += 1
                continue
            pred = r["parsed_answer"]
            if pred is None:
                n_no_pred += 1
            ok = (pred is not None) and (pred.upper() == gold.upper())
        else:
            gold_raw = item.get("answer", "")
            gold = _strip_outer_boxed(gold_raw)
            pred = r["parsed_answer"]
            if pred is None:
                n_no_pred += 1
            ok = (pred is not None) and equivalent_freeform(pred, gold)

        n_scored += 1
        correct += int(ok)
        cat = "/".join(item.get("category", "unknown").split("/")[:2])
        by_cat_total[cat] += 1
        by_cat[cat] += int(ok)
        scored.append({
            **r,
            "category": cat,
            "gold": gold,
            "is_correct": ok,
        })

    return {
        "accuracy": correct / n_scored if n_scored else 0.0,
        "n_correct": correct,
        "n_scored": n_scored,
        "n_total": len(records),
        "n_no_match": n_no_match,
        "n_no_gold": n_no_gold,
        "n_no_pred": n_no_pred,
        "per_category": {
            c: by_cat[c] / by_cat_total[c] for c in sorted(by_cat_total)
        },
        "per_category_total": dict(by_cat_total),
        "records": scored,
    }


# ── Driver ───────────────────────────────────────────────

def load_dataset_index(path: str) -> tuple[dict[str, dict], str]:
    """Read the dataset; return ``{title: row}`` map and inferred parse mode.

    `parse_mode = "letter"` if any row carries `gold_letter`, else `"boxed"`.
    """
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a JSON list")
    has_letter = any(it.get("gold_letter") for it in items)
    parse_mode = "letter" if has_letter else "boxed"
    by_title: dict[str, dict] = {}
    for it in items:
        title = it.get("problem") or it.get("title")
        if title:
            by_title[title] = it
    return by_title, parse_mode


def evaluate_one_cell(*, cell: str, policy: str, dataset: str,
                      secretary_model: str, gpus: str,
                      output: Path, name: str | None,
                      sessions_per_gpu: int, actors: int, judges: int,
                      max_turns: int, max_questions: int,
                      reuse_run_dir: Path | None,
                      include_comments: bool | None,
                      secretary_tp: int, secretary_max_len: int,
                      secretary_max_tokens: int,
                      secretary_temperature: float,
                      secretary_top_p: float,
                      secretary_thinking: bool,
                      secretary_gpu_mem: float) -> dict:
    """End-to-end: collect → extract → score → write.  Returns the report dict."""
    if cell not in CELL_ENV:
        raise ValueError(f"unknown cell {cell!r}; expected one of {list(CELL_ENV)}")
    name = name or f"frame_{cell}"

    # 1. Collect sessions (or reuse)
    if reuse_run_dir is not None:
        run_dir = reuse_run_dir
        print(f"[frame] cell {cell}: reusing existing run dir {run_dir}")
    else:
        run_dir = collect_sessions(
            cell=cell, policy=policy, dataset=dataset, name=name,
            gpus=gpus, sessions_per_gpu=sessions_per_gpu,
            actors=actors, judges=judges, max_turns=max_turns,
            max_questions=max_questions,
        )

    db_path = run_dir / "parliament.db"
    if not db_path.exists():
        raise RuntimeError(f"parliament.db not found in {run_dir}")

    # 2. Read sessions
    if include_comments is None:
        include_comments = (CELL_ENV[cell]["PRL_CONTEXT"] == "Parliament")
    sessions = load_sessions_from_db(db_path, include_comments=include_comments)
    if not sessions:
        raise RuntimeError(f"no sessions found in {db_path}")
    print(f"[frame] cell {cell}: {len(sessions)} sessions, "
          f"avg {sum(len(s['posts']) for s in sessions) / len(sessions):.1f} actor posts/session")

    # 3. Secretary inference + parse
    by_title, parse_mode = load_dataset_index(dataset)
    print(f"[frame] dataset {dataset}: {len(by_title)} items, "
          f"parse_mode={parse_mode}")

    secretary_records = run_secretary(
        secretary_model, sessions,
        parse_mode=parse_mode, include_comments=include_comments,
        tensor_parallel_size=secretary_tp,
        max_tokens=secretary_max_tokens,
        temperature=secretary_temperature, top_p=secretary_top_p,
        enable_thinking=secretary_thinking,
        gpu_memory_utilization=secretary_gpu_mem,
        max_model_len=secretary_max_len,
    )

    # 4. Score
    scored = score_records(secretary_records, by_title, parse_mode)

    # 5. Save
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "cell": cell,
        "cell_label": CELL_LABEL[cell],
        "policy": policy,
        "dataset": dataset,
        "secretary": secretary_model,
        "run_dir": str(run_dir),
        "parse_mode": parse_mode,
        "include_comments": include_comments,
        "sessions": len(sessions),
        "actor_posts_total": sum(len(s["posts"]) for s in sessions),
        "secretary_config": {
            "tensor_parallel_size": secretary_tp,
            "max_model_len": secretary_max_len,
            "max_tokens": secretary_max_tokens,
            "temperature": secretary_temperature,
            "top_p": secretary_top_p,
            "enable_thinking": secretary_thinking,
        },
        **scored,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print()
    print(f"=== Cell {cell} ({CELL_LABEL[cell]}) frame eval ===")
    print(f"  policy:    {policy}")
    print(f"  sessions:  {len(sessions)}  ({report['actor_posts_total']} actor posts)")
    print(f"  scored:    {scored['n_scored']}/{scored['n_total']}  "
          f"(no_match={scored['n_no_match']}, no_gold={scored['n_no_gold']}, "
          f"no_pred={scored['n_no_pred']})")
    print(f"  ACCURACY:  {scored['accuracy']:.4f}  ({scored['n_correct']}/{scored['n_scored']})")
    if scored["per_category"]:
        for c in sorted(scored["per_category"]):
            print(f"    {c:40s} {scored['per_category'][c]:.3f}  "
                  f"({scored['per_category_total'][c]} q)")
    print(f"  saved →    {output}")
    return report


# ── CLI ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", required=True, choices=list(CELL_ENV),
                   help="Which 2x2 cell setting to roll the policy under")
    p.add_argument("--policy", required=True,
                   help="Path to the trained policy (HF folder or model id)")
    p.add_argument("--dataset", required=True,
                   help="Test set JSON; rows must carry `problem` and either "
                        "`gold_letter` (MC) or `answer` (free-form)")
    p.add_argument("--secretary",
                   default=os.environ.get("PRL_MODEL_PATH", "Qwen/Qwen3.5-9B"),
                   help="Secretary model — fixed across cells. Default = $PRL_MODEL_PATH.")
    p.add_argument("--output", required=True, help="JSON report path")
    p.add_argument("--name", default=None,
                   help="Run-dir name prefix; defaults to frame_<cell>")
    p.add_argument("--reuse-run-dir", default=None,
                   help="Skip step 1 and reuse an existing data/<run>/ that "
                        "already contains parliament.db (useful for re-extraction)")

    # Rollout knobs (forwarded to scripts/run.py)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                   help="GPUs for the policy vLLM and (later) the secretary vLLM")
    p.add_argument("--sessions-per-gpu", type=int, default=2)
    p.add_argument("--actors", type=int, default=3)
    p.add_argument("--judges", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-questions", type=int, default=0,
                   help="Cap on dataset rows (0 = use all)")
    p.add_argument("--include-comments",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Pass actor comments to the secretary in coupled cells. "
                        "Default: True for A/B, False for C/D (no comments exist).")

    # Secretary knobs
    p.add_argument("--secretary-tp", type=int, default=1,
                   help="Secretary tensor-parallel size (1 = single GPU)")
    p.add_argument("--secretary-max-model-len", type=int, default=32768)
    p.add_argument("--secretary-max-tokens", type=int, default=2048,
                   help="Generation budget per session for the secretary")
    p.add_argument("--secretary-temperature", type=float, default=0.6)
    p.add_argument("--secretary-top-p", type=float, default=0.95)
    p.add_argument("--secretary-thinking",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Enable Qwen3.5 thinking mode for the secretary. "
                        "Off by default — the secretary's job is mechanical "
                        "extraction, not reasoning.")
    p.add_argument("--secretary-gpu-mem", type=float, default=0.85)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_one_cell(
        cell=args.cell, policy=args.policy, dataset=args.dataset,
        secretary_model=args.secretary, gpus=args.gpus,
        output=Path(args.output), name=args.name,
        sessions_per_gpu=args.sessions_per_gpu,
        actors=args.actors, judges=args.judges,
        max_turns=args.max_turns, max_questions=args.max_questions,
        reuse_run_dir=Path(args.reuse_run_dir) if args.reuse_run_dir else None,
        include_comments=args.include_comments,
        secretary_tp=args.secretary_tp,
        secretary_max_len=args.secretary_max_model_len,
        secretary_max_tokens=args.secretary_max_tokens,
        secretary_temperature=args.secretary_temperature,
        secretary_top_p=args.secretary_top_p,
        secretary_thinking=args.secretary_thinking,
        secretary_gpu_mem=args.secretary_gpu_mem,
    )


if __name__ == "__main__":
    main()
