#!/bin/bash
# Sweep all 4 cells through eval.frame using one fixed secretary.
#
# Each cell is run sequentially (vLLM cannot host multiple model
# loads on the same GPU pool simultaneously).  All 4 reports are
# written into one output dir; a final per-cell accuracy table is
# printed and saved to summary.json.
#
# Usage
# -----
#   bash eval/frame_sweep.sh \
#     --policies A:data/mainA_iter10/merged,B:.../merged,C:.../merged,D:.../merged \
#     --dataset datasets/sciencepedia_heldout_mc100.json \
#     --output  data/frame_eval/main_run \
#     [--secretary $PRL_MODEL_PATH] \
#     [--gpus 0,1,2,3,4,5,6,7] \
#     [--max-questions 0] \
#     [--max-turns 30]
#
# Notes
# -----
# - The secretary defaults to $PRL_MODEL_PATH (the original base) — do
#   NOT use one of the trained policies, or extraction bias would
#   silently favour that cell's style.
# - --policies entries list the cells to sweep, in any order.  Cells
#   not listed are skipped.
# - Each cell takes ~1-2 h on 8×A100 for a 100-q test set with
#   3 actors × 3 judges × max_turns=30.  Use --max-questions and
#   --max-turns to dial down for smoke tests.

set -euo pipefail

POLICIES=""
DATASET=""
OUTPUT=""
SECRETARY="${PRL_MODEL_PATH:-Qwen/Qwen3.5-9B}"
GPUS="0,1,2,3,4,5,6,7"
MAX_QUESTIONS=0
MAX_TURNS=30
SESSIONS_PER_GPU=2
ACTORS=3
JUDGES=3
SEC_TP=1
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policies)         POLICIES="$2";        shift 2 ;;
    --dataset)          DATASET="$2";         shift 2 ;;
    --output)           OUTPUT="$2";          shift 2 ;;
    --secretary)        SECRETARY="$2";       shift 2 ;;
    --gpus)             GPUS="$2";            shift 2 ;;
    --max-questions)    MAX_QUESTIONS="$2";   shift 2 ;;
    --max-turns)        MAX_TURNS="$2";       shift 2 ;;
    --sessions-per-gpu) SESSIONS_PER_GPU="$2";shift 2 ;;
    --actors)           ACTORS="$2";          shift 2 ;;
    --judges)           JUDGES="$2";          shift 2 ;;
    --secretary-tp)     SEC_TP="$2";          shift 2 ;;
    *)                  EXTRA+=("$1");        shift ;;
  esac
done

[[ -z "$POLICIES" ]] && { echo "FATAL: --policies tag:path[,tag:path,...] is required"; exit 1; }
[[ -z "$DATASET"  ]] && { echo "FATAL: --dataset is required"; exit 1; }
[[ -z "$OUTPUT"   ]] && { echo "FATAL: --output is required"; exit 1; }
[[ -f "$DATASET"  ]] || { echo "FATAL: dataset not found: $DATASET"; exit 1; }

mkdir -p "$OUTPUT"

PYTHON="${PRL_PYTHON:-python}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================================"
echo "Frame eval sweep"
echo "  policies:    $POLICIES"
echo "  dataset:     $DATASET"
echo "  secretary:   $SECRETARY"
echo "  output:      $OUTPUT"
echo "  gpus:        $GPUS"
echo "  max_q:       $MAX_QUESTIONS  max_turns: $MAX_TURNS"
echo "  agents:      $ACTORS actors + $JUDGES judges  ($SESSIONS_PER_GPU sessions/GPU)"
echo "============================================================"

# Run each (cell, policy) pair sequentially.
IFS=',' read -ra PAIRS <<< "$POLICIES"
for pair in "${PAIRS[@]}"; do
  cell="${pair%%:*}"
  policy="${pair#*:}"
  out="$OUTPUT/cell_${cell}.json"

  if [[ -s "$out" ]]; then
    acc=$("$PYTHON" -c "import json; print(json.load(open('$out'))['accuracy'])")
    echo "  [skip] cell $cell already evaluated (acc=$acc)"
    continue
  fi

  echo
  echo "=== Cell $cell  policy=$policy ==="
  cd "$PROJECT_DIR"
  "$PYTHON" -m eval.frame \
    --cell "$cell" \
    --policy "$policy" \
    --dataset "$DATASET" \
    --secretary "$SECRETARY" \
    --output "$out" \
    --gpus "$GPUS" \
    --sessions-per-gpu "$SESSIONS_PER_GPU" \
    --actors "$ACTORS" \
    --judges "$JUDGES" \
    --max-turns "$MAX_TURNS" \
    --max-questions "$MAX_QUESTIONS" \
    --secretary-tp "$SEC_TP" \
    "${EXTRA[@]}"
done

# Summary table
echo
echo "============================================================"
echo "Frame eval summary  (dataset: $DATASET)"
echo "============================================================"
"$PYTHON" - <<EOF
import json, os, sys
from pathlib import Path
out_dir = Path("$OUTPUT")
rows = []
for f in sorted(out_dir.glob("cell_*.json")):
    r = json.load(open(f))
    rows.append((r["cell"], r["cell_label"], r["accuracy"],
                 r["n_correct"], r["n_scored"], r.get("sessions", 0),
                 r.get("actor_posts_total", 0)))
print(f"{'cell':<6}{'label':<18}{'acc':>8}  {'correct/scored':>16}  {'sess':>6}  {'posts':>7}")
print("-" * 70)
for cell, label, acc, c, n, sess, posts in rows:
    print(f"{cell:<6}{label:<18}{acc:>8.4f}  {f'{c}/{n}':>16}  {sess:>6}  {posts:>7}")
print()

# Save summary.json
summary = {
    "dataset": "$DATASET",
    "secretary": "$SECRETARY",
    "rows": [
        {"cell": c, "cell_label": l, "accuracy": a,
         "n_correct": cor, "n_scored": n, "sessions": s,
         "actor_posts_total": p}
        for c, l, a, cor, n, s, p in rows
    ],
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"summary → {out_dir}/summary.json")
EOF
