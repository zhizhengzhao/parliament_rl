#!/bin/bash
# Sweep base + every iter's merged policy through eval/gpqa.py.
#
# `--subset` here is forwarded as `eval/gpqa.py --data`, so it accepts
# either a HF subset name (`gpqa_diamond` | `gpqa_main` | `gpqa_extended`)
# or a path to a local CSV.
#
# Usage:
#   eval/gpqa_sweep.sh --name nrun_v1                    # required
#   eval/gpqa_sweep.sh --name nrun_v1 --subset gpqa_main
#   eval/gpqa_sweep.sh --name nrun_v1 --n 8 --temperature 0.7
set -euo pipefail

NAME=""
RUN_DIR=""
SUBSET="gpqa_diamond"
N=1
# Qwen3.5 thinking mode: 0.6 is the recommended sampling temperature
# (matches eval/gpqa.py default). Use --temperature 0.0 to force greedy
# decoding for low-variance head-to-head comparisons.
TEMP=0.6
GPU=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)    NAME="$2";    shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --subset)  SUBSET="$2";  shift 2 ;;
    --n)       N="$2";       shift 2 ;;
    --temperature) TEMP="$2"; shift 2 ;;
    --gpu)     GPU="$2";     shift 2 ;;
    *)         EXTRA+=("$1"); shift ;;
  esac
done

DATA_ROOT="${PRL_DATA_ROOT:-data}"

if [[ -z "$RUN_DIR" ]]; then
  [[ -z "$NAME" ]] && { echo "FATAL: --name or --run-dir required"; exit 1; }
  # Pick the latest top-level dir whose state.json has completed > 0.
  RUN_DIR=$(ls -td "$DATA_ROOT"/${NAME}_0* 2>/dev/null | head -1)
fi
[[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]] && { echo "FATAL: run dir not found for '$NAME'"; exit 1; }

BASE_MODEL="${PRL_MODEL_PATH:-Qwen/Qwen3.5-9B}"
EVAL_DIR="$RUN_DIR/eval/$SUBSET"
mkdir -p "$EVAL_DIR"

echo "=== GPQA sweep ==="
echo "  RUN_DIR:  $RUN_DIR"
echo "  SUBSET:   $SUBSET"
echo "  n=$N  temperature=$TEMP  GPU=$GPU"

# Collect (tag, path) pairs: iter00 = base, iterNN = merged from each iteration.
declare -a PAIRS=("iter00:$BASE_MODEL")
for iter_run in $(ls -d "$DATA_ROOT"/${NAME}_iter*_* 2>/dev/null | sort); do
  merged="$iter_run/merged"
  if [[ -d "$merged" ]]; then
    tag=$(basename "$iter_run" | sed 's/.*_iter/iter/;s/_.*//')
    PAIRS+=("$tag:$merged")
  fi
done

echo "  models:"
for p in "${PAIRS[@]}"; do echo "    $p"; done
echo

# Sequential sweep — vLLM can't reload a different model in-process cheaply.
for pair in "${PAIRS[@]}"; do
  tag="${pair%%:*}"
  model="${pair#*:}"
  out="$EVAL_DIR/${tag}.json"

  if [[ -s "$out" ]]; then
    acc=$(python3 -c "import json; print(json.load(open('$out'))['accuracy'])")
    echo "  [skip] $tag already evaluated (acc=$acc)"
    continue
  fi

  echo "  [run]  $tag  ←  $model"
  CUDA_VISIBLE_DEVICES=$GPU python -m eval.gpqa \
    --model "$model" \
    --output "$out" \
    --data "$SUBSET" \
    --n "$N" \
    --temperature "$TEMP" \
    "${EXTRA[@]}"
done

# Summary table
echo
echo "=== Summary ($SUBSET, n=$N, temperature=$TEMP) ==="
printf "%-10s %-10s\n" tag accuracy
for pair in "${PAIRS[@]}"; do
  tag="${pair%%:*}"
  out="$EVAL_DIR/${tag}.json"
  if [[ -s "$out" ]]; then
    acc=$(python3 -c "import json; print(f\"{json.load(open('$out'))['accuracy']:.4f}\")")
    printf "%-10s %-10s\n" "$tag" "$acc"
  fi
done
