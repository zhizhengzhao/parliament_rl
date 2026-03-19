#!/bin/bash
# Launch vLLM instances — one per GPU.
#
# Usage:
#   bash launch_vllm.sh <MODEL_PATH> [GPU_IDS]
#
# Examples:
#   bash launch_vllm.sh /data/models/Qwen3.5-9B              # all 8 GPUs (0-7)
#   bash launch_vllm.sh /data/models/Qwen3.5-9B 0,1,2,3      # GPUs 0-3
#   bash launch_vllm.sh /data/models/Qwen3.5-9B 2,5,7         # specific GPUs
#
# GPU k → port (8000 + k).  Logs go to vllm_gpu{k}.log.
# Stop all:  pkill -f 'vllm serve'

set -e

if [ -z "$1" ]; then
    echo "Usage: bash launch_vllm.sh <MODEL_PATH> [GPU_IDS]"
    echo "  GPU_IDS: comma-separated, e.g. 0,1,2,3 (default: 0,1,2,3,4,5,6,7)"
    exit 1
fi

MODEL="$1"
GPU_IDS=${2:-"0,1,2,3,4,5,6,7"}

IFS=',' read -ra GPUS <<< "$GPU_IDS"

echo "Launching ${#GPUS[@]} vLLM instances"
echo "Model: $MODEL"
echo "GPUs:  $GPU_IDS"
echo ""

for GPU_ID in "${GPUS[@]}"; do
    PORT=$((8000 + GPU_ID))
    echo "GPU $GPU_ID → port $PORT"
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup vllm serve "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.90 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        > "vllm_gpu${GPU_ID}.log" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "All instances launched. Check vllm_gpu*.log for status."
echo "To stop all: pkill -f 'vllm serve'"
