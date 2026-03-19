#!/bin/bash
# Launch vLLM instances — one per GPU.
#
# Usage:
#   bash launch_vllm.sh <MODEL_PATH> [NUM_GPUS]
#
# Examples:
#   bash launch_vllm.sh /data/models/Qwen3-8B           # 8 GPUs (default)
#   bash launch_vllm.sh /data/models/Qwen3-8B 4         # 4 GPUs
#
# Each GPU k gets port (8000 + k).  Logs go to vllm_gpu{k}.log.
# Stop all:  pkill -f 'vllm serve'

set -e

if [ -z "$1" ]; then
    echo "Usage: bash launch_vllm.sh <MODEL_PATH> [NUM_GPUS]"
    exit 1
fi

MODEL="$1"
NUM_GPUS=${2:-8}
BASE_PORT=8000

echo "Launching $NUM_GPUS vLLM instances (ports $BASE_PORT–$((BASE_PORT + NUM_GPUS - 1)))"
echo "Model: $MODEL"
echo ""

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "GPU $i → port $PORT"
    CUDA_VISIBLE_DEVICES=$i nohup vllm serve "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.90 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        > "vllm_gpu${i}.log" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "All instances launched. Check vllm_gpu*.log for status."
echo "To stop all: pkill -f 'vllm serve'"
