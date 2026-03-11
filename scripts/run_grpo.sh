#!/bin/bash
set -e

CHECKPOINTS_BASE=$(dirname "$RV_CHECKPOINT_DIR")
SFT_MODEL_PATH="$CHECKPOINTS_BASE/chess-sft/merged"

if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "WARNING: SFT checkpoint not found at $SFT_MODEL_PATH"
    echo "Falling back to base model (results will be worse without SFT warmstart)"
    SFT_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
fi

echo "Model: $SFT_MODEL_PATH"
echo "Starting vLLM server on GPU 0..."

CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model "$SFT_MODEL_PATH" \
    --dtype bfloat16 \
    --gpu_memory_utilization 0.85 &
VLLM_PID=$!

echo "Waiting for vLLM server (PID: $VLLM_PID)..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready after ${WAITED}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server process died unexpectedly"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "vLLM server did not respond to health check within ${MAX_WAIT}s"
    echo "Proceeding anyway — TRL may handle connection internally"
fi

echo "Starting GRPO training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python scripts/grpo_train.py \
    --model_path "$SFT_MODEL_PATH" \
    "$@"
EXIT_CODE=$?

echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

exit $EXIT_CODE
