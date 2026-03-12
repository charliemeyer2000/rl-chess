#!/bin/bash
set -e

pip install vllm 2>&1 | tail -5

CHECKPOINTS_BASE="/scratch/$USER/.rv/checkpoints"
MODEL_PATH=""

for name in chess-grpo chess-sft; do
    path="$CHECKPOINTS_BASE/$name/merged"
    if [ -d "$path" ]; then
        MODEL_PATH="$path"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No trained checkpoint found. Run SFT or GRPO training first."
    exit 1
fi

echo "Serving model: $MODEL_PATH"
echo "To forward to your local machine:"
echo "  rv forward 8000 chess-serve"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9
