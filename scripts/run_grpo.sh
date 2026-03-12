#!/bin/bash
set -e

pip install vllm 2>&1 | tail -5

CHECKPOINTS_BASE=$(dirname "$RV_CHECKPOINT_DIR")
SFT_MODEL_PATH="$CHECKPOINTS_BASE/chess-sft/merged"

if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "WARNING: SFT checkpoint not found at $SFT_MODEL_PATH"
    echo "Falling back to base model (results will be worse without SFT warmstart)"
    SFT_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
fi

echo "Model: $SFT_MODEL_PATH"
echo "Starting GRPO training (colocate mode — vLLM shares GPU with trainer)..."

python scripts/grpo_train.py \
    --model_path "$SFT_MODEL_PATH" \
    "$@"
