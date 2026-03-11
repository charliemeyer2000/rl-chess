import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_from_disk
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from rewards import format_reward, legality_reward, quality_reward

DATA_DIR = os.path.join(f"/scratch/{os.environ['USER']}", "rl-chess", "data")
CHECKPOINT_DIR = os.environ.get(
    "RV_CHECKPOINT_DIR",
    os.path.join(f"/scratch/{os.environ['USER']}", ".rv", "checkpoints", "chess-grpo"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    args = parser.parse_args()

    print(f"Loading dataset from {DATA_DIR}/grpo_dataset")
    dataset = load_from_disk(os.path.join(DATA_DIR, "grpo_dataset"))
    print(f"GRPO dataset: {len(dataset)} examples")

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    output_dir = os.path.join(CHECKPOINT_DIR, "grpo_output")
    config = GRPOConfig(
        output_dir=output_dir,
        use_vllm=True,
        vllm_mode="server",
        num_generations=args.num_generations,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_steps=args.max_steps,
        max_completion_length=384,
        max_prompt_length=512,
        beta=0.0,
        reward_weights=[0.1, 0.4, 0.5],
        bf16=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=3,
        report_to="wandb",
        run_name="chess-grpo",
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        args=config,
        reward_funcs=[format_reward, legality_reward, quality_reward],
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("Merging adapter...")
    merged_path = os.path.join(CHECKPOINT_DIR, "merged")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged GRPO model saved to {merged_path}")


if __name__ == "__main__":
    main()
