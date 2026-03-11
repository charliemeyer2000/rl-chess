import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = os.path.join(f"/scratch/{os.environ['USER']}", "rl-chess", "data")
CHECKPOINT_DIR = os.environ.get(
    "RV_CHECKPOINT_DIR",
    os.path.join(f"/scratch/{os.environ['USER']}", ".rv", "checkpoints", "chess-sft"),
)


def main():
    print(f"Loading dataset from {DATA_DIR}/sft_dataset")
    dataset = load_from_disk(os.path.join(DATA_DIR, "sft_dataset"))
    print(f"SFT dataset: {len(dataset)} examples")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    output_dir = os.path.join(CHECKPOINT_DIR, "adapter")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        max_seq_length=512,
        report_to="wandb",
        run_name="chess-sft",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("Merging adapter into base model...")
    merged_path = os.path.join(CHECKPOINT_DIR, "merged")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to {merged_path}")


if __name__ == "__main__":
    main()
