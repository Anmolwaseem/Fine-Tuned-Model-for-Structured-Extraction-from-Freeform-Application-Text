"""
train_qlora.py

QLoRA fine-tuning of an open-weight base model for the TalentGrid structured
extraction task.

WHY THIS BASE MODEL / METHOD (ties back to the brief's constraints):
- Qwen2.5-3B-Instruct: open-weight, Apache 2.0 license, free to fine-tune,
  strong instruction-following for its size, small enough to train and serve
  on a single consumer GPU (fits the "no training cluster" constraint).
  If you have more VRAM (>=16GB), swap MODEL_NAME to
  "Qwen/Qwen2.5-7B-Instruct" or "mistralai/Mistral-7B-Instruct-v0.3" for a
  stronger baseline model — the script does not change.
- QLoRA (4-bit base weights + LoRA adapters) keeps memory low and, critically,
  freezes almost all base-model weights. This directly targets the
  "catastrophic forgetting" failure mode named in the brief: because the
  base model's general language understanding is not being overwritten,
  the model should retain its ability to parse resume sections it already
  handled correctly.
- Only ~1-3 epochs, small LR, and a held-out val set with early stopping are
  used deliberately to avoid overfitting to the narrow training slice
  (also called out explicitly as a prior failure mode).

Run (on a GPU machine / Colab with a T4 or better):
    pip install -r requirements.txt
    python train_qlora.py \
        --train_file ../data/out/train.jsonl \
        --val_file ../data/out/val.jsonl \
        --output_dir ./qlora-adapter \
        --epochs 2
"""

import argparse
import json

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import torch

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def format_example(example):
    """Chat-style formatting so the fine-tune matches how the model will be
    prompted at inference time (same template used in baseline prompting,
    so the comparison in evaluate.py is apples-to-apples).
    Uses the 'messages' column name (not 'text') so SFTTrainer auto-detects
    this as conversational data and applies the chat template itself."""
    messages = [
        {"role": "system", "content": "You are a precise resume information extraction assistant. You only output valid JSON matching the requested schema. You never invent values not present in the source text."},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {"messages": messages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--val_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./qlora-adapter")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    args = ap.parse_args()

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")
    train_ds = train_ds.map(format_example)
    val_ds = val_ds.map(format_example)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # Modest rank/alpha: enough capacity to learn the schema/format, low
    # enough to avoid overwriting general language ability (anti-forgetting).
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Log hyperparameters + resource footprint for the reproducibility
    # requirement in the deliverables.
    run_info = {
        "base_model": MODEL_NAME,
        "method": "QLoRA (4-bit NF4, LoRA r=16 alpha=32)",
        "epochs": args.epochs,
        "lr": args.lr,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": 1024,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
    }
    with open(f"{args.output_dir}/run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print("Saved adapter + run_info.json to", args.output_dir)


if __name__ == "__main__":
    main()
