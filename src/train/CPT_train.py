import argparse
import logging
import os
import shutil
from pathlib import Path
import datetime as dt
import numpy as np
import random

import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    set_seed,
)

LOG = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    # Path args
    p.add_argument("--tokenized_dataset_path", required=True, help="Path to saved arrow dataset")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--tag", default="CPT_finetune")
    
    # Train hyperparams
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--deepspeed_config", default=None)
    p.add_argument("--save_strategy", default="no", choices=["no", "steps", "epoch"])
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--bf16", action="store_true")
    # p.add_argument("--tf32", action="store_true")

    # Wandb
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    
    return p.parse_args()
def seed_everything(seed: int = 42):
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")


def main():
    args = parse_args()
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # set_seed(args.seed)
    seed_everything(args.seed)

    # if args.tf32:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    # --- Load Dataset ---
    LOG.info(f"Loading dataset from {args.tokenized_dataset_path}")
    train_ds = load_from_disk(args.tokenized_dataset_path)
    total_tokens = sum(sum(x) for x in train_ds["attention_mask"])
    LOG.info("Total tokens in train set (excluding padding): %d", total_tokens)
    # We need tokenizer for collator pad_token
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    collator = DataCollatorWithPadding(tokenizer)

    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else None,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Trainer ---
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "temp_checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.wandb_run_name,
        bf16=args.bf16 and torch.cuda.is_available(),
        deepspeed=args.deepspeed_config,
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=collator)
    
    LOG.info("Starting training...")
    trainer.train()

    # --- Save Final ---
    # 保存路径加上tag
    tag = (
        f"{args.tag}_{dt.datetime.now().strftime('%m%d_%H%M%S')}"
        if args.tag
        else dt.datetime.now().strftime('%m%d_%H%M%S')
    )
    final_dir = os.path.join(args.output_dir, tag, "final_model")
    if (not dist.is_initialized()) or (dist.get_rank() == 0):
        LOG.info(f"Saving final model to {final_dir}")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        # 保存args和数据相关信息比如total_tokens
        args_path = os.path.join(args.output_dir, tag, "training_args.txt")
        with open(args_path, "w") as f:
            for k, v in sorted(vars(args).items()):
                f.write(f"{k}: {v}\n")  
            f.write(f"total_tokens: {total_tokens}\n") 
            
    # Ensure processes sync before exiting
    if dist.is_initialized():
        dist.barrier()

if __name__ == "__main__":
    main()
