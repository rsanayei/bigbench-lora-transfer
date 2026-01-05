from __future__ import annotations
import json
import os
from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import get_peft_model
from .lora import build_lora_config


@dataclass
class TrainOutput:
    adapter_dir: str
    train_samples: int


def train_lora_sft(
    *,
    model_name: str,
    output_dir: str,
    train_texts: list[str],
    seed: int,
    max_length: int,
    per_device_batch_size: int,
    grad_accum_steps: int,
    lr: float,
    epochs: int,
    warmup_ratio: float,
    weight_decay: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> TrainOutput:
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.config.use_cache = False

    lora_cfg = build_lora_config(lora_r, lora_alpha, lora_dropout, target_modules)
    model = get_peft_model(model, lora_cfg)

    ds = Dataset.from_dict({"text": train_texts})

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    ds_tok = ds.map(tok, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),  # simple heuristic
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds_tok, data_collator=collator)
    trainer.train()

    # Save adapter only (PEFT)
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save run metadata
    meta = {
        "model_name": model_name,
        "train_samples": len(train_texts),
        "seed": seed,
        "max_length": max_length,
        "batch_size": per_device_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "epochs": epochs,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": target_modules,
        },
    }
    with open(os.path.join(output_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return TrainOutput(adapter_dir=adapter_dir, train_samples=len(train_texts))
