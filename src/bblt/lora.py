from __future__ import annotations
from peft import LoraConfig, TaskType

def build_lora_config(r: int, alpha: int, dropout: float, target_modules: list[str]) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
