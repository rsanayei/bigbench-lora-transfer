from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset
from .prompts import format_sft_example

@dataclass(frozen=True)
class MCExample:
    prompt: str
    options: List[str]
    label_idx: int

def load_mc_task(task_name: str, split: str = "train", max_examples: Optional[int] = None) -> List[MCExample]:
    ds = load_dataset("tasksource/bigbench", task_name, split=split)
    examples: List[MCExample] = []
    for ex in ds:
        opts = ex["multiple_choice_targets"]
        scores = ex["multiple_choice_scores"]
        if not opts or not scores:
            continue
        label_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        examples.append(MCExample(prompt=ex["inputs"], options=opts, label_idx=label_idx))
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples

def load_sft_texts(task_name: str, split: str = "train", max_examples: Optional[int] = None) -> List[str]:
    ds = load_dataset("tasksource/bigbench", task_name, split=split)
    texts: List[str] = []
    for ex in ds:
        opts = ex["multiple_choice_targets"]
        scores = ex["multiple_choice_scores"]
        if not opts or not scores:
            continue
        label_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        answer = opts[label_idx]
        texts.append(format_sft_example(ex["inputs"], answer))
        if max_examples is not None and len(texts) >= max_examples:
            break
    return texts

