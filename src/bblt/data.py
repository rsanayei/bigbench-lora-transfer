from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset

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
