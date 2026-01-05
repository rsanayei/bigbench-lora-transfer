from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import MCExample

@dataclass
class EvalResult:
    accuracy: float
    n: int

@torch.no_grad()
def score_option_logprob(model, tokenizer, prompt: str, option: str, max_length: int) -> float:
    # Encode prompt and full prompt+option
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids
    full_ids = tokenizer(prompt + option, return_tensors="pt", truncation=True, max_length=max_length).input_ids

    prompt_ids = prompt_ids.to(model.device)
    full_ids = full_ids.to(model.device)

    # We want log P(option_tokens | prompt)
    # So compute logits over full_ids[:-1], gather logprobs for full_ids[1:]
    outputs = model(full_ids)
    logits = outputs.logits  # [1, T, V]
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]
    target = full_ids[:, 1:]  # [1, T-1]

    # Only sum over positions that correspond to option tokens
    option_start = prompt_ids.shape[1]
    # token positions in target aligned with full_ids[1:], so option begins at index option_start-1 there
    start_in_target = option_start - 1
    if start_in_target < 0 or start_in_target >= target.shape[1]:
        return float("-inf")

    gathered = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    option_logprob = gathered[:, start_in_target:].sum().item()
    return float(option_logprob)

@torch.no_grad()
def eval_mc_accuracy(
    model_name: str,
    examples: List[MCExample],
    max_length: int = 512,
    adapter_path: str | None = None,
) -> EvalResult:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    correct = 0
    for ex in examples:
        scores = [score_option_logprob(model, tokenizer, ex.prompt, opt, max_length) for opt in ex.options]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        correct += int(pred == ex.label_idx)

    return EvalResult(accuracy=correct / max(1, len(examples)), n=len(examples))
