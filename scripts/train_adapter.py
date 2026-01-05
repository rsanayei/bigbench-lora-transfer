import os
import time
import json
import yaml
import argparse

from bblt.data import load_sft_texts
from bblt.train import train_lora_sft

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        with open(cfg["base"], "r") as f:
            base = yaml.safe_load(f)
        base.update({k: v for k, v in cfg.items() if k != "base"})
        return base
    return cfg

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cfg_path")
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    # optional LoRA overrides for Phase 2
    p.add_argument("--lora_r", type=int, default=None)
    args = p.parse_args()

    cfg = load_cfg(args.cfg_path)

    # Apply overrides (CLI beats YAML)
    if args.max_train_examples is not None:
        cfg["train"]["max_train_examples"] = args.max_train_examples
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r

    task_name = cfg["task_name"]
    base_outdir = cfg["output_dir"]
    model_name = cfg["model_name"]
    max_len = cfg.get("max_length", 512)
    seed = cfg.get("seed", 42)

    train_cfg = cfg["train"]
    lora_cfg = cfg["lora"]

    max_train = train_cfg.get("max_train_examples", None)

    # Unique run directory (include split-critical params)
    run_id = (
        f"{task_name}_n{max_train or 'full'}_lr{train_cfg['lr']}_"
        f"r{lora_cfg['r']}_a{lora_cfg['alpha']}_d{lora_cfg['dropout']}_"
        f"seed{seed}_{int(time.time())}"
    )
    outdir = os.path.join(base_outdir, "runs", run_id)
    os.makedirs(outdir, exist_ok=True)

    train_texts = load_sft_texts(task_name, split="train", max_examples=max_train)

    out = train_lora_sft(
        model_name=model_name,
        output_dir=outdir,
        train_texts=train_texts,
        seed=seed,
        max_length=max_len,
        per_device_batch_size=train_cfg["per_device_batch_size"],
        grad_accum_steps=train_cfg["grad_accum_steps"],
        lr=float(train_cfg["lr"]),
        epochs=int(train_cfg["epochs"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        lora_r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
    )

    print(f"Saved adapter to: {out.adapter_dir}")

    with open(os.path.join(outdir, "done.json"), "w") as f:
        json.dump({"adapter_dir": out.adapter_dir, "train_samples": out.train_samples}, f, indent=2)
