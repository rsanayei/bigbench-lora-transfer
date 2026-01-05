import argparse
import json
import os
import yaml

from bblt.data import load_mc_task
from bblt.eval_mc import eval_mc_accuracy


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
    p.add_argument("cfg_path", help="Path to eval config YAML (can reference base)")
    p.add_argument("--adapter", default=None, help="Path to PEFT adapter dir (optional)")
    p.add_argument("--split", default=None, choices=["train", "validation"], help="Override eval split")
    p.add_argument("--outdir", default=None, help="Override output directory")
    args = p.parse_args()

    cfg = load_cfg(args.cfg_path)
    model_name = cfg["model_name"]
    task_name = cfg["task_name"]
    max_len = cfg.get("max_length", 512)

    outdir = args.outdir or cfg["output_dir"]
    os.makedirs(outdir, exist_ok=True)

    # CLI split overrides YAML; otherwise use YAML or default validation
    split = args.split or cfg.get("eval_split", "validation")
    try:
        examples = load_mc_task(task_name, split=split)
    except Exception:
        # fallback
        split = "train"
        examples = load_mc_task(task_name, split=split)

    res = eval_mc_accuracy(
        model_name=model_name,
        examples=examples,
        max_length=max_len,
        adapter_path=args.adapter,
    )

    payload = {
        "model_name": model_name,
        "task": task_name,
        "split": split,
        "adapter_path": args.adapter,
        "accuracy": res.accuracy,
        "n": res.n,
    }

    tag = "base" if args.adapter is None else "adapter"
    outpath = os.path.join(outdir, f"{tag}_{task_name}_{split}.json")
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
