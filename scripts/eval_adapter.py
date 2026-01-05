import json
import os
import sys
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
    cfg_path = sys.argv[1]
    adapter_path = sys.argv[2] if len(sys.argv) > 2 else None

    cfg = load_cfg(cfg_path)
    model_name = cfg["model_name"]
    task_name = cfg["task_name"]
    outdir = cfg["output_dir"]
    max_len = cfg.get("max_length", 512)

    os.makedirs(outdir, exist_ok=True)

    # Prefer validation if it exists; otherwise fall back to train
    split = "validation"
    try:
        examples = load_mc_task(task_name, split=split)
    except Exception:
        split = "train"
        examples = load_mc_task(task_name, split=split)

    res = eval_mc_accuracy(model_name=model_name, examples=examples, max_length=max_len, adapter_path=adapter_path)

    payload = {
        "model_name": model_name,
        "task": task_name,
        "split": split,
        "adapter_path": adapter_path,
        "accuracy": res.accuracy,
        "n": res.n,
    }
    outpath = os.path.join(outdir, f"{'base' if adapter_path is None else 'adapter'}_{task_name}.json")
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
