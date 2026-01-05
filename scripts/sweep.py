import itertools
import json
import os
import subprocess
from pathlib import Path

TRAIN_CFGS = {
    "metaphor_boolean": "configs/train_metaphor_boolean.yaml",
    "implicatures": "configs/train_implicatures.yaml",
}

EVAL_CFG = "configs/eval_identify_odd_metaphor.yaml"

# Phase 1
PH1_SIZES = [200, None]  # None => full
PH1_LRS = [1e-4, 2e-4, 5e-4]
PH1_SEED = 42

# Phase 2 (capacity check on best config per adapter)
PH2_RS = [8, 16, 32]

# Phase 3 (stability check)
PH3_SEED = 123


def run(cmd):
    print(" ".join(cmd))
    out = subprocess.check_output(cmd)
    return out.decode("utf-8")


def parse_eval_json(stdout: str) -> dict:
    # eval_adapter prints a single JSON object
    return json.loads(stdout.strip())


def latest_run_dir(task_outdir: str) -> str:
    runs = Path(task_outdir) / "runs"
    candidates = [p for p in runs.iterdir() if p.is_dir()]
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def eval_both_splits(adapter_dir: str | None) -> dict:
    res = {}
    for split in ["train", "validation"]:
        stdout = run(["python", "scripts/eval_adapter.py", EVAL_CFG, "--split", split] + ([] if adapter_dir is None else ["--adapter", adapter_dir]))
        payload = parse_eval_json(stdout)
        res[split] = {"accuracy": payload["accuracy"], "n": payload["n"]}
    return res


def main():
    # Where to store one combined leaderboard
    os.makedirs("outputs/sweeps", exist_ok=True)
    leaderboard_path = "outputs/sweeps/leaderboard.jsonl"

    # Evaluate base once (both splits)
    base_eval = eval_both_splits(adapter_dir=None)
    with open(leaderboard_path, "a") as f:
        f.write(json.dumps({"kind": "base", "eval": base_eval}) + "\n")
    print("Base eval:", base_eval)

    best_per_task = {}

    # -------- Phase 1 --------
    for task, cfg_path in TRAIN_CFGS.items():
        best = None

        for max_train, lr in itertools.product(PH1_SIZES, PH1_LRS):
            epochs = 2 if (max_train is not None and max_train <= 200) else 1

            cmd = ["python", "scripts/train_adapter.py", cfg_path, "--seed", str(PH1_SEED), "--lr", str(lr), "--epochs", str(epochs)]
            if max_train is not None:
                cmd += ["--max_train_examples", str(max_train)]
            run(cmd)

            # find adapter dir
            task_outdir = load_task_outdir_from_cfg(cfg_path)
            run_dir = latest_run_dir(task_outdir)
            adapter_dir = os.path.join(run_dir, "adapter")

            eval_res = eval_both_splits(adapter_dir)

            record = {
                "kind": "phase1",
                "task": task,
                "max_train_examples": max_train,
                "lr": lr,
                "epochs": epochs,
                "seed": PH1_SEED,
                "lora_r": 16,
                "adapter_dir": adapter_dir,
                "eval": eval_res,
            }
            with open(leaderboard_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # choose best by validation accuracy; tie-break by train accuracy
            score = (eval_res["validation"]["accuracy"], eval_res["train"]["accuracy"])
            if best is None or score > best["score"]:
                best = {"score": score, "record": record}

        best_per_task[task] = best["record"]
        print(f"Best Phase 1 for {task}: {best['score']} -> {best['record']['adapter_dir']}")

    # -------- Phase 2 --------
    for task, best_rec in best_per_task.items():
        cfg_path = TRAIN_CFGS[task]
        max_train = best_rec["max_train_examples"]
        lr = best_rec["lr"]
        epochs = best_rec["epochs"]

        for r in PH2_RS:
            cmd = ["python", "scripts/train_adapter.py", cfg_path, "--seed", str(PH1_SEED), "--lr", str(lr), "--epochs", str(epochs), "--lora_r", str(r)]
            if max_train is not None:
                cmd += ["--max_train_examples", str(max_train)]
            run(cmd)

            task_outdir = load_task_outdir_from_cfg(cfg_path)
            run_dir = latest_run_dir(task_outdir)
            adapter_dir = os.path.join(run_dir, "adapter")

            eval_res = eval_both_splits(adapter_dir)

            record = {
                "kind": "phase2",
                "task": task,
                "max_train_examples": max_train,
                "lr": lr,
                "epochs": epochs,
                "seed": PH1_SEED,
                "lora_r": r,
                "adapter_dir": adapter_dir,
                "eval": eval_res,
            }
            with open(leaderboard_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    # -------- Phase 3 (stability) --------
    for task, best_rec in best_per_task.items():
        cfg_path = TRAIN_CFGS[task]
        max_train = best_rec["max_train_examples"]
        lr = best_rec["lr"]
        epochs = best_rec["epochs"]
        r = best_rec["lora_r"]

        cmd = ["python", "scripts/train_adapter.py", cfg_path, "--seed", str(PH3_SEED), "--lr", str(lr), "--epochs", str(epochs), "--lora_r", str(r)]
        if max_train is not None:
            cmd += ["--max_train_examples", str(max_train)]
        run(cmd)

        task_outdir = load_task_outdir_from_cfg(cfg_path)
        run_dir = latest_run_dir(task_outdir)
        adapter_dir = os.path.join(run_dir, "adapter")

        eval_res = eval_both_splits(adapter_dir)

        record = {
            "kind": "phase3",
            "task": task,
            "max_train_examples": max_train,
            "lr": lr,
            "epochs": epochs,
            "seed": PH3_SEED,
            "lora_r": r,
            "adapter_dir": adapter_dir,
            "eval": eval_res,
        }
        with open(leaderboard_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def load_task_outdir_from_cfg(cfg_path: str) -> str:
    import yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        with open(cfg["base"], "r") as f:
            base = yaml.safe_load(f)
        base.update({k: v for k, v in cfg.items() if k != "base"})
        cfg = base
    return cfg["output_dir"]


if __name__ == "__main__":
    main()
