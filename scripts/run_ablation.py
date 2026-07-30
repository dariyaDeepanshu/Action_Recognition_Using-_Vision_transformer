# =============================================================================
# ONE‑FACTOR‑AT‑A‑TIME ABLATION STUDY FOR VIDEOMAE (PART B)
# =============================================================================
# This script trains and evaluates the VideoMAE model on HMDB_simp while varying
# a single factor at a time, keeping all other hyperparameters fixed.
#
# Factors ablated:
#   1. Temporal sampling strategy (uniform, random, dense)
#   2. Data augmentation (spatial, temporal) – compared to baseline (none)
#   3. Learning rate scheduler (cosine, step, warmup_cosine)
#
# For each configuration, the script runs training (using train_classifier.py)
# and then evaluation (using evaluate_classifier.py), extracts top‑1/top‑5
# accuracy from the output, and saves the results to
# outputs/run_ablation/<run>/ablation_results.json.
#
# Usage:
#     python scripts/run_ablation.py
# =============================================================================

import os
import sys
import json
import yaml
import tempfile
import subprocess
from typing import Dict, Any, List, Optional

sys.path.append('.')

from utils.run_utils import make_run_dir

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_CONFIG = "configs/base.yaml"          # base hyperparameters
MODEL_CONFIG = "configs/videomae.yaml"     # model‑specific overrides
DEVICE = "cuda"                            # training device


def run_experiment(
    config_updates: Dict[str, Any],
    experiment_name: str,
    extra_args: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Run a single ablation experiment: train + evaluate with the given config
    overrides and optional command-line arguments.

    The function:
        1. Loads and merges base.yaml and videomae.yaml.
        2. Applies the config updates (e.g., {"data.sampling": "random"}).
        3. Writes the merged config to a temporary YAML file.
        4. Calls train_classifier.py with that temporary config.
        5. Calls evaluate_classifier.py on the resulting best checkpoint.
        6. Parses top-1 and top-5 accuracy from the evaluation output.
        7. Deletes the temporary config file.

    Args:
        config_updates : Dictionary of configuration overrides (dot-notation keys,
                         e.g., "data.sampling": "random").
        experiment_name: Name of the experiment (used for logging, not saved).
        extra_args     : Additional command-line arguments to pass to
                         train_classifier.py (e.g., ["--augment_mode", "spatial"]).

    Returns:
        dict with keys "top1" and "top5" (accuracy percentages as floats).
    """
    # 1. Load base and model configurations
    with open(BASE_CONFIG, 'r') as f:
        base = yaml.safe_load(f)
    with open(MODEL_CONFIG, 'r') as f:
        model_cfg = yaml.safe_load(f)

    # Merge: model_cfg overrides base
    merged = base.copy()
    for k, v in model_cfg.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k].update(v)
        else:
            merged[k] = v

    # 2. Apply the ablation overrides (nested keys supported, e.g., "training.lr_scheduler")
    for k, v in config_updates.items():
        keys = k.split('.')
        target = merged
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = v

    # 3. Write temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(merged, f)
        temp_config = f.name

    # 4. Build and run the training command. --run_tag threads the experiment
    #    name through to train_classifier.py so its output directory under
    #    outputs/ (training curves) is identifiable as belonging to this
    #    ablation variant rather than just "videomae_<timestamp>".
    train_cmd = (
        f"python scripts/train_classifier.py --config {temp_config} "
        f"--device {DEVICE} --run_tag {experiment_name}"
    )
    if extra_args:
        train_cmd += " " + " ".join(extra_args)
    subprocess.run(train_cmd, shell=True, check=True)

    # 5. Evaluate the trained model (checkpoint is always saved as videomae_best.pt)
    eval_cmd = (
        f"python scripts/evaluate_classifier.py --model_name videomae "
        f"--checkpoint checkpoints/videomae_best.pt --config {temp_config} "
        f"--device {DEVICE} --run_tag {experiment_name}"
    )
    proc = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)

    # 6. Parse top‑1 and top‑5 from the evaluation output
    top1 = top5 = None
    for line in proc.stdout.split('\n'):
        if "Top-1:" in line:
            top1 = float(line.split(':')[-1].strip().replace('%', ''))
        if "Top-5:" in line:
            top5 = float(line.split(':')[-1].strip().replace('%', ''))
    if top1 is None or top5 is None:
        raise RuntimeError(f"Could not parse evaluation output for {experiment_name}")

    # 7. Clean up temporary file
    os.unlink(temp_config)

    return {"top1": top1, "top5": top5}


def main() -> None:
    """
    Run the three ablation studies sequentially and save the results.
    """
    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Baseline (uniform sampling, no augmentation, cosine scheduler)
    # -------------------------------------------------------------------------
    print("\n=== Baseline (uniform, no augmentation, cosine) ===")
    baseline = run_experiment({}, "baseline")
    results["baseline"] = baseline

    # -------------------------------------------------------------------------
    # 1. Temporal sampling (uniform already baseline)
    # -------------------------------------------------------------------------
    sampling_variants = {
        "random": {"data.sampling": "random"},
        "dense":  {"data.sampling": "dense"}
    }
    results["sampling"] = {}
    for name, cfg in sampling_variants.items():
        print(f"\n--- Sampling: {name} ---")
        res = run_experiment(cfg, f"sampling_{name}")
        results["sampling"][name] = res

    # -------------------------------------------------------------------------
    # 2. Data augmentation (baseline is no augmentation)
    # -------------------------------------------------------------------------
    # Use the --augment_mode command‑line flag (spatial / temporal)
    aug_modes = {"spatial": "spatial", "temporal": "temporal"}
    results["augmentation"] = {}
    for name, mode in aug_modes.items():
        print(f"\n--- Augmentation: {name} ---")
        res = run_experiment(
            {}, f"aug_{name}",
            extra_args=[f"--augment_mode {mode}"]
        )
        results["augmentation"][name] = res

    # -------------------------------------------------------------------------
    # 3. Learning rate scheduler (cosine is baseline)
    # -------------------------------------------------------------------------
    scheduler_variants = {
        "step": {"training.lr_scheduler": "step", "training.step_size": 5, "training.gamma": 0.1},
        "warmup_cosine": {"training.lr_scheduler": "warmup_cosine", "training.warmup_epochs": 2}
    }
    results["scheduler"] = {}
    for name, cfg in scheduler_variants.items():
        print(f"\n--- LR scheduler: {name} ---")
        res = run_experiment(cfg, f"sched_{name}")
        results["scheduler"][name] = res

    # -------------------------------------------------------------------------
    # Save results to JSON
    # -------------------------------------------------------------------------
    run_dir = make_run_dir("run_ablation", "videomae")
    out_path = os.path.join(run_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # -------------------------------------------------------------------------
    # Print a summary table to the console
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS (Top-1 / Top-5)")
    print("=" * 60)
    print(f"Baseline (uniform, none, cosine): "
          f"{results['baseline']['top1']:.2f}% / {results['baseline']['top5']:.2f}%")
    print("\nSampling:")
    for name, m in results["sampling"].items():
        print(f"  {name:8s}: {m['top1']:.2f}% / {m['top5']:.2f}%")
    print("\nAugmentation:")
    for name, m in results["augmentation"].items():
        print(f"  {name:8s}: {m['top1']:.2f}% / {m['top5']:.2f}%")
    print("\nLR scheduler:")
    for name, m in results["scheduler"].items():
        print(f"  {name:12s}: {m['top1']:.2f}% / {m['top5']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()