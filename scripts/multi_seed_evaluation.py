# =============================================================================
# MULTI‑SEED STATISTICAL ROBUSTNESS CHECK (PART C.1)
# =============================================================================
# This script runs the `multi_seed_evaluation` function from `evaluation.metrics`
# to train and evaluate the best classification model (VideoMAE) with three
# different random seeds (42, 123, 456). It reports the mean ± standard deviation
# for top‑1 and top‑5 accuracy, as required by the project specification.
#
# The results are saved to `outputs/multi_seed_evaluation/<run>/multi_seed_results.json`
# for later use in the report.
#
# Usage:
#     python scripts/multi_seed_evaluation.py
# =============================================================================

import sys
import os
import json
import torch

# Add project root to Python path for local imports
sys.path.append('.')

from evaluation.metrics import multi_seed_evaluation
from models.videomae import build_videomae
from utils.run_utils import make_run_dir


def main() -> None:
    """
    Run the multi-seed evaluation for the VideoMAE model on HMDB_simp.
    """
    # ── Configuration (hyperparameters match the best model from Part A) ─────
    DATASET_ROOT = "HMDB_simp"                     # path to the dataset
    MODEL_NAME   = "videomae"                      # model identifier (for logging)
    SEEDS        = [42, 123, 456]                 # three random seeds
    EPOCHS       = 20                             # full training epochs per seed
    BATCH_SIZE   = 12                             # batch size used for VideoMAE
    LR           = 1e-4                           # initial learning rate
    PATIENCE     = 5                              # early stopping patience
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running multi‑seed evaluation on {DEVICE} …")

    # Call the multi‑seed evaluation function (defined in evaluation/metrics.py)
    results = multi_seed_evaluation(
        build_model_fn=lambda: build_videomae(num_classes=25),
        dataset_root=DATASET_ROOT,
        model_name=MODEL_NAME,
        seeds=SEEDS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE,
        device=DEVICE,
    )

    # ── Print the final summary to the console ───────────────────────────────
    print("\n" + "=" * 60)
    print("STATISTICAL ROBUSTNESS CHECK - VIDEOMAE (BEST MODEL)")
    print("=" * 60)
    print(f"Seeds used        : {SEEDS}")
    print(f"Top-1 Accuracy    : {results['top1_mean']:.2f}% ± {results['top1_std']:.2f}%")
    print(f"Top-5 Accuracy    : {results['top5_mean']:.2f}% ± {results['top5_std']:.2f}%")
    print("=" * 60)

    # ── Save the results as JSON for later reference in the report ────────────
    run_dir = make_run_dir("multi_seed_evaluation", MODEL_NAME)
    out = {
        "model":      MODEL_NAME,
        "seeds":      SEEDS,
        "top1_mean":  results["top1_mean"],
        "top1_std":   results["top1_std"],
        "top5_mean":  results["top5_mean"],
        "top5_std":   results["top5_std"],
        "per_seed":   results.get("per_seed", []),   # per‑seed records if available
    }
    save_path = os.path.join(run_dir, "multi_seed_results.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[INFO] Results saved to {save_path}")


if __name__ == "__main__":
    main()