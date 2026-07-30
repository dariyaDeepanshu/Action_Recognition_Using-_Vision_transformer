# =============================================================================
# EVALUATE A TRAINED CLASSIFICATION MODEL ON HMDB_simp
# =============================================================================
# This script evaluates one of the trained models (TimeSFormer, VideoMAE)
# on the HMDB_simp test set. It loads the model checkpoint, computes
# top‑1 / top‑5 accuracy, per‑class precision/recall/F1, macro/weighted F1,
# and a confusion matrix heatmap. Additionally, it reports the number of
# parameters and (optionally) FLOPs per clip using the `thop` library.
#
# Usage:
#     python scripts/evaluate_classifier.py \
#         --model_name videomae \
#         --checkpoint checkpoints/videomae_best.pt \
#         --config configs/videomae.yaml
#
# Dependencies (optional for FLOPs):
#     pip install thop
# =============================================================================

import argparse
import torch
import sys
import json
import os
from typing import Dict, Any

# Add project root to Python path for local imports
sys.path.append('.')

from data.hmdb_dataset import get_dataloaders
from evaluation.metrics import evaluate_model
from models.timesformer import TimeSFormer
from models.videomae import VideoMAE
from training.trainer import set_seed
from configs.config_utils import load_config
from utils.run_utils import make_run_dir


# Mapping from model name strings to the corresponding model class.
MODEL_BUILDERS: Dict[str, Any] = {
    "timesformer": TimeSFormer,
    "videomae": VideoMAE,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classification model on HMDB_simp."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["timesformer", "videomae"],
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved PyTorch checkpoint (.pt file).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model-specific YAML configuration file "
             "(e.g., configs/timesformer.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional extra tag identifying this run (e.g. an experiment "
             "name), used in the output directory name.",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load and merge configuration (base.yaml + model‑specific YAML)
    # -------------------------------------------------------------------------
    cfg = load_config(args.config)
    set_seed(42)   # ensure reproducibility of data split

    # Extract settings from merged config
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    dataset_root = data_cfg.get("dataset_root", "HMDB_simp")
    sampling = data_cfg.get("sampling", "uniform")
    num_workers = data_cfg.get("num_workers", 2)
    batch_size = train_cfg.get("batch_size", 8)
    num_frames = cfg.get("num_frames")   # from model‑specific override
    if num_frames is None:
        raise ValueError(
            "num_frames must be specified in the model configuration "
            "(e.g., in timesformer.yaml)."
        )

    # -------------------------------------------------------------------------
    # Create test DataLoader (only test set is needed)
    # -------------------------------------------------------------------------
    _, _, test_loader, class_names = get_dataloaders(
        dataset_root=dataset_root,
        model_name=args.model_name,
        batch_size=batch_size,
        seed=42,
        sampling=sampling,
        num_frames=num_frames,
        num_workers=num_workers,
    )

    # -------------------------------------------------------------------------
    # Build model and load the checkpoint
    # -------------------------------------------------------------------------
    model = MODEL_BUILDERS[args.model_name](num_classes=len(class_names))
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device)
    print(f"[INFO] Loading checkpoint from: {args.checkpoint}")

    # -------------------------------------------------------------------------
    # Parameter count (total and trainable)
    # -------------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters : {total_params:,} total | {trainable_params:,} trainable")

    # -------------------------------------------------------------------------
    # FLOPs measurement (optional, requires `thop`)
    # -------------------------------------------------------------------------
    gflops = None
    try:
        from thop import profile
        # Create a dummy input of the correct shape (B=1, T, C, H, W)
        image_size = cfg.get("image_size", 224)
        dummy_input = torch.zeros(
            1, num_frames, 3, image_size, image_size
        ).to(args.device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
        print(f"FLOPs per clip   : {gflops:.2f} GFLOPs")
    except ImportError:
        print("[WARN] 'thop' not installed - skipping FLOPs measurement. "
              "Run 'pip install thop' to enable.")
    except Exception as e:
        print(f"[WARN] FLOPs measurement failed: {e}")

    # -------------------------------------------------------------------------
    # Run evaluation (computes metrics and saves confusion matrix), writing
    # into a dedicated, timestamped run directory under outputs/
    # -------------------------------------------------------------------------
    run_dir = make_run_dir("evaluate_classifier", args.model_name, args.run_tag)
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=args.device,
        model_name=args.model_name,
        save_dir=run_dir,
    )

    # Print final test results to console
    print(f"\nFinal Test Results for {args.model_name}:")
    print(f"Top-1: {results['top1']*100:.2f}%")
    print(f"Top-5: {results['top5']*100:.2f}%")
    print(f"Macro F1: {results['macro_f1']*100:.2f}%")
    print(f"Weighted F1: {results['weighted_f1']*100:.2f}%")

    # -------------------------------------------------------------------------
    # Save all results to a JSON file (including parameter counts and FLOPs)
    # -------------------------------------------------------------------------
    save_path = os.path.join(run_dir, "metrics.json")

    results_serializable = {
        "top1": results["top1"],
        "top5": results["top5"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "per_class": results["report"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "gflops_per_clip": round(gflops, 3) if gflops is not None else None,
    }

    with open(save_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"[INFO] Metrics saved to {save_path}")


if __name__ == "__main__":
    main()