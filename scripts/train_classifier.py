# =============================================================================
# UNIFIED TRAINING SCRIPT FOR VIDEO CLASSIFICATION MODELS
# =============================================================================
# This script trains one of the video classification models (TimeSFormer,
# VideoMAE) on the HMDB_simp dataset. It uses configuration files
# (base.yaml + model-specific YAML) and supports command‑line overrides for
# dataset root, device, and augmentation mode.
#
# The training pipeline includes:
#   1. Loading and merging configuration (base + model-specific).
#   2. Creating train/val/test DataLoaders (stratified split, 70/15/15).
#   3. Building the model with the appropriate number of classes.
#   4. Setting up the trainer (optimizer, scheduler, early stopping, TensorBoard).
#   5. Running the training loop.
#   6. Saving training curves (loss/accuracy) as a PNG image.
#
# Usage:
#     python scripts/train_classifier.py --config configs/timesformer.yaml
#     python scripts/train_classifier.py --config configs/videomae.yaml --augment_mode spatial
# =============================================================================

import argparse
import torch
import sys
import os

# Add project root to Python path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.hmdb_dataset import get_dataloaders
from training.trainer import Trainer, set_seed
from models.timesformer import TimeSFormer
from models.videomae import VideoMAE
from configs.config_utils import load_config   # shared config loader
from utils.run_utils import make_run_dir

# Mapping from model name strings to the corresponding model class.
MODEL_BUILDERS = {
    "timesformer": TimeSFormer,
    "videomae": VideoMAE,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a video classification model on HMDB_simp."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model-specific YAML configuration file "
             "(e.g., configs/timesformer.yaml).",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Override the dataset root directory (absolute path).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the training device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--augment_mode",
        type=str,
        default=None,
        choices=["none", "spatial", "temporal"],
        help="Override the augmentation mode (none / spatial / temporal). "
             "If not provided, the value from the config file is used, "
             "with a default of 'none'.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional extra tag identifying this run (e.g. an experiment "
             "name), used in the output directory name.",
    )
    args = parser.parse_args()

    # 1. Load and merge configuration (base.yaml + model-specific YAML)
    config = load_config(args.config)
    set_seed(config.get("seed", 42))   # reproducibility

    # 2. Extract settings from the merged config
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})

    # Dataset root: command‑line override or value from config
    dataset_root = args.dataset_root if args.dataset_root else data_cfg.get(
        "dataset_root", "HMDB_simp"
    )
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(os.getcwd(), dataset_root)

    # Model‑specific parameters (must be present in the model YAML)
    model_name = config.get("model_name")
    num_frames = config.get("num_frames", 8)
    image_size = config.get("image_size", 224)

    # Data loading parameters
    sampling = data_cfg.get("sampling", "uniform")
    # Augmentation mode: command‑line overrides config, config overrides default "none"
    augment_mode = args.augment_mode   # if None, it will be passed as None (no augmentation)

    # 3. Create DataLoaders (train, val, test)
    #    The test loader is included for completeness but not used during training.
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        dataset_root=dataset_root,
        model_name=model_name,
        batch_size=train_cfg.get("batch_size", 8),
        seed=config.get("seed", 42),
        sampling=sampling,
        num_frames=num_frames,
        num_workers=data_cfg.get("num_workers", 2),
        augment_mode=augment_mode,
    )

    # 4. Build the model (the classification head is automatically sized to the
    #    number of classes in the dataset).
    model_class = MODEL_BUILDERS[model_name]
    model = model_class(num_classes=len(class_names))

    # 5. Prepare the trainer configuration (flattened for the Trainer class)
    trainer_config = {
        "epochs": train_cfg.get("epochs", 20),
        "lr": train_cfg.get("lr", 1e-4),
        "weight_decay": train_cfg.get("weight_decay", 0.01),
        "patience": train_cfg.get("patience", 5),
        "lr_scheduler": train_cfg.get("lr_scheduler", "cosine"),
        "warmup_epochs": train_cfg.get("warmup_epochs", 0),
        "checkpoint_path": os.path.join(
            log_cfg.get("checkpoint_dir", "checkpoints"),
            f"{model_name}_best.pt",
        ),
        "log_dir": os.path.join(log_cfg.get("log_dir", "logs"), model_name),
        "device": args.device if args.device else train_cfg.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        ),
    }

    # Print a summary of the effective configuration
    print(f"Dataset root: {dataset_root}")
    print(f"Model: {model_name} | Frames: {num_frames} | Image size: {image_size}")
    print(f"Device: {trainer_config['device']} | Batch size: {train_cfg.get('batch_size')}")

    # 6. Create the trainer and start training
    trainer = Trainer(model, train_loader, val_loader, trainer_config)
    trainer.train()

    # 7. Save the training curves (loss and accuracy) as a PNG image, inside a
    #    dedicated, timestamped run directory under outputs/
    run_dir = make_run_dir("train_classifier", model_name, args.run_tag)
    save_path = os.path.join(run_dir, "training_curves.png")
    trainer.plot_history(save_path=save_path)


if __name__ == "__main__":
    main()