# =============================================================================
# CLASSIFICATION METRICS SUITE
# =============================================================================
# This module provides functions to evaluate video classification models on the
# HMDB_simp dataset. It computes:
#   - Top‑1 and Top‑5 accuracy
#   - Per‑class precision, recall, F1‑score (printed as a table)
#   - Macro‑averaged and weighted‑averaged F1‑score
#   - Confusion matrix (saved as a heatmap image)
#   - Statistical robustness check (multi‑seed evaluation mean ± std)
#
# The evaluation uses scikit‑learn metrics and the project's custom plotting
# utilities. The multi‑seed evaluation re‑trains the model from scratch for
# each seed, ensuring fresh data splits and weight initialisation.
# =============================================================================

import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Local imports
from .plots import plot_confusion_matrix
from data.hmdb_dataset import get_dataloaders
from training.trainer import Trainer, set_seed


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: list,
    device: torch.device,
    model_name: str = "model",
    save_dir: str = "outputs"
) -> dict:
    """
    Evaluate a trained classification model on the test set.

    Args:
        model       : PyTorch model (e.g., TimeSFormer, VideoMAE)
        test_loader : DataLoader for the test set
        class_names : list of class name strings
        device      : torch.device ('cuda' or 'cpu')
        model_name  : string used in print statements and the saved confusion matrix filename
        save_dir    : directory where the confusion matrix image will be saved

    Returns:
        dict containing:
            top1           : top-1 accuracy (float)
            top5           : top-5 accuracy (float)
            macro_f1       : macro-averaged F1-score
            weighted_f1    : weighted-averaged F1-score
            report         : full classification_report dictionary (per-class metrics)
            confusion_matrix: raw confusion matrix (numpy array)
            all_preds      : all predicted class indices
            all_labels     : all ground-truth labels
            all_probs      : all class probabilities (for top-5)
    """
    print(f"\n[INFO] Evaluating {model_name} on test set...")
    model.eval()
    model = model.to(device)

    all_labels = []   # ground‑truth labels
    all_preds  = []   # predicted class indices
    all_probs  = []   # class probabilities (for top‑5)

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(test_loader):
            videos = videos.to(device)
            outputs = model(pixel_values=videos)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)   # convert to probabilities
            preds = probs.argmax(dim=-1)            # most likely class

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Evaluated {batch_idx+1}/{len(test_loader)} batches...")

    # Convert lists to NumPy arrays for efficient metric computation
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.vstack(all_probs)   # (N_samples, num_classes)

    # ─── Core metrics ─────────────────────────────────────────────────────────
    top1 = accuracy_score(all_labels, all_preds)
    top5 = top_k_accuracy_score(all_labels, all_probs, k=5)

    # Detailed per‑class metrics (precision, recall, F1, support)
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Averaged F1 scores
    macro_f1    = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # ─── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  RESULTS: {model_name}")
    print("=" * 55)
    print(f"  Top-1 Accuracy : {top1*100:.2f}%")
    print(f"  Top-5 Accuracy : {top5*100:.2f}%")
    print(f"  Macro F1       : {macro_f1*100:.2f}%")
    print(f"  Weighted F1    : {weighted_f1*100:.2f}%")

    # Check against the project's minimum performance requirements
    top1_pass = "✓ PASS" if top1 >= 0.45 else "✗ FAIL (need ≥45%)"
    top5_pass = "✓ PASS" if top5 >= 0.75 else "✗ FAIL (need ≥75%)"
    print(f"\n  Top-1 target (≥45%): {top1_pass}")
    print(f"  Top-5 target (≥75%): {top5_pass}")
    print("=" * 55)

    # ─── Per‑class breakdown (formatted table) ────────────────────────────────
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*65}")
    for cls in class_names:
        if cls in report_dict:
            r = report_dict[cls]
            print(f"  {cls:<25} {r['precision']*100:>9.1f}% {r['recall']*100:>9.1f}% "
                  f"{r['f1-score']*100:>9.1f}% {int(r['support']):>10}")

    # ─── Confusion matrix heatmap ────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, model_name, save_dir=save_dir)

    return {
        "top1": top1, "top5": top5,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "report": report_dict,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
    }


def multi_seed_evaluation(
    build_model_fn,           # function that returns a fresh untrained model
    dataset_root: str,
    model_name: str,
    seeds: list = [42, 123, 456],
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    patience: int = 5,
    device: str = "cuda",
) -> dict:
    """
    Perform a statistical robustness check by training and evaluating the model
    with multiple random seeds. Reports mean ± standard deviation for top-1 and
    top-5 accuracy, as required by Part C.1 of the project specification.

    For each seed, the procedure is:
        1. Fix the random seed (reproducible data split and initialisation).
        2. Create fresh train/val/test dataloaders (with stratified split).
        3. Build a new model instance (using the provided factory function).
        4. Train the model using the same hyperparameters.
        5. Evaluate the best checkpoint on the test set.
        6. Record the top-1 and top-5 accuracies.

    Args:
        build_model_fn : callable that returns a fresh, untrained model
                         (e.g., lambda: build_videomae(num_classes=25)).
        dataset_root   : path to the HMDB_simp dataset folder.
        model_name     : model identifier (used for logging and checkpoint naming).
        seeds          : list of random seeds to try (default [42, 123, 456]).
        epochs         : number of training epochs per run.
        batch_size     : batch size (videos per batch).
        lr             : learning rate.
        patience       : early stopping patience (number of epochs without improvement).
        device         : 'cuda' or 'cpu'.

    Returns:
        dict containing:
            top1_mean : mean top-1 accuracy (in percentage points)
            top1_std  : standard deviation of top-1 accuracy
            top5_mean : mean top-5 accuracy
            top5_std  : standard deviation of top-5 accuracy
            per_seed  : list of {"seed": seed, "top1": value, "top5": value}
    """
    top1_scores = []   # store raw accuracy (0–1 range) for each seed
    top5_scores = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        # Set seed for all random number generators (Python, NumPy, PyTorch, CUDA)
        set_seed(seed)

        # Create fresh data splits (stratified, different random_state per seed)
        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            dataset_root=dataset_root,
            model_name=model_name,
            batch_size=batch_size,
            seed=seed,
            num_workers=2,
        )

        # Build a new model instance (fresh weights)
        model = build_model_fn()

        # Train the model using the unified Trainer class
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config={
                "epochs":          epochs,
                "lr":              lr,
                "patience":        patience,
                "checkpoint_path": f"checkpoints/seed_{seed}.pt",
                "log_dir":         f"logs/{model_name}_seed_{seed}",
                "device":          device,
            }
        )
        trainer.train()

        # Load the best checkpoint (lowest validation loss) and evaluate
        model.load_state_dict(torch.load(f"checkpoints/seed_{seed}.pt"))
        results = evaluate_model(
            model, test_loader, class_names,
            device=torch.device(device),
            model_name=f"{model_name}_seed{seed}",
        )

        top1_scores.append(results["top1"])
        top5_scores.append(results["top5"])

        print(f"\n  Seed {seed} → Top-1: {results['top1']*100:.2f}%, "
              f"Top-5: {results['top5']*100:.2f}%")

    # Convert scores from 0–1 to percentages and compute statistics
    top1_mean = np.mean(top1_scores) * 100
    top1_std  = np.std(top1_scores)  * 100
    top5_mean = np.mean(top5_scores) * 100
    top5_std  = np.std(top5_scores)  * 100

    print(f"\n{'='*60}")
    print(f"  MULTI-SEED RESULTS ({model_name})")
    print(f"{'='*60}")
    print(f"  Seeds tested : {seeds}")
    print(f"  Top-1 Acc    : {top1_mean:.2f}% ± {top1_std:.2f}%")
    print(f"  Top-5 Acc    : {top5_mean:.2f}% ± {top5_std:.2f}%")
    print(f"{'='*60}")

    # Record per‑seed results for potential later analysis
    per_seed_records = [
        {"seed": seed, "top1": round(t1 * 100, 4), "top5": round(t5 * 100, 4)}
        for seed, t1, t5 in zip(seeds, top1_scores, top5_scores)
    ]

    return {
        "top1_mean": top1_mean, "top1_std": top1_std,
        "top5_mean": top5_mean, "top5_std": top5_std,
        "per_seed": per_seed_records,
    }