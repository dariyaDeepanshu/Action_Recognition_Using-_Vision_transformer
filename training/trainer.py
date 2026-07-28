# =============================================================================
# UNIFIED TRAINER FOR VIDEO CLASSIFICATION MODELS
# =============================================================================
# This module provides a `Trainer` class that implements the full training loop
# for any video classification model (TimeSFormer, VideoMAE). It handles:
#   - Training for a fixed number of epochs (with early stopping).
#   - Validation after each epoch.
#   - Learning rate scheduling.
#   - Checkpoint saving (best model based on validation accuracy).
#   - TensorBoard logging for loss, accuracy, and learning rate.
#   - Plotting training curves (loss and accuracy) as a PNG image.
#
# The trainer expects a configuration dictionary with fields such as:
#   "epochs", "lr", "weight_decay", "patience", "lr_scheduler", etc.
# =============================================================================

import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

# Local imports
from .lr_scheduler import get_scheduler
from .optimizer import get_optimizer
from .early_stopping import EarlyStopping


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    This function fixes seeds for Python's `random`, NumPy, PyTorch (CPU and CUDA),
    and configures cuDNN to use deterministic algorithms (at the cost of speed).

    Call this at the very beginning of any script or notebook.

    Args:
        seed: Integer seed value.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic behaviour (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Seed set to {seed}")


def extract_logits(outputs) -> torch.Tensor:
    """
    Extract logits from the output of a Hugging Face model.

    Different models may return the logits as an attribute `.logits`,
    a dictionary key 'logits', or the first element of a tuple.
    This function handles all common cases.

    Args:
        outputs: Output returned by the model's forward pass.

    Returns:
        Logits tensor of shape (batch_size, num_classes).
    """
    if hasattr(outputs, 'logits'):
        return outputs.logits
    elif isinstance(outputs, dict) and 'logits' in outputs:
        return outputs['logits']
    elif isinstance(outputs, tuple):
        return outputs[0]
    else:
        # Fallback: assume the output itself is the logits
        return outputs


class Trainer:
    """
    Unified trainer for video classification models.

    Args:
        model       : PyTorch model (must accept `pixel_values` argument
                      and return an object with logits).
        train_loader: DataLoader for the training set.
        val_loader  : DataLoader for the validation set.
        config      : Dictionary with training hyperparameters (see example below).

    Example config:
        {
            "epochs": 20,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "patience": 5,
            "lr_scheduler": "cosine",
            "warmup_epochs": 2,
            "checkpoint_path": "best_model.pt",
            "log_dir": "logs/experiment",
            "device": "cuda"
        }
    """

    def __init__(self, model: nn.Module, train_loader, val_loader, config: Dict[str, Any]):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device: use CUDA if available and requested, otherwise CPU
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device)

        # Loss and optimiser
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        self.early_stopping = EarlyStopping(patience=config.get("patience", 5))

        # Path and logging
        self.checkpoint_path = config.get("checkpoint_path", "best_model.pt")
        self.writer = SummaryWriter(log_dir=config.get("log_dir", "logs/experiment")) \
            if config.get("log_dir") else None
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_one_epoch(self, epoch: int) -> tuple:
        """
        Run one full training epoch.

        Args:
            epoch: Current epoch index (0-based, used only for printing).

        Returns:
            (average_loss, accuracy) for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos, labels = videos.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(pixel_values=videos)
            logits = extract_logits(outputs)
            loss = self.criterion(logits, labels)

            loss.backward()
            # Gradient clipping to avoid explosion (common in transformers)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} "
                    f"| Loss: {loss.item():.4f} | Time: {elapsed:.1f}s"
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def validate(self) -> tuple:
        """
        Run validation on the entire validation set.

        Returns:
            (average_loss, accuracy) for the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = self.model(pixel_values=videos)
                logits = extract_logits(outputs)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self) -> Dict[str, list]:
        """
        Execute the full training loop with early stopping.

        Returns:
            history: Dictionary containing lists of train/val loss and accuracy
                     for each epoch.
        """
        print(f"Training on {self.device}")
        for epoch in range(self.config["epochs"]):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            print(
                f"Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
                f"Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}% | LR {current_lr:.6f}"
            )

            # Checkpoint saving (best model based on validation accuracy)
            best_score = self.early_stopping.best_score
            if val_acc > (best_score if best_score is not None else -1):
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  ✓ Saved best model (val acc {val_acc*100:.2f}%)")

            # Early stopping
            if self.early_stopping(val_acc):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if self.writer:
            self.writer.close()

        return self.history

    def plot_history(self, save_path: str = "training_curves.png") -> None:
        """
        Plot training and validation loss / accuracy curves using matplotlib.

        The plot is saved as a PNG image. The output directory is created if needed.

        Args:
            save_path: Full path where the plot will be saved (e.g., outputs/model_curves.png).
        """
        import matplotlib.pyplot as plt

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        axes[0].plot(epochs, self.history["train_loss"], 'b-o', label='Train Loss', markersize=4)
        axes[0].plot(epochs, self.history["val_loss"],   'r-o', label='Val Loss',   markersize=4)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot (convert to percentage)
        train_acc_pct = [a * 100 for a in self.history["train_acc"]]
        val_acc_pct   = [a * 100 for a in self.history["val_acc"]]

        axes[1].plot(epochs, train_acc_pct, 'b-o', label='Train Acc', markersize=4)
        axes[1].plot(epochs, val_acc_pct,   'r-o', label='Val Acc',   markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Training curves saved to {save_path}")