# =============================================================================
# VideoMAE Model Wrapper
# =============================================================================
# This module provides a wrapper for the VideoMAE (Video Masked Autoencoder)
# video classification model from Hugging Face Transformers. It loads a
# pre‑trained checkpoint (default: fine‑tuned on Kinetics‑400) and adapts
# the classification head to output the required number of classes (e.g., 25
# for HMDB_simp). The wrapper is compatible with the unified trainer
# (training.trainer.Trainer).
#
# A factory function `build_videomae` is provided for multi‑seed robustness
# checks (Part C.1).
# =============================================================================
import torch
import torch.nn as nn
from transformers import AutoModelForVideoClassification


class VideoMAE(nn.Module):
    """
    Wrapper for the VideoMAE video classification model.

    The model loads a pre-trained checkpoint (default: VideoMAE-base fine-tuned
    on Kinetics-400) and replaces the classifier head to match the target number
    of classes. Optional dropout can be added before the final linear layer.

    Attributes:
        model: The underlying Hugging Face AutoModelForVideoClassification instance.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        dropout: float = 0.0,
    ) -> None:
        """
        Initialise the VideoMAE model.

        Args:
            num_classes: Number of output classes (e.g., 25 for HMDB_simp).
            model_name : Hugging Face model identifier (default is the base
                         VideoMAE pre-trained on Kinetics-400).
            dropout    : Dropout probability applied before the final linear layer
                         (if > 0). Useful for ablation studies (Part B).
        """
        super().__init__()
        print(f"[INFO] Loading VideoMAE: {model_name}")

        # Load the pre‑trained model, automatically resizing the classifier head
        self.model = AutoModelForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,          # adapt head to num_classes
            ignore_mismatched_sizes=True,    # required because the original head had 400 classes
        )

        # Optionally insert an extra dropout layer before the classifier head
        if dropout > 0.0:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
            print(f"[INFO] Added dropout {dropout} before classifier head")

        print(f"[INFO] Model ready for {num_classes} classes")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VideoMAE model.

        Args:
            pixel_values: Input tensor of shape (batch_size, num_frames, channels, height, width).

        Returns:
            Model output (usually an object with a `.logits` attribute).
        """
        return self.model(pixel_values=pixel_values)


# -----------------------------------------------------------------------------
# Factory function for multi‑seed evaluation (used in `multi_seed_evaluation`)
# -----------------------------------------------------------------------------
def build_videomae(num_classes: int) -> VideoMAE:
    """
    Factory function that returns a fresh VideoMAE instance.

    This function is used by the statistical robustness check
    (multi_seed_evaluation in evaluation/metrics.py) to create a new model
    for each random seed, ensuring that each run starts from the same
    pre-trained weights but with different random initialisations of the head
    (if any random layers are added).

    Args:
        num_classes: Number of output classes.

    Returns:
        VideoMAE: A new instance of the model.
    """
    return VideoMAE(num_classes=num_classes)