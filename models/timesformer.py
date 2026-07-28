# =============================================================================
# TimeSFormer Model Wrapper
# =============================================================================
# This module provides a wrapper for the TimeSFormer video classification model
# from Hugging Face Transformers. It loads a pre‑trained TimeSFormer checkpoint
# (default: fine‑tuned on Kinetics‑400) and adapts the classification head to
# output the required number of classes (e.g., 25 for HMDB_simp). The wrapper
# is compatible with the unified trainer (training.trainer.Trainer).
#
# A factory function `build_timesformer` is provided for multi‑seed robustness
# checks (Part C.1).
# =============================================================================
import torch
import torch.nn as nn
from transformers import AutoModelForVideoClassification


class TimeSFormer(nn.Module):
    """
    Wrapper for the TimeSFormer video classification model.

    The model loads a pre-trained checkpoint (default: TimeSFormer-base fine-tuned
    on Kinetics-400) and replaces the classifier head to match the target number
    of classes. Optional dropout can be added before the final linear layer.

    Attributes:
        model: The underlying Hugging Face AutoModelForVideoClassification instance.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "facebook/timesformer-base-finetuned-k400",
        dropout: float = 0.0,
    ) -> None:
        """
        Initialise the TimeSFormer model.

        Args:
            num_classes: Number of output classes (e.g., 25 for HMDB_simp).
            model_name : Hugging Face model identifier (default is the base
                         TimeSFormer pre-trained on Kinetics-400).
            dropout    : Dropout probability applied before the final linear layer
                         (if > 0). Useful for ablation studies (Part B).
        """
        super().__init__()
        print(f"[INFO] Loading TimeSFormer: {model_name}")

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
        Forward pass through the TimeSFormer model.

        Args:
            pixel_values: Input tensor of shape (batch_size, num_frames, channels, height, width).

        Returns:
            Model output (usually an object with a `.logits` attribute).
        """
        # The Hugging Face model expects keyword argument `pixel_values`
        return self.model(pixel_values=pixel_values)


# -----------------------------------------------------------------------------
# Factory function for multi‑seed evaluation (used in `multi_seed_evaluation`)
# -----------------------------------------------------------------------------
def build_timesformer(num_classes: int) -> TimeSFormer:
    """
    Factory function that returns a fresh TimeSFormer instance.

    This function is used by the statistical robustness check
    (multi_seed_evaluation in evaluation/metrics.py) to create a new model
    for each random seed, ensuring that each run starts from the same
    pre-trained weights but with different random initialisations of the head
    (if any random layers are added).

    Args:
        num_classes: Number of output classes.

    Returns:
        TimeSFormer: A new instance of the model.
    """
    return TimeSFormer(num_classes=num_classes)