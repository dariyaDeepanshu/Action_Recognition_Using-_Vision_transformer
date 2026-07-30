# =============================================================================
# PLOTTING UTILITIES FOR EVALUATION
# =============================================================================
# This module provides functions to visualise evaluation results, primarily
# the confusion matrix heatmap. It relies on matplotlib and seaborn for
# high‑quality publication‑ready figures.
# =============================================================================

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    model_name: str = "model",
    save_dir: str = "outputs"
) -> None:
    """
    Plot and save a normalised confusion matrix as a heatmap.

    The matrix is row-normalised so that each row sums to 1, making it easy
    to see the fraction of true class i that are predicted as each class j.
    The heatmap is saved as a PNG file named 'confusion_matrix_{model_name}.png'
    inside the specified directory.

    Args:
        cm          : Raw confusion matrix (square, shape (C, C)) where rows are
                      true classes and columns are predicted classes.
        class_names : List of class name strings (same order as the matrix indices).
        model_name  : Identifier used in the plot title and the output filename.
        save_dir    : Directory where the image will be saved (created if missing).
    """
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a new figure with a size appropriate for up to 25 classes
    plt.figure(figsize=(16, 13))

    # Normalise the confusion matrix row‑wise (each row sums to 1)
    # Adding a small epsilon to avoid division by zero when a row sum is 0
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # Draw the heatmap using seaborn
    sns.heatmap(
        cm_norm,
        annot=True,               # Show the numeric values in each cell
        fmt='.2f',                # Format as two‑decimal floats
        cmap='Blues',             # Colour map (light to dark blue)
        xticklabels=class_names,  # Labels on the x‑axis (predicted classes)
        yticklabels=class_names,  # Labels on the y‑axis (true classes)
        linewidths=0.3,           # Thin grid lines between cells
        linecolor='gray',         # Colour of the grid lines
        cbar_kws={'label': 'Proportion'},  # Colour bar label
        vmin=0, vmax=1,           # Colour scale fixed to [0,1]
    )

    # Add titles and axis labels
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)

    # Rotate x‑axis labels for better readability (especially with many classes)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()   # Free memory

    print(f"[INFO] Confusion matrix saved to {save_path}")