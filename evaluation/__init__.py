# =============================================================================
# EVALUATION PACKAGE INITIALISATION
# =============================================================================
# This package provides functions for evaluating classification and localisation
# performance. The `evaluate_model` function computes classification metrics
# (top‑1, top‑5, F1 scores, confusion matrix), while `plot_confusion_matrix`
# visualises the confusion matrix as a heatmap.
# =============================================================================

# Import the main evaluation function and the confusion matrix plotter
# so that they can be accessed directly from the evaluation package.
from .metrics import evaluate_model
from .plots import plot_confusion_matrix