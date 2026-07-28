# =============================================================================
# EARLY STOPPING FOR TRAINING
# =============================================================================
# This module provides a simple early stopping mechanism that monitors a
# validation metric (e.g., accuracy) and stops training when the metric
# stops improving for a given number of epochs.
# =============================================================================


class EarlyStopping:
    """
    Early stopping to halt training when validation performance stops improving.

    The class keeps track of the best observed validation score. If the score
    does not improve by at least `min_delta` after `patience` consecutive
    epochs, the `early_stop` flag is set to `True`.

    Args:
        patience (int): Number of epochs with no improvement after which
                        training will be stopped. Default: 5.
        min_delta (float): Minimum change in the monitored score to qualify
                           as an improvement. Default: 0.0 (any improvement counts).
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Initialise the early stopping handler.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum absolute change required to consider an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0          # number of epochs without improvement
        self.best_score = None    # best validation score seen so far
        self.early_stop = False   # flag indicating whether to stop training

    def __call__(self, val_acc: float) -> bool:
        """
        Update the early stopping state with a new validation score.

        Args:
            val_acc (float): The current validation accuracy (or any metric where
                             higher is better).

        Returns:
            bool: `True` if training should be stopped, `False` otherwise.
        """
        # First call: initialise best_score
        if self.best_score is None:
            self.best_score = val_acc
        # No significant improvement → increment counter
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        # Improvement detected → reset counter and update best_score
        else:
            self.best_score = val_acc
            self.counter = 0

        return self.early_stop