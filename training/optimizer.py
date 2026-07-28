# =============================================================================
# OPTIMISER FACTORY
# =============================================================================
# This module provides a simple factory function to create an AdamW optimizer
# with learning rate and weight decay taken from a configuration dictionary.
# =============================================================================

import torch
from typing import Dict, Any


def get_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer for the given model.

    AdamW is the recommended optimiser for transformer-based models because it
    decouples weight decay from the gradient update.

    Args:
        model: PyTorch model (all trainable parameters will be optimised).
        config: Dictionary with keys:
            - "lr" (float): learning rate (default 1e-4)
            - "weight_decay" (float): weight decay coefficient (default 0.01)

    Returns:
        AdamW optimizer instance.

    Note:
        The learning rate and weight decay values are explicitly converted to
        floats to avoid issues if they are read as strings from a YAML config.
    """
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 0.01)

    # Ensure numeric types (in case YAML loaded them as strings)
    lr = float(lr)
    weight_decay = float(weight_decay)

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )