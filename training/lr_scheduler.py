# =============================================================================
# LEARNING RATE SCHEDULERS
# =============================================================================
# This module provides a factory function `get_scheduler` that returns a
# PyTorch learning rate scheduler based on a configuration dictionary.
#
# Supported schedulers:
#   - "cosine"       : CosineAnnealingLR (default)
#   - "step"         : StepLR with step_size and gamma
#   - "warmup_cosine": Linear warmup + CosineAnnealingLR (warmup_epochs, then cosine decay)
# =============================================================================

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: Optional[int] = None
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Factory function to create a learning rate scheduler based on the configuration.

    Args:
        optimizer: PyTorch optimizer (e.g., AdamW).
        config: Dictionary containing scheduler parameters. Expected keys:
            - "lr_scheduler" (str): Type of scheduler ('cosine', 'step', 'warmup_cosine').
            - "epochs" (int): Total number of epochs (required for cosine schedulers).
            - "step_size" (int, optional): Step size for 'step' scheduler (default 5).
            - "gamma" (float, optional): Multiplicative factor for 'step' scheduler (default 0.1).
            - "warmup_epochs" (int, optional): Number of warmup epochs for 'warmup_cosine' (default 2).
        num_training_steps: Not used in this implementation (kept for compatibility).

    Returns:
        A PyTorch learning rate scheduler.

    Raises:
        ValueError: If an unknown scheduler type is provided.
    """
    scheduler_type = config.get("lr_scheduler", "cosine")

    if scheduler_type == "cosine":
        # Cosine annealing from initial LR to eta_min over `epochs` steps
        return CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
            eta_min=1e-6
        )

    elif scheduler_type == "step":
        # Step decay: multiply LR by gamma every `step_size` epochs
        step_size = config.get("step_size", 5)
        gamma = config.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "warmup_cosine":
        # Linear warmup followed by cosine annealing
        warmup_epochs = config.get("warmup_epochs", 2)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6,          # initial LR is extremely low
            end_factor=1.0,             # after warmup, LR reaches the base LR
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"] - warmup_epochs,
            eta_min=1e-6
        )
        # Chain schedulers: warmup first, then cosine
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")