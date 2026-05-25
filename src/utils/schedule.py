"""Scheduling functions for the dimensionality regularizer weight λ(t).

The regularizer is introduced gradually to avoid destabilizing early training
when the representation space is still forming.
"""

import math


def lambda_schedule(
    epoch: int,
    max_epochs: int,
    base_weight: float,
    schedule: str = "linear_warmup",
    warmup_epochs: int = 10,
) -> float:
    """Compute λ(t) for the dimensionality regularizer.

    Args:
        epoch: Current epoch (0-indexed)
        max_epochs: Total training epochs
        base_weight: Target λ value after warmup
        schedule: One of "constant", "linear_warmup", "cosine_warmup"
        warmup_epochs: Number of warmup epochs

    Returns:
        Current λ value
    """
    if schedule == "constant":
        return base_weight

    elif schedule == "linear_warmup":
        if epoch < warmup_epochs:
            return base_weight * (epoch / warmup_epochs)
        return base_weight

    elif schedule == "cosine_warmup":
        if epoch < warmup_epochs:
            # Cosine warmup: 0 -> base_weight over warmup_epochs
            return base_weight * 0.5 * (1.0 - math.cos(math.pi * epoch / warmup_epochs))
        return base_weight

    else:
        raise ValueError(f"Unknown schedule: {schedule}")
