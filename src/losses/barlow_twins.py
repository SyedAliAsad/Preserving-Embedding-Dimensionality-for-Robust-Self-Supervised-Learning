"""Barlow Twins loss from Zbontar et al., 2021.

Encourages the cross-correlation matrix between two augmented views
to be close to the identity matrix:
- Diagonal terms → 1 (invariance)
- Off-diagonal terms → 0 (redundancy reduction)
"""

import torch
import torch.nn as nn


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins: Self-Supervised Learning via Redundancy Reduction.

    Args:
        lambda_coeff: Weight for the off-diagonal (redundancy) term. Default: 0.005
        scale_loss: Scale factor applied to the total loss. Default: 0.025
    """

    def __init__(self, lambda_coeff: float = 0.005, scale_loss: float = 0.025):
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.scale_loss = scale_loss

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (B, D) projected embeddings from two views
        Returns:
            Scalar Barlow Twins loss
        """
        B, D = z1.shape

        # Normalize along the batch dimension
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-8)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-8)

        # Cross-correlation matrix (D x D)
        cross_corr = (z1_norm.T @ z2_norm) / B

        # Loss: diagonal → 1, off-diagonal → 0
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(cross_corr).pow_(2).sum()

        loss = self.scale_loss * (on_diag + self.lambda_coeff * off_diag)
        return loss

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """Return a flattened view of the off-diagonal elements of a square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
