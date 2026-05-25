"""VICReg loss from Bardes et al., 2022.

Three terms: Variance (prevent collapse), Invariance (align views),
Covariance (decorrelate dimensions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """VICReg: Variance-Invariance-Covariance Regularization.

    Args:
        sim_weight: Weight for invariance (MSE) term. Default: 25.0
        var_weight: Weight for variance (hinge) term. Default: 25.0
        cov_weight: Weight for covariance term. Default: 1.0
        var_target: Target std for variance hinge. Default: 1.0
    """

    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_target: float = 1.0,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_target = var_target

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Hinge loss on per-dimension std: max(0, target - std)."""
        std = z.std(dim=0)
        return F.relu(self.var_target - std).mean()

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Penalise off-diagonal covariance entries."""
        B, D = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (B - 1)  # (D, D)
        # Zero out diagonal, penalise off-diagonal
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / D

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (B, D) projected embeddings from two views
        Returns:
            Scalar loss
        """
        # Invariance: MSE between paired embeddings
        inv_loss = F.mse_loss(z1, z2)

        # Variance: apply to each view independently
        var_loss = self._variance_loss(z1) + self._variance_loss(z2)

        # Covariance: apply to each view independently
        cov_loss = self._covariance_loss(z1) + self._covariance_loss(z2)

        return (
            self.sim_weight * inv_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
        )
