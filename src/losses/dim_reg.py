"""Dimensionality-preserving regularizer via Von Neumann entropy.

This is the core contribution of the thesis. It encourages high intrinsic
dimensionality of the embedding space by maximizing the Von Neumann entropy
of the normalized embedding covariance matrix.

Von Neumann entropy: H_VN = -sum_i (p_i * log(p_i))
where p_i = lambda_i / sum(lambda_j) are the normalized eigenvalues of the
covariance matrix. This is maximized when all eigenvalues are equal (uniform
spectrum = full-rank representation) and minimized when a single eigenvalue
dominates (spectral collapse).
"""

import torch
import torch.nn as nn


class DimensionalityRegularizer(nn.Module):
    """Von Neumann entropy regularizer with moving-average covariance.

    The loss is: L_dim = -H_VN(C_ema)  (negative because we maximize entropy)

    A moving-average covariance estimate stabilizes the geometry signal across
    mini-batches, which is critical because per-batch covariance estimates are
    noisy (especially with small batch sizes on Colab).

    Args:
        embed_dim: Dimensionality of the embedding space.
        momentum: EMA momentum for covariance tracking. Default: 0.99
        eps: Small constant for numerical stability in log. Default: 1e-8
    """

    def __init__(self, embed_dim: int, momentum: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.embed_dim = embed_dim

        # Running covariance estimate (not a learnable parameter)
        self.register_buffer("cov_ema", torch.zeros(embed_dim, embed_dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _update_covariance(self, z: torch.Tensor):
        """Update the EMA covariance estimate with the current batch.

        Args:
            z: (B, D) embeddings (detached, no grad needed)
        """
        B = z.size(0)
        z_centered = z - z.mean(dim=0)
        batch_cov = (z_centered.T @ z_centered) / (B - 1)

        if not self.initialized:
            self.cov_ema.copy_(batch_cov)
            self.initialized.fill_(True)
        else:
            self.cov_ema.mul_(self.momentum).add_(batch_cov, alpha=1.0 - self.momentum)

    def compute_von_neumann_entropy(self, cov: torch.Tensor) -> torch.Tensor:
        """Compute Von Neumann entropy from a covariance matrix.

        Args:
            cov: (D, D) symmetric positive semi-definite covariance matrix
        Returns:
            Scalar Von Neumann entropy
        """
        # Eigendecomposition (covariance is symmetric -> use eigh for stability)
        eigenvalues = torch.linalg.eigvalsh(cov)  # sorted ascending

        # Clamp for numerical stability (remove near-zero / negative eigenvalues)
        eigenvalues = eigenvalues.clamp(min=self.eps)

        # Normalize to form a probability distribution
        p = eigenvalues / eigenvalues.sum()

        # Von Neumann entropy: -sum(p * log(p))
        entropy = -(p * p.log()).sum()

        return entropy

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute dimensionality regularization loss.

        Args:
            z: (B, D) embeddings from the projector (or encoder)
        Returns:
            Scalar loss = -H_VN (to be minimized, i.e., entropy is maximized)
        """
        # Update EMA covariance (no grad - this is a statistics tracker)
        self._update_covariance(z.detach())

        # Compute entropy on the EMA covariance (with grad through eigendecomp)
        # We recompute a batch covariance WITH gradients for backprop,
        # but blend it with the EMA for stability
        B = z.size(0)
        z_centered = z - z.mean(dim=0)
        batch_cov = (z_centered.T @ z_centered) / (B - 1)

        # Blend: use batch cov (has grad) regularized toward EMA shape
        # This gives gradient signal while benefiting from EMA stability
        alpha = 0.5
        cov_for_loss = alpha * batch_cov + (1.0 - alpha) * self.cov_ema.detach()

        entropy = self.compute_von_neumann_entropy(cov_for_loss)

        # Max entropy for D dimensions = log(D)
        max_entropy = torch.log(torch.tensor(float(self.embed_dim), device=z.device))

        # Return negative normalized entropy (minimize this -> maximize entropy)
        # Normalization by log(D) keeps the loss scale independent of embed_dim
        return -(entropy / max_entropy)

    @torch.no_grad()
    def get_diagnostics(self) -> dict:
        """Return diagnostic metrics for logging.

        Returns dict with:
            - von_neumann_entropy: current H_VN of EMA covariance
            - effective_rank: exp(H_VN), a measure of intrinsic dimensionality
            - top1_eigenvalue_ratio: fraction of variance in top eigenvalue
            - spectral_decay: eigenvalue spectrum (for plotting)
        """
        if not self.initialized:
            return {}

        eigenvalues = torch.linalg.eigvalsh(self.cov_ema)
        eigenvalues = eigenvalues.clamp(min=self.eps)
        p = eigenvalues / eigenvalues.sum()
        entropy = -(p * p.log()).sum()

        return {
            "von_neumann_entropy": entropy.item(),
            "effective_rank": torch.exp(entropy).item(),
            "top1_eigenvalue_ratio": (eigenvalues[-1] / eigenvalues.sum()).item(),
            "spectral_decay": eigenvalues.flip(0).cpu().numpy(),  # descending order
        }
