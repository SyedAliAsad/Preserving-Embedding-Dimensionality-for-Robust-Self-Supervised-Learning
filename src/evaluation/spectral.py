"""Spectral diagnostics for analyzing representation geometry.

Key metrics:
- Effective rank: exp(Von Neumann entropy), measures intrinsic dimensionality
- Spectral decay: eigenvalue distribution of the representation covariance
- Participation ratio: alternative dimensionality measure
- Condition number: ratio of largest to smallest eigenvalue
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def compute_spectral_diagnostics(
    model,
    loader: DataLoader,
    device: str = "cuda",
    use_projector: bool = False,
    max_samples: int = 10000,
) -> dict:
    """Compute spectral diagnostics on representations.

    Args:
        model: SSLModel instance
        loader: Data loader (eval transforms)
        device: Compute device
        use_projector: If True, analyze projected embeddings z;
                       if False, analyze backbone representations h
        max_samples: Cap on number of samples (for speed)

    Returns:
        Dict with eigenvalues, effective_rank, participation_ratio,
        von_neumann_entropy, condition_number, top_k_variance
    """
    model.eval()
    features = []
    n_collected = 0

    for x, _ in tqdm(loader, desc="Collecting features", leave=False):
        x = x.to(device)
        if use_projector:
            h = model(x)
        else:
            h = model.encode(x)
        features.append(h.cpu())
        n_collected += x.size(0)
        if n_collected >= max_samples:
            break

    features = torch.cat(features)[:max_samples]  # (N, D)
    N, D = features.shape

    # Center
    features = features - features.mean(dim=0)

    # Covariance
    cov = (features.T @ features) / (N - 1)  # (D, D)

    # Eigendecomposition
    eigenvalues = torch.linalg.eigvalsh(cov)  # ascending
    eigenvalues = eigenvalues.flip(0)  # descending
    eigenvalues = eigenvalues.clamp(min=1e-10)

    # Normalized eigenvalues (probability distribution)
    p = eigenvalues / eigenvalues.sum()

    # Von Neumann entropy
    vn_entropy = -(p * p.log()).sum().item()

    # Effective rank = exp(entropy)
    effective_rank = np.exp(vn_entropy)

    # Participation ratio = (sum(lambda))^2 / sum(lambda^2)
    participation_ratio = (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()

    # Condition number
    condition_number = (eigenvalues[0] / eigenvalues[-1]).item()

    # Variance explained by top-k eigenvalues
    cumvar = torch.cumsum(p, dim=0).numpy()
    top_k_variance = {
        f"top{k}_variance": float(cumvar[k - 1])
        for k in [1, 5, 10, 20, 50]
        if k <= D
    }

    return {
        "eigenvalues": eigenvalues.numpy(),
        "von_neumann_entropy": vn_entropy,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "condition_number": condition_number,
        "embed_dim": D,
        **top_k_variance,
    }
