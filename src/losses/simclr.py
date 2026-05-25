"""SimCLR NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
    """NT-Xent loss from Chen et al., 2020.

    Given a batch of N pairs (z_i, z_j) from two augmented views,
    constructs a 2N-sample contrastive problem.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) projected embeddings from view 1
            z2: (B, D) projected embeddings from view 2
        Returns:
            Scalar loss
        """
        B = z1.size(0)
        device = z1.device

        # L2 normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate: [z1_0, ..., z1_{B-1}, z2_0, ..., z2_{B-1}]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Pairwise cosine similarity
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # Mask out self-similarity (diagonal)
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim.masked_fill_(mask_self, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(device)
        positives = sim[torch.arange(2 * B, device=device), pos_idx]  # (2B,)

        # NT-Xent: -log(exp(pos) / sum(exp(all non-self)))
        loss = -positives + torch.logsumexp(sim, dim=1)
        return loss.mean()
