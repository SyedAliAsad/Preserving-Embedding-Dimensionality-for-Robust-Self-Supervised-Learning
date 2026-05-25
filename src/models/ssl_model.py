"""SSL model: encoder + projector, with easy access to representations."""

import torch.nn as nn

from .encoder import Encoder
from .projector import Projector


class SSLModel(nn.Module):
    """Self-supervised model = backbone encoder + projection head.

    - forward() returns projected embeddings (for SSL loss computation).
    - encode() returns backbone representations (for downstream evaluation).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 128,
        proj_layers: int = 3,
    ):
        super().__init__()
        self.encoder = Encoder(backbone=backbone)
        self.projector = Projector(
            in_dim=self.encoder.out_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
            num_layers=proj_layers,
        )

    @property
    def repr_dim(self):
        """Dimension of backbone representations (before projector)."""
        return self.encoder.out_dim

    @property
    def embed_dim(self):
        """Dimension of projected embeddings."""
        return self.projector.mlp[-1].out_features

    def forward(self, x):
        """Returns projected embeddings z."""
        h = self.encoder(x)
        z = self.projector(h)
        return z

    def encode(self, x):
        """Returns backbone representations h (for linear probing)."""
        return self.encoder(x)
