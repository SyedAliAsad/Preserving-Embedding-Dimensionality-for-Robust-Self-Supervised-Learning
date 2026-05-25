"""Backbone encoder: ResNet with projection-ready output."""

import torch.nn as nn
import torchvision.models as models


# Map name -> (torchvision constructor, output dim)
_BACKBONES = {
    "resnet18": (models.resnet18, 512),
    "resnet50": (models.resnet50, 2048),
}


class Encoder(nn.Module):
    """Wraps a torchvision ResNet, strips the final FC, exposes repr dim."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone {backbone}. Choose from {list(_BACKBONES)}")

        factory, self.out_dim = _BACKBONES[backbone]
        resnet = factory(weights="IMAGENET1K_V1" if pretrained else None)

        # Everything except the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, out_dim, 1, 1)

    def forward(self, x):
        h = self.features(x)  # (B, out_dim, 1, 1)
        return h.flatten(1)   # (B, out_dim)
