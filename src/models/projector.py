"""MLP projection head used in SimCLR / VICReg."""

import torch.nn as nn


class Projector(nn.Module):
    """2- or 3-layer MLP projector.

    Architecture follows VICReg convention:
        Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128, num_layers: int = 3):
        super().__init__()
        layers = []
        prev = in_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
