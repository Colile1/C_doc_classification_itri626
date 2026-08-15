# baseline_cnn.py — the small from-scratch CNN the other two models must beat.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Any

import torch
from torch import nn

CHANNELS = (32, 64, 128, 256)


class BaselineCNN(nn.Module):
    """Four convolutional blocks, global pooling, one classifier head.

    Deliberately plain: batch normalisation and ReLU, max pooling between
    blocks, and adaptive pooling so the same network accepts the smoke config's
    64-pixel pages and the full runs' 224-pixel pages without a code change.
    """

    def __init__(self, n_classes: int, dropout: float = 0.3) -> None:
        """Purpose: assemble the network.

        Inputs: the number of classes and the head dropout probability.
        Output: None.
        """
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in CHANNELS:
            blocks.append(self._block(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(CHANNELS[-1], n_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Purpose: one convolution-normalisation-activation-pool block.

        Inputs: input and output channel counts. Output: the block module.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Purpose: map a batch of pages to class logits.

        Input: a `(batch, 3, H, W)` tensor. Output: `(batch, n_classes)` logits.
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(
                f"BaselineCNN expects a (batch, 3, H, W) tensor, got {tuple(x.shape)}."
            )
        return self.classifier(self.pool(self.features(x)))


def build_baseline_cnn(n_classes: int, cfg: Any = None) -> nn.Module:
    """Purpose: build the from-scratch baseline (objective O3).

    Inputs: the class count and the RunConfig (unused; the baseline has no
    pretrained weights and no per-run architecture choices, which is the point).
    Output: a BaselineCNN.
    """
    del cfg
    return BaselineCNN(n_classes=n_classes)
