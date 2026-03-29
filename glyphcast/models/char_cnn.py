"""Lightweight CNN for ASCII character classification."""

from __future__ import annotations

import torch
from torch import nn


class AsciiCharCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.features(batch)
        flattened = features.view(features.shape[0], -1)
        return self.classifier(flattened)
