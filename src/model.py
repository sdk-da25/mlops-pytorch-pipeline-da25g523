"""Model factory for CIFAR-10 classification.

``get_model`` builds whichever architecture is named in
``configs/training_config.yaml`` (``model.architecture``) — the choice of
architecture is never hardcoded in the training/serving code paths.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """A small 3-conv-block CNN for CIFAR-10 classification."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_resnet18(num_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    # The default 7x7/stride-2 stem + maxpool is tuned for 224x224 ImageNet
    # input and downsamples 32x32 CIFAR images too aggressively.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def get_model(architecture: str, num_classes: int) -> nn.Module:
    """Build a model by name, as declared in ``configs/training_config.yaml``."""
    if architecture == "resnet18":
        return _build_resnet18(num_classes)
    if architecture == "simplecnn":
        return SimpleCNN(num_classes)
    raise ValueError(f"Unknown architecture: {architecture!r}")
