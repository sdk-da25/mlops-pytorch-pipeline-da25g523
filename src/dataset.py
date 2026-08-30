"""CIFAR-10 dataset and dataloader construction.

``get_transforms`` is shared by training, evaluation, and the serving
endpoint so preprocessing never drifts apart.
"""
from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_transforms(train: bool) -> transforms.Compose:
    """Return the CIFAR-10 preprocessing pipeline.

    ``train=True`` adds light augmentation for training; ``train=False``
    returns the deterministic eval transform, also used by the serving
    endpoint so preprocessing matches exactly.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def get_dataloaders(
    data_dir: str, batch_size: int, num_workers: int = 2
) -> tuple[DataLoader, DataLoader]:
    """Build CIFAR-10 train/test dataloaders."""
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_transforms(train=True)
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
