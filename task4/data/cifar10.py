"""CIFAR-10 loaders with separate optimization and unaugmented views."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .make_splits import ensure_split

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _transform(train=False, gcsc=False):
    operations = []
    if train:
        operations += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if gcsc:
            # The assignment places RandAugment after crop/flip and before tensor conversion.
            operations.append(transforms.RandAugment(num_ops=2, magnitude=9))
    operations += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(operations)


def _loader(dataset, batch_size, workers, shuffle=False, seed=6304):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      generator=generator)


def build_cifar10_loaders(data_root: Path, batch_size=128, workers=2, gcsc=False,
                          download=True, seed=6304):
    split = ensure_split(data_root, download=download)
    train_aug = CIFAR10(data_root, train=True, transform=_transform(True, gcsc), download=download)
    train_plain = CIFAR10(data_root, train=True, transform=_transform(), download=False)
    test = CIFAR10(data_root, train=False, transform=_transform(), download=download)
    return {
        "train": _loader(Subset(train_aug, split["train"]), batch_size, workers, True, seed),
        "train_plain": _loader(Subset(train_plain, split["train"]), batch_size, workers),
        "validation": _loader(Subset(train_plain, split["validation"]), batch_size, workers),
        "test": _loader(test, batch_size, workers),
    }
