"""Fixed CIFAR-100 test-only near and far unknown subsets."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from .cifar10 import CIFAR10_MEAN, CIFAR10_STD

NEAR_CLASSES = ("bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel")
FAR_CLASSES = ("bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe")


def build_unknown_loaders(data_root: Path, batch_size=128, workers=2, download=True):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    # train=False is intentional: CIFAR-100 training data is prohibited by the protocol.
    dataset = CIFAR100(data_root, train=False, transform=transform, download=download)
    loaders = {}
    for group, names in (("near", NEAR_CLASSES), ("far", FAR_CLASSES)):
        class_ids = {dataset.class_to_idx[name] for name in names}
        indices = [index for index, target in enumerate(dataset.targets) if target in class_ids]
        if len(indices) != 800:
            raise ValueError(f"Expected 800 {group} unknowns, found {len(indices)}")
        loaders[group] = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                                    num_workers=workers, pin_memory=torch.cuda.is_available(),
                                    persistent_workers=workers > 0)
    return loaders, dataset.classes
