"""Fixed STL-10 splits and class-balanced test identifiers."""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10


def prepare(root: Path, out: Path, seed: int = 6304, per_class: int = 50) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    train = STL10(root=str(root), split="train", download=True)
    test = STL10(root=str(root), split="test", download=True)
    train_ids, val_ids = train_test_split(
        np.arange(len(train)), test_size=0.2, stratify=train.labels, random_state=seed
    )
    rng = np.random.default_rng(seed)
    test_ids = []
    counts = {}
    for label, name in enumerate(train.classes):
        available = np.flatnonzero(np.asarray(test.labels) == label)
        chosen = rng.choice(available, min(per_class, len(available)), replace=False)
        test_ids.extend(map(int, chosen))
        counts[name] = len(chosen)
    manifest = {
        "seed": seed,
        "classes": list(train.classes),
        "train_ids": sorted(map(int, train_ids)),
        "val_ids": sorted(map(int, val_ids)),
        "test_ids": sorted(test_ids),
        "test_class_counts": counts,
        "identifier_note": "Zero-based index in torchvision's official STL10 split",
    }
    (out / "splits.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_dataset(root: Path, split: str):
    return STL10(root=str(root), split=split, download=False)


def image224(dataset, index: int):
    # STL-10 images are square. Resize once, before any intervention or normalization.
    from PIL import Image

    return dataset[index][0].convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
