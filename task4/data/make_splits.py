"""Create the fixed, stratified CIFAR-10 90/10 train/validation split."""

import argparse
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

BASE = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = BASE / "data" / "cifar10_seed6304.json"


def make_split(targets, seed=6304):
    """Return deterministic indices with 4,500 train and 500 validation per class."""
    labels = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for class_id in range(10):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        train.extend(indices[:4500].tolist())
        validation.extend(indices[4500:].tolist())
    # Shuffle across classes while retaining exact stratification.
    rng.shuffle(train)
    rng.shuffle(validation)
    return {"seed": seed, "train": train, "validation": validation}


def ensure_split(data_root, output=DEFAULT_SPLIT, download=True):
    output = Path(output)
    if output.exists():
        split = json.loads(output.read_text(encoding="utf-8"))
    else:
        dataset = CIFAR10(root=data_root, train=True, download=download)
        split = make_split(dataset.targets)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(split), encoding="utf-8")
    if len(split["train"]) != 45000 or len(split["validation"]) != 5000:
        raise ValueError("The CIFAR-10 split must contain 45,000 train and 5,000 validation images")
    if set(split["train"]) & set(split["validation"]):
        raise ValueError("Train and validation indices overlap")
    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT)
    args = parser.parse_args()
    result = ensure_split(args.data_root, args.output)
    print(f"Wrote {len(result['train'])} train and {len(result['validation'])} validation indices")
