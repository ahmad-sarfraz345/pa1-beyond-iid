"""Extract clipped-PROSER outputs only after its CIFAR-10 checkpoint is fixed."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from task4.data.cifar10 import build_cifar10_loaders
from task4.data.cifar100_unknowns import build_unknown_loaders
from task4.extract_outputs import extract
from task4.methods import make_proser
from task4.train_supplemental import CHECKPOINTS, CONFIG, RESULTS, VANILLA

BASE = Path(__file__).resolve().parent
CACHE = BASE / "supplemental" / "cache"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    checkpoint_path = CHECKPOINTS / "proser_clip5.pt"
    record_path = RESULTS / "proser_clip5_train.json"
    if not checkpoint_path.exists() or not record_path.exists():
        raise FileNotFoundError("Finish clipped PROSER before opening CIFAR-100")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["config_sha256"] != hashlib.sha256(CONFIG.read_bytes()).hexdigest():
        raise ValueError("Clipped PROSER configuration changed")
    if record["vanilla_checkpoint_sha256"] != hashlib.sha256(VANILLA.read_bytes()).hexdigest():
        raise ValueError("Original Vanilla checkpoint changed")
    if record["checkpoint_sha256"] != hashlib.sha256(checkpoint_path.read_bytes()).hexdigest():
        raise ValueError("Clipped PROSER checkpoint changed after training")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if hashlib.sha256(VANILLA.read_bytes()).hexdigest() != config["vanilla_checkpoint_sha256"]:
        raise ValueError("Vanilla checkpoint differs from the fixed Task 4 baseline")
    loaders = build_cifar10_loaders(args.data_root, 128, args.workers,
                                    download=not args.no_download)
    unknown, _ = build_unknown_loaders(args.data_root, 128, args.workers,
                                       download=not args.no_download)
    all_loaders = {"validation": loaders["validation"], "test": loaders["test"], **unknown}
    vanilla = torch.load(VANILLA, map_location="cpu", weights_only=False)
    model = make_proser(vanilla["model"], config["num_dummies"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    CACHE.mkdir(parents=True, exist_ok=True)
    for split, loader in all_loaders.items():
        np.savez_compressed(CACHE / f"proser_clip5_{split}.npz",
                            **extract(model, loader, device, proser=True))
    print(f"Saved clipped PROSER outputs to {CACHE}")


if __name__ == "__main__":
    main()
