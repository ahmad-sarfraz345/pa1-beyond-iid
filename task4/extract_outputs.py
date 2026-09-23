"""Freeze checkpoints, then extract logits/features for all Task 4 datasets once."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from task4.data.cifar10 import build_cifar10_loaders
from task4.data.cifar100_unknowns import build_unknown_loaders
from task4.methods import make_gcsc, make_proser, make_vanilla

BASE = Path(__file__).resolve().parent
CHECKPOINTS, CACHE = BASE / "checkpoints", BASE / "cache"


@torch.no_grad()
def extract(model, loader, device, proser=False):
    model.eval()
    logits, features, labels, dummy_logits = [], [], [], []
    for images, target in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        if proser:
            known, dummy, feature = model(images, return_features=True)
            dummy_logits.append(dummy.cpu().numpy())
        else:
            known, feature = model(images, return_features=True)
        logits.append(known.cpu().numpy())
        features.append(feature.cpu().numpy())
        labels.append(target.numpy())
    result = {"logits": np.concatenate(logits), "features": np.concatenate(features),
              "labels": np.concatenate(labels)}
    if proser:
        result["dummy_logits"] = np.concatenate(dummy_logits)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    missing = [name for name in ("vanilla", "gcsc", "proser")
               if not (CHECKPOINTS / f"{name}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Train all models before opening unknown data; missing: {missing}")
    proser_record_path = BASE / "results" / "proser_train.json"
    if not proser_record_path.exists():
        raise FileNotFoundError("Missing PROSER training record")
    proser_record = json.loads(proser_record_path.read_text(encoding="utf-8"))
    vanilla_hash = hashlib.sha256((CHECKPOINTS / "vanilla.pt").read_bytes()).hexdigest()
    if proser_record.get("vanilla_checkpoint_sha256") != vanilla_hash:
        raise ValueError("PROSER was initialized from a different Vanilla checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_cifar10_loaders(args.data_root, 128, args.workers,
                                    download=not args.no_download)
    unknown, cifar100_names = build_unknown_loaders(args.data_root, 128, args.workers,
                                                     download=not args.no_download)
    all_loaders = {"train": loaders["train_plain"], "validation": loaders["validation"],
                   "test": loaders["test"], **unknown}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "cifar100_classes.json").write_text(json.dumps(cifar100_names), encoding="utf-8")
    for method in ("vanilla", "gcsc", "proser"):
        checkpoint = torch.load(CHECKPOINTS / f"{method}.pt", map_location="cpu", weights_only=False)
        if method == "proser":
            config = yaml.safe_load((BASE / "configs" / "proser.yaml").read_text())
            vanilla_state = torch.load(CHECKPOINTS / "vanilla.pt", map_location="cpu", weights_only=False)["model"]
            model = make_proser(vanilla_state, config["num_dummies"])
        else:
            model = make_gcsc() if method == "gcsc" else make_vanilla()
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        for split, loader in all_loaders.items():
            np.savez_compressed(CACHE / f"{method}_{split}.npz",
                                **extract(model, loader, device, method == "proser"))
    print(f"Saved frozen outputs to {CACHE}")


if __name__ == "__main__":
    main()
