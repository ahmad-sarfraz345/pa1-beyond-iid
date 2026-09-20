"""Train PACS source-only, DAN, DANN, CDAN, and the fixed MMD study.

Training and model selection never request Sketch class labels. The source-only
checkpoint is also the Task 3 ERM baseline; keep it unchanged.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.pacs import PACSImages, SOURCES, unlabelled_records
from shared.pacs_protocol import create_split, load_split
from .methods import domain_loss, grl_alpha, mmd_loss
from .models import Classifier, DomainDiscriminator

BASE = Path(__file__).resolve().parent
SPLIT = BASE.parent / "shared" / "splits" / "pacs_sketch_seed6304.json"
CONFIG = BASE / "configs" / "base.json"
CHECKPOINTS = BASE / "checkpoints"
RESULTS = BASE / "results"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader(dataset, batch_size: int, seed: int, shuffle: bool, workers: int, drop_last=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      drop_last=drop_last)


def next_batch(iterator, data_loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        return next(iterator), iterator


@torch.no_grad()
def validate(model, root: Path, split: dict, device, workers: int):
    model.eval()
    metrics = {}
    for domain in SOURCES:
        dataset = PACSImages(root, split["sources"][domain]["val"], train=False)
        data_loader = loader(dataset, 64, 6304, False, workers)
        true, pred = [], []
        for images, labels, _ in data_loader:
            logits, _ = model(images.to(device))
            true.extend(labels.tolist())
            pred.extend(logits.argmax(1).cpu().tolist())
        metrics[domain] = {
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=list(range(7)), average="macro", zero_division=0)),
        }
    return metrics, float(np.mean([metrics[d]["macro_f1"] for d in SOURCES]))


def train_one(name: str, root: Path, config: dict, split: dict, workers: int, device):
    specification = config["methods"][name]
    kind = specification["kind"]
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    complete_path = RESULTS / f"{name}_train.json"
    checkpoint_path = CHECKPOINTS / f"{name}.pt"
    if complete_path.exists() and checkpoint_path.exists():
        record = json.loads(complete_path.read_text(encoding="utf-8"))
        if (record.get("config_sha256") != hashlib.sha256(CONFIG.read_bytes()).hexdigest()
                or record.get("split_sha256") != hashlib.sha256(SPLIT.read_bytes()).hexdigest()):
            raise ValueError(f"Completed {name} used a different config or source split")
        print(f"Skipping completed {name}")
        return

    set_seed(config["seed"])
    model = Classifier(pretrained=True).to(device)
    discriminator = None
    if kind in ("dann", "cdan"):
        discriminator = DomainDiscriminator(512 * (7 if kind == "cdan" else 1)).to(device)
    parameters = list(model.parameters()) + (list(discriminator.parameters()) if discriminator else [])
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])

    source_datasets = {
        domain: PACSImages(root, split["sources"][domain]["train"], True,
                           config["seed"] + i * 10_000_000)
        for i, domain in enumerate(SOURCES)
    }
    source_loaders = {
        domain: loader(dataset, config["source_batch_per_domain"], config["seed"] + i,
                       True, workers, drop_last=True)
        for i, (domain, dataset) in enumerate(source_datasets.items())
    }
    steps = max(math.ceil(len(ds) / config["source_batch_per_domain"]) for ds in source_datasets.values())
    if any(len(dl) == 0 for dl in source_loaders.values()):
        raise ValueError("Each source domain needs at least eight training images")
    target_dataset = target_loader = None
    if kind != "source_only":
        target_dataset = PACSImages(root, unlabelled_records(root), True, config["seed"] + 100)
        target_loader = loader(target_dataset, config["target_batch"], config["seed"] + 100,
                               True, workers, drop_last=True)
        if len(target_loader) == 0:
            raise ValueError("Sketch needs at least 24 unlabelled images")

    history = []
    best, stale = -1.0, 0
    total_steps = config["max_epochs"] * steps
    for epoch in range(config["max_epochs"]):
        model.train()
        if discriminator:
            discriminator.train()
        for dataset in source_datasets.values():
            dataset.epoch = epoch
        source_iterators = {domain: iter(dl) for domain, dl in source_loaders.items()}
        if target_dataset is not None:
            target_dataset.epoch = epoch
            target_iterator = iter(target_loader)
        sums = {"class_loss": 0.0, "alignment_loss": 0.0, "domain_accuracy": 0.0}
        for step in tqdm(range(steps), desc=f"{name} epoch {epoch + 1}", leave=False):
            source_images, source_labels = [], []
            for domain in SOURCES:
                (images, labels, _), source_iterators[domain] = next_batch(
                    source_iterators[domain], source_loaders[domain])
                source_images.append(images)
                source_labels.append(labels)
            images = torch.cat(source_images).to(device, non_blocking=True)
            labels = torch.cat(source_labels).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            source_logits, source_features = model(images)
            class_loss = nn.functional.cross_entropy(source_logits, labels)
            alignment = class_loss.new_zeros(())
            domain_accuracy = class_loss.new_zeros(())
            if target_loader is not None:
                (target_images, _, _), target_iterator = next_batch(target_iterator, target_loader)
                target_logits, target_features = model(target_images.to(device, non_blocking=True))
                if kind == "dan":
                    alignment = mmd_loss(source_features, target_features)
                    loss = class_loss + specification["mmd_weight"] * alignment
                else:
                    progress = (epoch * steps + step) / max(total_steps - 1, 1)
                    alignment, domain_accuracy = domain_loss(
                        discriminator, source_features, target_features,
                        source_logits, target_logits, kind,
                        grl_alpha(progress, specification["max_grl"]),
                    )
                    loss = class_loss + alignment
            else:
                loss = class_loss
            loss.backward()
            optimizer.step()
            sums["class_loss"] += class_loss.item()
            sums["alignment_loss"] += alignment.item()
            sums["domain_accuracy"] += domain_accuracy.item()

        val, score = validate(model, root, split, device, workers)
        row = {"epoch": epoch + 1, "source_val_mean_macro_f1": score,
               "source_val": val, **{key: value / steps for key, value in sums.items()}}
        history.append(row)
        print(f"{name} epoch {epoch + 1}: source val mean macro-F1={score:.4f}")
        if score > best + 1e-12:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "discriminator":
                        discriminator.state_dict() if discriminator else None,
                        "method": name, "epoch": epoch + 1, "source_val_mean_macro_f1": score}, checkpoint_path)
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    complete_path.write_text(json.dumps({"method": name, "configuration": specification,
                                         "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                                         "split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
                                         "best_source_val_mean_macro_f1": best,
                                         "history": history}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--method", default="all")
    train.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        existed = SPLIT.exists()
        create_split(args.data_root, SPLIT)
        print(f"{'Reused' if existed else 'Created'} {SPLIT}")
        return
    if not SPLIT.exists():
        raise FileNotFoundError(f"Run prepare first: {SPLIT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    split = load_split(SPLIT)
    names = list(config["methods"]) if args.method == "all" else [args.method]
    if any(name not in config["methods"] for name in names):
        parser.error(f"Method must be one of: {', '.join(config['methods'])}, all")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    for name in names:
        train_one(name, args.data_root, config, split, args.workers, device)


if __name__ == "__main__":
    main()
