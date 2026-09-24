"""Train Task 3 models using source domains only.

This module deliberately has no Sketch inventory or final-evaluation import.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.pacs import PACSImages, SOURCES
from shared.pacs_protocol import load_split
from task2.models import Classifier
from task2.train import set_seed, validate, validation_prediction_histogram
from .methods import pairwise_source_mmd, sam_step

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
CONFIG = BASE / "configs" / "base.json"
SPLIT = ROOT / "shared" / "splits" / "pacs_sketch_seed6304.json"
TASK2_ERM = ROOT / "task2" / "checkpoints" / "source_only.pt"
CHECKPOINTS = BASE / "checkpoints"
RESULTS = BASE / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loader(dataset, batch_size, seed, workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))


def next_batch(iterator, data_loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        return next(iterator), iterator


def verify_erm(config):
    if not TASK2_ERM.exists():
        raise FileNotFoundError(
            f"Missing {TASK2_ERM}. Restore it from task2_artifacts.zip; Task 3 must reuse it unchanged.")
    actual = sha256(TASK2_ERM)
    if actual != config["erm_checkpoint_sha256"]:
        raise ValueError(f"Task 2 ERM checkpoint hash mismatch: {actual}")


def train_one(name, root, config, split, workers, device, config_path=CONFIG,
              checkpoints=CHECKPOINTS, results=RESULTS):
    specification = config["methods"][name]
    checkpoints.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints / f"{name}.pt"
    record_path = results / f"{name}_train.json"
    config_hash, split_hash = sha256(config_path), sha256(SPLIT)
    if checkpoint_path.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("config_sha256") != config_hash or record.get("split_sha256") != split_hash:
            raise ValueError(f"Completed {name} used a different config or source split")
        print(f"Skipping completed {name}")
        return

    set_seed(config["seed"])
    model = Classifier(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    datasets = {
        domain: PACSImages(root, split["sources"][domain]["train"], train=True,
                           seed=config["seed"] + index * 10_000_000)
        for index, domain in enumerate(SOURCES)
    }
    loaders = {domain: loader(dataset, config["source_batch_per_domain"],
                              config["seed"] + index, workers)
               for index, (domain, dataset) in enumerate(datasets.items())}
    if any(len(data_loader) == 0 for data_loader in loaders.values()):
        raise ValueError("Each source domain needs at least eight training images")
    steps = max(math.ceil(len(dataset) / config["source_batch_per_domain"])
                for dataset in datasets.values())
    best, stale, history = -1.0, 0, []
    for epoch in range(config["max_epochs"]):
        model.train()  # Classifier.train keeps BatchNorm running statistics frozen.
        for dataset in datasets.values():
            dataset.epoch = epoch
        iterators = {domain: iter(data_loader) for domain, data_loader in loaders.items()}
        totals = {"classification_loss": 0.0, "mmd_penalty": 0.0,
                  "sam_perturbed_loss": 0.0}
        for _ in tqdm(range(steps), desc=f"{name} epoch {epoch + 1}", leave=False):
            images_by_domain, labels_by_domain = [], []
            for domain in SOURCES:
                (images, labels, _), iterators[domain] = next_batch(iterators[domain], loaders[domain])
                images_by_domain.append(images.to(device, non_blocking=True))
                labels_by_domain.append(labels.to(device, non_blocking=True))
            images = torch.cat(images_by_domain)
            labels = torch.cat(labels_by_domain)
            if specification["kind"] == "sam":
                class_loss, perturbed_loss = sam_step(
                    model, optimizer, images, labels, specification["rho"])
                totals["classification_loss"] += class_loss.item()
                totals["sam_perturbed_loss"] += perturbed_loss.item()
                continue

            optimizer.zero_grad(set_to_none=True)
            logits, features = model(images)
            class_loss = F.cross_entropy(logits, labels)
            chunks = list(features.split(config["source_batch_per_domain"]))
            penalty = pairwise_source_mmd(
                chunks, specification.get("normalize_mmd_features", False))
            loss = class_loss + specification["mmd_weight"] * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {name} at epoch {epoch + 1}")
            loss.backward()
            clip_norm = specification.get("gradient_clip_norm")
            if clip_norm is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_norm, error_if_nonfinite=True)
                totals.setdefault("gradient_norm", 0.0)
                totals["gradient_norm"] += float(gradient_norm.detach().cpu())
            optimizer.step()
            totals["classification_loss"] += class_loss.item()
            totals["mmd_penalty"] += penalty.item()

        source_val, score = validate(model, root, split, device, workers)
        row = {"epoch": epoch + 1, "source_val_mean_macro_f1": score,
               "source_val": source_val,
               **{key: value / steps for key, value in totals.items()}}
        if specification.get("record_prediction_histogram", False):
            row["source_val_prediction_histogram"] = validation_prediction_histogram(
                model, root, split, device, workers)
        history.append(row)
        print(f"{name} epoch {epoch + 1}: source val mean macro-F1={score:.4f}")
        if score > best + 1e-12:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "method": name, "epoch": epoch + 1,
                        "source_val_mean_macro_f1": score}, checkpoint_path)
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    record_path.write_text(json.dumps({
        "method": name, "configuration": specification,
        "config_sha256": config_hash, "split_sha256": split_hash,
        "best_source_val_mean_macro_f1": best, "history": history,
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method", default="all")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    verify_erm(config)
    split = load_split(SPLIT)
    names = list(config["methods"]) if args.method == "all" else [args.method]
    if any(name not in config["methods"] for name in names):
        parser.error(f"Method must be one of: {', '.join(config['methods'])}, all")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}; Sketch is not loaded by this program")
    for name in names:
        train_one(name, args.data_root, config, split, args.workers, device)


if __name__ == "__main__":
    main()
