"""Train Vanilla, GCSC, and PROSER using CIFAR-10 only."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from tqdm import tqdm

from task4.data.cifar10 import build_cifar10_loaders
from task4.methods import make_gcsc, make_proser, make_vanilla, proser_loss

BASE = Path(__file__).resolve().parent
CHECKPOINTS = BASE / "checkpoints"
RESULTS = BASE / "results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(method):
    path = BASE / "configs" / f"{method}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")), hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def accuracy(model, loader, device, proser=False):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        logits = output[0] if proser else output
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def save_checkpoint(model, method, epoch, validation_accuracy, config_hash):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "method": method, "epoch": epoch,
                "validation_accuracy": validation_accuracy,
                "config_sha256": config_hash}, CHECKPOINTS / f"{method}.pt")


def completed(checkpoint, record_path, config_hash):
    """Skip only artifacts produced with the current experiment configuration."""
    if not checkpoint.exists() or not record_path.exists():
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if record.get("config_sha256") != config_hash or saved.get("config_sha256") != config_hash:
        raise ValueError(f"Existing artifacts for {checkpoint.stem} use a different configuration")
    return True


def train_closed_set(method, data_root, device, download):
    config, config_hash = load_config(method)
    checkpoint = CHECKPOINTS / f"{method}.pt"
    record_path = RESULTS / f"{method}_train.json"
    if completed(checkpoint, record_path, config_hash):
        print(f"Skipping completed {method}")
        return
    set_seed(config["seed"])
    loaders = build_cifar10_loaders(data_root, config["batch_size"], config["num_workers"],
                                    gcsc=method == "gcsc", download=download, seed=config["seed"])
    model = (make_gcsc() if method == "gcsc" else make_vanilla()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    best, history = -1.0, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running = 0.0
        for images, labels in tqdm(loaders["train"], desc=f"{method} {epoch}/{config['epochs']}", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        validation_accuracy = accuracy(model, loaders["validation"], device)
        history.append({"epoch": epoch, "train_loss": running / len(loaders["train"]),
                        "validation_accuracy": validation_accuracy,
                        "learning_rate": scheduler.get_last_lr()[0]})
        if validation_accuracy > best:
            best = validation_accuracy
            save_checkpoint(model, method, epoch, best, config_hash)
        scheduler.step()
        print(f"{method} epoch {epoch}: val accuracy={validation_accuracy:.4f}, best={best:.4f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({"method": method, "config_sha256": config_hash,
                                      "best_validation_accuracy": best, "history": history}, indent=2),
                           encoding="utf-8")


def train_proser(data_root, device, download):
    config, config_hash = load_config("proser")
    checkpoint = CHECKPOINTS / "proser.pt"
    record_path = RESULTS / "proser_train.json"
    if completed(checkpoint, record_path, config_hash):
        print("Skipping completed proser")
        return
    vanilla_path = CHECKPOINTS / "vanilla.pt"
    if not vanilla_path.exists():
        raise FileNotFoundError("Train Vanilla before PROSER")
    set_seed(config["seed"])
    vanilla = torch.load(vanilla_path, map_location="cpu", weights_only=False)
    model = make_proser(vanilla["model"], config["num_dummies"]).to(device)
    loaders = build_cifar10_loaders(data_root, config["batch_size"], config["num_workers"],
                                    download=download, seed=config["seed"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    best, history = -1.0, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        totals = {"total": 0.0, "known": 0.0, "classifier_placeholder": 0.0,
                  "data_placeholder": 0.0}
        for images, labels in tqdm(loaders["train"], desc=f"proser {epoch}/{config['epochs']}", leave=False):
            # Drop a final odd example so the prescribed half-batch split stays exact.
            if images.size(0) % 2:
                images, labels = images[:-1], labels[:-1]
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = proser_loss(model, images, labels, config["beta"], config["gamma"],
                                      config["mixup_alpha"])
            loss.backward()
            optimizer.step()
            totals["total"] += loss.item()
            for key, value in parts.items():
                totals[key] += value.item()
        validation_accuracy = accuracy(model, loaders["validation"], device, proser=True)
        history.append({"epoch": epoch, "validation_accuracy": validation_accuracy,
                        "learning_rate": scheduler.get_last_lr()[0],
                        **{key: value / len(loaders["train"]) for key, value in totals.items()}})
        if validation_accuracy > best:
            best = validation_accuracy
            save_checkpoint(model, "proser", epoch, best, config_hash)
        scheduler.step()
        print(f"proser epoch {epoch}: val accuracy={validation_accuracy:.4f}, best={best:.4f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({"method": "proser", "config_sha256": config_hash,
                                      "vanilla_checkpoint_sha256": hashlib.sha256(vanilla_path.read_bytes()).hexdigest(),
                                      "best_validation_accuracy": best, "history": history}, indent=2),
                           encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--method", choices=("vanilla", "gcsc", "proser", "all"), default="all")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}. CIFAR-100 is not loaded by this program.")
    methods = ("vanilla", "gcsc", "proser") if args.method == "all" else (args.method,)
    for method in methods:
        if method == "proser":
            train_proser(args.data_root, device, not args.no_download)
        else:
            train_closed_set(method, args.data_root, device, not args.no_download)


if __name__ == "__main__":
    main()
