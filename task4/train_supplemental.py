"""Train clipped PROSER in isolated supplemental artifact directories."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from task4.data.cifar10 import build_cifar10_loaders
from task4.methods import make_proser, proser_loss
from task4.train import set_seed

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "configs" / "proser_clip5.yaml"
CHECKPOINTS = BASE / "supplemental" / "checkpoints"
RESULTS = BASE / "supplemental" / "results"
VANILLA = BASE / "checkpoints" / "vanilla.pt"


@torch.no_grad()
def validation_diagnostics(model, loader, device):
    """Measure known accuracy and how often a dummy beats every known logit."""
    model.eval()
    correct = dummy_wins = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        known, dummy = model(images)
        correct += known.argmax(1).eq(labels).sum().item()
        dummy_wins += (dummy.max(1).values > known.max(1).values).sum().item()
        total += labels.numel()
    return correct / total, dummy_wins / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    checkpoint_path = CHECKPOINTS / "proser_clip5.pt"
    record_path = RESULTS / "proser_clip5_train.json"
    if checkpoint_path.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["config_sha256"] != config_hash:
            raise ValueError("Existing clipped PROSER used another configuration")
        if record.get("checkpoint_sha256") != hashlib.sha256(checkpoint_path.read_bytes()).hexdigest():
            raise ValueError("Existing clipped PROSER checkpoint changed after training")
        if not VANILLA.exists() or record.get("vanilla_checkpoint_sha256") != hashlib.sha256(VANILLA.read_bytes()).hexdigest():
            raise ValueError("Existing clipped PROSER used another Vanilla checkpoint")
        print("Skipping completed proser_clip5")
        return
    if not VANILLA.exists():
        raise FileNotFoundError("Restore the original selected Vanilla checkpoint first")
    vanilla_hash = hashlib.sha256(VANILLA.read_bytes()).hexdigest()
    if vanilla_hash != config["vanilla_checkpoint_sha256"]:
        raise ValueError("Vanilla checkpoint differs from the fixed Task 4 baseline")
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vanilla = torch.load(VANILLA, map_location="cpu", weights_only=False)
    model = make_proser(vanilla["model"], config["num_dummies"]).to(device)
    loaders = build_cifar10_loaders(args.data_root, config["batch_size"], config["num_workers"],
                                    download=not args.no_download, seed=config["seed"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        totals = {"total": 0.0, "known": 0.0, "classifier_placeholder": 0.0,
                  "data_placeholder": 0.0, "gradient_norm": 0.0}
        batches = 0
        for images, labels in tqdm(loaders["train"], desc=f"proser_clip5 {epoch}/50", leave=False):
            if images.size(0) % 2:
                images, labels = images[:-1], labels[:-1]
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = proser_loss(model, images, labels, config["beta"], config["gamma"],
                                      config["mixup_alpha"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite PROSER loss at epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            totals["total"] += loss.item()
            totals["gradient_norm"] += float(gradient_norm.detach().cpu())
            for key, value in parts.items():
                totals[key] += value.item()
            batches += 1
        validation_accuracy, validation_dummy_win_rate = validation_diagnostics(
            model, loaders["validation"], device)
        history.append({"epoch": epoch, "validation_accuracy": validation_accuracy,
                        "validation_dummy_win_rate": validation_dummy_win_rate,
                        "learning_rate": scheduler.get_last_lr()[0],
                        **{key: value / batches for key, value in totals.items()}})
        if validation_accuracy > best:
            best = validation_accuracy
            torch.save({"model": model.state_dict(), "method": "proser_clip5", "epoch": epoch,
                        "validation_accuracy": best, "config_sha256": config_hash}, checkpoint_path)
        scheduler.step()
        print(f"epoch {epoch}: val={validation_accuracy:.4f}, best={best:.4f}")
    record_path.write_text(json.dumps({
        "method": "proser_clip5", "configuration": config,
        "config_sha256": config_hash,
        "vanilla_checkpoint_sha256": vanilla_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "best_validation_accuracy": best, "history": history,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
