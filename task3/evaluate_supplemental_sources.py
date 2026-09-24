"""Run source-only diagnostics for fixed Task 3 stabilization variants."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shared.pacs import SOURCES
from shared.pacs_protocol import load_split
from task2.models import Classifier
from task2.train import validate
from task3.diagnostics import (fixed_sharpness_batch, local_sharpness,
                               source_domain_separability, source_features)
from task3.train import SPLIT, TASK2_ERM, sha256
from task3.train_supplemental import CHECKPOINTS, CONFIG, RESULTS


def plot_training(records):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, record in records.items():
        history = record["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["source_val_mean_macro_f1"] for row in history], label=name)
        axes[1].plot(epochs, [row["classification_loss"] for row in history], label=name)
        axes[2].plot(epochs, [row["mmd_penalty"] for row in history], label=name)
    for axis, title in zip(axes, ("Source validation macro-F1", "Classification loss", "Pairwise MMD")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(RESULTS / "training_curves.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--methods", default="dan_dg_clip5")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    names = list(config["methods"]) if args.methods == "all" else [x.strip() for x in args.methods.split(",")]
    if any(name not in config["methods"] for name in names):
        parser.error("Unknown supplemental method")
    if sha256(TASK2_ERM) != config["erm_checkpoint_sha256"]:
        raise ValueError("Shared ERM checkpoint hash mismatch")
    config_hash, split_hash = sha256(CONFIG), sha256(SPLIT)
    split = load_split(SPLIT)
    records = {}
    for name in names:
        checkpoint, record_path = CHECKPOINTS / f"{name}.pt", RESULTS / f"{name}_train.json"
        if not checkpoint.exists() or not record_path.exists():
            raise FileNotFoundError(f"Finish training before diagnostics: {name}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["config_sha256"] != config_hash or record["split_sha256"] != split_hash:
            raise ValueError(f"Config or split changed after training {name}")
        records[name] = record
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = fixed_sharpness_batch(args.data_root, split)
    paths = {"erm": TASK2_ERM, **{name: CHECKPOINTS / f"{name}.pt" for name in names}}
    results = {}
    for name, path in paths.items():
        model = Classifier(pretrained=False).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        per_domain, mean_f1 = validate(model, args.data_root, split, device, args.workers)
        accuracies = [per_domain[d]["accuracy"] for d in SOURCES]
        macro_f1s = [per_domain[d]["macro_f1"] for d in SOURCES]
        features = source_features(model, args.data_root, split, device, args.workers)
        results[name] = {
            "checkpoint_sha256": sha256(path), "source_val": per_domain,
            "mean_accuracy": float(np.mean(accuracies)), "mean_macro_f1": mean_f1,
            "worst_accuracy": float(min(accuracies)), "worst_macro_f1": float(min(macro_f1s)),
            "worst_accuracy_domain": SOURCES[int(np.argmin(accuracies))],
            "worst_macro_f1_domain": SOURCES[int(np.argmin(macro_f1s))],
            "source_domain_separability": source_domain_separability(features),
            "sharpness": local_sharpness(model, images, labels, device),
        }
        print(f"{name}: source mean F1={mean_f1:.4f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "source_diagnostics.json").write_text(json.dumps({
        "config_sha256": config_hash, "split_sha256": split_hash,
        "sketch_loaded": False, "selected_variants": names, "methods": results,
    }, indent=2), encoding="utf-8")
    plot_training(records)
    print(f"Saved source-only diagnostics to {RESULTS}")


if __name__ == "__main__":
    main()
