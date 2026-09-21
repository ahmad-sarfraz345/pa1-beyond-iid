"""Evaluate Task 3 source domains without loading any Sketch image."""

import argparse
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
from .diagnostics import (fixed_sharpness_batch, local_sharpness,
                          source_domain_separability, source_features)
from .train import CHECKPOINTS, CONFIG, RESULTS, ROOT, SPLIT, TASK2_ERM, sha256, verify_erm


def artifacts(config):
    yield "erm", TASK2_ERM, ROOT / "task2" / "results" / "source_only_train.json"
    for name in config["methods"]:
        yield name, CHECKPOINTS / f"{name}.pt", RESULTS / f"{name}_train.json"


def plot_training(records):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, record in records.items():
        history = record["history"]
        epoch = [row["epoch"] for row in history]
        axes[0].plot(epoch, [row["source_val_mean_macro_f1"] for row in history], label=name)
        class_key = "class_loss" if name == "erm" else "classification_loss"
        axes[1].plot(epoch, [row[class_key] for row in history], label=name)
        if name.startswith("dan_dg"):
            axes[2].plot(epoch, [row["mmd_penalty"] for row in history], label=name)
    for axis, title in zip(axes, ("Source validation macro-F1", "Classification loss",
                                  "Pairwise source MMD")):
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    verify_erm(config)
    split = load_split(SPLIT)
    config_hash, split_hash = sha256(CONFIG), sha256(SPLIT)
    records = {}
    for name, checkpoint_path, record_path in artifacts(config):
        if not checkpoint_path.exists() or not record_path.exists():
            raise FileNotFoundError(f"Missing completed artifact for {name}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if name != "erm" and (record.get("config_sha256") != config_hash
                              or record.get("split_sha256") != split_hash):
            raise ValueError(f"Configuration or split changed after training {name}")
        records[name] = record

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sharp_images, sharp_labels = fixed_sharpness_batch(args.data_root, split)
    results = {}
    for name, checkpoint_path, _ in artifacts(config):
        model = Classifier(pretrained=False).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        per_domain, mean_f1 = validate(model, args.data_root, split, device, args.workers)
        accuracies = [per_domain[d]["accuracy"] for d in SOURCES]
        macro_f1s = [per_domain[d]["macro_f1"] for d in SOURCES]
        features = source_features(model, args.data_root, split, device, args.workers)
        results[name] = {
            "checkpoint_sha256": sha256(checkpoint_path), "source_val": per_domain,
            "mean_accuracy": float(np.mean(accuracies)), "mean_macro_f1": mean_f1,
            "worst_accuracy": float(min(accuracies)), "worst_macro_f1": float(min(macro_f1s)),
            "worst_accuracy_domain": SOURCES[int(np.argmin(accuracies))],
            "worst_macro_f1_domain": SOURCES[int(np.argmin(macro_f1s))],
            "source_domain_separability": source_domain_separability(features),
            "sharpness": local_sharpness(model, sharp_images, sharp_labels, device),
        }
        print(f"{name}: mean/worst source F1={mean_f1:.4f}/{min(macro_f1s):.4f}, "
              f"separability={results[name]['source_domain_separability']['accuracy']:.4f}")
    (RESULTS / "source_diagnostics.json").write_text(json.dumps({
        "config_sha256": config_hash, "split_sha256": split_hash,
        "sketch_loaded": False, "methods": results,
    }, indent=2), encoding="utf-8")
    plot_training(records)
    print(f"Saved source-only diagnostics in {RESULTS}")


if __name__ == "__main__":
    main()
