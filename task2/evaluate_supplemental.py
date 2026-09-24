"""Evaluate fixed Task 2 stabilization variants after source-only selection."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from shared.pacs import CLASSES, SOURCES, labelled_records, unlabelled_records
from shared.pacs_protocol import load_split
from task2.evaluate_final import domain_probe, infer, metrics
from task2.models import Classifier
from task2.train import SPLIT, validate
from task2.train_supplemental import CHECKPOINTS, CONFIG, RESULTS, SOURCE_ONLY

BASE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_training(records):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, record in records.items():
        history = record["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["source_val_mean_macro_f1"] for row in history], label=name)
        axes[1].plot(epochs, [row["class_loss"] for row in history], label=name)
        axes[2].plot(epochs, [row["alignment_loss"] for row in history], label=name)
    axes[0].set_ylabel("Score")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[2].set_yscale("symlog", linthresh=1.0)
    for axis, title in zip(axes, ("Source validation macro-F1", "Classification loss",
                                  "Alignment/domain loss")):
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
    parser.add_argument("--methods", default="clip_all")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not SOURCE_ONLY.exists() or sha256(SOURCE_ONLY) != config["source_only_checkpoint_sha256"]:
        raise ValueError("The fixed Task 2 source-only checkpoint is missing or changed")
    groups = {
        "clip_all": ["dan_clip5", "dann_clip5", "cdan_clip5"],
        "fallback": ["dan_norm_clip5"],
        "all": list(config["methods"]),
    }
    names = groups.get(args.methods, [item.strip() for item in args.methods.split(",")])
    if any(name not in config["methods"] for name in names):
        parser.error("Unknown supplemental method")
    config_hash, split_hash = sha256(CONFIG), sha256(SPLIT)
    split = load_split(SPLIT)
    records = {}
    for name in names:
        record_path = RESULTS / f"{name}_train.json"
        checkpoint_path = CHECKPOINTS / f"{name}.pt"
        if not record_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"Finish source-only training before evaluation: {name}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["config_sha256"] != config_hash or record["split_sha256"] != split_hash:
            raise ValueError(f"Config or split changed after training {name}")
        records[name] = record

    original_path = BASE / "results" / "final_metrics.json"
    if not original_path.exists():
        raise FileNotFoundError("Original Task 2 results are required for the fixed baseline")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    baseline = original["methods"]["source_only"]

    # Target labels are opened only after all supplemental records have passed the gates above.
    target_records = labelled_records(args.data_root, "sketch")
    if {r["path"] for r in target_records} != {r["path"] for r in unlabelled_records(args.data_root)}:
        raise ValueError("Sketch inventory changed between adaptation and evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, class_rows = {}, []
    for name in names:
        checkpoint_path = CHECKPOINTS / f"{name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = Classifier(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        source_val, mean_f1 = validate(model, args.data_root, split, device, args.workers)
        source_features = []
        for domain in SOURCES:
            _, _, features, _ = infer(model, args.data_root, split["sources"][domain]["val"],
                                      device, args.workers)
            source_features.append(features)
        actual, predicted, target_features, paths = infer(
            model, args.data_root, target_records, device, args.workers)
        matrix = confusion_matrix(actual, predicted, labels=list(range(7)))
        per_class = {}
        for index, class_name in enumerate(CLASSES):
            total = int(matrix[index].sum())
            value = float(matrix[index, index] / total) if total else None
            per_class[class_name] = value
            class_rows.append({"method": name, "class": class_name,
                               "correct": int(matrix[index, index]), "total": total,
                               "accuracy": value})
        errors = Counter((CLASSES[int(a)], CLASSES[int(p)])
                         for a, p in zip(actual, predicted) if a != p)
        results[name] = {
            "checkpoint_sha256": sha256(checkpoint_path), "best_epoch": checkpoint["epoch"],
            "configuration": config["methods"][name], "source_val": source_val,
            "source_val_mean_accuracy": float(np.mean([source_val[d]["accuracy"] for d in SOURCES])),
            "source_val_mean_macro_f1": mean_f1, "target": metrics(actual, predicted),
            "target_accuracy_delta_vs_source_only": float(
                metrics(actual, predicted)["accuracy"] - baseline["target"]["accuracy"]),
            "target_per_class_accuracy": per_class,
            "target_confusion_matrix": matrix.tolist(),
            "dominant_confusions": [{"actual": a, "predicted": p, "count": count}
                                    for (a, p), count in errors.most_common(5)],
            "domain_probe": domain_probe(np.concatenate(source_features), target_features),
        }
        print(f"{name}: source F1={mean_f1:.4f}, target={results[name]['target']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = {"protocol": "Supplemental stabilization; source validation selects checkpoints",
              "config_sha256": config_hash, "split_sha256": split_hash,
              "baseline": baseline, "methods": results}
    (RESULTS / "final_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    rows = []
    for name, result in results.items():
        rows.append({"method": name, "best_epoch": result["best_epoch"],
                     "source_val_mean_accuracy": result["source_val_mean_accuracy"],
                     "source_val_mean_macro_f1": result["source_val_mean_macro_f1"],
                     "target_accuracy": result["target"]["accuracy"],
                     "target_macro_f1": result["target"]["macro_f1"],
                     "target_accuracy_delta_vs_source_only": result["target_accuracy_delta_vs_source_only"],
                     "domain_probe_accuracy": result["domain_probe"]["accuracy"]})
    for filename, data in (("summary.csv", rows), ("target_per_class.csv", class_rows)):
        with (RESULTS / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    plot_training(records)
    print(f"Saved supplemental Task 2 results to {RESULTS}")


if __name__ == "__main__":
    main()
