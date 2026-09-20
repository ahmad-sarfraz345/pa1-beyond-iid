"""Final PACS evaluation. Run only after all configs and checkpoints are fixed."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from shared.pacs import CLASSES, PACSImages, SOURCES, labelled_records, unlabelled_records
from shared.pacs_protocol import load_split
from .models import Classifier
from .train import CHECKPOINTS, CONFIG, RESULTS, SPLIT


@torch.no_grad()
def infer(model, root, records, device, workers):
    dataset = PACSImages(root, records, train=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    model.eval()
    actual, predicted, features, paths = [], [], [], []
    for images, labels, names in loader:
        logits, representation = model(images.to(device, non_blocking=True))
        actual.extend(labels.tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
        features.append(representation.cpu().numpy())
        paths.extend(names)
    return np.array(actual), np.array(predicted), np.concatenate(features), paths


def metrics(actual, predicted):
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=list(range(7)),
                                       average="macro", zero_division=0))}


def domain_probe(source_features, target_features):
    rng = np.random.default_rng(6304)
    size = min(len(source_features), len(target_features))
    source = source_features[rng.choice(len(source_features), size, replace=False)]
    target = target_features[rng.choice(len(target_features), size, replace=False)]
    x = np.concatenate((source, target))
    y = np.concatenate((np.zeros(size, dtype=int), np.ones(size, dtype=int)))
    train_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=6304, stratify=y)
    probe = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=6304))
    probe.fit(x[train_indices], y[train_indices])
    return {"accuracy": float(probe.score(x[test_indices], y[test_indices])),
            "samples_per_domain": size, "train_fraction": 0.7}


def plot_training(records):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for name, record in records.items():
        history = record["history"]
        epoch = [row["epoch"] for row in history]
        axes[0].plot(epoch, [row["source_val_mean_macro_f1"] for row in history], label=name)
        axes[1].plot(epoch, [row["class_loss"] for row in history], label=name)
        if name != "source_only":
            axes[2].plot(epoch, [row["alignment_loss"] for row in history], label=name)
        if name in ("dann", "cdan"):
            axes[3].plot(epoch, [row["domain_accuracy"] for row in history], label=name)
    for axis, title in zip(axes, ("Source validation macro-F1", "Source class loss",
                                  "Alignment/domain loss", "Train domain accuracy")):
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=7)
    axes[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    split = load_split(SPLIT)
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    split_hash = hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    training = {}
    for name in config["methods"]:
        complete = RESULTS / f"{name}_train.json"
        checkpoint = CHECKPOINTS / f"{name}.pt"
        if not complete.exists() or not checkpoint.exists():
            raise FileNotFoundError(f"Finish training {name} before target evaluation")
        record = json.loads(complete.read_text(encoding="utf-8"))
        if record["config_sha256"] != config_hash or record["split_sha256"] != split_hash:
            raise ValueError(f"Configuration or source split changed after training {name}")
        training[name] = record

    # This is the only point in Task 2 that converts Sketch folders to labels.
    target_records = labelled_records(args.data_root, "sketch")
    if {r["path"] for r in target_records} != {r["path"] for r in unlabelled_records(args.data_root)}:
        raise ValueError("Sketch inventory differs between training and final evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, class_rows, confusion_rows = {}, [], []
    for name in config["methods"]:
        model = Classifier(pretrained=False).to(device)
        checkpoint = torch.load(CHECKPOINTS / f"{name}.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        source_scores, source_features = {}, []
        for domain in SOURCES:
            actual, predicted, features, _ = infer(
                model, args.data_root, split["sources"][domain]["val"], device, args.workers)
            source_scores[domain] = metrics(actual, predicted)
            source_features.append(features)
        target_true, target_pred, target_features, paths = infer(
            model, args.data_root, target_records, device, args.workers)
        matrix = confusion_matrix(target_true, target_pred, labels=list(range(7)))
        class_accuracies = {}
        for index, class_name in enumerate(CLASSES):
            total = int(matrix[index].sum())
            class_accuracies[class_name] = float(matrix[index, index] / total) if total else None
            class_rows.append({"method": name, "class": class_name, "correct": int(matrix[index, index]),
                               "total": total, "accuracy": class_accuracies[class_name]})
            for predicted_index, count in enumerate(matrix[index]):
                confusion_rows.append({"method": name, "actual": class_name,
                                       "predicted": CLASSES[predicted_index], "count": int(count)})
        errors = Counter((CLASSES[int(a)], CLASSES[int(p)]) for a, p in zip(target_true, target_pred) if a != p)
        examples = {f"{a}->{p}": [path for path, true, pred in zip(paths, target_true, target_pred)
                                   if CLASSES[int(true)] == a and CLASSES[int(pred)] == p][:3]
                    for (a, p), _ in errors.most_common(5)}
        results[name] = {
            "best_epoch": checkpoint["epoch"],
            "source_val": source_scores,
            "source_val_mean_accuracy": float(np.mean([source_scores[d]["accuracy"] for d in SOURCES])),
            "source_val_mean_macro_f1": float(np.mean([source_scores[d]["macro_f1"] for d in SOURCES])),
            "target": metrics(target_true, target_pred),
            "target_per_class_accuracy": class_accuracies,
            "target_confusion_matrix": matrix.tolist(),
            "dominant_confusions": [{"actual": a, "predicted": p, "count": count}
                                    for (a, p), count in errors.most_common(5)],
            "confusion_examples": examples,
            "domain_probe": domain_probe(np.concatenate(source_features), target_features),
        }
        print(f"{name}: target accuracy={results[name]['target']['accuracy']:.4f}, "
              f"macro-F1={results[name]['target']['macro_f1']:.4f}")

    baseline = results["source_only"]
    for name, result in results.items():
        result["target_accuracy_delta_vs_source_only"] = (
            result["target"]["accuracy"] - baseline["target"]["accuracy"])
        result["target_per_class_accuracy_delta_vs_source_only"] = {
            cls: result["target_per_class_accuracy"][cls] - baseline["target_per_class_accuracy"][cls]
            for cls in CLASSES
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "final_metrics.json").write_text(json.dumps({
        "protocol": "Photo/Art Painting/Cartoon to Sketch; source validation selects checkpoints",
        "config_sha256": config_hash, "split_sha256": split_hash,
        "study_hypothesis": config["study_hypothesis"], "methods": results,
    }, indent=2), encoding="utf-8")
    for filename, rows in (("target_per_class.csv", class_rows), ("target_confusions.csv", confusion_rows)):
        with (RESULTS / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summary_rows = []
    for name, result in results.items():
        row = {"method": name}
        for domain in SOURCES:
            row[f"{domain}_val_accuracy"] = result["source_val"][domain]["accuracy"]
            row[f"{domain}_val_macro_f1"] = result["source_val"][domain]["macro_f1"]
        row.update({
            "source_val_mean_accuracy": result["source_val_mean_accuracy"],
            "source_val_mean_macro_f1": result["source_val_mean_macro_f1"],
            "target_accuracy": result["target"]["accuracy"],
            "target_macro_f1": result["target"]["macro_f1"],
            "target_accuracy_delta_vs_source_only": result["target_accuracy_delta_vs_source_only"],
            "domain_probe_accuracy": result["domain_probe"]["accuracy"],
        })
        summary_rows.append(row)
    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_training(training)
    print(f"Saved final results in {RESULTS}")


if __name__ == "__main__":
    main()
