"""Final Task 3 Sketch evaluation after all source decisions are fixed."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from shared.pacs import CLASSES, PACSImages, SOURCES, labelled_records
from task2.models import Classifier
from .train import CHECKPOINTS, CONFIG, RESULTS, ROOT, SPLIT, TASK2_ERM, sha256, verify_erm


@torch.no_grad()
def infer(model, root, records, device, workers):
    dataset = PACSImages(root, records, train=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    model.eval()
    actual, predicted, paths = [], [], []
    for images, labels, names in loader:
        logits, _ = model(images.to(device, non_blocking=True))
        actual.extend(labels.tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
        paths.extend(names)
    return np.asarray(actual), np.asarray(predicted), paths


def checkpoint_paths(config):
    paths = {"erm": TASK2_ERM}
    paths.update({name: CHECKPOINTS / f"{name}.pt" for name in config["methods"]})
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    verify_erm(config)
    config_hash, split_hash = sha256(CONFIG), sha256(SPLIT)
    source_path = RESULTS / "source_diagnostics.json"
    if not source_path.exists():
        raise FileNotFoundError("Run task3.evaluate_sources before revealing Sketch labels")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source["config_sha256"] != config_hash or source["split_sha256"] != split_hash:
        raise ValueError("Task 3 config or source split changed after source diagnostics")
    paths = checkpoint_paths(config)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing fixed checkpoint for {name}: {path}")
        if source["methods"][name]["checkpoint_sha256"] != sha256(path):
            raise ValueError(f"Checkpoint {name} changed after source diagnostics")

    task2_path = ROOT / "task2" / "results" / "final_metrics.json"
    if not task2_path.exists():
        raise FileNotFoundError("Task 2 final metrics are required for the assigned comparison")
    task2 = json.loads(task2_path.read_text(encoding="utf-8"))

    # Target labels are first requested only after every gate above succeeds.
    target_records = labelled_records(args.data_root, "sketch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, class_rows, confusion_rows = {}, [], []
    for name, checkpoint_path in paths.items():
        model = Classifier(pretrained=False).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        actual, predicted, sample_paths = infer(
            model, args.data_root, target_records, device, args.workers)
        matrix = confusion_matrix(actual, predicted, labels=list(range(7)))
        per_class = {}
        for index, class_name in enumerate(CLASSES):
            total = int(matrix[index].sum())
            accuracy = float(matrix[index, index] / total) if total else None
            per_class[class_name] = accuracy
            class_rows.append({"method": name, "class": class_name,
                               "correct": int(matrix[index, index]),
                               "total": total, "accuracy": accuracy})
            for prediction, count in enumerate(matrix[index]):
                confusion_rows.append({"method": name, "actual": class_name,
                                       "predicted": CLASSES[prediction], "count": int(count)})
        errors = Counter((CLASSES[int(a)], CLASSES[int(p)])
                         for a, p in zip(actual, predicted) if a != p)
        examples = {f"{a}->{p}": [path for path, true, pred
                                   in zip(sample_paths, actual, predicted)
                                   if CLASSES[int(true)] == a and CLASSES[int(pred)] == p][:3]
                    for (a, p), _ in errors.most_common(5)}
        results[name] = {
            **source["methods"][name],
            "sketch": {
                "accuracy": float(accuracy_score(actual, predicted)),
                "macro_f1": float(f1_score(actual, predicted, labels=list(range(7)),
                                           average="macro", zero_division=0)),
            },
            "sketch_per_class_accuracy": per_class,
            "sketch_confusion_matrix": matrix.tolist(),
            "dominant_confusions": [{"actual": a, "predicted": p, "count": count}
                                    for (a, p), count in errors.most_common(5)],
            "confusion_examples": examples,
        }
        print(f"{name}: Sketch accuracy={results[name]['sketch']['accuracy']:.4f}, "
              f"macro-F1={results[name]['sketch']['macro_f1']:.4f}")

    baseline = results["erm"]
    for result in results.values():
        result["sketch_accuracy_delta_vs_erm"] = (
            result["sketch"]["accuracy"] - baseline["sketch"]["accuracy"])
        result["sketch_per_class_accuracy_delta_vs_erm"] = {
            class_name: result["sketch_per_class_accuracy"][class_name]
                        - baseline["sketch_per_class_accuracy"][class_name]
            for class_name in CLASSES
        }

    task2_comparison = {
        "erm": {
            **task2["methods"]["source_only"]["target"],
            "per_class_accuracy": task2["methods"]["source_only"]["target_per_class_accuracy"],
        },
        "target_aware_dan": {
            **task2["methods"]["dan"]["target"],
            "per_class_accuracy": task2["methods"]["dan"]["target_per_class_accuracy"],
            "dominant_confusions": task2["methods"]["dan"]["dominant_confusions"],
        },
        "target_free_dan_dg": {
            **results["dan_dg"]["sketch"],
            "per_class_accuracy": results["dan_dg"]["sketch_per_class_accuracy"],
            "dominant_confusions": results["dan_dg"]["dominant_confusions"],
        },
    }
    output = {
        "protocol": "Task 3 source-only PACS DG; Sketch loaded only in this final script",
        "config_sha256": config_hash, "split_sha256": split_hash,
        "study_hypothesis": config["study_hypothesis"],
        "methods": results, "task2_comparison": task2_comparison,
    }
    (RESULTS / "final_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    summary_rows = []
    for name, result in results.items():
        row = {"method": name}
        for domain in SOURCES:
            row[f"{domain}_val_accuracy"] = result["source_val"][domain]["accuracy"]
            row[f"{domain}_val_macro_f1"] = result["source_val"][domain]["macro_f1"]
        row.update({
            "source_mean_accuracy": result["mean_accuracy"],
            "source_mean_macro_f1": result["mean_macro_f1"],
            "source_worst_accuracy": result["worst_accuracy"],
            "source_worst_accuracy_domain": result["worst_accuracy_domain"],
            "source_worst_macro_f1": result["worst_macro_f1"],
            "source_worst_macro_f1_domain": result["worst_macro_f1_domain"],
            "sketch_accuracy": result["sketch"]["accuracy"],
            "sketch_macro_f1": result["sketch"]["macro_f1"],
            "sketch_accuracy_delta_vs_erm": result["sketch_accuracy_delta_vs_erm"],
            "source_domain_separability": result["source_domain_separability"]["accuracy"],
            "sharpness_loss_increase": result["sharpness"]["loss_increase"],
        })
        summary_rows.append(row)
    outputs = (("summary.csv", summary_rows), ("sketch_per_class.csv", class_rows),
               ("sketch_confusions.csv", confusion_rows))
    for filename, rows in outputs:
        with (RESULTS / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    comparison_rows = [{"method": name, "accuracy": values["accuracy"],
                        "macro_f1": values["macro_f1"]}
                       for name, values in task2_comparison.items()]
    with (RESULTS / "task2_task3_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)
    study_rows = [row for row in summary_rows if row["method"].startswith("dan_dg")]
    with (RESULTS / "controlled_study.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=study_rows[0].keys())
        writer.writeheader()
        writer.writerows(study_rows)
    print(f"Saved final Task 3 results in {RESULTS}")


if __name__ == "__main__":
    main()
