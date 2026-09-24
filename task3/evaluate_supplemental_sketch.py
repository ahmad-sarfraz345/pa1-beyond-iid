"""Reveal Sketch labels only after supplemental source diagnostics are fixed."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from shared.pacs import CLASSES, SOURCES, labelled_records
from task2.models import Classifier
from task3.evaluate_sketch import infer
from task3.train import SPLIT, TASK2_ERM, sha256
from task3.train_supplemental import CHECKPOINTS, CONFIG, RESULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_hash, split_hash = sha256(CONFIG), sha256(SPLIT)
    source_path = RESULTS / "source_diagnostics.json"
    if not source_path.exists():
        raise FileNotFoundError("Run evaluate_supplemental_sources before revealing Sketch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source["config_sha256"] != config_hash or source["split_sha256"] != split_hash:
        raise ValueError("Config or split changed after source diagnostics")
    names = source["selected_variants"]
    paths = {"erm": TASK2_ERM, **{name: CHECKPOINTS / f"{name}.pt" for name in names}}
    for name, path in paths.items():
        if source["methods"][name]["checkpoint_sha256"] != sha256(path):
            raise ValueError(f"Checkpoint changed after source diagnostics: {name}")

    target_records = labelled_records(args.data_root, "sketch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, class_rows, confusion_rows = {}, [], []
    for name, path in paths.items():
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = Classifier(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        actual, predicted, sample_paths = infer(model, args.data_root, target_records, device, args.workers)
        matrix = confusion_matrix(actual, predicted, labels=list(range(7)))
        per_class = {}
        for index, class_name in enumerate(CLASSES):
            total = int(matrix[index].sum())
            value = float(matrix[index, index] / total) if total else None
            per_class[class_name] = value
            class_rows.append({"method": name, "class": class_name,
                               "correct": int(matrix[index, index]), "total": total,
                               "accuracy": value})
            for prediction, count in enumerate(matrix[index]):
                confusion_rows.append({"method": name, "actual": class_name,
                                       "predicted": CLASSES[prediction], "count": int(count)})
        errors = Counter((CLASSES[int(a)], CLASSES[int(p)])
                         for a, p in zip(actual, predicted) if a != p)
        score = {"accuracy": float(accuracy_score(actual, predicted)),
                 "macro_f1": float(f1_score(actual, predicted, labels=list(range(7)),
                                            average="macro", zero_division=0))}
        results[name] = {**source["methods"][name], "sketch": score,
                         "sketch_per_class_accuracy": per_class,
                         "sketch_confusion_matrix": matrix.tolist(),
                         "dominant_confusions": [{"actual": a, "predicted": p, "count": count}
                                                 for (a, p), count in errors.most_common(5)]}
        print(f"{name}: Sketch={score}")
    erm = results["erm"]
    for result in results.values():
        result["sketch_accuracy_delta_vs_erm"] = result["sketch"]["accuracy"] - erm["sketch"]["accuracy"]
    output = {"protocol": "Supplemental Task 3 stabilization; Sketch opened after source gates",
              "config_sha256": config_hash, "split_sha256": split_hash, "methods": results}
    (RESULTS / "final_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    rows = []
    for name, result in results.items():
        rows.append({"method": name, "source_mean_accuracy": result["mean_accuracy"],
                     "source_mean_macro_f1": result["mean_macro_f1"],
                     "source_worst_accuracy": result["worst_accuracy"],
                     "source_worst_macro_f1": result["worst_macro_f1"],
                     "sketch_accuracy": result["sketch"]["accuracy"],
                     "sketch_macro_f1": result["sketch"]["macro_f1"],
                     "sketch_accuracy_delta_vs_erm": result["sketch_accuracy_delta_vs_erm"],
                     "source_domain_separability": result["source_domain_separability"]["accuracy"],
                     "sharpness_loss_increase": result["sharpness"]["loss_increase"]})
    for filename, data in (("summary.csv", rows), ("sketch_per_class.csv", class_rows),
                           ("sketch_confusions.csv", confusion_rows)):
        with (RESULTS / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    print(f"Saved supplemental Task 3 results to {RESULTS}")


if __name__ == "__main__":
    main()
