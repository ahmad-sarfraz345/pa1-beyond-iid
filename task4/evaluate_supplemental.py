"""Evaluate clipped PROSER with validation-only MLS and placeholder thresholds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from task4.evaluate_osr import placeholder_score
from task4.evaluation.metrics import osr_metrics
from task4.evaluation.thresholds import validation_threshold
from task4.scores import mls
from task4.train_supplemental import CONFIG, RESULTS

BASE = Path(__file__).resolve().parent
CACHE = BASE / "supplemental" / "cache"


def load(split):
    path = CACHE / f"proser_clip5_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run extract_supplemental first: {path}")
    return np.load(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    bundles = {split: load(split) for split in ("validation", "test", "near", "far")}
    rows = []
    for score_name, function in (
        ("MLS", lambda bundle: mls(bundle["logits"])),
        ("Placeholder", lambda bundle: placeholder_score(bundle, config["placeholder_temperature"])),
    ):
        scores = {split: function(bundle) for split, bundle in bundles.items()}
        threshold = validation_threshold(scores["validation"])
        for group, unknown in (("Near", scores["near"]), ("Far", scores["far"]),
                               ("All", np.concatenate((scores["near"], scores["far"])))):
            rows.append({"method": "PROSER clip5", "score": score_name,
                         "unknown_group": group, "threshold": threshold,
                         **osr_metrics(scores["test"], unknown, threshold)})
    test_accuracy = float(np.mean(bundles["test"]["logits"].argmax(1) == bundles["test"]["labels"]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "osr_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"method": "PROSER clip5", "cifar10_test_accuracy": test_accuracy,
               "configuration": config, "metrics": rows}
    (RESULTS / "final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("CIFAR-10 test accuracy:", test_accuracy)
    print(f"Saved supplemental Task 4 results to {RESULTS}")


if __name__ == "__main__":
    main()
