"""Evaluate frozen Task 4 outputs using validation-only thresholds."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from torchvision.datasets import CIFAR10

from task4.evaluation.failure_analysis import save_failures
from task4.evaluation.metrics import osr_metrics
from task4.evaluation.thresholds import validation_threshold
from task4.scores import energy, fit_diagonal_mahalanobis, mahalanobis, mls, msp

BASE = Path(__file__).resolve().parent
CACHE, RESULTS = BASE / "cache", BASE / "results"


def load(method, split):
    path = CACHE / f"{method}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run extract_outputs first")
    return np.load(path)


def base_scores(name, bundle, mah_params=None):
    if name == "MSP":
        return msp(bundle["logits"])
    if name == "MLS":
        return mls(bundle["logits"])
    if name == "Energy":
        return energy(bundle["logits"])
    return mahalanobis(bundle["features"], *mah_params)


def placeholder_score(bundle, temperature):
    known, dummy = bundle["logits"], bundle["dummy_logits"].max(axis=1, keepdims=True)
    all_logits = np.concatenate((known, dummy), axis=1) / temperature
    shifted = all_logits - all_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return probabilities[:, -1] - probabilities[:, :-1].max(axis=1)


def add_rows(rows, method, score_name, validation, known, near, far):
    threshold = validation_threshold(validation)
    for group, unknown in (("Near", near), ("Far", far),
                           ("All", np.concatenate((near, far)))):
        metrics = osr_metrics(known, unknown, threshold)
        rows.append({"method": method, "score": score_name, "unknown_group": group,
                     "threshold": threshold, **metrics})
    return threshold


def plot_distributions(vanilla_scores):
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, score_name in zip(axes, ("MSP", "MLS", "Mahalanobis")):
        values = vanilla_scores[score_name]
        for split, label in (("test", "CIFAR-10"), ("near", "Near"), ("far", "Far")):
            axis.hist(values[split], bins=50, density=True, histtype="step", label=label)
        axis.set_title(score_name)
        axis.set_xlabel("Unknownness")
    axes[0].set_ylabel("Density")
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(RESULTS / "vanilla_score_distributions.png", dpi=180)
    plt.close(figure)


def write_required_tables(rows, known_accuracies):
    """Write the score-comparison and trained-model comparison tables."""
    lookup = {(row["method"], row["score"], row["unknown_group"]): row for row in rows}
    vanilla_rows = []
    for score in ("MSP", "MLS", "Energy", "Mahalanobis"):
        near, far, all_unknown = (lookup[("Vanilla", score, group)]
                                  for group in ("Near", "Far", "All"))
        vanilla_rows.append({"score": score, "near_auroc": near["auroc"],
                             "far_auroc": far["auroc"], "all_auroc": all_unknown["auroc"],
                             "known_test_acceptance": near["known_test_acceptance"],
                             "near_rejection": near["unknown_rejection"],
                             "far_rejection": far["unknown_rejection"]})
    comparison_rows = []
    for method, score in (("Vanilla", "MLS"), ("GCSC", "MLS"),
                          ("PROSER", "MLS"), ("PROSER", "Placeholder")):
        near, far, all_unknown = (lookup[(method, score, group)]
                                  for group in ("Near", "Far", "All"))
        comparison_rows.append({"method": method, "score": score,
                                "cifar10_test_accuracy": known_accuracies[method],
                                "near_auroc": near["auroc"], "far_auroc": far["auroc"],
                                "all_auroc": all_unknown["auroc"],
                                "known_test_acceptance": near["known_test_acceptance"],
                                "near_rejection": near["unknown_rejection"],
                                "far_rejection": far["unknown_rejection"]})
    for filename, data in (("vanilla_score_comparison.csv", vanilla_rows),
                           ("trained_model_comparison.csv", comparison_rows)):
        with (RESULTS / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, score_cache = [], {}

    vanilla = {split: load("vanilla", split) for split in ("train", "validation", "test", "near", "far")}
    mah_params = fit_diagonal_mahalanobis(vanilla["train"]["features"], vanilla["train"]["labels"])
    for score_name in ("MSP", "MLS", "Energy", "Mahalanobis"):
        params = mah_params if score_name == "Mahalanobis" else None
        scores = {split: base_scores(score_name, bundle, params) for split, bundle in vanilla.items()}
        score_cache[score_name] = scores
        add_rows(rows, "Vanilla", score_name, scores["validation"], scores["test"],
                 scores["near"], scores["far"])

    for method, display in (("gcsc", "GCSC"), ("proser", "PROSER")):
        bundles = {split: load(method, split) for split in ("validation", "test", "near", "far")}
        scores = {split: mls(bundle["logits"]) for split, bundle in bundles.items()}
        add_rows(rows, display, "MLS", scores["validation"], scores["test"],
                 scores["near"], scores["far"])
        if method == "proser":
            config = yaml.safe_load((BASE / "configs" / "proser.yaml").read_text())
            scores = {split: placeholder_score(bundle, config["placeholder_temperature"])
                      for split, bundle in bundles.items()}
            add_rows(rows, display, "Placeholder", scores["validation"], scores["test"],
                     scores["near"], scores["far"])

    fields = ("method", "score", "unknown_group", "threshold", "auroc",
              "known_test_acceptance", "unknown_rejection", "fpr_at_95_tpr")
    with (RESULTS / "osr_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_distributions(score_cache)

    vanilla_mls = score_cache["MLS"]
    threshold = validation_threshold(vanilla_mls["validation"])
    cifar10_names = CIFAR10(args.data_root, train=False, download=False).classes
    failures = save_failures(args.data_root, vanilla_mls["near"], vanilla_mls["far"],
                             vanilla["near"]["logits"], vanilla["far"]["logits"], threshold,
                             cifar10_names, RESULTS)
    known_accuracies = {"Vanilla": float(np.mean(vanilla["test"]["logits"].argmax(1) ==
                                                     vanilla["test"]["labels"]))}
    for method, display in (("gcsc", "GCSC"), ("proser", "PROSER")):
        bundle = load(method, "test")
        known_accuracies[display] = float(np.mean(bundle["logits"].argmax(1) == bundle["labels"]))
    write_required_tables(rows, known_accuracies)
    summary = {"cifar10_test_accuracy": known_accuracies,
               "threshold_policy": "95th percentile of CIFAR-10 validation unknownness",
               "unknown_positive": True, "metrics_rows": len(rows), "failure_examples": failures}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("CIFAR-10 test accuracy:", known_accuracies)
    print(f"Wrote {RESULTS / 'osr_metrics.csv'}")


if __name__ == "__main__":
    main()
