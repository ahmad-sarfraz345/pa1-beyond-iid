"""Save confidently accepted Vanilla-MLS unknowns for manual semantic analysis."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR100

from task4.data.cifar100_unknowns import FAR_CLASSES, NEAR_CLASSES


def selected_indices(dataset, names):
    ids = {dataset.class_to_idx[name] for name in names}
    return [index for index, target in enumerate(dataset.targets) if target in ids]


def save_failures(data_root, near_scores, far_scores, near_logits, far_logits, threshold,
                  cifar10_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "vanilla_mls_failures.csv"
    existing_labels = {}
    if output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing_labels[(row["group"], int(row["cifar100_test_index"]))] = (
                    row.get("student_classification", ""))
    dataset = CIFAR100(data_root, train=False, download=False)
    rows, images = [], []
    for group, names, scores, logits in (("near", NEAR_CLASSES, near_scores, near_logits),
                                         ("far", FAR_CLASSES, far_scores, far_logits)):
        source_indices = selected_indices(dataset, names)
        accepted = np.flatnonzero(scores <= threshold)
        if len(accepted) < 3:
            raise ValueError(f"Vanilla MLS accepted fewer than three {group} examples")
        # Lowest unknownness gives the clearest incorrectly accepted examples.
        chosen = accepted[np.argsort(scores[accepted])[:3]]
        for row_index in chosen:
            source_index = source_indices[int(row_index)]
            image, target = dataset[source_index]
            rows.append({"group": group, "cifar100_test_index": source_index,
                         "unknown_class": dataset.classes[target],
                         "predicted_cifar10_class": cifar10_names[int(logits[row_index].argmax())],
                         "mls_score": float(scores[row_index]), "threshold": threshold,
                         "student_classification": existing_labels.get((group, source_index), "")})
            images.append((image, rows[-1]))
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 3, figsize=(9, 6))
    for axis, (image, row) in zip(axes.flat, images):
        axis.imshow(image)
        axis.set_title(f"{row['unknown_class']} -> {row['predicted_cifar10_class']}\n"
                       f"score={row['mls_score']:.3f}, tau={threshold:.3f}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "vanilla_mls_failures.png", dpi=180)
    plt.close(figure)
    return rows
