"""One joint t-SNE fit per backbone and intervention."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_projection(clean, changed, labels, class_names, output: Path, seed: int, perplexity: int):
    count = len(labels)
    combined = np.concatenate((clean, changed), axis=0)
    perplexity = min(perplexity, len(combined) - 1)
    xy = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
              random_state=seed).fit_transform(combined)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for label, name in enumerate(class_names):
        selected = np.asarray(labels) == label
        ax.scatter(xy[:count][selected, 0], xy[:count][selected, 1], s=13,
                   marker="o", color=cmap(label), alpha=0.7, label=name)
        ax.scatter(xy[count:][selected, 0], xy[count:][selected, 1], s=16,
                   marker="x", color=cmap(label), alpha=0.7)
    ax.set_title("Clean (circles) and transformed (crosses); color = true class")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(ncol=2, fontsize=7, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_translation(rows, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for model, model_rows in rows.items():
        xs = [0, 8, 16, 32]
        axes[0].plot(xs, [model_rows[str(x)]["accuracy"] for x in xs], marker="o", label=model)
        axes[1].plot(xs, [model_rows[str(x)]["consistency"] for x in xs], marker="o", label=model)
    for ax, title in zip(axes, ("Accuracy", "Prediction consistency")):
        ax.set_title(title)
        ax.set_xlabel("Displacement (pixels)")
        ax.set_xticks([0, 8, 16, 32])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
