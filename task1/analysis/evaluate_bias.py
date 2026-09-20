"""Classification metrics, paired consistency, and cue decision counts."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def metrics(y_true, probabilities):
    prediction = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "mean_max_confidence": float(np.max(probabilities, axis=1).mean()),
    }


def consistency(clean_probabilities, changed_probabilities):
    return float(np.mean(np.argmax(clean_probabilities, axis=1) ==
                         np.argmax(changed_probabilities, axis=1)))


def cue_counts(predictions, content_labels, style_labels):
    pred = np.asarray(predictions)
    shape = int(np.sum(pred == content_labels))
    texture = int(np.sum(pred == style_labels))
    other = int(len(pred) - shape - texture)
    covered = shape + texture
    return {
        "shape": shape, "texture": texture, "other": other,
        "shape_bias_pct": 100 * shape / covered if covered else None,
        "coverage_pct": 100 * covered / len(pred) if len(pred) else None,
    }
