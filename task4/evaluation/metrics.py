import numpy as np
from sklearn.metrics import roc_auc_score


def osr_metrics(known_scores, unknown_scores, threshold):
    labels = np.concatenate((np.zeros(len(known_scores)), np.ones(len(unknown_scores))))
    scores = np.concatenate((known_scores, unknown_scores))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "known_test_acceptance": float(np.mean(known_scores <= threshold)),
        "unknown_rejection": float(np.mean(unknown_scores > threshold)),
        "fpr_at_95_tpr": float(np.mean(unknown_scores <= threshold)),
    }
