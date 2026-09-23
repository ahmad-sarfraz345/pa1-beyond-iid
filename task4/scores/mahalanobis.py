import numpy as np


def fit_diagonal_mahalanobis(features, labels, epsilon=1e-6):
    means = np.stack([features[labels == class_id].mean(axis=0) for class_id in range(10)])
    residuals = features - means[labels]
    inverse_variance = 1.0 / (np.mean(residuals ** 2, axis=0) + epsilon)
    return means, inverse_variance


def mahalanobis(features, means, inverse_variance):
    distances = ((features[:, None, :] - means[None, :, :]) ** 2 * inverse_variance).sum(axis=2)
    return distances.min(axis=1)
