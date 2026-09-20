"""Cosine stability between paired clean and transformed embeddings."""

import numpy as np


def cosine_stability(clean, changed):
    assert clean.shape == changed.shape
    a = clean / np.maximum(np.linalg.norm(clean, axis=1, keepdims=True), 1e-12)
    b = changed / np.maximum(np.linalg.norm(changed, axis=1, keepdims=True), 1e-12)
    return float(np.sum(a * b, axis=1).mean())
