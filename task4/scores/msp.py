import numpy as np


def msp(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return 1.0 - probabilities.max(axis=1)
