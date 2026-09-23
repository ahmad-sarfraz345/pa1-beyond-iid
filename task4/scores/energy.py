import numpy as np


def energy(logits):
    maximum = logits.max(axis=1, keepdims=True)
    return -(maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1)))
