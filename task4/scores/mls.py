def mls(logits):
    return -logits.max(axis=1)
