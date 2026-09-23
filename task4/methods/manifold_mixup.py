"""Manifold mixup helpers for PROSER's data placeholders."""

import torch


def different_class_permutation(labels: torch.Tensor) -> torch.Tensor:
    """Pair every feature with a different-class feature in the same half-batch."""
    size = labels.numel()
    _, counts = torch.unique(labels, return_counts=True)
    largest_class = int(counts.max().item())
    if largest_class * 2 > size:
        raise ValueError("Half-batch class imbalance prevents all different-class mixup pairs")
    # Sorting creates contiguous class blocks. Rotating by the largest block size gives
    # a one-to-one assignment and cannot map an element back into its own class block.
    order = torch.argsort(labels)
    partners = torch.roll(order, shifts=-largest_class)
    permutation = torch.empty_like(order)
    permutation[order] = partners
    if not torch.all(labels != labels[permutation]):
        raise RuntimeError("Different-class pairing invariant failed")
    return permutation


def mix_features(features, labels, alpha=2.0):
    permutation = different_class_permutation(labels)
    lam = torch.distributions.Beta(alpha, alpha).sample().to(features.device)
    return lam * features + (1.0 - lam) * features[permutation], lam
