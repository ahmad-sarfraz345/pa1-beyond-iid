"""DAN, DANN, and CDAN objectives on pre-head ResNet features."""

import math

import torch
from torch.nn import functional as F


def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Biased empirical MMD² with three median-scaled Gaussian kernels."""
    both = torch.cat((source, target), dim=0)
    distances = torch.cdist(both, both).square()
    n, m = len(source), len(target)
    # Exclude self-distances from the bandwidth estimate; detach the scale.
    mask = ~torch.eye(n + m, dtype=torch.bool, device=both.device)
    median = distances.detach()[mask].median().clamp_min(1e-8)
    kernel = sum(torch.exp(-distances / (2 * multiplier * median))
                 for multiplier in (0.5, 1.0, 2.0))
    return kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()


def grl_alpha(progress: float, maximum: float = 1.0) -> float:
    return maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def domain_input(features: torch.Tensor, logits: torch.Tensor, method: str):
    if method == "cdan":
        probabilities = F.softmax(logits, dim=1)
        # Keep both factors attached to the graph for conditional alignment.
        return torch.bmm(probabilities.unsqueeze(2), features.unsqueeze(1)).flatten(1)
    return features


def domain_loss(discriminator, source_features, target_features,
                source_logits, target_logits, method: str, alpha: float):
    features = torch.cat((source_features, target_features), dim=0)
    logits = torch.cat((source_logits, target_logits), dim=0)
    inputs = domain_input(features, logits, method)
    # Reversal changes only the upstream gradient; discriminator learns normally.
    from .models import reverse_gradient
    reverse = reverse_gradient(inputs, alpha)
    predictions = discriminator(reverse)
    labels = torch.cat((
        torch.zeros(len(source_features), dtype=torch.long, device=features.device),
        torch.ones(len(target_features), dtype=torch.long, device=features.device),
    ))
    return F.cross_entropy(predictions, labels), (predictions.argmax(1) == labels).float().mean()
