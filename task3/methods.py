"""Source-domain alignment and standard non-adaptive SAM updates."""

import torch
from torch.nn import functional as F

from task2.methods import mmd_loss


def pairwise_source_mmd(features: list[torch.Tensor]) -> torch.Tensor:
    """Average MMD over the three unordered pairs of source domains."""
    penalties = [mmd_loss(features[i], features[j])
                 for i in range(len(features)) for j in range(i + 1, len(features))]
    return torch.stack(penalties).mean()


def sam_step(model, optimizer, images, labels, rho: float):
    """Perform one two-pass SAM update and return both unperturbed losses."""
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(images)
    first_loss = F.cross_entropy(logits, labels)
    first_loss.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    grad_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters]))
    scale = rho / grad_norm.clamp_min(1e-12)
    perturbations = []
    with torch.no_grad():
        for parameter in parameters:
            perturbation = parameter.grad * scale
            parameter.add_(perturbation)
            perturbations.append(perturbation)
    try:
        optimizer.zero_grad(set_to_none=True)
        perturbed_logits, _ = model(images)
        perturbed_loss = F.cross_entropy(perturbed_logits, labels)
        perturbed_loss.backward()
    finally:
        with torch.no_grad():
            for parameter, perturbation in zip(parameters, perturbations):
                parameter.sub_(perturbation)
    optimizer.step()
    return first_loss.detach(), perturbed_loss.detach()
