"""Source-only metrics shared by the Task 3 diagnostic script."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader

from shared.pacs import PACSImages, SOURCES


@torch.no_grad()
def source_features(model, root, split, device, workers):
    output = {}
    model.eval()
    for domain in SOURCES:
        dataset = PACSImages(root, split["sources"][domain]["val"], train=False)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda")
        blocks = []
        for images, _, _ in loader:
            _, features = model(images.to(device, non_blocking=True))
            blocks.append(features.cpu().numpy())
        output[domain] = np.concatenate(blocks)
    return output


def source_domain_separability(features):
    rng = np.random.default_rng(6304)
    count = min(len(features[domain]) for domain in SOURCES)
    blocks, labels = [], []
    for label, domain in enumerate(SOURCES):
        indices = rng.choice(len(features[domain]), count, replace=False)
        blocks.append(features[domain][indices])
        labels.append(np.full(count, label, dtype=int))
    x, y = np.concatenate(blocks), np.concatenate(labels)
    train, test = train_test_split(np.arange(len(y)), test_size=0.3,
                                   random_state=6304, stratify=y)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                           random_state=6304),
    )
    classifier.fit(x[train], y[train])
    return {"accuracy": float(classifier.score(x[test], y[test])),
            "samples_per_domain": count, "train_fraction": 0.7,
            "chance_accuracy": 1 / 3}


def fixed_sharpness_batch(root, split):
    rng = np.random.default_rng(6304)
    records = []
    for domain in SOURCES:
        candidates = split["sources"][domain]["val"]
        records.extend(candidates[int(i)] for i in rng.choice(len(candidates), 32, replace=False))
    dataset = PACSImages(root, records, train=False)
    images, labels, _ = next(iter(DataLoader(dataset, batch_size=96, shuffle=False)))
    return images, labels


def local_sharpness(model, images, labels, device, radius=0.05):
    """Loss increase from the assignment's one normalized ascent step."""
    model.eval()
    images, labels = images.to(device), labels.to(device)
    model.zero_grad(set_to_none=True)
    logits, _ = model(images)
    base_loss = F.cross_entropy(logits, labels)
    base_loss.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters]))
    scale = radius / norm.clamp_min(1e-12)
    perturbations = []
    with torch.no_grad():
        for parameter in parameters:
            perturbation = parameter.grad * scale
            parameter.add_(perturbation)
            perturbations.append(perturbation)
    try:
        with torch.no_grad():
            perturbed_logits, _ = model(images)
            perturbed_loss = F.cross_entropy(perturbed_logits, labels)
    finally:
        with torch.no_grad():
            for parameter, perturbation in zip(parameters, perturbations):
                parameter.sub_(perturbation)
        model.zero_grad(set_to_none=True)
    return {"radius": radius, "base_loss": float(base_loss.detach().cpu()),
            "perturbed_loss": float(perturbed_loss.cpu()),
            "loss_increase": float((perturbed_loss - base_loss.detach()).cpu())}
