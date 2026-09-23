"""PROSER initialization and the assignment-specified placeholder losses."""

import torch
from torch.nn import functional as F

from task4.models import PROSERResNet18


def make_proser(vanilla_state, num_dummies=5):
    model = PROSERResNet18(num_classes=10, num_dummies=num_dummies)
    # Vanilla has no dummy head; all backbone and known-head parameters transfer exactly.
    missing, unexpected = model.load_state_dict(vanilla_state, strict=False)
    if set(missing) != {"dummy_head.weight", "dummy_head.bias"} or unexpected:
        raise ValueError(f"Unexpected Vanilla-to-PROSER keys: missing={missing}, unexpected={unexpected}")
    return model


def aggregate_dummy(dummy_logits):
    """The reference method represents the strongest dummy as its unknown response."""
    return dummy_logits.max(dim=1, keepdim=True).values


def proser_loss(model, images, labels, beta=1.0, gamma=0.1, mixup_alpha=2.0):
    """Compute classifier-placeholder and layer2 manifold-placeholder objectives."""
    from .manifold_mixup import mix_features

    midpoint = images.size(0) // 2
    classifier_images, classifier_labels = images[:midpoint], labels[:midpoint]
    manifold_images, manifold_labels = images[midpoint:], labels[midpoint:]

    known, dummy = model(classifier_images)
    dummy_response = aggregate_dummy(dummy)
    # All five dummy logits participate in ordinary classification, as in the
    # reference implementation's augmented classifier output.
    classifier_logits = torch.cat((known, dummy), dim=1)
    known_loss = F.cross_entropy(classifier_logits, classifier_labels)

    # Mask the ground-truth class: the strongest placeholder must become second largest.
    masked_known = known.clone()
    masked_known.scatter_(1, classifier_labels[:, None], float("-inf"))
    placeholder_logits = torch.cat((masked_known, dummy_response), dim=1)
    dummy_target = torch.full_like(classifier_labels, 10)
    classifier_placeholder_loss = F.cross_entropy(placeholder_logits, dummy_target)

    layer2 = model.stages(manifold_images)
    mixed, _ = mix_features(layer2, manifold_labels, mixup_alpha)
    mixed_known, mixed_dummy = model.heads(model.after_layer2(mixed))
    mixed_logits = torch.cat((mixed_known, aggregate_dummy(mixed_dummy)), dim=1)
    data_placeholder_loss = F.cross_entropy(mixed_logits, torch.full_like(manifold_labels, 10))
    total = known_loss + beta * classifier_placeholder_loss + gamma * data_placeholder_loss
    return total, {"known": known_loss.detach(),
                   "classifier_placeholder": classifier_placeholder_loss.detach(),
                   "data_placeholder": data_placeholder_loss.detach()}
