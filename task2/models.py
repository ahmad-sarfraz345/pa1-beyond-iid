"""Pretrained ResNet-18 and the domain discriminator."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class Classifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, 7)

    def forward(self, images):
        features = self.backbone(images)
        return self.head(features), features

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Keep ImageNet running means/variances fixed; affine weights still train.
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self


class DomainDiscriminator(nn.Sequential):
    def __init__(self, input_dim: int):
        super().__init__(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2),
        )


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def reverse_gradient(x, scale: float):
    return _GradientReverse.apply(x, scale)
