"""ResNet-18 variants adapted to 32x32 CIFAR images."""

import torch
from torch import nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        network = resnet18(weights=None)
        network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        network.maxpool = nn.Identity()
        network.fc = nn.Linear(network.fc.in_features, num_classes)
        self.network = network

    def stages(self, x):
        net = self.network
        x = net.relu(net.bn1(net.conv1(x)))
        x = net.layer1(x)
        layer2 = net.layer2(x)
        return layer2

    def after_layer2(self, x):
        net = self.network
        x = net.layer3(x)
        x = net.layer4(x)
        return torch.flatten(net.avgpool(x), 1)

    def forward(self, x, return_features=False):
        features = self.after_layer2(self.stages(x))
        logits = self.network.fc(features)
        return (logits, features) if return_features else logits


class PROSERResNet18(CIFARResNet18):
    """CIFAR ResNet-18 with five classifier placeholders."""

    def __init__(self, num_classes=10, num_dummies=5):
        super().__init__(num_classes)
        self.dummy_head = nn.Linear(self.network.fc.in_features, num_dummies)

    def heads(self, features):
        return self.network.fc(features), self.dummy_head(features)

    def forward(self, x, return_features=False):
        features = self.after_layer2(self.stages(x))
        known, dummy = self.heads(features)
        return (known, dummy, features) if return_features else (known, dummy)
