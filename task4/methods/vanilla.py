from task4.models import CIFARResNet18


def make_vanilla():
    """Randomly initialized ten-class CIFAR ResNet-18."""
    return CIFARResNet18(num_classes=10)
