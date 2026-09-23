from task4.models import CIFARResNet18


def make_gcsc():
    """GCSC changes training augmentation; its architecture matches Vanilla."""
    return CIFARResNet18(num_classes=10)
