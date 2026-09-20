"""PACS file inventory and deterministic image preprocessing.

Target images are enumerated without assigning or returning a class label during
training. Only the final evaluator calls ``labelled_records`` on Sketch.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
SOURCES = DOMAINS[:3]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def domain_dir(root: Path, domain: str) -> Path:
    """Accept the common PACS root or a parent containing a PACS directory."""
    root = Path(root)
    candidates = (root / domain, root / "PACS" / domain, root / "pacs" / domain)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Missing PACS domain {domain!r} under {root}")


def labelled_records(root: Path, domain: str) -> list[dict]:
    folder = domain_dir(root, domain)
    records = []
    for label, name in enumerate(CLASSES):
        class_dir = folder / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing PACS class directory: {class_dir}")
        files = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"No images in {class_dir}")
        records.extend({"path": p.relative_to(root).as_posix(), "label": label} for p in files)
    return records


def unlabelled_records(root: Path, domain: str = "sketch") -> list[dict]:
    # The folder names may encode labels, but no label is parsed or returned.
    folder = domain_dir(root, domain)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f"No images in {folder}")
    return [{"path": p.relative_to(root).as_posix()} for p in files]


class PACSImages(Dataset):
    def __init__(self, root: Path, records: list[dict], train: bool, seed: int = 6304):
        self.root = Path(root)
        self.records = records
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            image = image.convert("RGB")
            image = F.resize(image, [256, 256], interpolation=InterpolationMode.BILINEAR)
            if self.train:
                # Stateless RNG keeps each source crop identical across methods.
                rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
                top, left = (int(v) for v in rng.integers(0, 33, size=2))
                image = F.crop(image, top, left, 224, 224)
                if rng.random() < 0.5:
                    image = F.hflip(image)
            else:
                image = F.center_crop(image, [224, 224])
            tensor = F.normalize(F.to_tensor(image), MEAN, STD)
        return tensor, record.get("label", -1), record["path"]
