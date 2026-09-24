"""Train isolated Task 2 stabilization variants without overwriting prescribed runs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from shared.pacs_protocol import load_split
from task2.train import SPLIT, train_one

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "configs" / "supplemental.json"
CHECKPOINTS = BASE / "supplemental" / "checkpoints"
RESULTS = BASE / "supplemental" / "results"
SOURCE_ONLY = BASE / "checkpoints" / "source_only.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method", default="clip_all")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not SOURCE_ONLY.exists():
        raise FileNotFoundError("Restore task2/checkpoints/source_only.pt first")
    actual_hash = hashlib.sha256(SOURCE_ONLY.read_bytes()).hexdigest()
    if actual_hash != config["source_only_checkpoint_sha256"]:
        raise ValueError("Task 2 source-only checkpoint differs from the fixed baseline")
    groups = {
        "clip_all": ["dan_clip5", "dann_clip5", "cdan_clip5"],
        "fallback": ["dan_norm_clip5"],
        "all": list(config["methods"]),
    }
    names = groups.get(args.method, [args.method])
    if any(name not in config["methods"] for name in names):
        parser.error(f"Unknown method. Choose {', '.join(config['methods'])}, clip_all, fallback, or all")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(SPLIT)
    print(f"Training supplemental variants on {device}; outputs: {RESULTS}")
    for name in names:
        train_one(name, args.data_root, config, split, args.workers, device,
                  config_path=CONFIG, checkpoints=CHECKPOINTS, results=RESULTS)


if __name__ == "__main__":
    main()
