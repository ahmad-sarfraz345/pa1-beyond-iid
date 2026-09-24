"""Train isolated Task 3 DAN-DG stabilization variants without loading Sketch."""

import argparse
import json
from pathlib import Path

import torch

from shared.pacs_protocol import load_split
from task3.train import SPLIT, TASK2_ERM, sha256, train_one

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "configs" / "supplemental.json"
CHECKPOINTS = BASE / "supplemental" / "checkpoints"
RESULTS = BASE / "supplemental" / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method", default="dan_dg_clip5")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if sha256(TASK2_ERM) != config["erm_checkpoint_sha256"]:
        raise ValueError("Task 2 ERM checkpoint differs from the fixed shared baseline")
    names = list(config["methods"]) if args.method == "all" else [args.method]
    if any(name not in config["methods"] for name in names):
        parser.error(f"Choose {', '.join(config['methods'])} or all")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(SPLIT)
    print(f"Training on {device}; Sketch is not loaded; outputs: {RESULTS}")
    for name in names:
        train_one(name, args.data_root, config, split, args.workers, device,
                  config_path=CONFIG, checkpoints=CHECKPOINTS, results=RESULTS)


if __name__ == "__main__":
    main()
