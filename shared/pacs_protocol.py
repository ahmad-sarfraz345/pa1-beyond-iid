"""One immutable source split used by Tasks 2 and 3."""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from .pacs import CLASSES, SOURCES, labelled_records

SEED = 6304


def create_split(root: Path, output: Path) -> dict:
    if output.exists():
        # A later notebook session must reuse exactly this split for Task 3.
        split = load_split(output)
        for domain in SOURCES:
            for part in ("train", "val"):
                for record in split["sources"][domain][part]:
                    if not (Path(root) / record["path"]).is_file():
                        raise FileNotFoundError(f"Saved split image missing under {root}: {record['path']}")
        return split
    sources = {}
    for domain in SOURCES:
        records = labelled_records(root, domain)
        indices = list(range(len(records)))
        train, val = train_test_split(
            indices, test_size=0.2, stratify=[r["label"] for r in records],
            random_state=SEED,
        )
        sources[domain] = {
            "train": [records[i] for i in sorted(train)],
            "val": [records[i] for i in sorted(val)],
        }
    split = {"seed": SEED, "classes": list(CLASSES), "sources": sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def load_split(path: Path) -> dict:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    if split["seed"] != SEED or split["classes"] != list(CLASSES):
        raise ValueError("Unexpected PACS split protocol")
    return split
