# Supplemental Stabilization Runs on Kaggle

This workflow preserves the original prescribed artifacts. New checkpoints and results are written under each task's `supplemental/` directory.

## 1. Notebook setup

Create a fresh Kaggle notebook, enable a GPU and Internet, and attach:

- the PACS dataset;
- the expanded Task 2 artifact dataset containing `source_only.pt`;
- the expanded Task 4 artifact dataset containing `vanilla.pt`;
- `local-cifar` and `local-cifar100`.

Clone and install:

```python
!git clone YOUR_REPO_URL /kaggle/working/pa1-beyond-iid
%cd /kaggle/working/pa1-beyond-iid
!git checkout YOUR_BRANCH_OR_COMMIT
!python -m pip install -q -r requirements.txt
```

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
```

## 2. Locate PACS

```python
from pathlib import Path

domains = ("photo", "art_painting", "cartoon", "sketch")
matches = [p for p in Path("/kaggle/input").rglob("images")
           if p.is_dir() and all((p / domain).is_dir() for domain in domains)]
print(matches)
assert len(matches) == 1
PACS_ROOT = str(matches[0])
```

## 3. Restore the immutable Task 2 ERM checkpoint

```python
import shutil

matches = list(Path("/kaggle/input").rglob("source_only.pt"))
print(matches)
assert len(matches) >= 1
destination = Path("task2/checkpoints/source_only.pt")
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(matches[0], destination)
```

```python
import hashlib, json
expected = json.loads(Path("task2/configs/supplemental.json").read_text())["source_only_checkpoint_sha256"]
actual = hashlib.sha256(Path("task2/checkpoints/source_only.pt").read_bytes()).hexdigest()
print(actual)
assert actual == expected
```

## 4. Train Task 2 clipping-only variants

This trains DAN lambda=1, DANN, and CDAN with global gradient clipping at 5.0:

```python
!python -m task2.train_supplemental \
    --data-root {PACS_ROOT} \
    --method clip_all
```

Inspect source-only stability before revealing Sketch labels:

```python
import json
from pathlib import Path

for name in ("dan_clip5", "dann_clip5", "cdan_clip5"):
    record = json.loads(Path(f"task2/supplemental/results/{name}_train.json").read_text())
    best = record["best_source_val_mean_macro_f1"]
    best_row = max(record["history"], key=lambda row: row["source_val_mean_macro_f1"])
    histogram = best_row["source_val_prediction_histogram"]
    largest_fraction = max(histogram) / sum(histogram)
    print(name, "best source F1=", best,
          "largest predicted-class fraction=", largest_fraction,
          "best epoch=", best_row["epoch"])
```

A largest-class fraction above 0.90 or source F1 below roughly 0.88 indicates continued collapse.

## 5. Optional Task 2 normalized-MMD fallback

Run this only if `dan_clip5` still collapses according to source validation:

```python
!python -m task2.train_supplemental \
    --data-root {PACS_ROOT} \
    --method fallback
```

Inspect it without target labels:

```python
record = json.loads(Path(
    "task2/supplemental/results/dan_norm_clip5_train.json"
).read_text())
best_row = max(record["history"], key=lambda row: row["source_val_mean_macro_f1"])
histogram = best_row["source_val_prediction_histogram"]
print("best source F1:", record["best_source_val_mean_macro_f1"])
print("largest predicted-class fraction:", max(histogram) / sum(histogram))
```

Freeze the list of variants before target evaluation:

```python
TASK2_VARIANTS = ["dan_clip5", "dann_clip5", "cdan_clip5"]

# Add the fallback only if the source-only rule above required it.
if Path("task2/supplemental/checkpoints/dan_norm_clip5.pt").exists():
    TASK2_VARIANTS.append("dan_norm_clip5")

TASK2_METHODS = ",".join(TASK2_VARIANTS)
print(TASK2_METHODS)
```

## 6. Final Task 2 supplemental evaluation

This is the first supplemental Task 2 command that reads Sketch labels:

```python
!python -m task2.evaluate_supplemental \
    --data-root {PACS_ROOT} \
    --methods {TASK2_METHODS}
```

```python
!cat task2/supplemental/results/summary.csv
```

Save Task 2 immediately so a later session timeout cannot remove it:

```python
!zip -q -r /kaggle/working/task2_supplemental_artifacts.zip task2/supplemental
!ls -lh /kaggle/working/task2_supplemental_artifacts.zip
```

## 7. Train Task 3 clipped DAN-DG

This command never loads Sketch:

```python
!python -m task3.train_supplemental \
    --data-root {PACS_ROOT} \
    --method dan_dg_clip5
```

Inspect source-only stability:

```python
record = json.loads(Path(
    "task3/supplemental/results/dan_dg_clip5_train.json"
).read_text())
best_row = max(record["history"], key=lambda row: row["source_val_mean_macro_f1"])
histogram = best_row["source_val_prediction_histogram"]
largest_fraction = max(histogram) / sum(histogram)
print("best source F1:", record["best_source_val_mean_macro_f1"])
print("largest predicted-class fraction:", largest_fraction)
```

## 8. Optional Task 3 normalized-MMD fallback

Run only if the clipped lambda=1 model still collapses on source validation:

```python
!python -m task3.train_supplemental \
    --data-root {PACS_ROOT} \
    --method dan_dg_norm_clip5
```

Freeze the Task 3 variant list before Sketch is loaded:

```python
TASK3_VARIANTS = ["dan_dg_clip5"]
if Path("task3/supplemental/checkpoints/dan_dg_norm_clip5.pt").exists():
    TASK3_VARIANTS.append("dan_dg_norm_clip5")
TASK3_METHODS = ",".join(TASK3_VARIANTS)
print(TASK3_METHODS)
```

## 9. Task 3 source diagnostics

```python
!python -m task3.evaluate_supplemental_sources \
    --data-root {PACS_ROOT} \
    --methods {TASK3_METHODS}
```

Confirm that Sketch was not loaded:

```python
source = json.loads(Path(
    "task3/supplemental/results/source_diagnostics.json"
).read_text())
print(source["sketch_loaded"])
assert source["sketch_loaded"] is False
```

## 10. Final Task 3 Sketch evaluation

Only after the source diagnostic file is fixed:

```python
!python -m task3.evaluate_supplemental_sketch \
    --data-root {PACS_ROOT}
```

```python
!cat task3/supplemental/results/summary.csv
```

Save Task 3 immediately:

```python
!zip -q -r /kaggle/working/task3_supplemental_artifacts.zip task3/supplemental
!ls -lh /kaggle/working/task3_supplemental_artifacts.zip
```

## 11. Restore local CIFAR-10 and CIFAR-100

```python
import shutil
from pathlib import Path

input_root = Path("/kaggle/input")
c10 = [p.parent for p in input_root.rglob("data_batch_1")
       if (p.parent / "data_batch_5").is_file() and (p.parent / "test_batch").is_file()]
c100 = [p.parent for p in input_root.rglob("meta")
        if (p.parent / "train").is_file() and (p.parent / "test").is_file()]
assert len(c10) == 1 and len(c100) == 1

data_root = Path("/kaggle/working/cifar-data")
dest10 = data_root / "cifar-10-batches-py"
dest100 = data_root / "cifar-100-python"
dest10.mkdir(parents=True, exist_ok=True)
dest100.mkdir(parents=True, exist_ok=True)
for item in c10[0].iterdir():
    if item.is_file(): shutil.copy2(item, dest10 / item.name)
for item in c100[0].iterdir():
    if item.is_file(): shutil.copy2(item, dest100 / item.name)
```

## 12. Restore the original selected Vanilla checkpoint

```python
matches = list(Path("/kaggle/input").rglob("vanilla.pt"))
print(matches)
assert len(matches) >= 1
destination = Path("task4/checkpoints/vanilla.pt")
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(matches[0], destination)
```

If several `vanilla.pt` files are printed, select the one from the Task 4 artifact dataset.

Verify that it is the original selected checkpoint:

```python
import hashlib, yaml

expected = yaml.safe_load(Path("task4/configs/proser_clip5.yaml").read_text())["vanilla_checkpoint_sha256"]
actual = hashlib.sha256(destination.read_bytes()).hexdigest()
print(actual)
assert actual == expected
```

## 13. Train clipped PROSER

This uses CIFAR-10 only:

```python
!python -m task4.train_supplemental \
    --data-root /kaggle/working/cifar-data \
    --no-download
```

Inspect its CIFAR-10 validation trajectory before loading CIFAR-100:

```python
record = json.loads(Path(
    "task4/supplemental/results/proser_clip5_train.json"
).read_text())
best = max(record["history"], key=lambda row: row["validation_accuracy"])
print("best epoch:", best["epoch"])
print("best validation accuracy:", best["validation_accuracy"])
print("best-epoch dummy-win rate:", best["validation_dummy_win_rate"])
print("final validation accuracy:", record["history"][-1]["validation_accuracy"])
```

## 14. Extract and evaluate clipped PROSER

These commands open CIFAR-100 only after the checkpoint is fixed:

```python
!python -m task4.extract_supplemental \
    --data-root /kaggle/working/cifar-data \
    --no-download
```

```python
!python -m task4.evaluate_supplemental
!cat task4/supplemental/results/osr_metrics.csv
!cat task4/supplemental/results/final_metrics.json
```

Save Task 4 immediately:

```python
!zip -q -r /kaggle/working/task4_supplemental_artifacts.zip task4/supplemental
!ls -lh /kaggle/working/task4_supplemental_artifacts.zip
```

## 15. Save supplemental artifacts

```python
!zip -q -r /kaggle/working/task2_supplemental_artifacts.zip task2/supplemental
!zip -q -r /kaggle/working/task3_supplemental_artifacts.zip task3/supplemental
!zip -q -r /kaggle/working/task4_supplemental_artifacts.zip task4/supplemental
!ls -lh /kaggle/working/*_supplemental_artifacts.zip
```

Download all three named ZIP files before the session ends. Keep large checkpoints/caches out of Git; commit the small JSON, CSV, and PNG result files under each `supplemental/results/` directory.
