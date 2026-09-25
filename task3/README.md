# Task 3: Domain Generalization on PACS

Task 3 reuses the exact Task 2 protocol and ERM checkpoint. Photo, Art Painting, and Cartoon are the only domains available to training, source diagnostics, checkpoint selection, and study design. Sketch is opened only by `evaluate_sketch.py` after every checkpoint and source diagnostic is fixed. This file records the experiment design and commands.

## Fixed design

- The source split is `shared/splits/pacs_sketch_seed6304.json`. All new models use the same ImageNet ResNet-18 initialization, seven-class head, deterministic source augmentation, domain-balanced batches of eight examples per source, frozen BatchNorm running statistics, AdamW settings, 30-epoch limit, five-epoch patience, and seed 6304 used by Task 2.
- ERM is loaded unchanged from `task2/checkpoints/source_only.pt`. Its expected SHA-256 is stored in `configs/base.json`, so training stops if a different checkpoint is supplied.
- DAN-DG adds the average MMD over Photo–Art, Photo–Cartoon, and Art–Cartoon feature pairs. It uses Task 2's 512-dimensional feature MMD and three current-batch median-scaled RBF kernels. The main comparison uses weight 1.
- SAM uses standard non-adaptive two-pass updates with radius 0.05. Both passes preserve the shared frozen BatchNorm running-statistics policy.
- Controlled study: DAN-DG weights 0.1, 1, and 10. The hypothesis is fixed in `configs/base.json`; Sketch is not used to choose among them.
- `evaluate_sources.py` reports source accuracy and macro-F1 by domain, mean and worst-domain values, balanced three-way source-domain separability with a seeded 70/30 logistic-regression split, and the common radius-0.05 sharpness proxy on a fixed batch of 32 examples per source.
- `evaluate_sketch.py` checks the config, split, source diagnostics, and checkpoint hashes before loading Sketch labels. It writes final aggregate and class metrics, dominant confusions, the controlled-study table, and the required Task 2 DAN comparison.

## Fresh Kaggle notebook

### 1. Configure and clone

Create a Kaggle notebook, attach the same PACS dataset, enable a GPU and Internet, and run:

```python
!git clone https://github.com/ahmad-sarfraz345/pa1-beyond-iid.git /kaggle/working/pa1-beyond-iid
%cd /kaggle/working/pa1-beyond-iid
!python -m pip install -q -r requirements.txt
!python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Locate PACS `images/`

```python
from pathlib import Path

domains = ("photo", "art_painting", "cartoon", "sketch")
matches = [p for p in Path("/kaggle/input").rglob("images")
           if p.is_dir() and all((p / domain).is_dir() for domain in domains)]
print(matches)
assert len(matches) == 1, "Select the PACS images directory from the printed matches"
DATA_ROOT = str(matches[0])
print("Using:", DATA_ROOT)
```

The expected path ends in `pacs/images`. Do not use `pacs/splits`; Task 3 reuses the committed assignment split.

### 3. Restore the Task 2 ERM checkpoint

Attach the saved Task 2 artifacts as a Kaggle dataset. Kaggle expands the uploaded archive under `/kaggle/input`, which is read-only. Locate the unchanged ERM checkpoint and copy it into the cloned repository:

```python
import shutil

artifact_root = Path("/kaggle/input/datasets/ahmadsarfraz345/task2-artifacts")
checkpoint_matches = list(artifact_root.rglob("source_only.pt"))
print(checkpoint_matches)
assert len(checkpoint_matches) == 1, "Expected exactly one source_only.pt"

destination = Path("/kaggle/working/pa1-beyond-iid/task2/checkpoints/source_only.pt")
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(checkpoint_matches[0], destination)
print("Copied to:", destination)
```

Verify that this is the unchanged checkpoint used by Task 2:

```python
!python -c "import hashlib,pathlib,json; p=pathlib.Path('task2/checkpoints/source_only.pt'); c=json.loads(pathlib.Path('task3/configs/base.json').read_text()); h=hashlib.sha256(p.read_bytes()).hexdigest(); print(h); assert h==c['erm_checkpoint_sha256']"
```

### 4. Train the Task 3 models

This command trains three DAN-DG weights and the main SAM model. It does not load Sketch:

```python
!python -m task3.train --data-root {DATA_ROOT} --method all
```

To run one method at a time, use `dan_dg_0p1`, `dan_dg`, `dan_dg_10`, and `sam` as `--method` values. A completed method is skipped when both its checkpoint and training record remain available.

### 5. Run source-only evaluation

```python
!python -m task3.evaluate_sources --data-root {DATA_ROOT}
```

Confirm that `task3/results/source_diagnostics.json` was created. This step still does not load Sketch.

### 6. Run final Sketch evaluation

Only after training and source diagnostics finish:

```python
!python -m task3.evaluate_sketch --data-root {DATA_ROOT}
!cat task3/results/summary.csv
!cat task3/results/task2_task3_comparison.csv
```

### 7. Save artifacts before the session ends

```python
!zip -q -r /kaggle/working/task3_artifacts.zip task3/checkpoints task3/results task2/checkpoints/source_only.pt shared/splits/pacs_sketch_seed6304.json
!ls -lh /kaggle/working/task3_artifacts.zip
```

Download `task3_artifacts.zip` from the notebook Output panel or save a notebook version with outputs. Checkpoints are ignored by Git; commit only the small files under `task3/results/` and the source split if it is not already in the repository.
