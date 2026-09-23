# Task 4: Open-Set Recognition on CIFAR

This implementation follows the fixed Task 4 protocol. Training and model selection use only CIFAR-10. `train.py` cannot import or load CIFAR-100; unknown examples are first opened by `extract_outputs.py` after all checkpoints exist. Write the PDF discussion and plausible/surprising labels yourself.

## Implemented protocol

- A committed, stratified 90/10 CIFAR-10 split uses seed 6304. The official test set is reserved for final known-class evaluation.
- All methods use a random-initialized CIFAR ResNet-18 with a 3x3 stride-one stem and no max pool.
- Vanilla uses standard crop/flip augmentation. GCSC adds `RandAugment(2, 9)` in the required position.
- PROSER starts from the selected Vanilla checkpoint, adds five random dummy classifiers, and fine-tunes the full network for 50 epochs. Each batch is split in half for the classifier-placeholder and layer2 manifold-mixup losses.
- CIFAR-100 **test** supplies exactly 800 fixed Near and 800 fixed Far unknowns. Its training set is never constructed.
- Frozen arrays are reused for MSP, MLS, Energy, diagonal shared-covariance Mahalanobis, common-MLS comparison, and the PROSER dummy-minus-known placeholder score.
- Every threshold is the 95th percentile of the corresponding CIFAR-10 validation unknownness score. Unknown examples never determine a threshold or checkpoint.

## Fresh Kaggle notebook: complete run

### 0. Push the implementation from your computer

From the repository root, review and push the source plus the small Task 3 results. Large checkpoints and artifact ZIPs are ignored:

```bash
git status
git add .gitignore README.md requirements.txt task3/README.md task3/results task4
git commit -m "Implement Task 4 open-set recognition"
git push
```

### 1. Create the notebook

1. Open Kaggle and choose **Create > New Notebook**.
2. Open **Notebook options** on the right.
3. Set **Accelerator** to **GPU T4 x2** or **GPU P100**. One GPU is sufficient.
4. Turn **Internet** on so the repository, packages, and CIFAR archives can be downloaded.
5. Keep the notebook language set to Python.

Task 4 does not need the PACS or Task 3 artifact dataset.

### 2. Clone the latest repository

Run this in the first cell, replacing the URL:

```python
!git clone YOUR_REPO_URL /kaggle/working/pa1-beyond-iid
%cd /kaggle/working/pa1-beyond-iid
!git log -1 --oneline
```

For a private repository, use Kaggle Secrets for a GitHub token. Do not paste a token into a saved notebook.

### 3. Install and verify the environment

```python
!python -m pip install -q -r requirements.txt
```

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
assert torch.cuda.is_available(), "Enable a GPU in Notebook options, then restart the session"
```

### 4. Create and inspect the fixed CIFAR-10 split

The command downloads CIFAR-10 and writes the deterministic indices:

```python
!python -m task4.data.make_splits --data-root /kaggle/working/cifar-data
```

Check the class balance:

```python
import json
from pathlib import Path

split = json.loads(Path("task4/data/cifar10_seed6304.json").read_text())
print(len(split["train"]), len(split["validation"]), split["seed"])
assert len(split["train"]) == 45000
assert len(split["validation"]) == 5000
assert not (set(split["train"]) & set(split["validation"]))
```

### 5. Train Vanilla

```python
!python -m task4.train --data-root /kaggle/working/cifar-data --method vanilla
```

This runs 100 epochs and saves the checkpoint with the highest CIFAR-10 validation accuracy. Do not run extraction yet.

### 6. Train GCSC

```python
!python -m task4.train --data-root /kaggle/working/cifar-data --method gcsc
```

This runs another 100 epochs with RandAugment. It starts from a fresh random initialization.

### 7. Fine-tune PROSER

```python
!python -m task4.train --data-root /kaggle/working/cifar-data --method proser
```

This loads the selected Vanilla checkpoint, initializes five dummy heads, and fine-tunes for 50 epochs. At this point all model choices are fixed without CIFAR-100.

If the runtime disconnects, attach the artifact ZIP made in step 10 to a new notebook, restore it as shown in **Restarting later**, and rerun the same method command. A method with both its checkpoint and training record is skipped.

### 8. Extract all frozen outputs

Only now load the fixed CIFAR-100 test unknowns:

```python
!python -m task4.extract_outputs --data-root /kaggle/working/cifar-data
```

This creates compressed caches for CIFAR-10 train/validation/test and CIFAR-100 Near/Far. Caches can be large and are ignored by Git.

### 9. Evaluate and inspect the deliverables

```python
!python -m task4.evaluate_osr --data-root /kaggle/working/cifar-data
!cat task4/results/osr_metrics.csv
!cat task4/results/vanilla_score_comparison.csv
!cat task4/results/trained_model_comparison.csv
!cat task4/results/summary.json
!ls -lh task4/results
```

Display the two required figures:

```python
from IPython.display import display
from PIL import Image

display(Image.open("task4/results/vanilla_score_distributions.png"))
display(Image.open("task4/results/vanilla_mls_failures.png"))
```

Open `task4/results/vanilla_mls_failures.csv` and fill its `student_classification` column with your own plausible/surprising judgment before using it in your report. The assignment requires your own analysis.

### 10. Save everything before the Kaggle session ends

```python
!zip -q -r /kaggle/working/task4_artifacts.zip \
    task4/checkpoints task4/results task4/cache task4/data/cifar10_seed6304.json
!ls -lh /kaggle/working/task4_artifacts.zip
```

Download `task4_artifacts.zip` from the Files pane, or choose **Save Version** and include notebook outputs. The `.pt` checkpoints and `cache/` are intentionally ignored by Git. Commit the source, configs, fixed split, and small result files; keep the large artifact ZIP as a Kaggle dataset.

## Restarting later in a new Kaggle session

Attach the uploaded Task 4 artifact dataset, clone the latest repository, install requirements, and copy its expanded contents back into the clone:

```python
from pathlib import Path
import shutil

repo = Path("/kaggle/working/pa1-beyond-iid")
matches = list(Path("/kaggle/input").rglob("task4_artifacts.zip"))
if matches:
    !unzip -q -o {str(matches[0])} -d /kaggle/working/pa1-beyond-iid
else:
    roots = [p for p in Path("/kaggle/input").rglob("task4")
             if (p / "checkpoints").is_dir()]
    print(roots)
    assert len(roots) == 1, "Select the printed expanded task4 artifact folder"
    shutil.copytree(roots[0], repo / "task4", dirs_exist_ok=True)
```

Then use the same train, extract, and evaluate commands. Completed training stages are skipped.
