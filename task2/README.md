# Task 2: PACS unsupervised adaptation to Sketch

This directory implements the assignment's fixed Photo / Art Painting / Cartoon → Sketch protocol. It is code and an experiment plan, not PDF report text. Write the assignment report yourself.

## Fixed protocol

- Use PACS with the seven classes in `shared/pacs.py`. `prepare` makes a separate stratified 80/20 train/validation split in each source domain with seed 6304. Save `shared/splits/pacs_sketch_seed6304.json` and reuse it for Task 3.
- Source-only ERM uses only labeled source images. DAN, DANN, and CDAN use the same labeled source images plus unlabeled Sketch images. The training code never reads Sketch class labels. Final evaluation reads them only after all six checkpoints and training records exist under the original config and split hashes.
- All runs start from `ResNet18_Weights.IMAGENET1K_V1` with a new seven-class head. The entire network is trainable, including BatchNorm affine parameters; ImageNet BatchNorm running statistics stay frozen. All runs use AdamW (learning rate 1e-4, weight decay 1e-4), up to 30 epochs, and patience 5 on the mean source-validation macro-F1.
- Images are resized to 256 × 256. Training uses a random 224 × 224 crop and horizontal flip; validation and final evaluation use a center crop. ImageNet normalization is applied. Each update draws eight examples from each source and, for adaptation, 24 from Sketch. Loaders cycle as needed. Source crops are deterministic per sample and epoch across methods.
- DAN uses empirical MMD² on the 512-dimensional pre-head features, with three Gaussian kernels at 0.5, 1, and 2 times the current combined-batch median pairwise squared distance. The main DAN weight is 1. DANN uses a 512 → 256 → 2 discriminator; CDAN uses a 3584 → 256 → 2 discriminator on the outer product of features and softmax class probabilities. Both use ReLU, 0.5 dropout, unit domain-loss weight, and the scheduled gradient reversal `2/(1+exp(-10p))-1`.
- Controlled study: DAN weights 0.1, 1, and 10. The hypothesis is recorded in `configs/base.json` before target evaluation. Checkpoint selection uses source validation only; target results are analysis, never tuning feedback.
- Final outputs include per-method source and target accuracy/macro-F1, target accuracy change against source-only, class accuracy changes and dominant confusions, training curves, and a balanced logistic-regression domain probe. The probe uses equal numbers of source-validation and target features, a seeded 70/30 split, standardized features, and `C=1`.

## Dataset layout

Point `--data-root` to the `pacs/images/` directory containing `photo/`, `art_painting/`, `cartoon/`, and `sketch/`, each with seven class directories such as `dog/`, `elephant/`, and so on. Keep the same data root for `prepare`, `train`, and `evaluate_final`. This layout is documented by [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#pacs). Ignore the dataset's supplied `pacs/splits/` files: this assignment creates its own seeded source split. Raw images stay in Kaggle Input or ignored `data/`, outside Git.

## Kaggle GPU, from a fresh notebook

1. Push these code changes to your Git repository. Create a Kaggle notebook, enable a GPU accelerator and internet in notebook settings. In separate code cells, replace `YOUR_REPO_URL` with your repository URL and run:

   ```python
   !git clone YOUR_REPO_URL /kaggle/working/pa1-beyond-iid
   %cd /kaggle/working/pa1-beyond-iid
   !python -m pip install -q -r requirements.txt
   !python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

2. Attach the PACS dataset containing `pacs/images/` through Kaggle's **Add Input** panel. Locate its exact image root (the Kaggle input slug may differ):

   ```python
   from pathlib import Path
   domains = ("photo", "art_painting", "cartoon", "sketch")
   matches = [p for p in Path("/kaggle/input").rglob("images")
              if p.is_dir() and all((p / d).is_dir() for d in domains)]
   print(matches)
   assert len(matches) == 1, "Select the PACS images directory from the printed matches"
   DATA_ROOT = str(matches[0])
   print("Using:", DATA_ROOT)
   ```

3. Create or validate the saved source split. If the split file was committed from an earlier session, `prepare` reuses it. It does not overwrite it.

   ```python
   !python -m task2.train prepare --data-root {DATA_ROOT}
   ```

4. Train the six fixed runs. The source-only checkpoint is the unchanged ERM checkpoint for Task 3. Each completed method writes `task2/results/<method>_train.json` and `task2/checkpoints/<method>.pt`; rerunning `train` skips completed methods in the current session. An interrupted method restarts from its original seed.

   ```python
   !python -m task2.train train --data-root {DATA_ROOT} --method all
   ```

5. Once all six runs complete and their config and checkpoints are fixed, reveal Sketch labels to the final evaluator:

   ```python
   !python -m task2.evaluate_final --data-root {DATA_ROOT}
   !cat task2/results/summary.csv
   ```

6. **Save before the Kaggle session ends.** The ignored `.pt` checkpoints are essential for Task 3; the source-only checkpoint must be reused unchanged. Download the archive from Kaggle's Output panel or use Save Version with outputs enabled. Also commit the small `shared/splits/*.json` and `task2/results/*` files to Git after checking them. Do not commit raw PACS images or checkpoints.

   ```python
   !zip -q -r /kaggle/working/task2_artifacts.zip task2/checkpoints task2/results shared/splits/pacs_sketch_seed6304.json
   !ls -lh /kaggle/working/task2_artifacts.zip
   ```

For a new Kaggle session after training, restore the archive into the cloned repository before running the final evaluator or Task 3. A fresh session without that archive has no checkpoints, even if the small results were pushed to Git.

## Files and attribution

`task2/train.py` runs training, `task2/evaluate_final.py` runs the final labeled evaluation, `task2/models.py` and `task2/methods.py` define the objectives, and `shared/` holds the PACS loader and saved source split. ResNet-18 and pretrained weights are loaded through [torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html); PACS folder structure follows [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#pacs). These are external dependencies, not copied implementations.
