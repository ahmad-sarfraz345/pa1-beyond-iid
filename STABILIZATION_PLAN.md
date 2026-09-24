# Supplemental Stabilization Plan

This is an experiment-control record, not report prose. It records the stabilization choices before supplemental target evaluation.

Context: the teaching assistant explicitly permits gradient clipping, normalization, and hyperparameter changes when failures occur, provided every deviation from the assignment manual is documented and reasoned about.

Plan date: 2026-09-25

## Rules

1. Preserve all original checkpoints, histories, tables, and figures.
2. Write stabilized artifacts under separate names/directories. Never overwrite the prescribed runs.
3. Diagnose and select using permitted validation data only:
   - Task 2: source validation macro-F1 and unlabeled domain-loss behavior; no Sketch labels.
   - Task 3: source validation macro-F1 only; do not load any Sketch image.
   - Task 4: CIFAR-10 validation accuracy only; do not load CIFAR-100 during selection.
4. Freeze the stabilization recipe before evaluating any target labels.
5. Report the prescribed and stabilized settings separately if both are used.

## Stage A: diagnostics only

Add the following logging without changing optimization:

- total gradient norm before clipping;
- maximum absolute gradient;
- finite/non-finite loss and gradient checks;
- prediction histogram on source validation;
- classification and alignment loss separately;
- for DANN/CDAN, discriminator accuracy;
- for PROSER, known-only accuracy and dummy-win rate.

This stage is not a protocol deviation because it does not change training.

## Stage B: minimal stabilization

Use global gradient clipping with `max_norm = 5.0` immediately after `loss.backward()` and before `optimizer.step()`.

Apply it only to the failed methods in the first pass:

- Task 2: DAN λ=1, DAN λ=10, DANN, and CDAN.
- Task 3: DAN-DG λ=1 and DAN-DG λ=10.
- Task 4: PROSER.

Keep every other manual setting unchanged. Name these variants with `_clip5` and store them separately.

This is a deviation because the manual does not specify gradient clipping.

## Stage C: method-specific fallback

Run this stage only if clipping still produces collapse according to permitted validation metrics.

### MMD methods in Tasks 2 and 3

L2-normalize the 512-dimensional features before computing pairwise distances in MMD, while retaining the manual's three median-scaled kernels and prescribed λ values. Continue clipping at 5.0. Name these variants `_norm_clip5`.

Reason: normalization limits feature-scale-driven kernel gradients while retaining angular class/domain structure.

Deviation: the manual specifies MMD on the pre-head feature and does not request feature normalization.

### DANN and CDAN in Task 2

If clipping alone fails, retain the manual's GRL schedule but use a discriminator learning rate of `2e-5` and the original `1e-4` model learning rate, with separate AdamW optimizers. Name these variants `_clip5_disc2e5`.

Reason: the observed minimax dynamics diverged, with class and domain losses reaching hundreds. A slower discriminator update can reduce oscillation.

Deviation: the manual requests the shared optimizer recipe and does not specify a separate discriminator learning rate.

### PROSER in Task 4

If clipping alone still causes CIFAR-10 validation deterioration, use learning rate `1e-4` instead of `1e-3`, retain β=1, γ=0.1, five dummies, full-model fine-tuning, cosine decay, and 50 epochs. Name this variant `proser_clip5_lr1e4`.

Reason: the original selected epoch 1 and deteriorated substantially thereafter, indicating that the prescribed fine-tuning step size may be too aggressive for this implementation/model combination.

Deviation: the manual fixes PROSER learning rate at `1e-3`.

## Stop conditions

A supplemental model is considered non-collapsed only if all of the following hold on permitted validation data:

- losses and gradients remain finite;
- no single predicted class accounts for more than 90% of source/known validation predictions;
- source macro-F1 (Tasks 2/3) or CIFAR-10 validation accuracy (Task 4) remains within five percentage points of its corresponding non-adapted starting point;
- the selected checkpoint is determined only by the manual's validation metric.

These are diagnostic safeguards, not new target-selection criteria.

## Recommended run order

1. Task 2 DAN λ=1 `_clip5`.
2. Task 3 DAN-DG λ=1 `_clip5` using the same clipping implementation.
3. Task 2 DANN and CDAN `_clip5`.
4. Task 4 PROSER `_clip5` only if compute time permits; its original selected checkpoint is already valid.
5. Run Stage C only for methods that still collapse.

Do not rerun healthy ERM, DAN/DAN-DG λ=0.1, SAM, Vanilla, or GCSC unless a fair supplemental table explicitly requires the same optimizer modification for every row.

## Reporting checklist for each supplemental variant

- original manual setting;
- changed setting and exact value;
- validation-only reason for the change;
- whether target data or labels were unavailable during selection;
- original versus stabilized training curves;
- original versus stabilized source/known performance;
- final target/unknown evaluation performed only after the recipe was frozen;
- statement that the run is supplemental if the prescribed result is also retained.
