# Experimental Results Audit

This is a technical QA record for the code and artifacts. It is not report prose. Any interpretation included in the submitted PDF must be written independently by the student.

Audit date: 2026-09-25

## Overall status

| Task | Artifact/protocol status | Training status | Action |
|---|---|---|---|
| 1 | Complete and internally consistent | Healthy | Keep results |
| 2 | Complete; split/config hashes match | DAN 1/10 and DANN collapsed; CDAN became unstable after its selected epoch | Keep required results and disclose failures; use only clearly marked supplemental stabilization runs if time permits |
| 3 | Complete; Task 2 ERM is reused byte-for-byte and Sketch isolation is recorded | DAN-DG 1/10 collapsed; DAN-DG 0.1 and SAM are healthy | Keep prescribed main result and disclose collapse; do not replace main λ=1 with post-hoc λ=0.1 |
| 4 | Complete; dataset sizes, split, caches, thresholds, and metrics verify | Vanilla/GCSC healthy; PROSER deteriorated after epoch 1, but best-checkpoint selection protected the final model | Keep results; describe PROSER instability and the selected epoch |

## Task 1: Inductive biases and representations

### Checks passed

- Fixed stratified split: 4,000 train, 1,000 validation, and 500 test examples; test set is exactly 50 images per class with no duplicate identifiers.
- All three linear heads trained with early stopping and retained sensible best epochs: ResNet-50 epoch 13, ViT-B/16 epoch 6, and CLIP ViT-B/32 epoch 14.
- Clean test accuracy is high but consistent across the four decision rules: 97.8% ResNet, 97.4% ViT, 98.4% CLIP head, and 98.0% CLIP zero-shot.
- Cue review contains 400 candidate decisions: 348 accepted and 52 rejected before prediction evaluation.
- Final cue set is exactly balanced: 20 examples for each of ten pair/direction groups, totaling 200.
- Cue arithmetic, coverage, prediction files, translation curve, representation similarities, and 12 required t-SNE figures are internally consistent.
- No training collapse or suspicious discontinuity was found.

### Minor cautions

- Clean accuracies are close, so intervention deltas and consistency are more informative than raw clean ranking.
- CLIP head and CLIP zero-shot representation-stability values are identical by design because they use the same frozen image representation with different decision rules.
- Cue conclusions should retain coverage values; shape bias alone omits predictions assigned to neither intended class.

### Fixes

No rerun is recommended. Preserve the review CSV and rejection counts as evidence that filtering occurred before model evaluation.

## Task 2: Unsupervised domain adaptation

### Checks passed

- PACS source split is stratified, non-overlapping, and shared with Task 3.
- Config and split hashes match every training record and final evaluation.
- Checkpoints were selected using mean source-validation macro-F1; target labels are opened only in final evaluation.
- Source-only is healthy: best source macro-F1 0.9334 and Sketch accuracy/macro-F1 0.5610/0.5745.
- DAN λ=0.1 is healthy and improves Sketch accuracy to 0.6872 and macro-F1 to 0.6349 while retaining source macro-F1 0.9469.

### Troublesome results

1. **DAN λ=1 collapsed.** Best source macro-F1 is 0.0877, Sketch accuracy is 0.0204, and every Sketch image is predicted as `house`.
2. **DAN λ=10 collapsed.** Source macro-F1 is 0.0507, Sketch accuracy is 0.0407, and every Sketch image is predicted as `person`.
3. **DANN diverged.** Its source class/domain losses become extremely large, its best source macro-F1 is only 0.1276, and final predictions are almost entirely `house` or `person`. Domain-probe accuracy remains 0.9986, so the intended domain confusion was not achieved.
4. **CDAN is unstable after epoch 2.** The selected epoch-2 checkpoint is valid, but later class loss reaches hundreds. The selected checkpoint still transfers negatively: Sketch accuracy 0.3489 versus 0.5610 for source-only. Domain separability remains high at 0.9945.
5. The current training-curve plot uses one linear y-axis. The exploding DANN/CDAN losses compress the healthy curves and make them hard to read.

### Recommended handling

- Retain these runs as the required fixed-protocol results. They directly demonstrate the manual's warning that alignment can destroy class information or become unstable.
- Do not present low separability from collapsed DAN models as successful invariance. Their poor source classification and one-class predictions show that discriminative structure was lost.
- Treat DAN λ=0.1 as a controlled-study result, not a replacement for the prescribed main λ=1 result.
- For presentation, plot unstable losses on a log scale or in separate panels. This changes only visualization and does not deviate from the manual.

### Supplemental stabilization options

Run these only as additional diagnostics while preserving the original required rows:

1. Log gradient norms, prediction histograms, and finite-value checks. Logging alone does not change the manual protocol.
2. Apply global gradient clipping to backbone, classifier, and discriminator parameters. **Deviation:** gradient clipping is not specified in the manual.
3. Ramp the alignment coefficient or GRL strength more slowly, or use a lower discriminator learning rate/separate discriminator optimizer. **Deviation:** these change the prescribed optimization schedule or shared optimizer settings.
4. Use an MMD warm-up before reaching λ=1. **Deviation:** the main objective no longer uses constant λ=1 throughout training.

Any stabilized run must be labeled supplemental and must not silently replace the required result.

## Task 3: Domain generalization

### Checks passed

- The Task 2 source-only checkpoint hash exactly matches the required Task 3 ERM hash.
- `source_diagnostics.json` records `sketch_loaded: false`.
- Config, split, training records, and all checkpoint hashes match.
- ERM is healthy: source macro-F1 0.9334 and Sketch accuracy/macro-F1 0.5610/0.5745.
- DAN-DG λ=0.1 is healthy: source macro-F1 0.9529 and Sketch accuracy/macro-F1 0.7109/0.7199.
- SAM is healthy: source macro-F1 0.9567 and Sketch accuracy/macro-F1 0.7045/0.7466.

### Troublesome results

1. **Main DAN-DG λ=1 collapsed immediately.** Source macro-F1 is 0.0507 and all 3,929 Sketch images are predicted as `person`.
2. **DAN-DG λ=10 also collapsed to the same one-class solution.**
3. The collapsed models show very small sharpness increases (about 0.01). This does not indicate a useful flat solution: a near-constant classifier can have low local loss sensitivity while being unusable.
4. The lower source-domain separability of the collapsed models must not be interpreted as successful domain invariance without the accompanying source and class metrics.

### Recommended handling

- Keep λ=1 as the main prescribed comparison even though it failed.
- Keep λ=0.1 and λ=10 in the controlled study and state that λ=0.1 is the successful bounded-study condition.
- Use the one-class confusion matrix, source macro-F1, and class loss near log(7) as direct collapse evidence.
- Do not claim that the collapsed λ=1/10 models are meaningfully flatter than SAM.

### Supplemental stabilization options

- Gradient-norm logging is protocol-neutral.
- Gradient clipping, MMD warm-up, normalized-feature MMD, or a smaller learning rate may stabilize λ=1. **All are deviations** from the fixed main recipe and must be reported only as supplemental variants.
- Do not choose a stabilized variant using Sketch results. Selection must remain source-only.

## Task 4: Open-set recognition

### Checks passed

- CIFAR-10 split is exactly stratified: 4,500 train and 500 validation examples per class, with 45,000/5,000 totals and no overlap.
- CIFAR-10 test has 1,000 examples per class.
- Near and Far caches each contain exactly 800 examples; PROSER caches contain five dummy logits per example.
- Config hashes match training records, and PROSER records the exact Vanilla checkpoint from which it was initialized.
- All 21 OSR metric rows pass range, shared-threshold, and rejection/FPR complement checks.
- Vanilla and GCSC training are healthy. Test accuracy is 94.67% and 95.35%, respectively.
- The required score table, model table, plots, and six failure examples are present.

### Troublesome result

**PROSER training deteriorates after the first epoch.** Validation accuracy is 94.38% at epoch 1, 91.84% at epoch 20, 79.10% at epoch 30, and 82.62% at epoch 50. The saved checkpoint correctly retains epoch 1 and achieves 94.07% test accuracy, so the reported final checkpoint is not collapsed. However, the trajectory shows that prolonged placeholder optimization damaged known-class recognition.

PROSER also fails to beat Vanilla in this run. The placeholder score improves over PROSER MLS on Far rejection and Far AUROC, but remains below Vanilla MLS overall.

### Recommended handling

- Keep the selected epoch-1 checkpoint because the manual explicitly requires selection by CIFAR-10 validation accuracy.
- Record both the 50-epoch deterioration and the selected checkpoint epoch. Do not describe the final reported PROSER checkpoint as collapsed.
- Do not tune using CIFAR-100 outcomes. Any stabilization must use CIFAR-10 training/validation only.
- Early stopping would save computation but does not alter the already selected result. **Deviation:** ending before the prescribed 50 fine-tuning epochs would change the stated training procedure, so retain the completed run as the required result.

### Supplemental stabilization options

- Add gradient and known/dummy prediction diagnostics without changing training.
- Gradient clipping, a lower learning rate, freezing the backbone temporarily, or changing β/γ could improve stability. **Deviation:** each changes a manual-specified setting or the full-model fine-tuning procedure.
- Preserve the required β=1, γ=0.1, learning rate 1e-3 result as the main row.

## Deviation ledger

The following proposed fixes require explicit disclosure if run:

| Proposed change | Affected tasks | Why it is a deviation |
|---|---|---|
| Gradient clipping | 2, 3, 4 | Not included in the fixed optimizer recipe |
| Alignment/MMD warm-up | 2, 3 | Changes constant alignment pressure or prescribed schedule |
| Lower/separate discriminator learning rate | 2 | Breaks the shared optimizer settings |
| Normalizing features before MMD | 2, 3 | Changes the specified MMD input and objective |
| Lower PROSER learning rate or different β/γ | 4 | Manual fixes these values |
| Temporarily freezing the PROSER backbone | 4 | Manual requires full-model fine-tuning |
| Stopping PROSER before 50 epochs | 4 | Manual specifies 50 epochs, although best-checkpoint selection already protects evaluation |

Pure logging, additional plots, log-scaled axes, and class-frequency summaries do not change training and do not constitute experimental deviations.
