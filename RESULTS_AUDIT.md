# Experimental Results Audit

This is a technical QA record for the code and artifacts. It is not report prose. Any interpretation included in the submitted PDF must be written independently by the student.

Audit date: 2026-09-25

## Overall status

| Task | Artifact/protocol status | Training status | Action |
|---|---|---|---|
| 1 | Complete and internally consistent | Healthy | Keep results |
| 2 | Original and supplemental artifacts complete; hashes and target-isolation gates verify | Clipping recovers DAN lambda=1 and CDAN; DANN remains failed | Keep prescribed rows as primary and present successful/failed stabilization attempts as supplemental |
| 3 | Original and supplemental artifacts complete; ERM reuse and Sketch isolation verify | Clipping alone fails; normalized-feature MMD plus clipping recovers DAN-DG lambda=1 | Keep prescribed lambda=1 failure as primary and label the recovered model supplemental |
| 4 | Original and supplemental artifacts complete; checkpoint and validation-only selection verify | Clipping reduces late known-accuracy deterioration but does not prevent late dummy dominance or materially change the selected result | Keep the original result as primary; clipping is a supplemental stability diagnostic |

## Supplemental run integrity and report readiness

- `all_supplemental_artifacts.zip` contained 42 safe relative paths and was extracted into the corresponding `task2/`, `task3/`, `task4/`, and `supplemental_overnight/` directories.
- All three tasks completed on a Tesla T4 using repository commit `98d45036b4ce85950cca612481a8159afd8fb9d4` between 22:03 and 23:20 UTC on 2026-09-24.
- No traceback or non-finite metric was found in the records or logs.
- Task 2 and Task 3 supplemental config hashes match the current configs. Their PACS split hash matches the original shared split.
- All six supplemental checkpoint hashes match the hashes recorded during evaluation.
- Selection manifests were written before target evaluation. Task 2 records `target_labels_opened: false`, Task 3 records `sketch_loaded: false`, and Task 4 records `cifar100_loaded: false` at selection time.
- The work is ready for the report. Preserve the manual runs as the required results and put the stabilization experiments in a clearly labeled supplemental or ablation subsection. Do not present the remaining DANN, clipped DAN-DG, or late-PROSER failures as resolved.

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

### Supplemental results

All supplemental Task 2 methods used global gradient clipping with `max_norm=5.0`. The decision to run a fallback was based only on source validation. DAN passed the preregistered source gate, so normalized-feature DAN was not run.

| Method | Source macro-F1 | Sketch accuracy | Sketch macro-F1 | Change in accuracy from corresponding original |
|---|---:|---:|---:|---:|
| DAN lambda=1, original | 0.0877 | 0.0204 | 0.0057 | -- |
| DAN lambda=1 + clip5 | 0.9331 | 0.6406 | 0.5982 | +0.6203 |
| DANN, original | 0.1276 | 0.0402 | 0.0217 | -- |
| DANN + clip5 | 0.5929 | 0.0593 | 0.0363 | +0.0191 |
| CDAN, original | 0.9051 | 0.3489 | 0.4201 | -- |
| CDAN + clip5 | 0.9313 | 0.6826 | 0.6277 | +0.3337 |

- **DAN is recovered by clipping.** Its selected epoch is 9, the largest source-validation predicted-class fraction is 0.2168, and all recorded metrics are finite. It improves over source-only by 7.97 accuracy points on Sketch. It remains 4.66 points below the healthy manual DAN lambda=0.1 result in Sketch accuracy.
- **CDAN is recovered by clipping.** Its selected epoch is 11, source macro-F1 remains near ERM, and Sketch accuracy rises to 0.6826. This is only 0.46 points below DAN lambda=0.1. Domain-probe accuracy remains high at 0.9753, so the gain does not require full domain confusion.
- **DANN is not recovered.** Although clipping raises source macro-F1, it remains 34.05 points below ERM and Sketch predictions remain almost entirely `person`. Epoch-mean pre-clip gradient norm reaches 362,803, classification loss reaches 1,242, and alignment loss reaches 1,988. Clipping bounds the applied update but does not repair the unstable adversarial dynamics.
- The successful DAN and CDAN results support gradient clipping as a useful stabilization step for this implementation. The failed DANN result shows that clipping alone is not a universal fix.

### Remaining optional work

No further rerun is required for report readiness. A lower or separate discriminator learning rate could be tested if additional time is available, but it would be another supplemental deviation and is unnecessary to document the observed DANN failure.

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

### Supplemental results

The first supplemental model added clipping at `max_norm=5.0`. It failed the source-only gate, so the preregistered fallback also L2-normalized the 512-dimensional features before MMD while retaining lambda=1 and clipping.

| Method | Source macro-F1 | Sketch accuracy | Sketch macro-F1 | Source-domain separability | Sharpness increase |
|---|---:|---:|---:|---:|---:|
| DAN-DG lambda=1, original | 0.0507 | 0.0407 | 0.0112 | 0.5847 | 0.0103 |
| DAN-DG lambda=1 + clip5 | 0.0507 | 0.0407 | 0.0112 | 0.5914 | 0.0088 |
| DAN-DG lambda=1 + normalized MMD + clip5 | 0.9284 | 0.6261 | 0.6321 | 0.6611 | 46.9051 |

- **Clipping alone does not help DAN-DG.** It selects epoch 1, predicts `person` for every source-validation and Sketch example, and reproduces the failed original accuracy.
- **Normalized-feature MMD plus clipping recovers a discriminative model.** The selected epoch is 3, the largest source predicted-class fraction is 0.2094, source macro-F1 is within 0.50 points of ERM, and Sketch accuracy improves by 58.54 points over the original lambda=1 run and by 6.52 points over ERM.
- The recovered model still trails the manual lambda=0.1 run by 8.48 Sketch accuracy points and trails SAM by 7.84 points. It is useful evidence about the failure mechanism, but it is not the best Task 3 model.
- Feature normalization changes the MMD objective and must be disclosed. It likely prevents feature scale from dominating the distance kernels, but this is a mechanism hypothesis rather than a directly proven cause.
- **Sharpness remains a serious caveat.** The recovered model's fixed-radius loss increase is 46.91, compared with 0.334 for ERM, 0.118 for DAN-DG lambda=0.1, and 0.112 for SAM. Do not describe it as flat or uniformly robust. Its source/Sketch classification is healthy, but it is highly sensitive under the assignment's one-step parameter perturbation diagnostic.
- Epoch-mean pre-clip gradient norm reaches 395.65. The selected checkpoint is protected by source validation, but the final epoch's source macro-F1 falls to 0.6870; include the selected epoch rather than implying stable convergence through the last epoch.

### Remaining optional work

No further rerun is required for report readiness. The selection was source-only and the fallback succeeded according to its preregistered gate. A lower learning rate or MMD warm-up could be explored, but would add more post-hoc variants without being necessary for the assignment conclusions.

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

### Supplemental results

The supplemental PROSER run retained all manual hyperparameters and 50 epochs, adding only global gradient clipping at `max_norm=5.0`. Its checkpoint was fixed using CIFAR-10 validation before CIFAR-100 was loaded.

| Quantity | Original PROSER | PROSER + clip5 | Change |
|---|---:|---:|---:|
| Best validation accuracy | 0.9438 | 0.9446 | +0.0008 |
| Final epoch validation accuracy | 0.8262 | 0.9040 | +0.0778 |
| Selected CIFAR-10 test accuracy | 0.9407 | 0.9408 | +0.0001 |
| MLS All AUROC | 0.8413 | 0.8414 | +0.0001 |
| Placeholder All AUROC | 0.8497 | 0.8501 | +0.0003 |
| Placeholder All unknown rejection | 0.4175 | 0.4181 | +0.0006 |

- Clipping substantially improves the late known-class trajectory, raising epoch-50 validation accuracy from 82.62% to 90.40%.
- It does not materially improve the selected checkpoint or OSR metrics. Both original and clipped runs select epoch 1, and changes in test accuracy, AUROC, and rejection are at most a few hundredths of a percentage point.
- The best checkpoint's dummy-win rate on known validation images is 0.0802. Dummy wins rise sharply after epoch 8, exceed 0.81 at epoch 10, and become essentially 1.0 from epoch 22 onward. Thus, clipping does not prevent late placeholder dominance even though it reduces degradation of the known logits.
- Recorded epoch-mean pre-clip gradient norms range from 1.12 to 4.68, below the clipping threshold on average. Some individual batches may still have been clipped, but the diagnostic explains why the selected epoch-1 model is almost unchanged.
- Keep the original PROSER result as the main manual row. Use the clipped trajectory to show partial optimization stabilization, while stating that the late dummy-head pathology and inferior OSR performance relative to Vanilla remain.

### Remaining optional work

No further rerun is required for report readiness. A lower learning rate could target the remaining dummy dominance, but the selected checkpoint is already valid and the additional tuning would not be justified by the negligible change in final OSR performance observed here.

## Deviation ledger

| Change | Tasks | Run status | Why it is a deviation |
|---|---|---|---|
| Global gradient clipping, `max_norm=5.0` | 2, 3, 4 | Run | Not included in the fixed optimizer recipes |
| L2-normalizing features before MMD | 3 | Run only after clipped DAN-DG failed its source gate | Changes the specified MMD inputs and objective |
| Prediction histograms, finite checks, gradient-norm logging, and dummy-win logging | 2, 3, 4 | Run | Diagnostic only; does not change optimization |
| Alignment/MMD warm-up | 2, 3 | Not run | Would change constant alignment pressure or the prescribed schedule |
| Lower/separate discriminator learning rate | 2 | Not run | Would break the shared optimizer settings |
| Lower PROSER learning rate or different beta/gamma | 4 | Not run | Would change manual-specified hyperparameters |
| Temporarily freezing the PROSER backbone | 4 | Not run | Would conflict with full-model fine-tuning |
| Stopping PROSER before 50 epochs | 4 | Not run | Would change the prescribed 50-epoch procedure |

Pure logging, additional plots, log-scaled axes, and class-frequency summaries do not constitute experimental deviations. The two optimization changes that were actually run must be disclosed and labeled supplemental.

## Final report checklist

- Present the original manual experiments first. Their failures are valid experimental outcomes.
- State that the teaching assistant permitted clipping, normalization, and hyperparameter adjustments for convergence problems if deviations were documented.
- Identify `max_norm=5.0`, its placement after backpropagation and before the optimizer step, and the methods to which it was applied.
- For normalized DAN-DG, state that each 512-dimensional feature was L2-normalized before pairwise-distance MMD computation, while lambda=1 and the three median-scaled kernels were retained.
- State that source validation alone triggered the Task 3 fallback; Task 2/3 Sketch labels and Task 4 CIFAR-100 were unavailable during supplemental model selection.
- Report both successes and failures: DAN/CDAN recovered, DANN did not, DAN-DG needed normalization in addition to clipping, and PROSER gained late known-class stability without a material OSR gain.
- Include selected epochs and avoid using final-epoch performance as the reported checkpoint performance.
- Retain the sharpness warning for normalized DAN-DG and the late dummy-dominance warning for clipped PROSER.
- Keep supplemental rows visually separate from the required manual tables and do not imply that they were part of the original prescribed protocol.
