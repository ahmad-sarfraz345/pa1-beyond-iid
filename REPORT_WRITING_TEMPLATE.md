# PA1 Report Writing Template — NeurIPS, 12-Page Main Paper

This is a fill-in structure, not report prose. Replace every bracketed instruction with your own writing. The **main paper must end within 12 NeurIPS-style pages**. Put supporting experiments and large diagnostic figures in the appendix. Keep original manual experiments visually separate from supplemental stabilization results.

[The detailed tables below are a result bank. Do not paste every table into the main paper. Follow each task's `MAIN TABLE`, `SOURCE FOR`, and `APPENDIX TABLE` labels.]

## Strict page budget

| Main-paper material | Target space | Hard guidance |
|---|---:|---|
| Title, abstract, and introduction | 1.0 page | Use the NeurIPS title block; do not create a separate title page. |
| Task 1 | 2.5 pages | Keep two compact tables and at most one main figure. |
| Task 2 | 2.2 pages | Keep the core comparison and one compact training figure. |
| Task 3 | 2.0 pages | Keep one results table; refer to appendix diagnostics. |
| Task 4 | 2.3 pages | Keep two compact tables and one figure. |
| Cross-task discussion | 0.8 page | Synthesize rather than repeat results. |
| Deviations and reproducibility | 0.5 page | Use a small ledger table. |
| Conclusion | 0.4 page | One short paragraph. |
| Safety margin | 0.3 page | Leave room for float movement and captions. |

[Aim for about 11.5 pages before references so NeurIPS float placement does not push main content past page 12. Start references immediately after the main-paper conclusion. Place appendices after references unless the course template says otherwise. Verify whether references count toward the 12-page limit in the assignment instructions; this scaffold conservatively limits the actual paper body to 12 pages.]

## What stays in the main paper

- [The central question and protocol for each task.]
- [The minimum tables needed to support the main claims.]
- [The most informative figure for each task, subject to the page budget.]
- [Original manual results and a compact summary of supplemental stabilization.]
- [Direct interpretation, failure cases, limitations, and disclosed deviations.]

## What moves to the appendix

- [Full per-class results, confusion matrices, all t-SNE panels, extra training curves, all cue-conflict examples, candidate review details, complete OSR metrics, and detailed run diagnostics.]
- [Do not move a result that is necessary to justify a main-paper claim. Summarize it in the main paper and give its full version in the appendix.]

## Formatting rules to apply throughout

- [Use the official NeurIPS LaTeX style supplied or permitted by the course. Do not change margins, font size, line spacing, or caption size to force material under the page limit.]
- [Use a two-column main paper. Keep the abstract as one compact paragraph beneath the title.]
- [Prefer `table`/`figure` for one-column items and `table*`/`figure*` only when a result is genuinely unreadable in one column.]
- [Use `\small` inside dense tables if needed, but keep all values and captions readable at 100% PDF zoom.]
- [Do not use a standalone contents page, list of figures, or separate title page.]
- [Use one consistent convention: percentages to two decimal places in prose/tables, or decimals to four places.]
- [Call differences between percentages **percentage points**, not percent improvements.]
- [Number every table and figure and refer to it before it appears.]
- [Every caption should state what was measured, which split/data were used, and the main visual encoding.]
- [Bold only the best valid value within a comparable group. Do not bold collapsed-model values merely because a diagnostic such as separability or sharpness is numerically small.]
- [Label every stabilization table or row **Supplemental**.]
- [Use appendix references such as “Appendix A.2 provides the full per-class results.”]
- [Do not repeat the same table in the main paper and appendix. Use a condensed main table and an expanded appendix table.]

# NeurIPS title block (top of page 1; no separate title page)

**Title:** [Write a title connecting inductive bias, domain shift, generalization, and open-set recognition.]

**Student:** [Name and identifier]

**Course/assignment:** [Course name and Programming Assignment 1]

**Date:** [Include only if the supplied NeurIPS/course template requests it.]

# Abstract

[Write 120–180 words after completing the report.]

[Sentence 1: state that the work studies model behavior beyond IID test conditions.]

[Sentence 2: name the four settings: controlled image interventions, unsupervised domain adaptation, domain generalization, and open-set recognition.]

[Sentence 3: summarize Task 1: clean accuracies were similar, while patch shuffle and cue conflicts exposed different behavior.]

[Sentence 4: summarize Tasks 2–3: mild MMD and SAM worked; excessive MMD caused collapse; stabilization helped selected methods.]

[Sentence 5: summarize Task 4: Near unknowns were harder; improved closed-set accuracy did not ensure improved OSR; PROSER was unstable.]

[Final sentence: state the overall lesson that validation protocol and optimization stability are essential when evaluating robustness.]

# 1. Introduction

## 1.1 Motivation

[Explain why IID test accuracy alone can hide reliance on color, texture, position, source-domain artifacts, or closed-set assumptions.]

[Introduce the four questions addressed by the assignment:]

1. [How do CNN, ViT, and CLIP representations respond to controlled image transformations?]
2. [Can unlabeled target data improve transfer to PACS Sketch?]
3. [Can a model generalize to Sketch without seeing any Sketch data?]
4. [Can a CIFAR-10 classifier detect semantically Near and Far unknown classes?]

## 1.2 Contributions and report map

[Give one sentence per task describing the experiment, without reporting all results here.]

[Mention that every split and random choice used seed 6304 and that model selection was separated from final target/unknown evaluation.]

[End with a one-sentence map of Sections 2–5.]

# 2. Task 1 — Inductive Biases and Representations

**Main-paper allocation: 2.5 pages.**

[Use three short method paragraphs, one condensed performance table, one cue-bias table, and one two-part figure. Combine the essential values from the clean, color, translation, and patch tables below into a single main-paper table with columns `Clean`, `Gray`, `Hue`, `Translate 32`, and `Patch shuffle`. Put consistencies, every displacement/direction value, cosine similarities, and the full t-SNE grid in Appendix A.]

[Main Figure 1 should be a compact two-part figure: (a) `task1/results/translation.png`, cropped only to remove unused whitespace, and (b) three or four cue-conflict examples. If this becomes unreadable, keep the translation plot in the main paper and move the cue examples to Appendix A.]

## 2.1 Objective and hypotheses

[State that Task 1 compares ResNet-50, ViT-B/16, and CLIP ViT-B/32 frozen representations.]

[Clarify that CLIP is evaluated with a trained linear head and with a fixed zero-shot prompt, giving four decision rules but only three image representations.]

[List the hypotheses: modest degradation under color changes, increasing sensitivity with translation magnitude, strong degradation after patch shuffle if global structure matters, and model-dependent shape/texture decisions under cue conflict.]

## 2.2 Dataset and fixed protocol

[Describe STL-10 and insert the split table below.]

**[APPENDIX TABLE A.1 unless the split fits in one sentence] — STL-10 split**

| Split | Count |
|---|---:|
| Train | 4,000 |
| Validation | 1,000 |
| Fixed test subset | 500, exactly 50 per class |

[Mention seed 6304, saved test identifiers, 224×224 RGB resizing before intervention, and model-specific normalization.]

[Describe the backbones and representations: ResNet global-pooled 2,048-D features, ViT final 768-D class token, and normalized CLIP visual embedding.]

[State that backbones were frozen.]

[Describe linear-head training: AdamW, learning rate 0.001, weight decay 0.0001, 50-epoch maximum, five-epoch patience, validation-accuracy selection.]

[Mention the selected epochs: ResNet 13, ViT 6, CLIP head 14.]

[State the fixed CLIP prompt exactly: `a photo of a {class}.`]

## 2.3 Clean classification results

[Introduce this as a check that the frozen representations are informative before studying interventions.]

**[SOURCE FOR CONDENSED MAIN TABLE 1] — Clean STL-10 performance**

| Decision rule | Accuracy | Macro-F1 | Mean max confidence |
|---|---:|---:|---:|
| ResNet-50 head | 97.80% | 97.80% | 0.977 |
| ViT-B/16 head | 97.40% | 97.39% | 0.968 |
| CLIP ViT-B/32 head | **98.40%** | **98.40%** | 0.850 |
| CLIP zero-shot | 98.00% | 97.99% | 0.955 |

[Discuss that all methods are close, CLIP head is numerically highest, and zero-shot CLIP is competitive without an STL-10-trained classifier.]

[Do not compare confidence as if it were calibrated across decision rules; the scores are produced differently.]

## 2.4 Color interventions

[Explain grayscale and the fixed 45-degree HSV hue rotation. Define accuracy change and paired prediction consistency.]

**[SOURCE FOR CONDENSED MAIN TABLE 1; move full consistency columns to Appendix A.2] — Color intervention results**

| Model | Gray acc. | Gray delta | Gray consistency | Hue acc. | Hue delta | Hue consistency |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 95.60% | −2.20 pp | 96.00% | 95.20% | −2.60 pp | 96.00% |
| ViT-B/16 | 95.60% | −1.80 pp | 95.60% | 96.40% | −1.00 pp | 97.00% |
| CLIP head | 97.00% | −1.40 pp | 97.40% | 96.40% | −2.00 pp | 97.20% |
| CLIP zero-shot | 97.00% | −1.00 pp | 97.20% | 96.40% | −1.60 pp | 97.40% |

[Discuss that color affects every model but only modestly; ResNet has the largest hue drop and CLIP zero-shot the smallest grayscale drop.]

[Conclude that color contributes but is not the dominant STL-10 signal. Avoid claiming perfect invariance.]

## 2.5 Translation sensitivity

[Explain reflection padding, four directions, displacements of 8/16/32 pixels, and averaging over directions.]

**[INSERT AS MAIN FIGURE 1(a): `task1/results/translation.png`]**

**Caption instructions:** [State that the left panel shows mean accuracy and the right panel paired prediction consistency over four translation directions; zero displacement is the clean reference.]

[Discuss: all accuracies remain above 96%; degradation is small at 32 px; ResNet has high consistency; CLIP head has the highest 32-px accuracy; ViT's small 8-px improvement is sampling variation.]

[Give the 32-px summary in prose: ResNet 96.75%/98.60% consistency, ViT 96.45%/97.55%, CLIP head 97.80%/98.65%, CLIP zero-shot 96.65%/98.25%.]

## 2.6 Patch-shuffle sensitivity

[Explain the seeded nonidentity permutation of a 4×4 grid of 56×56 patches. State that it preserves local pixels while disrupting global arrangement.]

**[SOURCE FOR CONDENSED MAIN TABLE 1; move full delta/consistency table to Appendix A.3] — Patch-shuffle results**

| Model | Patch accuracy | Change from clean | Consistency |
|---|---:|---:|---:|
| ResNet-50 | 91.60% | −6.20 pp | 92.80% |
| ViT-B/16 | 90.00% | −7.40 pp | 91.20% |
| CLIP head | 83.40% | −15.00 pp | 84.20% |
| CLIP zero-shot | 83.40% | −14.60 pp | 83.80% |

[Discuss that patch shuffle is much more damaging than color or translation and therefore global organization matters.]

[Discuss that CLIP decision rules are most affected, while ResNet retains the highest patch accuracy.]

[Do not equate patch robustness with cue-conflict shape bias; they measure different behavior.]

## 2.7 Cue-conflict construction and review

[Describe AdaIN at strength 0.8 and the five pairs: cat/dog, deer/horse, car/truck, airplane/bird, airplane/ship, evaluated in both directions.]

[State that 40 candidates per direction produced 400 candidates.]

[Explain the model-blind rule: reject if the content outline is unrecognizable, artifacts are severe, or style texture is not recognizable.]

[Report 348 accepted and 52 rejected: 30 severe artifacts, 17 unrecognizable outlines, four both, one severe artifact plus distorted outline.]

[State that the first 20 accepted images from every one of the ten groups formed a balanced 200-image evaluation set.]

**[INSERT AS MAIN FIGURE 1(b), if legible: a compact grid of 3–4 reviewed cue-conflict examples. Otherwise place it in Appendix A.4.]**

**Suggested examples:**

- [`p0_d0_00`: all four chose shape.]
- [`p0_d1_11` or `p1_d0_03`: ResNet chose texture while the transformer-based rules chose shape.]
- [`p1_d0_14`, `p2_d1_19`, or `p3_d0_08`: all four chose texture.]
- [`p0_d1_02` or `p3_d0_21`: all four chose another class.]

**Caption instructions:** [For each image, name the content class, style class, and predictions. State that screening occurred before predictions were viewed.]

## 2.8 Cue-conflict results

[Define shape, texture, and other predictions. Define shape bias as shape/(shape+texture) and coverage as (shape+texture)/200.]

**[MAIN TABLE 2] — Shape/texture decisions**

| Model | Shape | Texture | Other | Shape bias | Coverage |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 144 | 39 | 17 | 78.69% | 91.50% |
| ViT-B/16 | 171 | 16 | 13 | 91.44% | 93.50% |
| CLIP head | 172 | 17 | 11 | 91.01% | **94.50%** |
| CLIP zero-shot | 168 | 14 | 18 | **92.31%** | 91.00% |

[Discuss that all are shape-oriented overall, but ViT and CLIP are more shape-oriented than ResNet.]

[Mention 120/200 unanimous shape decisions, 24 cases where only ResNet chose texture and the other three chose shape, and only three unanimous texture decisions.]

[Discuss coverage with shape bias; do not quote shape bias alone.]

[Discuss the ResNet airplane/bird asymmetry: airplane content with bird style gives 0 shape/18 texture/2 other, while bird content with airplane style gives 17 shape/0 texture/3 other. State that cue bias is pair- and direction-dependent.]

## 2.9 Representation stability and t-SNE

[Define paired cosine similarity and state that CLIP head and zero-shot share identical representation values.]

**[APPENDIX TABLE A.5; summarize only the ordering in the main text] — Mean paired cosine similarity**

| Intervention | ResNet-50 | ViT-B/16 | CLIP ViT-B/32 |
|---|---:|---:|---:|
| Grayscale | 0.7884 | 0.7393 | **0.8983** |
| Patch shuffle | 0.7180 | 0.6572 | **0.7281** |
| Cue conflict | 0.4872 | 0.5213 | **0.7628** |
| Translation 8 | 0.9685 | 0.9471 | **0.9810** |
| Translation 16 | 0.9525 | **0.9743** | 0.9680 |
| Translation 32 | 0.9327 | 0.9418 | **0.9699** |

[Discuss that translation is most stable, cue conflicts produce the largest changes, and CLIP is generally most stable.]

[Discuss the representation/decision mismatch: CLIP has the highest patch cosine but the largest patch accuracy loss.]

**[INSERT AS APPENDIX FIGURE A.1: a 3×4 t-SNE grid using all 12 files in `task1/results/`; rows = ResNet, ViT, CLIP and columns = grayscale, patch, cue, translate-32-right.]**

**Caption instructions:** [Circles are clean, crosses transformed, color is true class; clean and transformed features were fitted jointly within each panel using 20 examples per class, seed 6304, perplexity 30, PCA initialization.]

[Discuss within-panel overlap: grayscale and translation mostly preserve neighborhoods; patch reorganizes neighborhoods; cue conflicts cause systematic shifts; CLIP cue representations remain more similar quantitatively.]

[Do not compare t-SNE axes or absolute positions across panels because every panel was fitted separately.]

## 2.10 Task 1 conclusion and limitations

[Conclude: clean performance is saturated; color dependence is modest; translation robustness is high; global arrangement matters; ViT/CLIP show stronger shape preference; CLIP representations are generally more stable.]

[Limitations: one seed/subset, architecture confounded with pretraining, fixed zero-shot prompt, subjective cue review, AdaIN limitations, t-SNE qualitative distortion.]

# 3. Task 2 — Unsupervised Domain Adaptation to PACS Sketch

**Main-paper allocation: 2.2 pages.**

[Use one protocol paragraph, one method paragraph, one combined results table, and one compact training-curve figure. Combine the original and supplemental Task 2 tables below into Main Table 3: retain all original methods, then add three visibly separated rows under `Supplemental (clip5)`. Keep source macro-F1, Sketch accuracy, Sketch macro-F1, and domain probe where available. Move dataset counts, per-class results, confusion matrices, full gradient logs, and expanded curves to Appendix B.]

## 3.1 Objective and data protocol

[State the UDA setting: labeled Photo/Art Painting/Cartoon and unlabeled Sketch during training; Sketch labels only at final evaluation.]

**[APPENDIX TABLE B.1; give counts compactly in main prose] — PACS source split**

| Domain | Train | Validation |
|---|---:|---:|
| Photo | 1,336 | 334 |
| Art Painting | 1,638 | 410 |
| Cartoon | 1,875 | 469 |
| Sketch target | Unlabeled during training | 3,929 final examples |

[Mention seven classes, stratified 80/20 source splits, seed 6304, source-validation model selection, and no target-label tuning.]

## 3.2 Shared model and training settings

[Describe ImageNet-pretrained ResNet-18, new seven-class head, trainable network, frozen BatchNorm running statistics, AdamW 1e-4/1e-4, 30 epochs, patience 5.]

[Describe 256 resize, random 224 crop/flip for training, center crop for evaluation, and batches of 8 per source plus 24 unlabeled Sketch images for adaptation.]

## 3.3 Methods

[Explain source-only ERM.]

[Explain DAN: classification plus MMD on 512-D pre-head features, three median-scaled Gaussian kernels, lambda 0.1/1/10.]

[Explain DANN: binary discriminator, gradient reversal schedule, source-versus-target confusion.]

[Explain CDAN: discriminator conditions on the outer product of features and softmax probabilities.]

[State the preregistered hypothesis that moderate alignment may help while excessive alignment may remove class structure.]

## 3.4 Metrics

[Define source and Sketch accuracy/macro-F1, delta from source-only, per-class accuracy, dominant confusion, and balanced source-target logistic-regression domain probe.]

[State that lower probe accuracy is meaningful only when source/target classification remains healthy.]

## 3.5 Original results

**[SOURCE FOR MAIN TABLE 3] — Original Task 2 results**

| Method | Best epoch | Source macro-F1 | Sketch acc. | Sketch macro-F1 | Domain probe |
|---|---:|---:|---:|---:|---:|
| Source-only | 7 | 0.9334 | 0.5610 | 0.5745 | 0.9959 |
| DAN 0.1 | 8 | **0.9469** | **0.6872** | **0.6349** | 0.9657 |
| DAN 1 | 1 | 0.0877 | 0.0204 | 0.0057 | 0.6484 |
| DAN 10 | 1 | 0.0507 | 0.0407 | 0.0112 | 0.8022 |
| DANN | 2 | 0.1276 | 0.0402 | 0.0217 | 0.9986 |
| CDAN | 2 | 0.9051 | 0.3489 | 0.4201 | 0.9945 |

**[INSERT AS MAIN FIGURE 2(a): the most informative panels from `task2/results/training_curves.png`; place the complete figure in Appendix B.2.]**

**Caption instructions:** [Name the source-validation selection metric and note that exploding DANN/CDAN losses compress healthy curves on the linear axes.]

[Discuss source-only transfer and its class imbalance: strong elephant/house, weak dog/horse/person.]

[Discuss DAN 0.1 as the successful manual run: +12.62 pp over source-only, major person/horse/guitar improvements, source performance retained.]

[Discuss DAN 1 collapse to house and DAN 10 collapse to person.]

[Discuss DANN: exploding losses, poor source/target classification, probe remains 0.9986.]

[Discuss CDAN: valid selected epoch 2 but negative transfer and later instability.]

[State that low probe accuracy for collapsed DAN is not useful domain invariance.]

## 3.6 Supplemental stabilization

[Introduce teaching-assistant permission and state the deviation: global gradient clipping at max norm 5 after backward and before optimizer step.]

**[ADD TO MAIN TABLE 3 under a clear 'Supplemental: clip5' divider] — Supplemental Task 2 results**

| Variant | Source macro-F1 | Sketch acc. | Sketch macro-F1 | Accuracy change from original |
|---|---:|---:|---:|---:|
| DAN 1 + clip5 | 0.9331 | 0.6406 | 0.5982 | +62.03 pp |
| DANN + clip5 | 0.5929 | 0.0593 | 0.0363 | +1.91 pp |
| CDAN + clip5 | 0.9313 | 0.6826 | 0.6277 | +33.37 pp |

**[INSERT AS MAIN FIGURE 2(b) only if readable beside Figure 2(a): selected panels from `task2/supplemental/results/training_curves.png`; place the complete figure in Appendix B.3.]**

**Caption instructions:** [State that losses use a symmetric log scale and that these are supplemental clipped runs.]

[Discuss: DAN and CDAN recover; CDAN nearly matches DAN 0.1; DANN remains failed; DANN pre-clip gradient norm reaches 362,803.]

[State that clipping is useful for some methods but does not repair every adversarial optimization failure.]

## 3.7 Task 2 conclusion and limitations

[Conclude that moderate alignment helps, excessive alignment collapses, and optimization stability is central.]

[Limitations: one seed/backbone/target direction, linear domain probe, source-validation selection imperfect, supplemental methods post-failure and must remain separate.]

# 4. Task 3 — Domain Generalization to Unseen Sketch

**Main-paper allocation: 2.0 pages.**

[Use one protocol/method paragraph, Main Table 4 containing the original Task 3 comparison plus the normalized supplemental recovery, and a short Task 2-versus-Task 3 comparison in prose. Move full training curves, sharpness details, separability breakdowns, and the clip-only failed fallback to Appendix C. Keep the sharpness 46.9051 caveat in the main text even though the detailed diagnostic is in the appendix.]

## 4.1 Objective and protocol

[Contrast with Task 2: only Photo/Art/Cartoon are available; no Sketch images or labels during training, diagnostics, selection, or fallback decisions.]

[State that the exact Task 2 split and source-only ERM checkpoint were reused and verified by SHA-256.]

[Refer to the Task 2 protocol or Appendix B.1 rather than repeating the source counts.]

## 4.2 Methods

[Explain ERM reuse.]

[Explain DAN-DG: average MMD across Photo–Art, Photo–Cartoon, and Art–Cartoon feature pairs; lambda 0.1/1/10.]

[Explain SAM: nonadaptive two-pass update at radius 0.05, seeking lower nearby loss.]

## 4.3 Diagnostics

[Define mean/worst source accuracy and macro-F1, Sketch metrics, balanced three-way source-domain separability with chance 1/3, and the fixed-radius sharpness proxy.]

[Warn that low separability or low sharpness is meaningless if the classifier has collapsed.]

## 4.4 Original results

**[SOURCE FOR MAIN TABLE 4] — Original Task 3 results**

| Method | Source F1 | Worst source F1 | Sketch acc. | Sketch F1 | Separability | Sharpness increase |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.9334 | 0.8995 | 0.5610 | 0.5745 | 0.8638 | 0.3340 |
| DAN-DG 0.1 | 0.9529 | 0.9249 | **0.7109** | 0.7199 | 0.7907 | 0.1180 |
| DAN-DG 1 | 0.0507 | 0.0421 | 0.0407 | 0.0112 | 0.5847 | 0.0103 |
| DAN-DG 10 | 0.0507 | 0.0421 | 0.0407 | 0.0112 | 0.5415 | 0.0112 |
| SAM | **0.9567** | **0.9394** | 0.7045 | **0.7466** | 0.8638 | **0.1117** |

**[INSERT AS APPENDIX FIGURE C.1: `task3/results/training_curves.png`; summarize the collapse and healthy trajectories in the main text.]**

**Caption instructions:** [Show source-validation F1 and objective trajectories; identify the lambda=1/10 one-class failures.]

[Discuss DAN-DG 0.1: +14.99 pp over ERM, lower healthy source-domain separability, strong person/horse/dog improvements.]

[Discuss SAM: slightly lower accuracy than DAN-DG 0.1 but highest macro-F1 and best worst-domain F1; meaningful sharpness reduction relative to ERM.]

[Discuss DAN-DG 1/10: every Sketch image becomes person, class loss near log(7), low sharpness reflects a constant unusable solution.]

## 4.5 Task 2 versus Task 3

**[APPENDIX TABLE C.2; give the four-number comparison in main prose] — Target-aware versus target-free transfer**

| Method | Target access | Sketch acc. | Sketch F1 |
|---|---|---:|---:|
| ERM | None | 0.5610 | 0.5745 |
| Task 2 DAN 0.1 | Unlabeled Sketch | 0.6872 | 0.6349 |
| Task 3 DAN-DG 0.1 | No Sketch | **0.7109** | 0.7199 |
| Task 3 SAM | No Sketch | 0.7045 | **0.7466** |

[Discuss that target-free methods are competitive in this single run; do not generalize that DG is always better than UDA.]

[Mention that the manual lambda=1 Task 2 and Task 3 models both collapsed, showing that target access cannot rescue an excessively strong objective.]

## 4.6 Supplemental stabilization

[State the staged decision: clip5 first; source-only gate failed; then L2-normalize each 512-D feature before MMD while retaining lambda=1 and clipping.]

**[ADD THE NORMALIZED RECOVERY ROW TO MAIN TABLE 4; put all three rows in Appendix C.3] — Supplemental DAN-DG lambda=1**

| Variant | Source F1 | Sketch acc. | Sketch F1 | Separability | Sharpness |
|---|---:|---:|---:|---:|---:|
| Original | 0.0507 | 0.0407 | 0.0112 | 0.5847 | 0.0103 |
| Clip5 | 0.0507 | 0.0407 | 0.0112 | 0.5914 | 0.0088 |
| Normalized MMD + clip5 | 0.9284 | 0.6261 | 0.6321 | 0.6611 | 46.9051 |

**[INSERT AS APPENDIX FIGURE C.2: `task3/supplemental/results/training_curves.png`.]**

**Caption instructions:** [Identify the normalized fallback and state that selection remained source-only.]

[Discuss that normalization recovers discrimination and exceeds ERM by 6.52 pp, but remains below DAN-DG 0.1/SAM.]

[Discuss the serious caveat: sharpness 46.91 and deterioration after selected epoch 3. Do not describe the recovered model as flat.]

[State that feature normalization changes the MMD objective and is a disclosed supplemental deviation.]

## 4.7 Task 3 conclusion and limitations

[Conclude: mild alignment and SAM work; strong alignment collapses; low diagnostic values can be misleading; target-free methods can be competitive; normalized fallback trades recovery for high sharpness.]

[Limitations: one unseen domain/seed/backbone, linear separability, one-batch one-direction sharpness proxy, source selection imperfect, post-failure supplemental changes.]

# 5. Task 4 — Open-Set Recognition

**Main-paper allocation: 2.3 pages.**

[Use one protocol/method paragraph, Main Table 5 for Vanilla score comparison, Main Table 6 for model comparison, and one two-part figure. Main Figure 3 should contain the score distributions plus a small row of representative MLS failures. Move the full failure gallery, complete threshold metrics, training trajectories, dummy-win diagnostics, and expanded supplemental PROSER table to Appendix D.]

## 5.1 Objective and data protocol

[Explain closed-set accuracy versus detecting inputs outside the known ten classes.]

[State that training/selection used only CIFAR-10 and CIFAR-100 was first opened after checkpoints were frozen.]

**[APPENDIX TABLE D.1; summarize counts and class lists in main prose] — CIFAR data protocol**

| Data | Use | Count |
|---|---|---:|
| CIFAR-10 train | Optimization | 45,000, 4,500/class |
| CIFAR-10 validation | Checkpoint and threshold selection | 5,000, 500/class |
| CIFAR-10 test | Final known evaluation | 10,000, 1,000/class |
| CIFAR-100 Near test subset | Unknown evaluation only | 800 |
| CIFAR-100 Far test subset | Unknown evaluation only | 800 |

[List Near classes: bus, pickup truck, motorcycle, tractor, wolf, fox, leopard, camel.]

[List Far classes: bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe.]

## 5.2 Models and training

[Describe the CIFAR ResNet-18 stem: 3×3 stride one, no max pool.]

[Describe Vanilla: random initialization, crop/flip, SGD, lr 0.1, momentum 0.9, weight decay 5e-4, cosine decay, 100 epochs.]

[Describe GCSC: same model plus RandAugment(2,9), separately initialized.]

[Describe PROSER: initialized from selected Vanilla, five dummy classifiers, full-network fine-tuning for 50 epochs at lr 0.001, beta 1, gamma 0.1, manifold mixup alpha 2.]

## 5.3 Scores, threshold, and metrics

[Define MSP unknownness = 1 − maximum softmax probability.]

[Define MLS unknownness = negative maximum logit.]

[Define Energy as negative log-sum-exp.]

[Define Mahalanobis as minimum diagonal shared-covariance distance to a CIFAR-10 class mean fitted on training features.]

[Define PROSER placeholder score as strongest dummy response relative to strongest known response after temperature scaling.]

[State that every threshold is the 95th percentile of the corresponding CIFAR-10 validation unknownness score; unknown data never selects a threshold.]

[Define AUROC, known acceptance, unknown rejection, and accepted-unknown/FPR quantity.]

## 5.4 Known-class training results

**[APPENDIX TABLE D.2; state the three test accuracies in main prose] — CIFAR-10 performance**

| Model | Selected epoch | Best val. acc. | Final val. acc. | Test acc. |
|---|---:|---:|---:|---:|
| Vanilla | 100 | 94.82% | 94.82% | 94.67% |
| GCSC | 99 | 95.32% | 95.30% | **95.35%** |
| PROSER | 1 | 94.38% | 82.62% | 94.07% |

[Discuss healthy Vanilla/GCSC training, GCSC's +0.68 pp test improvement, PROSER deterioration, and best-checkpoint protection.]

[Do not call the selected PROSER checkpoint collapsed; call the 50-epoch trajectory unstable.]

## 5.5 Vanilla score comparison

**[MAIN TABLE 5] — Vanilla OSR scores**

| Score | Near AUROC | Far AUROC | All AUROC | Known accept. | Near reject. | Far reject. |
|---|---:|---:|---:|---:|---:|---:|
| MSP | **0.8265** | 0.9061 | **0.8663** | 0.9500 | 0.2900 | 0.4638 |
| MLS | 0.8117 | 0.9075 | 0.8596 | 0.9466 | 0.3563 | 0.5775 |
| Energy | 0.8117 | 0.9081 | 0.8599 | 0.9444 | **0.3663** | **0.5875** |
| Mahalanobis | 0.8007 | **0.9201** | 0.8604 | 0.9523 | 0.2675 | 0.5275 |

**[INSERT AS MAIN FIGURE 3(a): `task4/results/vanilla_score_distributions.png`.]**

**Caption instructions:** [Show CIFAR-10 versus Near/Far unknownness distributions for MSP, MLS, and Mahalanobis; larger score means more unknown.]

[Discuss: Near is harder; MSP best Near/overall AUROC; Mahalanobis best Far AUROC; Energy best thresholded rejection; score rankings differ by metric.]

## 5.6 Model comparison

**[MAIN TABLE 6] — Trained-model OSR comparison**

| Model/score | Known acc. | Near AUROC | Far AUROC | All AUROC | Near reject. | Far reject. |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla MLS | 0.9467 | 0.8117 | **0.9075** | **0.8596** | **0.3563** | **0.5775** |
| GCSC MLS | **0.9535** | 0.8076 | 0.9078 | 0.8577 | 0.3300 | 0.5775 |
| PROSER MLS | 0.9407 | **0.8117** | 0.8708 | 0.8413 | 0.3125 | 0.4550 |
| PROSER placeholder | 0.9407 | 0.8067 | 0.8927 | 0.8497 | 0.3138 | 0.5213 |

[Discuss: GCSC improves known accuracy but not OSR; Vanilla MLS is best overall trained-model row; placeholder improves PROSER's Far behavior over its MLS score but remains below Vanilla.]

[State explicitly that stronger closed-set classification does not guarantee stronger open-set detection.]

## 5.7 Failure analysis

**[INSERT 4–6 representative examples AS MAIN FIGURE 3(b); place the complete `task4/results/vanilla_mls_failures.png` in Appendix D.3.]**

**Caption instructions:** [State that these are high-confidence Vanilla MLS failures beyond the validation threshold; top/first group is Near and second group Far, with predicted CIFAR-10 labels.]

[Discuss plausible Near failures: three buses predicted as trucks; shared wheels, large vehicle shape, and road context make these semantically plausible.]

[Discuss surprising Far failures: keyboard→bird, sunflower→bird, mushroom→dog; categories are semantically unrelated, so local texture/silhouette/background may have produced feature similarity.]

[Do not state a definitive causal explanation without attribution evidence.]

## 5.8 Supplemental PROSER clipping

[State the deviation: global clipping at max norm 5; all other hyperparameters and 50 epochs retained; checkpoint selected on CIFAR-10 before opening CIFAR-100.]

**[APPENDIX TABLE D.4; summarize the negligible selected-checkpoint/OSR changes in main prose] — Original versus clipped PROSER**

| Quantity | Original | Clip5 | Change |
|---|---:|---:|---:|
| Best validation acc. | 0.9438 | 0.9446 | +0.0008 |
| Epoch-50 validation acc. | 0.8262 | 0.9040 | +0.0778 |
| Test acc. | 0.9407 | 0.9408 | +0.0001 |
| MLS All AUROC | 0.8413 | 0.8414 | +0.0001 |
| Placeholder All AUROC | 0.8497 | 0.8501 | +0.0003 |
| Placeholder All rejection | 0.4175 | 0.4181 | +0.0006 |

[Discuss: clipping improves late known-class trajectory but has negligible selected-checkpoint/OSR effect.]

[Mention dummy-win rate 8.02% at selected epoch 1, >81% by epoch 10, approximately 100% from epoch 22; late placeholder dominance remains.]

[Conclude that clipping partially stabilizes training but does not solve PROSER's OSR weakness.]

## 5.9 Task 4 conclusion and limitations

[Conclude: Near unknowns harder; no score dominates; known accuracy and OSR differ; Vanilla remains strongest overall; PROSER placeholder helps only relative to its own MLS; clipping has negligible final OSR effect.]

[Limitations: one seed/architecture, fixed class lists, CIFAR datasets share visual format, one threshold policy, diagonal covariance, qualitative failure explanations, PROSER instability.]

# 6. Cross-task discussion

**Main-paper allocation: 0.8 page.**

## 6.1 Main findings across tasks

[Connect Task 1's controlled shifts to Tasks 2–3's domain shifts: high clean accuracy does not guarantee stable behavior after structure/style/domain changes.]

[Connect Task 2 and Task 3: mild alignment helps, excessive alignment removes class information, and target access alone does not prevent collapse.]

[Connect Task 3 SAM and Task 4: optimization geometry/stability affects generalization, but a single diagnostic such as sharpness or gradient norm is insufficient alone.]

[Connect Task 4: stronger closed-set accuracy does not guarantee better unknown detection, paralleling how lower domain separability does not guarantee better target classification.]

## 6.2 General methodological lesson

[Discuss that robustness claims require multiple metrics: accuracy plus consistency, shape bias plus coverage, invariance plus class discrimination, and AUROC plus operating-point rejection.]

[Discuss the role of validation-only selection and frozen final evaluation in preventing target leakage.]

# 7. Deviations and reproducibility

**Main-paper allocation: 0.5 page.**

## 7.1 Fixed reproducibility choices

[List seed 6304, saved STL/PACS/CIFAR splits, fixed test subsets, checkpoint hashes, fixed prompts, model-blind cue review, and validation-only thresholds.]

## 7.2 Supplemental deviations

**[MAIN TABLE 7; keep compact] — Deviation ledger**

| Change | Tasks | Status | Disclosure |
|---|---|---|---|
| Global gradient clipping, max norm 5 | 2, 3, 4 | Run | Not in manual optimizer recipe; supplemental only |
| L2 feature normalization before MMD | 3 | Run after clip-only source failure | Changes MMD inputs/objective; supplemental only |
| Gradient/prediction/dummy diagnostics | 2, 3, 4 | Run | Logging only; no optimization change |

[State that the teaching assistant permitted stabilization changes if documented with reasoning.]

[State that original results were retained and never silently replaced.]

[State that Task 2/3 target labels and Task 4 unknown data were unavailable during supplemental selection.]

# 8. Conclusion

**Main-paper allocation: 0.4 page. Keep this to one compact paragraph.**

[Paragraph 1: summarize Task 1—similar clean performance but different transformation and cue behavior.]

[Paragraph 2: summarize Tasks 2–3—moderate alignment and SAM work; excessive alignment collapses; clipping/normalization help selectively.]

[Paragraph 3: summarize Task 4—Near unknowns remain difficult; score/model choice trades known accuracy against rejection; PROSER underperforms Vanilla.]

[Final sentence: state that evaluation beyond IID accuracy requires controlled interventions, leakage-safe protocols, and diagnostics that preserve class utility.]

**[THE 12-PAGE MAIN PAPER ENDS HERE. If this line falls on page 13, shorten prose/captions or move supporting material to the appendix. Do not alter the NeurIPS style.]**

# References

[Add the citations required by the manual for STL-10/PACS/CIFAR, ResNet, ViT, CLIP, AdaIN, DAN/MMD, DANN, CDAN, SAM, GCSC, and PROSER.]

[Use one consistent bibliography style. Verify author names, titles, venues, and years against the manual or original papers.]

# Appendix

[Start the appendix after the references. Reset appendix section labels to A, B, C, and D using the NeurIPS template's appendix command. Each appendix item should be referenced at least once from the main paper. Appendix pages may contain fuller tables and figures, but still need captions and readable text.]

## Appendix A — Task 1 supporting results

### A.1 Data split and complete protocol

[Insert the STL-10 split table and implementation details omitted from the main paper: preprocessing order, feature dimensions, head optimization, selected epochs, and fixed CLIP prompt.]

### A.2 Complete color and translation results

[Insert full grayscale/hue accuracy, delta, and consistency results. Add translation values for all four directions and all three nonzero displacements.]

### A.3 Complete patch-shuffle results

[Insert patch accuracy, delta, and consistency. Include the exact seeded nonidentity permutation or its saved identifier if available.]

### A.4 Cue-conflict review and examples

[Insert accepted/rejected totals and counts by rejection reason, the full pair/direction breakdown, and a larger accepted-example grid. State again that review was model-blind.]

### A.5 Representation diagnostics

[Insert the paired cosine-similarity table and the complete 3×4 t-SNE grid. Repeat the warning that axes cannot be compared across separately fitted panels.]

## Appendix B — Task 2 supporting results

### B.1 PACS counts and training protocol

[Insert domain counts, complete optimizer/augmentation settings, batch composition, early stopping, and model-selection rule.]

### B.2 Original-run diagnostics

[Insert the complete original training curves, per-class Sketch results, dominant confusions, and domain-probe details. Highlight the DAN 1/10 and DANN failures without treating low probe accuracy as success.]

### B.3 Supplemental clipped runs

[Insert the complete supplemental training curves, gradient-norm traces or summary values, and the full clipped-result table. State that DANN still failed after clipping.]

## Appendix C — Task 3 supporting results

### C.1 Original training curves and diagnostics

[Insert the complete training curves, per-source and worst-source results, class breakdowns, separability details, and sharpness protocol.]

### C.2 Task 2 versus Task 3

[Insert the compact target-aware versus target-free table. Explain that it is a single-run comparison and does not establish that domain generalization is generally superior to adaptation.]

### C.3 Supplemental fallback sequence

[Show Original, clip5, and normalized-MMD-plus-clip5 rows. Insert the supplemental curves and document the source-only fallback gate. Emphasize the recovered model's sharpness value of 46.9051 and deterioration after epoch 3.]

## Appendix D — Task 4 supporting results

### D.1 Data protocol and class lists

[Insert CIFAR split counts and the complete Near/Far CIFAR-100 class lists. State that CIFAR-100 was opened only after checkpoint freezing.]

### D.2 Training details and known-class results

[Insert Vanilla, GCSC, and PROSER training settings and known-class validation/test results. Include the distinction between PROSER's selected epoch and deteriorated final epoch.]

### D.3 Complete score and failure analysis

[Insert the complete `task4/results/osr_metrics.csv`, full score-distribution plots, and full `task4/results/vanilla_mls_failures.png`. Include thresholds, known acceptance, Near/Far rejection, and AUROC.]

### D.4 Supplemental PROSER clipping

[Insert the full original-versus-clip5 table, training trajectory, gradient summary, and dummy-win progression. State that clipping improved late validation behavior but barely changed selected-checkpoint OSR.]

## Appendix E — Reproducibility inventory (optional)

[List artifact filenames, checkpoint hashes, commands/configuration files, seed 6304, software versions, and hardware. Use this appendix only for useful reproducibility information rather than raw logs.]

## Final pre-export checklist

- [The official NeurIPS/course style file is active and has not been modified.]
- [The main-paper conclusion ends on or before page 12.]
- [References and appendices begin only after the main paper.]
- [Every appendix section containing evidence is cited from the main paper.]
- [Every bracketed instruction has been replaced or deleted.]
- [Every table/figure is numbered, captioned, and cited in the text.]
- [All supplemental rows are labeled.]
- [Accuracy changes use percentage points.]
- [Cue shape bias is always paired with coverage.]
- [Collapsed models are never praised for low separability/sharpness.]
- [Normalized DAN-DG's sharpness caveat is present.]
- [PROSER selected checkpoint is distinguished from its unstable later trajectory.]
- [Target/unknown data isolation is stated for Tasks 2–4.]
- [No claim generalizes beyond the single seed/domain setup.]
- [References and external implementation attribution are complete.]
- [The exported PDF has no cropped tables or unreadable figures.]
