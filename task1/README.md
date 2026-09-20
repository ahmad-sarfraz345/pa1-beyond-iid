# Task 1: Inductive Biases and Representations

## Protocol fixed before evaluation

- Dataset: STL-10 official train/test partitions. The training partition is stratified 80/20 with seed 6304. The test subset has 50 examples per class, sampled with seed 6304; their zero-based official indices are saved in `results/splits.json`.
- The same 224 × 224 RGB image is used before each model's normalization. ResNet-50 uses `IMAGENET1K_V2`; ViT-B/16 uses `IMAGENET1K_V1`; CLIP ViT-B/32 uses the OpenAI pretrained weights. Backbones remain frozen. CLIP zero-shot uses exactly `a photo of a {class}.` and the model's learned logit scale.
- The linear heads use AdamW, learning rate 0.001, weight decay 0.0001, at most 50 epochs, and five-epoch patience on validation accuracy. No backbone fine-tuning or prompt search is performed.
- Color intervention: grayscale and a fixed 45° HSV hue rotation. Hue rotation changes chromatic hue while preserving pixel positions and approximately preserving saturation and brightness. Hypothesis: accuracy and paired prediction consistency will fall under color removal or hue rotation if a classifier relies on chromatic cues. Measure accuracy change from clean and prediction consistency.
- Cue conflicts: AdaIN with strength 0.8 on the five class pairs in `configs/task1.json`, in both directions. Hypothesis: different models may differ in their content/shape versus style/texture choices. Measure shape, texture, other counts, shape bias among shape/texture choices, and coverage across all valid conflicts. AdaIN may transfer style without a recognizable class-specific texture, so every candidate is screened using the saved visual rule before model evaluation.
- Translation: reflection padding and shifted crops at 8, 16, and 32 pixels, four cardinal directions. Hypothesis: positional sensitivity will reduce accuracy or consistency as displacement grows. Metrics are mean accuracy and consistency over directions, with zero displacement as the clean reference.
- Patch structure: one seeded non-identity 4 × 4 pixel-space patch permutation per image. Hypothesis: disrupting global organization will reduce accuracy and consistency while retaining local pixels. Metrics are accuracy change and paired prediction consistency.
- Representation analysis: mean paired cosine similarity for grayscale, cue conflicts, 8/16/32-pixel translation, and patch shuffle. A t-SNE plot is fitted jointly to clean and transformed features for each backbone and intervention, with seed 6304, perplexity 30, PCA initialization, and automatic learning rate. The fixed visualization subset is the first 20 sorted test indices per class. Cue plots use 200 balanced accepted conflicts and their content counterparts. Compare neighborhood mixing within a plot, not coordinates across separately fitted plots.

These hypotheses are **design records**, not report prose. The student must write the PDF report and interpretation independently.

## Setup and commands

Use Python 3.10+ on a machine with an NVIDIA GPU for practical runtime. Install a compatible PyTorch/torchvision build first if your runtime does not provide one. Then, from the repository root:

```bash
python -m pip install -r requirements.txt
python -m task1.scripts.run_task1 prepare
```

`prepare` downloads STL-10 into ignored `data/` and writes `task1/results/splits.json`.

For AdaIN, clone the public implementation and download the `vgg_normalised.pth` and `decoder.pth` assets from its [v0.0.0 release](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0). Keep the repository and weights outside this Git repository (for example in `/tmp/pytorch-AdaIN`). Record the clone commit and the asset filenames in your experiment notes.

```bash
git clone https://github.com/naoto0804/pytorch-AdaIN.git /tmp/pytorch-AdaIN
mkdir -p /tmp/pytorch-AdaIN/models
curl -L https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth -o /tmp/pytorch-AdaIN/models/vgg_normalised.pth
curl -L https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth -o /tmp/pytorch-AdaIN/models/decoder.pth
python -m task1.scripts.run_task1 conflicts --adain-repo /tmp/pytorch-AdaIN --vgg-weights /tmp/pytorch-AdaIN/models/vgg_normalised.pth --decoder-weights /tmp/pytorch-AdaIN/models/decoder.pth
python -m task1.scripts.run_task1 train
```

**Visual review is required before evaluation.** Open all `task1/results/cue_review_*.jpg` sheets. For each row in `task1/results/cue_review.csv`, set `accept` to `yes` or `no` using the rejection rule in `cue_rule.txt`. Record a short `reason` for each rejection. The evaluator refuses pending decisions and requires at least 20 accepted candidates for each of the 10 pair/direction groups. It takes the first 20 accepted per group for exactly 200 balanced conflicts. Screening must be done without model predictions.

```bash
python -m task1.scripts.run_task1 evaluate
```

The outputs include `metrics.json`, `cue_review_counts.json`, `cue_comparison.csv`, per-model cue predictions, `translation.png`, and t-SNE plots. `metrics.json` contains clean top-1, macro-F1 and mean maximum confidence; transformed accuracy changes and paired consistency; direction-averaged translation curves; cue decision counts with shape bias and coverage; and cosine stability. Inspect `cue_comparison.csv` alongside the candidate sheets to choose informative agreements, disagreements, and failures for your own report.

## Kaggle or Colab GPU

1. Create a GPU notebook and enable internet access for the STL-10, backbone, CLIP, and AdaIN downloads. In Kaggle, select a GPU accelerator in notebook settings; in Colab, select a GPU runtime.
2. Clone your public repository into the notebook working directory, then run the setup and commands above in notebook shell cells (prefix each line with `!` or use a `%%bash` cell). In Kaggle, use `/kaggle/working/pytorch-AdaIN` instead of `/tmp/pytorch-AdaIN` if you want the AdaIN files in the notebook output; in Colab, `/content/pytorch-AdaIN` is convenient.
3. Download the contact sheets and `cue_review.csv`, complete the visual review, upload the edited CSV to the same `task1/results/` path, then run `evaluate`.
4. Save `task1/results/splits.json`, the edited review CSV, JSON metrics, prediction files, and plots. Download them or commit the small files to your repository before the notebook session ends. Do not commit raw STL-10 files, cue images, or checkpoints.

Run `python -c "import torch; print(torch.cuda.is_available())"` to confirm the notebook sees a GPU. Training only the heads is light; extracting features for all translated variants and generating AdaIN images makes the full experiment GPU appropriate.

## Limitations and audit trail

All random choices, hyperparameters, and subset IDs are fixed or saved. Library versions can be recorded with `python -m pip freeze > task1/results/environment.txt`. The AdaIN visual rejection rule is subjective; keep the review CSV and rejected counts. Zero-shot CLIP is a separate decision rule on the same CLIP image features. The model comparison cannot isolate architecture from pretraining data or objectives.
