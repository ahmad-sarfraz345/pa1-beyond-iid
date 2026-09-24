# ATML Programming Assignment 1

This repository contains the assignment manual and the Task 1–4 implementations.

See [Task 1 instructions](task1/README.md) for the experimental choices, commands, outputs, and cloud GPU workflow. Raw datasets, generated images, and large model checkpoints are excluded from Git. Small JSON/CSV result files and figures can be committed after an experiment.

See [Task 2 instructions](task2/README.md) for the PACS adaptation protocol and Kaggle GPU commands. Task 2's source-only checkpoint and source split must be retained for Task 3.

See [Task 3 instructions](task3/README.md) for the target-free PACS domain-generalization protocol and Kaggle GPU commands.

See [Task 4 instructions](task4/README.md) for the CIFAR open-set-recognition protocol and a complete fresh-session Kaggle workflow.

See [Supplemental stabilization plan](STABILIZATION_PLAN.md) for the preregistered clipping and normalized-feature fallbacks, and [Supplemental Kaggle guide](SUPPLEMENTAL_KAGGLE.md) for the complete rerun workflow.

The course permits LLM coding assistance but requires the student to understand all submitted code. **The PDF report must be written entirely by the student without generative AI.**

## External implementation attribution

- ResNet-50, ViT-B/16, and STL-10 loading use [torchvision](https://github.com/pytorch/vision).
- CLIP ViT-B/32 and its OpenAI weights use [OpenCLIP](https://github.com/mlfoundations/open_clip).
- Cue-conflict generation imports the unmodified [naoto0804 PyTorch AdaIN implementation](https://github.com/naoto0804/pytorch-AdaIN) and its released weights at runtime. The external files are kept outside this repository. The wrapper and evaluation code in `task1/` are original to this assignment workflow.
- Task 2 loads ImageNet ResNet-18 weights through [torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). PACS can be obtained through the dataset URL in [DomainBed's downloader](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py). The adaptation and evaluation code under `task2/` is original to this assignment workflow.
- Task 4 uses torchvision's ResNet-18 and CIFAR loaders. Its PROSER loss and placeholder score follow the authors' [official reference implementation](https://github.com/LAMDA-CL/CVPR21-Proser); the assignment-specific pipeline is implemented under `task4/`.
