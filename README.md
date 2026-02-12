# Semantic Correspondence with Visual Foundation Models

A framework for **semantic keypoint correspondence** across object instances, built as a project for the Advanced Machine Learning (AML) course.

Given two images of the same object category, the system predicts pixel-level correspondences between semantically equivalent keypoints (e.g., the left eye of one cat ↔ left eye of another cat), even under significant appearance and pose variations.

---

## Overview

The pipeline works in the following stages:

1. **Feature Extraction** — A pre-trained vision backbone extracts dense feature maps from both source and target images.
2. **Correspondence Matching** — Cosine similarity is computed between source keypoint features and all target feature positions; the best match is found via argmax.
3. **Evaluation** — Predicted correspondences are evaluated using the **PCK** (Percentage of Correct Keypoints) metric at multiple thresholds (α = 0.05, 0.10, 0.15, 0.20).
4. **Fine-Tuning** — The last *N* transformer blocks of the backbone are unfrozen and trained with a Gaussian-smoothed cross-entropy loss for improved keypoint localization.
5. **Window Soft-Argmax** — A refined correspondence decoding strategy that replaces the standard hard argmax with a local softmax around the peak, achieving sub-pixel precision.
6. **Robustness Testing** — Models are additionally evaluated on **PF-Pascal**, **PF-Willow**, and **AP-10k** to measure cross-dataset generalization.

## Supported Backbones

| Backbone | Architecture | Resolution | Key |
|---|---|---|---|
| **DINOv2** | ViT-S/14, ViT-B/14 | 518 × 518 | `dinov2_vits14`, `dinov2_vitb14` |
| **DINOv3** | ViT-S/16, ViT-B/16 | 592 × 592 | `dinov3_vits16`, `dinov3_vitb16` |
| **SAM** | ViT-B | 592 × 592 | `sam_vitb` |

DINOv3 and SAM are included as **git submodules** under `external/`.

## Supported Datasets

| Dataset | Task | Source |
|---|---|---|
| **SPair-71k** | Keypoint correspondence| [link](http://cvlab.postech.ac.kr/research/SPair-71k/) |
| **PF-Pascal** | Keypoint correspondence | [link](https://www.di.ens.fr/willow/research/proposalflow/) |
| **PF-Willow** | Keypoint correspondence | [link](https://www.di.ens.fr/willow/research/proposalflow/) |
| **AP-10k** | Animal pose estimation | via `prepare_ap10k.ipynb` (adapted from [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC)) |

## Project Structure

```
├── models/                 # Feature extractors
│   ├── dinov2/             # DINOv2 wrapper
│   ├── dinov3/             # DINOv3 wrapper
│   ├── SAM/                # SAM wrapper
│   └── models_factory.py   # Backbone builder
├── utils/
│   ├── matching.py         # Similarity + argmax / window soft-argmax
│   ├── loss.py             # Gaussian-smoothed cross-entropy loss
│   ├── metrics.py          # PCK computation
│   ├── geometry.py         # Coordinate transforms
│   ├── cli.py              # CLI argument parsers
│   ├── train_utils.py      # Checkpointing & drive sync
│   └── validation.py       # Validation loop
├── dataset/                # Dataset loaders (SPair, PF-Pascal, PF-Willow, AP-10k)
├── external/               # Git submodules (DINOv3, Segment Anything)
├── train.py                # Training entry point
├── eval.py                 # Evaluation entry point
├── project_config.py       # Global configuration
└── requirements.txt        # Python dependencies
```

## Fine-Tuning

The framework supports partial fine-tuning of the last *N* transformer blocks while keeping earlier layers frozen. Training uses a **Gaussian-smoothed cross-entropy loss** over the spatial correlation map, which provides smoother gradients around the ground-truth keypoint location compared to a standard hard-target loss. The loss implementation is adapted from [SD4Match](https://github.com/ActiveVisionLab/SD4Match) (Li et al., CVPR 2024).

## Window Soft-Argmax

During inference, standard hard argmax selects the single highest-scoring position on the feature map, which limits predictions to the discrete feature grid. **Window Soft-Argmax** improves on this by first identifying the hard argmax peak and then applying a temperature-scaled softmax within a local window around it. The predicted keypoint is then computed as the weighted centroid (center of mass) of this distribution, yielding **sub-pixel accurate** correspondences without any additional training cost. The technique is adapted from Zhang et al., CVPR 2024 — *Telling Left from Right: Identifying Geometry-Aware Semantic Correspondence*.


