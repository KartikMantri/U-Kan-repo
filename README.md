# Glaucoma Detection via Deep-Learning Segmentation of the Optic Disc & Cup

Joint **optic disc (OD)** and **optic cup (OC)** segmentation on retinal fundus images, for automatic estimation of the **cup-to-disc ratio (CDR)** used in glaucoma screening. This repository contains a full, week-by-week progression of segmentation architectures — from a plain U-Net to attention-guided dual-decoder networks — implemented and evaluated on the **REFUGE**, **G1020**, and **ORIGA** datasets.

> **Project target:** a mean disc + cup Dice score of **0.90** on REFUGE.
> **Best result:** dual-decoder U-Net + cross-attention + containment loss — **test mean Dice 0.9165** (disc 0.9645, cup 0.8684). ✅ target met.

**Author:** Kartik Shivkumar Mantri (Roll No. 24UCS246)

---

## Table of Contents
- [Background](#background)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Architectures Implemented](#architectures-implemented)
- [Loss Functions](#loss-functions)
- [Results](#results)
- [Best Model](#best-model)
- [How to Run](#how-to-run)
- [State-of-the-Art Context](#state-of-the-art-context)
- [Key Learnings](#key-learnings)
- [Citation & Acknowledgements](#citation--acknowledgements)

---

## Background

Glaucoma is a leading cause of irreversible blindness and is assessed largely from the **cup-to-disc ratio (CDR)** on a retinal fundus photograph. Computing the CDR requires pixel-level segmentation of two nested structures: the **optic disc** (the bright region where the optic nerve exits) and the **optic cup** (the brighter depression inside it). This is framed as a **3-class problem** — background (0), optic disc / rim (1), optic cup (2).

The **optic cup is the limiting structure** throughout this project: it is small, low-contrast, and always nested inside the disc, so most of the engineering effort targets accurate cup segmentation without degrading the disc.

## Datasets

| Dataset | Images | Split used | Notes |
|---|---|---|---|
| **REFUGE** | 1,200 | 400 train / 400 val / 400 test (official) | Primary benchmark; cropped disc-region fundus images |
| **G1020** | 1,020 | 714 / 153 / 153 | Cross-dataset validation |
| **ORIGA** | 650 | 276 / 49 / 325 (Set-A/B) | Cross-dataset validation |

All images/masks resized to **256×256** (384×384 in later experiments). Masks encoded as `0 = background, 1 = optic disc, 2 = optic cup`.

> **Note:** The datasets are **not** redistributed in this repo. Obtain REFUGE / G1020 / ORIGA from their official sources (e.g. the Kaggle *glaucoma-datasets* collection) and point the notebooks' config paths at them.

## Files in this Repository

All notebooks live in the repository root. Use this index to find the code for each architecture (filenames may vary slightly from your local copies):

**U-Net (baseline)**
- `Glaucoma_UNet_REFUGE_Colab.ipynb` — plain U-Net, 3-class REFUGE segmentation (pipeline baseline).
- `unet_with_3_classes_with_sobelfilter.ipynb` — U-Net with an added Sobel edge-magnitude input channel.

**U-KAN (Kolmogorov–Arnold Network)**
- `ukan_with_2kan_layer_scratch_trained_onrefuge_week2.ipynb` — U-KAN from scratch, 2 KANLinear/block, REFUGE.
- `ukan_with_2kan_layer_scratch_trained_on_G1020_week2.ipynb` — same, G1020.
- `origa_2kanlinearperblockcode_week2.ipynb` — same, ORIGA.
- `ukan_with_3kanlayers_refuge_with_2_lrs_week3.ipynb` — 3 KANLinear + differential (two-rate) LR, REFUGE.
- `updated_g1020_unique2_lr_3kanlinear_week3.ipynb` / `updatedandtrainedonORIGA_..._week3.ipynb` — same on G1020 / ORIGA.
- `U-KAN_REFUGE_official_architecture_3kanlinearlahyers_with1lr.ipynb` / `U-KAN_REFUGE_full_pipeline_eta_min_updated codee.ipynb` — official-style U-KAN pipeline variants.
- `UKAN_REFUGE_3class_Kaggle_tta+postprocees_week5.ipynb` — U-KAN + Test-Time Augmentation + anatomical post-processing.
- `ResNet34_UKAN_REFUGE_3class_tta_postprocess_week5.ipynb` — ImageNet-pretrained ResNet34 encoder + KAN bottleneck.

**W-Net (two-stage cascade)**
- `wnet_unet_plus_ukan_week4.ipynb` — U-Net disc localizer → disc-guided crop → U-KAN cup segmenter.

**Dual-Decoder (Y-Net) family**
- `DualDecoder_UNet_REFUGE_week6.ipynb` — dual-decoder U-Net baseline (shared encoder, disc + cup heads).
- `DualDecoder_KANbottleneck_REFUGE_week6.ipynb` — dual-decoder with a KAN bottleneck.

**Attention variants (best models)**
- `DualDecoder_UNet_CrossAttn_Containment_REFUGE_1.ipynb` — ⭐ **best model**: cross-attention + containment loss.
- `DualDecoder_UNet_SpatialAttention_Containment_REFUGE.ipynb` — spatial (CBAM) attention + containment loss.
- `DualDecoder_UNet_REFUGE_withdim_384_384_week7.ipynb` — dual-decoder at 384×384.
- `3attention_kan_newest_architecture.ipynb` / `CSP_KAN_UNet_REFUGE.ipynb` — CSP-KAN-UNet (Channel→Spatial→Pixel attention decoder).
- `DualDecoder_SpatialAttn_CrossAttn_REFUGE_384.ipynb` — combined spatial (every stage) + cross-attention (first & last), 384×384.

**Reports & docs** — `*_Report*.docx` / `*.pdf` weekly reports document the theory, code, and results for each stage; `framework-1.jpg` and concept PDFs cover U-KAN internals.

> **Tip:** if you'd like the repo organized into `1_unet/`, `2_ukan/`, `3_wnet/`, `4_dual_decoder/`, `5_attention/`, `reports/` folders, see [Suggested Organization](#suggested-organization) below.

## Suggested Organization

The repo currently keeps every notebook in the root. An optional cleaner layout groups them by architecture:

```
├── 1_unet/            U-Net baseline (+ Sobel edge variant)
├── 2_ukan/            U-KAN from scratch, 3-KANLinear, differential-LR, TTA, ResNet34 hybrid
├── 3_wnet/            Two-stage W-Net cascade
├── 4_dual_decoder/    Y-Net baseline + KAN-bottleneck
├── 5_attention/       Cross-Attention, Spatial-Attention, CSP-KAN-UNet, 384 combined model
├── reports/           Weekly .docx / .pdf reports
└── docs/              Reference papers & figures
```

A ready-to-run reorganization script (`organize_repo.sh`) can be added on request.

## Architectures Implemented

| # | Architecture | Idea | Week |
|---|---|---|---|
| 1 | **U-Net** | Convolutional encoder–decoder with skip connections (baseline) | 1 |
| 2 | **U-KAN** | U-Net with a tokenized **Kolmogorov–Arnold Network** bottleneck (learnable B-spline edge activations, implemented from scratch) | 2–3 |
| 3 | **U-KAN modifications** | 3 KANLinear sublayers, differential (two-rate) LR, Sobel edge channel, TTA, anatomical post-processing, ResNet34-pretrained encoder | 3, 5 |
| 4 | **W-Net** | Two-stage cascade: U-Net disc localizer → disc-guided crop → U-KAN cup segmenter | 4 |
| 5 | **Y-Net (Dual-Decoder U-Net)** | One shared encoder, two parallel decoders (disc head + cup head), single forward pass | 6 |
| 6 | **Dual-Decoder + Cross-Attention** | Bidirectional cross-attention lets the cup decoder consult the disc decoder + **containment loss** | 7 |
| 7 | **Dual-Decoder + Spatial Attention** | CBAM-style spatial attention gating in the decoders + containment loss | 7 |
| 8 | **CSP-KAN-UNet** | KAN bottleneck + triple (Channel→Spatial→Pixel) attention decoder | 7 |
| 9 | **Dual-Decoder + Spatial (every stage) + Cross (first & last) @384** | Combined attention design at higher resolution | 8 |

## Loss Functions

- **Cross-Entropy** — stable per-pixel base term.
- **Soft Dice** — region-overlap objective, robust to class imbalance.
- **Focal loss** — down-weights easy pixels via `(1 − pₜ)^γ` to focus on the hard cup boundary.
- **Focal-Tversky** — weights false negatives harder (β > α) to push cup recall.
- **Containment loss** — `λ · ReLU(P(cup) − P(disc)).mean()`, encoding the anatomical constraint `cup ⊆ disc` directly into training.

Workhorse objective: `0.5·CE + 0.5·Dice` per head, plus containment (λ = 0.5) for the dual-decoder attention models.

## Results

**REFUGE test set — mean disc + cup Dice** (official 400/400/400 split):

| Model | Disc Dice | Cup Dice | Mean (D+C) | Notes |
|---|---|---|---|---|
| U-KAN (2 KANLinear, scratch) | 0.8982 | 0.8630 | 0.8806 | week 2 |
| U-KAN (3 KANLinear + diff-LR) | 0.9004 | 0.8609 | 0.8807 | week 3 |
| U-Net + Sobel edge | 0.7901 | 0.7681 | 0.7791 | week 5 |
| U-KAN + TTA + post-proc | 0.8785 | 0.8603 | 0.8694 | week 5 |
| ResNet34 + KAN bottleneck | 0.8962 | 0.8479 | 0.8721 | week 5 |
| Two-Stage W-Net | 0.8873 | 0.8433 | 0.8653 | week 4 |
| Dual-Decoder U-KAN (KAN bottleneck) | 0.9374 | 0.8083 | 0.8729 | week 6 |
| Dual-Decoder U-Net (Y-Net) | 0.9618 | 0.8491 | 0.9055 | week 6 |
| **Dual-Decoder + Spatial Attention** | 0.9647 | 0.8644 | **0.9146** | week 7 |
| **Dual-Decoder + Cross-Attention** ⭐ | **0.9645** | **0.8684** | **0.9165** | week 7 — best |

**Cross-dataset (mean disc + cup Dice):** U-KAN — G1020 0.8621, ORIGA 0.8882.

## Best Model

**Dual-Decoder U-Net + Cross-Attention + Containment Loss** (≈45.3 M params, REFUGE 256×256):

| Split | Disc Dice | Cup Dice | Mean (D+C) |
|---|---|---|---|
| Train (peak) | 0.9789 | 0.8952 | — |
| Validation (best ckpt) | 0.9667 | 0.8796 | **0.9231** |
| Test | 0.9645 | 0.8684 | **0.9165** |

One shared encoder feeds a disc decoder and a cup decoder that exchange information via **bidirectional cross-attention** at their coarsest stage; a differentiable **containment loss** enforces `cup ⊆ disc` during training. This is the first model to clear the 0.90 project target.

## How to Run

Notebooks are self-contained (Colab / Kaggle). Typical steps:

```bash
# 1. Open the notebook for the model you want (e.g. week7/…CrossAttention…ipynb)
# 2. Select a GPU runtime
# 3. In the CONFIG cell, set the paths to your REFUGE/G1020/ORIGA image & mask folders
#    (or use the Kaggle-API / Drive-mount cells provided)
# 4. Run all — it trains, plots curves, evaluates on test, and renders visual outputs
```

Core dependencies: `torch`, `torchvision`, `numpy`, `opencv-python`, `matplotlib` (installed at the top of each notebook).

## State-of-the-Art Context

On the official REFUGE benchmark, published methods cluster at **disc ≈ 0.96** and **cup ≈ 0.87–0.88** (e.g. FunduSAM 2025: OC 0.867; Segtran: OC 0.872) — the cup ceiling reflects inter-grader disagreement on the cup boundary. This project's best model (**disc 0.9645, cup 0.8684**) is therefore **at state of the art on the disc and within the SOTA band on the cup**, on the official split.

> ⚠️ Cross-paper comparisons are protocol-sensitive: different splits, resolutions, disc-cropping/polar transforms, and label orderings can shift cup Dice by 0.05–0.15. All numbers here use the **official REFUGE 400/400/400 split, per-structure Dice, on the held-out test set**.

## Key Learnings

- The **optic cup is the bottleneck** for the mean score; raising it came less from bigger backbones and more from **decoder coupling** (cross-attention) and **anatomy-aware losses** (containment).
- A **single-pass dual-decoder** matches a two-stage cascade while removing crop-error propagation.
- **Loss design and inference-time tricks** (Focal-Tversky, TTA, post-processing, threshold calibration) matter as much as architecture for a small, imbalanced structure.

## Citation & Acknowledgements

If you use this work, please cite the repository. Datasets belong to their respective owners (REFUGE, G1020, ORIGA). This work was completed as a Summer Research Internship at **PDPM IIITDM Jabalpur**.

---

*Maintained by Kartik Shivkumar Mantri · 2026*
