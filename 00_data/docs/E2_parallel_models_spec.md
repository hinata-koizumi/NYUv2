# E2: Parallel Model Specifications & Development Instructions

> [!IMPORTANT]
> This document defines the **contract** for developing Models A, B, C, and D.
> All models must be strictly compatible with `01_nearest` for the final ensemble.

---

## 1. Purpose / Goal
- **Objective**: Develop 4 specialized models to complement `01_nearest`.
- **Strategy**: Maximize **ΔmIoU** (Improvement over `01_nearest` when ensembled).
- **Key Requirement**: Diversify error patterns (Specialize in Small Objects, Non-Structure, or Depth-ambiguous regions).

## 2. Terminology & Coordinate Systems
To prevent resolution mismatch/confusion:

| Term | Resolution | Description |
| :--- | :--- | :--- |
| **TRAIN_SIZE** | **Flexible** (e.g., 720×960) | The resolution used internally by your model during training. |
| **EVAL_SIZE** | **480 $\times$ 640** | **Unified Coordinate System**. Matches `01_nearest` golden logits. |
| **SUB_SIZE** | **480 $\times$ 640** | Final submission resolution. Identical to EVAL_SIZE. |

> [!WARNING]
> All submitted logits (`test_logits.npy`, `oof_logits.npy`) **MUST** be strictly **480 $\times$ 640**.
> If your model trains/infers at `TRAIN_SIZE`, you must `bilinear` resize to `EVAL_SIZE` before saving.
> Do NOT save at 720x960.
> *Verification*: `EVAL_SIZE` assumes consistency with `01_nearest/golden_artifacts/test_logits.npy` (`[654,13,480,640]`). If the golden artifact changes, `EVAL_SIZE` updates to match it.

## 3. Common Contract (Required for All Models)

### Outputs
Models must verify and save the following artifacts:

1.  **`test_logits.npy`**
    *   **Shape**: `[654, 13, EVAL_H, EVAL_W]` (Current: `480x640`)
    *   **Dtype**: `float16`
    *   **Content**: Raw Logits (Before Softmax/Argmax).
    *   **Order**: Matches **`test_ids.txt` (Fixed repo manifest)**.

2.  **`oof_logits.npy`**
    *   **Shape**: `[795, 13, EVAL_H, EVAL_W]` (Current: `480x640`)
    *   **Dtype**: `float16`
    *   **Content**: OOF predictions (Cross-Validation).
    *   **Construction**: Concatenation of validation folds per image.
    *   **Sorting**: **CRITICAL**. Must match **`train_ids.txt` (Fixed repo manifest)**.
    *   *Note*: Do not rely solely on "sorted filenames". Use the ID list as the source of truth.
    *   *ID Generation Rule*:
        *   `train_ids.txt`: Source `data/NYUv2/train/image/`. Sort filenames (ASCII ascending). **Include extension**. Assert `len==795`.
        *   `test_ids.txt`: Source `data/NYUv2/test/image/`. Sort filenames (ASCII ascending). **Include extension**. Assert `len==654`.
        *   Files are **FROZEN** once generated.

### Evaluation Protocol
1.  **Resize**: Bilinear interpolation to `EVAL_SIZE`.
    *   **Parameters**: `align_corners=False`, `antialias=False` (unless otherwise specified).
    *   **Prohibited**: Nearest Neighbor for logits.
2.  **Ignore Mask**: `255`. All metrics must mask `GT == 255`.
3.  **Metrics**:
    *   **Single Model**: mIoU, Class-wise IoU.
    *   **Ensemble**: $\Delta mIoU = mIoU(ens) - mIoU(01\_nearest)$.

### Reproducibility
*   **Manifest**: `sha256_manifest.txt` recommended for `npy` files.

## 4. Model Specifications (Parallel Development)

### Model A: DetailCrop Specialist
*   **Role**: Fix "Small/Far" objects (Books, Picture, TV) where `01_nearest` fails.
*   **Strategy**: "Recall over Precision" for small objects.
*   **Input**: RGB (Recommended).
*   **Resolution**:
    *   **TRAIN_SIZE**: **High-Res Random Crop** (e.g., 768~1024 crop).
    *   **Smart Crop**: Significantly increase sampling of tiles containing `books`, `picture`, `tv`.
*   **Augmentation**: Strong Photometric (Color Jitter / Blur / Noise). Conservative Geometry (Don't destroy small shapes).
*   **Inference**:
    *   **Tiling/Overlap**: PERMITTED if it improves LB.
    *   **TTA**: Flip + 1 Scale (Keep it light).

### Model B: Non-Struct Specialist
*   **Role**: Fix "Granularity Confusion" (Furniture vs Objects). Win on Non-Structure.
*   **Strategy**: Let `01_nearest` handle the walls/floors.
*   **Input**: RGB.
*   **Data Sampling**: Prioritize images/crops with high "Non-Structure" pixel counts.
*   **Loss**:
    *   **Structure Classes (7)**: Lower weight (e.g., 0.5 or lower).
    *   **Non-Structure**: Focal Loss / Dice Loss with higher weight.
    *   **Label Smoothing**: Permitted (to soften confusion).
*   **Augmentation**: Medium-Strong Photometric. Copy-Paste (Increase small/mid object density).

### Model C: Depth Specialist
*   **Role**: Disambiguate RGB-confusing regions using Geometry.
*   **Strategy**: "Internal Class Certainty".
*   **Input**: **RGB-D (4ch)**.
*   **Architecture Constraint**:
    *   **Fusion**: **Gated Fusion** or **Late Fusion** preferred over early 4ch input (to force Depth feature usage).
    *   *Alternative*: Separate Stem for Depth.
*   **Augmentation (MANDATORY)**:
    *   **Depth Dropout**: Randomly zero out depth channels.
    *   **Depth Noise/Jitter**: Prevent overfitting to perfect depth.
    *   **Depth Masks/NaNs**: Explicitly handle invalid depth / NaNs.
        *   **Requirement**: Maintain a separate `invalid_mask` input, OR ensure normalization treats '0/NaN' as explicitly 'invalid' (distinct from 'near' or 'far'). Do not leave this open to interpretation.

### Model D: Context Specialist
*   **Role**: Resolve ambiguity via Global Context.
*   **Architecture**: **DeepLabV3+** (Fixed).
*   **Input**: RGB (Focus on diversity).
*   **Resolution**: Medium-High (Full 480x640 or Mild Upscale).
*   **Augmentation**: Medium Photometric, Medium Geometry.
*   **Constraint**: If improvements overlap >80% with Model B, discard one.

## 5. Adoption Criteria (The "Gates")

A model is adopted **ONLY** if it passes these gates:

1.  **Gate 1 (Performance)**: $\Delta mIoU(01\_nearest + Model\_X) > +0.003$ (OOF).
    *   **Step 1**: Check with **Equal Weight Ensemble (0.5/0.5)** of `01_nearest` + `Model_X`.
    *   **Step 2**: Check with **Class-wise Optimization** (If it helps).
    *   *Note*: Single model mIoU is secondary.
2.  **Gate 2 (Target)**: Improvement is concentrated in:
    *   **Non-Structure Classes** (Model B/D) OR
    *   **Small/Far Objects** (Model A).
3.  **Gate 3 (Correlation)**: Error Correlation with `01_nearest` should be $< 0.95$ (Ideal).
4.  **Gate 4 (Safety)**: Ensemble does **NOT** degrade **Structure Classes**.
    *   *Definition*: **bed, chair, floor, sofa, table, wall, window**.
    *   *Reasoning*: If these foundations collapse, global geometry is lost.

## 6. Training "Base Recipe" (Recommended)
*   **Optimizer**: AdamW + Cosine Schedule + Warmup.
*   **EMA**: Enabled (Crucial for stability).
*   **AMP**: Enabled (bf16 preferred).
*   **Folds**: **MUST** use same 5-fold split as `01_nearest`.
    *   *Source*: Reuse the split definition (e.g., `03_ensemble/splits` or `folds.json`) defined in `01_nearest`. **Do NOT re-implement split logic.**

## 7. Recommended Directory Structure
To ensure isolation and clear contracts:

```
repo_root/
├── 00_data/
│   ├── NYUv2/...
│   ├── ids/ (train_ids.txt, test_ids.txt)
│   └── splits/ (folds_v1.json - frozen copy)
├── 01_nearest/
├── 02_detailcrop/ (Model A)
├── 03_nonstruct/  (Model B)
├── 04_depth/      (Model C)
├── 05_context/    (Model D)
└── 03_ensemble/
```

---
**Verified by Antigravity**
**Status**: Ready for Implementation
