# Pipeline Methods Documentation

This document provides a detailed overview of the semantic segmentation pipelines implemented in `src/pipeline-1.py` through `src/pipeline-9.py`. All pipelines are based on the DeepLabV3+ architecture and are designed for the NYUv2 dataset, utilizing various strategies for input processing, multi-task learning, and loss optimization.

## Summary Table

| Pipeline | Encoder | Input / Features | Key Strategy | Losses |
| :--- | :--- | :--- | :--- | :--- |
| **1** | EfficientNetV2-L | 4-channel (RGB+Depth) | High-res, 4ch Input | CE + Dice + Lovasz |
| **2** | EfficientNet-B7 | 3-channel (RGB) | Gradient Accumulation, Weighted Sampling | CE + Dice + Lovasz |
| **3** | EfficientNet-B7 | 3-channel (RGB) | Rare Class Sampling (Flag-based) | CE + Dice + Lovasz |
| **4** | EfficientNet-B7 | 7-channel (RGB+Geo) | Geometric Features Input (Concatenated) | CE + Dice + Lovasz |
| **5** | EfficientNet-B7 | 7-channel (RGB+Geo) | Advanced Augmentation (Crop/Cutout) | CE + Dice + Lovasz |
| **6** | EfficientNet-B7 | 7-channel (RGB+Geo) | **Multi-task** (Depth Recon), Resume Support | CE + Dice + Lovasz + Depth(L1) |
| **7** | EfficientNet-B7 | 3-channel (RGB) | **Depth Gating** (Attention via Aux Depth) | CE + Dice + Lovasz + Depth(L1) |
| **8** | EfficientNet-B7 | 3-channel (RGB) | **Edge Detection Head** + Depth Gating | CE + Dice + Lovasz + Depth(L1) + Edge(BCE) |
| **9** | EfficientNet-B4 | 4-channel (RGB+Depth) | **Simplified**, Early Fusion | CE + Lovasz (No Dice, No Aux) |

---

## Detailed Descriptions

### pipeline-1.py: Baseline 4-Channel Model
*   **Architecture**: DeepLabV3+ with `timm-efficientnet-v2-l` encoder.
*   **Input**: 4-channel input (RGB + Depth). The depth channel is concatenated to the RGB image before entering the encoder.
*   **Loss Function**: Combination of CrossEntropy, Dice Loss, and Lovasz-Softmax Loss.
*   **Training**:
    *   Uses K-Fold Cross Validation.
    *   Standard augmentations (Flip, Affine, ColorJitter).
*   **Inference**: Test-Time Augmentation (TTA) with horizontal flips.

### pipeline-2.py: Gradient Accumulation & Weighted Sampling
*   **Architecture**: DeepLabV3+ with `timm-efficientnet-b7` encoder.
*   **Input**: Standard 3-channel RGB input.
*   **Key Features**:
    *   **Gradient Accumulation**: Uses `accum_steps` to simulate larger batch sizes.
    *   **Weighted Sampling**: Computes sample weights based on rare class presence to balance the training distribution.
*   **Loss Function**: CE + Dice + Lovasz.

### pipeline-3.py: Rare Class Sampling
*   **Architecture**: DeepLabV3+ with `timm-efficientnet-b7`.
*   **Input**: Standard 3-channel RGB input.
*   **Key Feature**: Explicit `use_rare_sampler` flag to control weighted sampling.
*   **Sampling**: Implements a `WeightedRandomSampler` that significantly upsamples images containing rare classes (e.g., classes 1, 7, 10).
*   **Loss Function**: CE + Dice + Lovasz.

### pipeline-4.py: Geometric Features Input
*   **Architecture**: DeepLabV3+ with `timm-efficientnet-b7`.
*   **Input**: **7-channel input**.
    *   RGB (3 channels)
    *   Geometric Features (4 channels): Log-depth, Normalized depth, dx (horizontal gradient), dy (vertical gradient).
*   **Mechanism**: Geometric features are concatenated with RGB at the input level (`in_channels=7`).
*   **Loss Function**: CE + Dice + Lovasz.

### pipeline-5.py: Advanced Augmentation
*   **Architecture**: DeepLabV3+ with `timm-efficientnet-b7`.
*   **Input**: **7-channel input** (RGB + Geometric Features).
*   **Augmentation**: Introduces more aggressive data augmentation techniques:
    *   `RandomResizedCrop`: Crops and resizes regions of the image.
    *   `CoarseDropout`: Randomly masks out rectangular regions (Cutout-like).
*   **Loss Function**: CE + Dice + Lovasz.

### pipeline-6.py: Multi-Task with Geometric Features
*   **Architecture**: DeepLabV3+ with `timm-efficientnet-b7`.
*   **Input**: **7-channel input** (RGB + Geometric Features).
*   **Multi-Task Learning**: Adds a secondary decoder head for **Depth Reconstruction**.
*   **Key Feature**: Enhanced checkpoint management to resume training.
*   **Loss Function**:
    *   Segmentation: CE + Dice + Lovasz.
    *   Auxiliary: L1 Loss for depth reconstruction.

### pipeline-7.py: Depth Gating (Attention)
*   **Architecture**: DeepLabV3+ with `timm-efficientnet-b7`.
*   **Input**: Standard 3-channel RGB input (Depth is passed separately for gating).
*   **Key Mechanism**: **Depth Gating Module**.
    *   Computes an attention gate using log-depth and depth gradient magnitude (edges).
    *   This gate modulates the encoder's intermediate feature maps via a `DepthGatedEncoder` wrapper.
    *   Formula: `feat = feat * (1 + alpha * gate)`.
*   **Loss Function**: CE + Dice + Lovasz + Depth L1 (Depth head is present).

### pipeline-8.py: Edge Detection Multi-Task
*   **Architecture**: DeepLabV3+ with `timm-efficientnet-b7`.
*   **Input**: Standard 3-channel RGB input (Depth for gating).
*   **Key Mechanism**: **Triple-Head Architecture**.
    1.  Segmentation Head.
    2.  Depth Reconstruction Head.
    3.  **Edge Detection Head**: Predicts binary edge maps.
*   **Depth Gating**: Retains the Depth Gating module from pipeline-7.
*   **Loss Function**:
    *   Segmentation: CE + Dice + Lovasz.
    *   Depth: L1 Loss.
    *   Edge: **Binary Cross Entropy (BCE)** with positive weighting for edge pixels.
*   **Targets**: Edge targets are generated on-the-fly from segmentation masks using Sobel-like filters.

### pipeline-9.py: Simplified Early Fusion
*   **Architecture**: DeepLabV3+ with `timm-efficientnet-b4` (lighter encoder).
*   **Input**: **4-Channel Early Fusion** (RGB + Depth).
*   **Simplification**:
    *   Removes auxiliary heads (No Depth Recon, No Edge Head).
    *   Removes Dice Loss (uses only CE + Lovasz).
    *   Lovasz Loss is applied from the start (Epoch 1), not delayed.
    *   Simplified Augmentations (Reduced dropout, no vertical flip).
*   **Goal**: A streamlined, efficient model focusing purely on segmentation performance with depth as a direct input.
