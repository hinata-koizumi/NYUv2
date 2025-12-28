# Exp000 Baseline Report

## Experiment Configuration
- **Script**: `src/exp000_baseline.py`
- **Model Architecture**: DeepLabV3+ with ResNet50 encoder (Pretrained on ImageNet)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8

## Results Summary
The model achieved its best performance at **Epoch 17**.

- **LB Score**: 0.52136
- **Best mIoU**: 0.5676
- **Pixel Accuracy**: 0.7816
- **Validation Loss**: 0.7991

### Per-Class IoU (Epoch 17)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.636 |
| 1 | 0.301 |
| 2 | 0.630 |
| 3 | 0.529 |
| 4 | 0.851 |
| 5 | 0.563 |
| 6 | 0.483 |
| 7 | 0.575 |
| 8 | 0.548 |
| 9 | 0.382 |
| 10 | 0.466 |
| 11 | 0.791 |
| 12 | 0.623 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.6204 | 1.1514 | 0.3342 | 0.6488 |
| 5 | 0.4307 | 0.7921 | 0.5105 | 0.7518 |
| 10 | 0.2048 | 0.7947 | 0.5419 | 0.7627 |
| 15 | 0.1270 | 0.8042 | 0.5616 | 0.7776 |
| **17** | **0.1083** | **0.7991** | **0.5676** | **0.7816** |
| 20 | 0.0970 | 0.8410 | 0.5484 | 0.7763 |
| 25 | 0.0811 | 0.8341 | 0.5598 | 0.7793 |
| 30 | 0.0749 | 0.8392 | 0.5614 | 0.7794 |

## Observations
- The model reached peak mIoU at Epoch 17.
- Validation loss started to plateau and slightly increase after Epoch 17, suggesting potential overfitting, although mIoU remained relatively stable around 0.56.
- Class 4 achieved the highest IoU (0.851), while Class 1 had the lowest (0.301).

# Exp001 RGB Only CE Weighted Report

## Experiment Configuration
- **Script**: `src/exp001.py`
- **Model Architecture**: DeepLabV3+ with ResNet50 encoder (Pretrained on ImageNet)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss Function**: Weighted CrossEntropyLoss

### Class Weights
| Class ID | Weight |
| :---: | :---: |
| 0 | 0.0382 |
| 1 | 0.2609 |
| 2 | 0.0962 |
| 3 | 0.0436 |
| 4 | 0.0159 |
| 5 | 0.0104 |
| 6 | 0.0111 |
| 7 | 0.0889 |
| 8 | 0.0585 |
| 9 | 0.0491 |
| 10 | 0.2923 |
| 11 | 0.0063 |
| 12 | 0.0286 |

## Results Summary
The model achieved its best performance at **Epoch 27**.

- **LB Score**: 0.5259
- **Best mIoU**: 0.5600
- **Pixel Accuracy**: 0.7691
- **Validation Loss**: 1.1127

### Per-Class IoU (Epoch 27)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU | Diff vs Baseline (Exp000) |
| :---: | :---: | :---: |
| 0 | 0.644 | +0.008 |
| 1 | 0.371 | +0.070 |
| 2 | 0.623 | -0.007 |
| 3 | 0.536 | +0.007 |
| 4 | 0.845 | -0.006 |
| 5 | 0.545 | -0.018 |
| 6 | 0.467 | -0.016 |
| 7 | 0.580 | +0.005 |
| 8 | 0.540 | -0.008 |
| 9 | 0.380 | -0.002 |
| 10 | 0.404 | -0.062 |
| 11 | 0.764 | -0.027 |
| 12 | 0.581 | -0.042 |

*Positive diff indicates improvement over Exp000.*

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.8178 | 1.3453 | 0.2660 | 0.4745 |
| 5 | 0.4887 | 0.9138 | 0.4774 | 0.7113 |
| 10 | 0.2230 | 0.9755 | 0.5281 | 0.7464 |
| 15 | 0.1370 | 1.0280 | 0.5512 | 0.7619 |
| 20 | 0.1066 | 1.1026 | 0.5528 | 0.7651 |
| 25 | 0.0888 | 1.1391 | 0.5561 | 0.7668 |
| **27** | **0.0842** | **1.1127** | **0.5600** | **0.7691** |
| 30 | 0.0833 | 1.1449 | 0.5599 | 0.7679 |

## Observations
- The model reached peak mIoU at Epoch 27 with 0.5600, which is slightly lower than the baseline of 0.5676.
- **Class Imbalance Impact**:
    - Class 1 (Weight: 0.2609) saw a significant improvement in IoU (+0.070).
    - Class 10 (Weight: 0.2923) strangely saw a significant decrease in IoU (-0.062).
    - Class 4 (Weight: 0.0159) maintained high performance, dropping only slightly.
- **Validation Loss**: The validation loss is consistently higher than the baseline (around 1.1 vs 0.8), likely due to the weighted loss function amplifying errors in weighted classes, or simply the scale of the loss being different.
- **Overfitting**: Train loss decreases steadily, while Valid loss increases after around epoch 5-10, suggesting overfitting to the training data, despite the mIoU slowly improving until Epoch 27.

# Exp010 Report: RGB-D Early Fusion

## Experiment Configuration
- **Script**: `src/exp010.py`
- **Model Architecture**: DeepLabV3+ with ResNet50 encoder (Pretrained on ImageNet)
    - **Input Channels**: 4 (RGB + Depth)
    - **Fusion**: Early Fusion (Depth concatenated as 4th channel)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss**: CrossEntropyLoss (No class weights, IGNORE_INDEX=255)
- **Depth Processing**:
    - Clipped to [0.71m, 10.0m]
    - Normalized to [0, 1] relative to min/max range

## Results Summary
The model achieved its best performance at **Epoch 17**.

- **LB Score**: 0.53647
- **Best mIoU**: 0.5771
- **Pixel Accuracy**: 0.7879
- **Validation Loss**: 0.7589

### Per-Class IoU (Epoch 17)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.645 |
| 1 | 0.383 |
| 2 | 0.661 |
| 3 | 0.538 |
| 4 | 0.879 |
| 5 | 0.567 |
| 6 | 0.493 |
| 7 | 0.568 |
| 8 | 0.531 |
| 9 | 0.365 |
| 10 | 0.443 |
| 11 | 0.803 |
| 12 | 0.628 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.6172 | 1.1479 | 0.3198 | 0.6501 |
| 5 | 0.4366 | 0.7740 | 0.5075 | 0.7546 |
| 10 | 0.1994 | 0.7531 | 0.5650 | 0.7754 |
| 15 | 0.1230 | 0.7586 | 0.5703 | 0.7858 |
| **17** | **0.1074** | **0.7589** | **0.5771** | **0.7879** |
| 20 | 0.0944 | 0.7912 | 0.5651 | 0.7840 |
| 25 | 0.0803 | 0.7889 | 0.5735 | 0.7868 |
| 30 | 0.0747 | 0.7912 | 0.5752 | 0.7874 |

## Observations
- **Performance**: The RGB-D early fusion model (LB 0.5365) outperformed the RGB-only baseline (LB 0.5214), confirming the value of depth information.
- **Best Epoch**: Peak performance was reached at Epoch 17, similar to the baseline.
- **Overfitting**: Validation loss and mIoU fluctuated after Epoch 17, with loss slightly increasing, indicating potential overfitting in later epochs similar to the baseline.
- **Class Performance**: Class 4 remains the highest performing class (0.879), further improved from baseline (0.851). Most classes showed improvement.

# Exp011 Report: RGB-D Early Fusion + Geometric Augmentation

## Experiment Configuration
- **Script**: `src/exp011.py`
- **Model Architecture**: DeepLabV3+ with ResNet50 encoder (Pretrained on ImageNet)
    - **Input Channels**: 4 (RGB + Depth)
    - **Fusion**: Early Fusion (Depth concatenated as 4th channel)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss**: CrossEntropyLoss (No class weights, IGNORE_INDEX=255)
- **Depth Processing**:
    - Clipped to [0.71m, 10.0m]
    - Normalized to [0, 1] relative to min/max range
- **Augmentations** (New in Exp011):
    - **Library**: Albumentations (Synchronized for RGB, Depth, Label)
    - **Geometric**:
        - Resize (480x640)
        - HorizontalFlip (p=0.5)
        - ShiftScaleRotate (shift=0.05, scale=0.1, rotate=5, p=1.0)
        - RandomCrop (480x640, p=0.5)

## Results Summary
The model achieved its best performance at **Epoch 21**.

- **LB Score**: 0.55426
- **Best mIoU**: 0.5945
- **Pixel Accuracy**: 0.8050
- **Validation Loss**: 0.6838

### Per-Class IoU (Epoch 21)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.670 |
| 1 | 0.320 |
| 2 | 0.682 |
| 3 | 0.585 |
| 4 | 0.884 |
| 5 | 0.608 |
| 6 | 0.519 |
| 7 | 0.611 |
| 8 | 0.543 |
| 9 | 0.408 |
| 10 | 0.453 |
| 11 | 0.817 |
| 12 | 0.629 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.6272 | 1.2152 | 0.2869 | 0.6266 |
| 5 | 0.5415 | 0.7855 | 0.4927 | 0.7489 |
| 10 | 0.2894 | 0.6653 | 0.5718 | 0.7923 |
| 15 | 0.1927 | 0.7056 | 0.5756 | 0.7925 |
| 20 | 0.1522 | 0.6931 | 0.5880 | 0.8037 |
| **21** | **0.1434** | **0.6838** | **0.5945** | **0.8050** |
| 25 | 0.1313 | 0.6938 | 0.5934 | 0.8042 |
| 30 | 0.1232 | 0.6986 | 0.5925 | 0.8047 |

## Observations
- **Performance**: The addition of geometric augmentations (Albumentations) significantly improved performance. LB Score increased from 0.5365 (Exp010) to 0.55426. Best mIoU improved from 0.5771 to 0.5945.
- **Best Epoch**: Peak performance was reached at Epoch 21. The model maintained high performance (mIoU > 0.59) from Epoch 21 onwards.
- **Stability**: Compared to Exp010, the validation mIoU was more stable in the later epochs, suggesting that augmentations helped with generalization and reduced overfitting.
- **Class Performance**: Major improvements seen in several classes. Class 4 remains the highest at 0.884.
