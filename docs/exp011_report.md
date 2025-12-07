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
