# Exp050 Report: RGB-D-XY Early Fusion + Geometric Augmentation + ResNet101 + Inverse Depth

## Experiment Configuration
- **Script**: `src/exp050.py`
- **Model Architecture**: DeepLabV3+ (Early Fusion with XY Coordinates)
    - **Backbone**: ResNet101 (Pretrained on ImageNet, in_channels=6)
    - **Input**: 6 channels (RGB + Depth + X + Y)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss**: CrossEntropyLoss (IGNORE_INDEX=255)
- **Depth Processing**:
    - Clipped to [0.71m, 10.0m]
    - Normalized using **Inverse Depth** encoding:
        - `inv = 1.0 / depth`
        - `inv_min = 1.0 / 10.0`
        - `inv_max = 1.0 / 0.71`
        - `depth = (inv - inv_min) / (inv_max - inv_min)`
- **Coordinate Processing**:
    - **X-channel**: Linear 0 to 1 (left to right)
    - **Y-channel**: Linear 0 to 1 (top to bottom)
    - Concatenated as 2 additional channels.
- **Augmentations**:
    - **Library**: Albumentations
    - **Geometric**:
        - Resize (480x640)
        - HorizontalFlip (p=0.5)
        - ShiftScaleRotate (shift=0.05, scale=0.1, rotate=5, p=1.0)
        - RandomCrop (480x640, p=0.5)

## Results Summary
The model achieved its best performance at **Epoch 22**.

- **LB Score**: 0.58362
- **Best mIoU**: 0.6220 (Validation)
- **Pixel Accuracy**: 0.8217 (Validation)
- **Validation Loss**: 0.6571

### Per-Class IoU (Epoch 22)

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.715 |
| 1 | 0.370 |
| 2 | 0.631 |
| 3 | 0.605 |
| 4 | 0.915 |
| 5 | 0.629 |
| 6 | 0.549 |
| 7 | 0.643 |
| 8 | 0.561 |
| 9 | 0.457 |
| 10 | 0.502 |
| 11 | 0.831 |
| 12 | 0.679 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.5551 | 1.1457 | 0.3664 | 0.6626 |
| 5 | 0.5169 | 0.6892 | 0.5196 | 0.7731 |
| 10 | 0.2762 | 0.6573 | 0.5743 | 0.7973 |
| 15 | 0.1712 | 0.6567 | 0.6044 | 0.8145 |
| 20 | 0.1294 | 0.6550 | 0.6197 | 0.8197 |
| **22** | **0.1177** | **0.6571** | **0.6220** | **0.8217** |
| 25 | 0.1098 | 0.6632 | 0.6200 | 0.8227 |
| 30 | 0.1044 | 0.6670 | 0.6165 | 0.8229 |

## Observations
- **Performance**: The LB score (0.58362) is slightly lower than Exp040 (0.58571), despite a slightly higher Validation mIoU (0.6220 vs 0.6185).
- **XY Coordinates**: Adding explicit X and Y coordinate channels did not result in a significant performance boost on the leaderboard compared to the 4-channel (RGB-D) input. It's possible the ResNet backbone already implicitly learns spatial information effectively, or the simple linear encoding wasn't distinctive enough.
- **Overfitting**: The gap between training loss (0.1177) and validation loss (0.6571) suggests the model might be overfitting slightly more than previous experiments, potentially due to the increased dimensionality of the input layer without corresponding regularization adjustments.
- **Class Performance**:
    - Class 4 (0.915) remains very high.
    - Class 1 (0.370) saw a small improvement over Exp040 (0.353).
