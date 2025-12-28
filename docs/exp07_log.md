# Exp070 Report: RGB-D + Normals (7ch) + ResNet101

## Experiment Configuration
- **Script**: `src/exp070.py`
- **Model Architecture**: DeepLabV3+
    - **Backbone**: ResNet101 (Pretrained on ImageNet, in_channels=7)
    - **Encoder**: ResNet101
    - **Input**: 7 channels (RGB + Depth + Normals)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss**: CrossEntropyLoss (IGNORE_INDEX=255)
- **Depth Processing**:
    - **Input**:
        - **Depth**: Inverse Depth normalized [0, 1].
        - **Normals**: Computed from depth map, 3 channels (x, y, z), normalized to [0, 1].
    - **Total Channels**: 7 (RGB=3, Depth=1, Normals=3)
- **Augmentations**:
    - **Library**: Albumentations
    - **Geometric**:
        - Resize (480x640)
        - HorizontalFlip (p=0.5)
        - ShiftScaleRotate (shift=0.05, scale=0.1, rotate=5, p=1.0)
        - RandomCrop (480x640, p=0.5)

## Results Summary
The model achieved its best mIoU at **Epoch 18**.

- **LB Score**: 0.58144
- **Best mIoU**: 0.6191 (Validation)
- **Pixel Accuracy**: 0.8153 (Validation)
- **Validation Loss**: 0.6805

### Per-Class IoU (Epoch 18)

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.719 |
| 1 | 0.360 |
| 2 | 0.660 |
| 3 | 0.610 |
| 4 | 0.911 |
| 5 | 0.594 |
| 6 | 0.545 |
| 7 | 0.621 |
| 8 | 0.574 |
| 9 | 0.472 |
| 10 | 0.479 |
| 11 | 0.824 |
| 12 | 0.679 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.5421 | 1.1429 | 0.3420 | 0.6523 |
| 5 | 0.4991 | 0.7302 | 0.5108 | 0.7648 |
| 10 | 0.2657 | 0.6626 | 0.5620 | 0.7992 |
| 15 | 0.1703 | 0.6673 | 0.6073 | 0.8128 |
| **18** | **0.1382** | **0.6805** | **0.6191** | **0.8153** |
| 26 | 0.1069 | 0.6812 | 0.6187 | 0.8211 |
| 30 | 0.1037 | 0.6832 | 0.6183 | 0.8207 |

## Observations
- **Performance**: The LB score (0.58144) is slightly lower than Exp060 (0.586) and Exp040 (0.585).
- **Normals Input**: Explicitly adding surface normals as input channels (increasing input from 4 to 7 channels) did not yield an improvement over the 4-channel RGB-D (Inverse) or Multi-Task approaches. It seems the ResNet backbone might already be learning necessary geometric features from the depth map, or the calculated normals (which are noisy if depth is noisy) added more noise than signal.
- **Overfitting**: The model fits the training data well (Loss ~0.10) but validation loss plateaus around ~0.66-0.68, similar to other experiments.
