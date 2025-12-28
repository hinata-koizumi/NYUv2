# Exp030 Report: RGB-D Early Fusion + Geometric Augmentation + ResNet101

## Experiment Configuration
- **Script**: `src/exp030.py`
- **Model Architecture**: DeepLabV3+ (Early Fusion)
    - **Backbone**: ResNet101 (Pretrained on ImageNet, in_channels=4, early fusion)
    - **Input**: 4 channels (RGB + Depth)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss**: CrossEntropyLoss (No class weights, IGNORE_INDEX=255)
- **Depth Processing**:
    - Clipped to [0.71m, 10.0m]
    - Normalized to [0, 1] relative to min/max range
- **Augmentations**:
    - **Library**: Albumentations (Synchronized for RGB, Depth, Label) - Same as Exp011/Exp020
    - **Geometric**:
        - Resize (480x640)
        - HorizontalFlip (p=0.5)
        - ShiftScaleRotate (shift=0.05, scale=0.1, rotate=5, p=1.0)
        - RandomCrop (480x640, p=0.5)

## Results Summary
The model achieved its best performance at **Epoch 20**.

- **LB Score**: 0.57291
- **Best mIoU**: 0.6253 (Validation)
- **Pixel Accuracy**: 0.8184 (Validation)
- **Validation Loss**: 0.6627

### Per-Class IoU (Epoch 20)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.683 |
| 1 | 0.342 |
| 2 | 0.691 |
| 3 | 0.592 |
| 4 | 0.898 |
| 5 | 0.626 |
| 6 | 0.543 |
| 7 | 0.647 |
| 8 | 0.563 |
| 9 | 0.468 |
| 10 | 0.576 |
| 11 | 0.821 |
| 12 | 0.680 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.5556 | 1.1011 | 0.3481 | 0.6754 |
| 5 | 0.5140 | 0.6933 | 0.5274 | 0.7766 |
| 10 | 0.2822 | 0.6714 | 0.5649 | 0.7947 |
| 15 | 0.1745 | 0.6634 | 0.6077 | 0.8103 |
| **20** | **0.1294** | **0.6627** | **0.6253** | **0.8184** |
| 25 | 0.1122 | 0.6682 | 0.6225 | 0.8208 |
| 30 | 0.1066 | 0.6811 | 0.6211 | 0.8186 |

## Observations
- **Performance**: LB Score improved significantly to 0.57291 (compared to ~0.554 in Exp011/Exp020).
- **Validation mIoU**: Validation mIoU also saw a large jump to 0.6253 (vs ~0.59).
- **Model Architecture**: Changing the backbone from ResNet50 to ResNet101 had a strong positive impact. This suggests the model was somewhat underfitting with ResNet50 or needed more capacity to handle the complex 4-channel input effectively.
- **Early Fusion Efficacy**: Early fusion continues to work well, and with a deeper backbone, it outperforms the Late Fusion approach attempted in Exp020.
- **Class Performance**: Class 1 (0.342) shows improvement over Exp020 (0.256). Class 4 remains high (0.898).
- **Convergence**: Best performance was reached at Epoch 20, slightly earlier than typical (Exp020 peaked at 26). The model maintained high performance afterwards without severe overfitting, although validation loss plateaued/slighty increased after epoch 15.
