# Exp060 Report: Multi-Task Learning (Segmentation + Depth) + ResNet101

## Experiment Configuration
- **Script**: `src/exp060.py`
- **Model Architecture**: Multi-Task DeepLabV3+
    - **Backbone**: ResNet101 (Pretrained on ImageNet, in_channels=4)
    - **Encoder**: Shared ResNet101
    - **Heads**:
        1. **Segmentation**: Standard DeepLabV3+ Head (13 classes)
        2. **Depth**: Auxiliary Convolutional Head (1 channel)
    - **Input**: 4 channels (RGB + Inverse Depth)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss**: Combined Loss
    - **Segmentation**: CrossEntropyLoss (IGNORE_INDEX=255)
    - **Depth**: L1 Loss (Masked)
    - **Formula**: `Loss = Loss_Seg + 0.1 * Loss_Depth`
- **Depth Processing**:
    - **Input**: Normalized using **Inverse Depth** encoding (same as Exp040).
    - **Target**: Linear normalized depth [0, 1] for auxiliary task.
- **Augmentations**:
    - **Library**: Albumentations
    - **Geometric**:
        - Resize (480x640)
        - HorizontalFlip (p=0.5)
        - ShiftScaleRotate (shift=0.05, scale=0.1, rotate=5, p=1.0)
        - RandomCrop (480x640, p=0.5)

## Results Summary
The model achieved its best performance at **Epoch 22**.

- **LB Score**: 0.58623
- **Best mIoU**: 0.6228 (Validation)
- **Pixel Accuracy**: 0.8215 (Validation)
- **Validation Loss**: 0.6647 (Combined)

### Per-Class IoU (Epoch 22)

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.707 |
| 1 | 0.351 |
| 2 | 0.663 |
| 3 | 0.620 |
| 4 | 0.908 |
| 5 | 0.635 |
| 6 | 0.549 |
| 7 | 0.620 |
| 8 | 0.552 |
| 9 | 0.476 |
| 10 | 0.520 |
| 11 | 0.821 |
| 12 | 0.677 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.5668 | 1.1412 | 0.3455 | 0.6634 |
| 5 | 0.5194 | 0.7312 | 0.5221 | 0.7635 |
| 10 | 0.2769 | 0.6538 | 0.5898 | 0.8029 |
| 15 | 0.1745 | 0.6666 | 0.6020 | 0.8125 |
| 20 | 0.1349 | 0.6684 | 0.6124 | 0.8190 |
| **22** | **0.1256** | **0.6647** | **0.6228** | **0.8215** |
| 25 | 0.1173 | 0.6676 | 0.6151 | 0.8208 |
| 30 | 0.1113 | 0.6720 | 0.6126 | 0.8205 |

## Observations
- **Performance**: The LB score (0.58623) is the highest among recent experiments (Exp050: 0.58362, Exp040: 0.58571).
- **Multi-Task Learning**: Adding an auxiliary depth prediction task (`Loss_Depth` weight=0.1) seems to have a positive regularizing effect or helps the shared encoder learn more robust geometric features.
- **Class Performance**:
    - Class 1 (0.351) is still challenging but consistent with previous runs (Exp050: 0.370).
    - Class 4 (0.908) remains strong.
    - Class 12 (0.677) shows solid performance.
- **Overfitting**: The gap between training loss (0.1256) and validation loss (0.6647) is significant, typical of these ResNet101 models on this dataset, but the multi-task constraint might be preventing it from degrading performance compared to single-task models.
