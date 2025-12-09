# Exp080 Report: Multi-Task Learning (Seg + Depth + Edge) + ResNet101

## Experiment Configuration
- **Script**: `src/exp080.py`
- **Model Architecture**: DeepLabV3+
    - **Backbone**: ResNet101 (Pretrained on ImageNet, in_channels=4)
    - **Encoder**: ResNet101
    - **Input**: 4 channels (RGB + Inverse Depth)
    - **Auxiliary Heads**:
        - **Depth Head**: Predicts depth map (Regresssion).
        - **Edge Head**: Predicts binary edge map (Classification).
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss**:
    - **Segmentation**: CrossEntropyLoss (IGNORE_INDEX=255)
    - **Depth**: L1 Loss (Masked, Lambda=0.1)
    - **Edge**: BCEWithLogitsLoss (Lambda=1.0)
    - **Total**: Seg + 0.1 * Depth + 1.0 * Edge
- **Data Processing**:
    - **Depth Input**: Inverse Depth normalized [0, 1].
    - **Edge Target**: Generated on-the-fly from segmentation labels using morphological erosion (kernel 3x3).
- **Augmentations**:
    - **Library**: Albumentations
    - **Geometric**:
        - Resize (480x640)
        - HorizontalFlip (p=0.5)
        - ShiftScaleRotate (shift=0.05, scale=0.1, rotate=5, p=1.0)
        - RandomCrop (480x640, p=0.5)

## Results Summary
The model achieved its best mIoU at **Epoch 25**.

- **LB Score**: 0.58272
- **Best mIoU**: 0.6245 (Validation)
- **Pixel Accuracy**: 0.8222 (Validation)
- **Validation Loss**: 0.8028 (Total), 0.6644 (Seg Only)

### Per-Class IoU (Epoch 25)

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.694 |
| 1 | 0.347 |
| 2 | 0.709 |
| 3 | 0.599 |
| 4 | 0.905 |
| 5 | 0.629 |
| 6 | 0.559 |
| 7 | 0.653 |
| 8 | 0.527 |
| 9 | 0.451 |
| 10 | 0.534 |
| 11 | 0.830 |
| 12 | 0.680 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.8209 | 1.2510 | 0.3310 | 0.6761 |
| 5 | 0.6718 | 0.8572 | 0.5208 | 0.7718 |
| 10 | 0.4024 | 0.7948 | 0.5885 | 0.8060 |
| 15 | 0.3056 | 0.7937 | 0.6079 | 0.8137 |
| 20 | 0.2636 | 0.7918 | 0.6208 | 0.8189 |
| **25** | **0.2461** | **0.8028** | **0.6245** | **0.8222** |
| 30 | 0.2407 | 0.7967 | 0.6236 | 0.8229 |

## Observations
- **Performance**: The LB score (0.58272) is slightly lower than Exp060 (0.586) but validation mIoU (0.6245) is higher than Exp070 (0.6191) and Exp060 (0.6214 - from prev reports).
- **Edge Task**: Adding edge detection as an auxiliary task provided a strong signal, pushing validation metrics higher. However, the LB score didn't improve proportionally, suggesting potential overfitting to the specific edge generation method or that the edge boundaries in the test set (which might be cleaner or different) are not perfectly aligned with the training set's noisy label boundaries.
- **Validation Loss**: The total validation loss is higher (~0.80) compared to Exp070 (~0.68) because it includes the Edge Loss component (Lambda=1.0). The Segmentation part of the loss (~0.66) is comparable.
