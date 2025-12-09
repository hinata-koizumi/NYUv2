# Exp062 Report: Class Weighted Loss + Multi-Task Learning

## Experiment Configuration
- **Script**: `src/exp062.py`
- **Model Architecture**: Multi-Task DeepLabV3+
    - **Backbone**: ResNet101
    - **Input**: 4 channels (RGB + Inverse Depth)
- **Loss Function**:
    - **Segmentation**: `CrossEntropyLoss(weight=CLASS_WEIGHTS)`
    - **Depth**: `0.1 * L1Loss(Masked)`
- **Class Weights**: Used Sqrt Median Frequency Balancing to assign higher weights to underrepresented classes (e.g., Wall=2.46, Floor=1.48, Chair=1.04).

## Results Summary
The model achieved its best mIoU at **Epoch 27**.

- **LB Score**: 0.58118
- **Best mIoU**: 0.6337 (Validation)
- **Pixel Accuracy**: 0.8197 (Validation)
- **Validation Loss**: 0.8100

### Per-Class IoU (Epoch 27)
| Class ID | IoU | Change vs Exp060 |
| :---: | :---: | :---: |
| 0 | 0.740 | +0.033 |
| 1 (**Wall**) | 0.385 | +0.034 |
| 2 | 0.681 | +0.018 |
| 3 | 0.619 | -0.001 |
| 4 | 0.910 | +0.002 |
| 5 | 0.620 | -0.015 |
| 6 | 0.537 | -0.012 |
| 7 | 0.628 | +0.008 |
| 8 | 0.572 | +0.020 |
| 9 | 0.485 | +0.009 |
| 10 | 0.571 | +0.051 |
| 11 | 0.827 | +0.006 |
| 12 | 0.662 | -0.015 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.6782 | 1.2415 | 0.3970 | 0.6502 |
| 5 | 0.5596 | 0.8141 | 0.5433 | 0.7561 |
| 10 | 0.2921 | 0.7851 | 0.5734 | 0.7781 |
| 15 | 0.1816 | 0.7672 | 0.6119 | 0.8064 |
| 20 | 0.1387 | 0.7963 | 0.6289 | 0.8186 |
| **27** | **0.1182** | **0.8100** | **0.6337** | **0.8197** |

## Observations
- **Performance**: High Validation mIoU (0.6337) compared to Exp060 (0.6228), but LB score (0.58118) is lower than Exp060 (0.586). This suggests potential overfitting to the validation set or that the class weighting strategy doesn't generalize perfectly to the test distribution.
- **Class Improvements**: Significant improvements in minority classes like Wall (Class 1) and Class 10.
- **Trade-off**: While average mIoU increased on validation, the general pixel accuracy (0.8197) is slightly lower than Exp060 (0.8215), which is expected when correcting for imbalance (sacrificing majority class accuracy for minority class coverage).
