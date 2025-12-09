# Exp061 Report: Soft IoU Loss + Multi-Task Learning

## Experiment Configuration
- **Script**: `src/exp061.py`
- **Model Architecture**: Multi-Task DeepLabV3+
    - **Backbone**: ResNet101 (Pretrained on ImageNet, in_channels=4)
    - **Input**: 4 channels (RGB + Inverse Depth)
    - **Heads**:
        1. **Segmentation**: DeepLabV3+ Head
        2. **Depth**: Auxiliary Head (1 channel)
- **Loss Function**:
    - **Segmentation**: `CrossEntropyLoss + 0.5 * SoftIoULoss`
    - **Depth**: `0.1 * L1Loss(Masked)`
    - **Total**: `Loss_Seg + 0.1 * Loss_Depth`
- **Why Soft IoU?**: To directly optimize the metric we care about (IoU) and potentially handle class imbalance better than CE alone.

## Results Summary
The model achieved its best mIoU at **Epoch 28**.

- **LB Score**: 0.5833
- **Best mIoU**: 0.6204 (Validation)
- **Pixel Accuracy**: 0.8205 (Validation)
- **Validation Loss**: 1.0496 (Combined)

### Per-Class IoU (Epoch 28)
| Class ID | IoU |
| :---: | :---: |
| 0 | 0.716 |
| 1 | 0.358 |
| 2 | 0.673 |
| 3 | 0.602 |
| 4 | 0.909 |
| 5 | 0.625 |
| 6 | 0.550 |
| 7 | 0.637 |
| 8 | 0.547 |
| 9 | 0.446 |
| 10 | 0.512 |
| 11 | 0.828 |
| 12 | 0.660 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 2.0200 | 1.5970 | 0.3470 | 0.6572 |
| 5 | 0.8734 | 1.1123 | 0.5052 | 0.7652 |
| 10 | 0.5647 | 1.0313 | 0.5873 | 0.8000 |
| 15 | 0.4431 | 1.0481 | 0.6062 | 0.8105 |
| 20 | 0.3944 | 1.0313 | 0.6186 | 0.8208 |
| **28** | **0.3658** | **1.0496** | **0.6204** | **0.8205** |

## Observations
- **Performance**: The LB score (0.5833) is comparable to Exp050 (0.5836) and slightly lower than Exp060 (0.5862).
- **Soft IoU Effect**: Adding Soft IoU Loss did not significantly boost the performance compared to standard CE Loss (Exp060).
- **Loss Magnitude**: The loss values are higher than previous experiments because of the additional `0.5 * SoftIoU` term (where SoftIoU ≈ 0.4-0.6).
- **Class 1 (Wall)**: 0.358 vs 0.351 (Exp060). Slight improvement.
- **Class 9 (Chair)**: 0.446 vs 0.476 (Exp060). Decrease.
