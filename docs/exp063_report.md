# Exp063 Report: Boundary Aware Loss + Multi-Task Learning

## Experiment Configuration
- **Script**: `src/exp063.py`
- **Model Architecture**: Multi-Task DeepLabV3+
    - **Backbone**: ResNet101
    - **Input**: 4 channels (RGB + Inverse Depth)
- **Loss Function**:
    - **Segmentation**: `CrossEntropyLoss` with **Boundary Weighting**.
    - **Depth**: `0.1 * L1Loss(Masked)`
- **Boundary Weighting**: Pixels at detected edges (using 3x3 min/max pooling on labels) are weighted by 2.0, while others are weighted 1.0. This emphasizes learning sharp transitions between semantic classes.

## Results Summary
The model achieved its best mIoU at **Epoch 28**.

- **LB Score**: 0.58433
- **Best mIoU**: 0.6267 (Validation)
- **Pixel Accuracy**: 0.8215 (Validation)
- **Validation Loss**: 0.7126

### Per-Class IoU (Epoch 28)
| Class ID | IoU | Change vs Exp060 |
| :---: | :---: | :---: |
| 0 | 0.724 | +0.017 |
| 1 (**Wall**) | 0.332 | -0.019 |
| 2 | 0.681 | +0.018 |
| 3 | 0.601 | -0.019 |
| 4 | 0.904 | -0.004 |
| 5 | 0.631 | -0.004 |
| 6 | 0.540 | -0.009 |
| 7 | 0.632 | +0.012 |
| 8 | 0.563 | +0.011 |
| 9 | 0.493 | +0.017 |
| 10 | 0.562 | +0.042 |
| 11 | 0.827 | +0.006 |
| 12 | 0.657 | -0.020 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.5709 | 1.1727 | 0.3523 | 0.6616 |
| 5 | 0.5481 | 0.7370 | 0.4990 | 0.7705 |
| 10 | 0.2868 | 0.6966 | 0.5811 | 0.7984 |
| 15 | 0.1907 | 0.6927 | 0.6051 | 0.8134 |
| 20 | 0.1511 | 0.6994 | 0.6203 | 0.8175 |
| **28** | **0.1278** | **0.7126** | **0.6267** | **0.8215** |

## Observations
- **Performance**: This experiment yielded a competitive LB score (0.58433), very close to the best Exp060 (0.58623).
- **Validation Comparison**: Valid mIoU (0.6267) is slightly better than Exp060 (0.6228), and Pixel Accuracy (0.8215) is identical.
- **Boundary Effect**: The boundary weighting likely helped edge precision, contributing to the higher mIoU. However, it negatively impacted the wide-area "Wall" class (IoU dropped to 0.332), possibly because the internal texture of walls is less emphasized compared to their edges with floors/ceilings.
- **Conclusion**: Boundary awareness is a strong candidate for fine-tuning, but might need to be balanced (e.g., lower weight or combined with class weighting) to avoid degrading performance on large homogenous regions like walls.
