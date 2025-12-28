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

# Exp064 Report: Ensemble Search (Exp060 + Exp062 + Exp063)

## Experiment Configuration
- **Script**: `src/exp064.py`
- **Method**: Weighted Ensemble (Grid Search)
- **Models Ensembled**:
    1. **Exp060**: Multi-Task ResNet101 (Baseline)
    2. **Exp062**: Class Weighted Loss
    3. **Exp063**: Boundary Aware Loss
- **Search Strategy**: Evaluated predefined combinations of weights on the validation set to maximize mIoU.

## Results Summary
The ensemble found an optimal combination that significantly outperformed individual models.

- **LB Score**: 0.60274
- **Best mIoU**: 0.6461 (Validation)
- **Best Weights**: `(0.4, 0.3, 0.3)` corresponding to `0.4 * Exp060 + 0.3 * Exp062 + 0.3 * Exp063`

### Per-Class IoU (Best Ensemble)
| Class ID | IoU |
| :---: | :---: |
| 0 | 0.740 |
| 1 (**Wall**) | 0.375 |
| 2 | 0.696 |
| 3 | 0.641 |
| 4 | 0.914 |
| 5 | 0.644 |
| 6 | 0.559 |
| 7 | 0.646 |
| 8 | 0.578 |
| 9 | 0.510 |
| 10 | 0.594 |
| 11 | 0.832 |
| 12 | 0.672 |

## Observations
- **Performance**: The ensemble approach successfully pushed the validation mIoU to **0.6461**, a significant improvement over individual models (typically ~0.62-0.63).
- **LB Correlation**: The improvement translates well to the Leaderboard, breaking the 0.60 barrier with a score of **0.60274**.
- **Contribution**: All three models contribute roughly equally (0.4/0.3/0.3), suggesting that each model captures complementary information (Baseline structure, Minority classes from Exp062, Boundaries from Exp063).

# Exp065 Report: Ensemble + Test Time Augmentation (TTA)

## Experiment Configuration
- **Script**: `src/exp065.py`
- **Method**: Weighted Ensemble + Test Time Augmentation (Horizontal Flip)
- **Base Ensemble**:
    - Weights: `(0.6, 0.2, 0.2)` for `(Exp060, Exp062, Exp063)`
    - This was a candidate weight set (close to the best found in Exp064).
- **TTA Strategy**:
    - Average of predictions from `Original Image` and `Flipped Image` (Horizontal Flip).
    - `Final Logits = 0.5 * (Logits_Original + Flip_Back(Logits_Flipped))`

## Results Summary
TTA provided a small but consistent improvement on the validation set and performed strongly on the Leaderboard.

- **LB Score**: 0.60445
- **Validation mIoU (With TTA)**: 0.6426 (+0.0022 vs No TTA)
- **Validation mIoU (No TTA)**: 0.6403

### TTA Comparison (Validation)
| Metric | No TTA | With TTA | Difference |
| :--- | :---: | :---: | :---: |
| **mIoU** | 0.64031 | 0.64256 | **+0.00225** |
| **Pixel Acc** | 0.82887 | 0.82975 | +0.00088 |

### Per-Class IoU (With TTA)
| Class ID | IoU | Change vs No TTA |
| :---: | :---: | :---: |
| 0 | 0.724 | -0.004 |
| 1 (**Wall**) | 0.353 | -0.014 |
| 2 | 0.689 | +0.005 |
| 3 | 0.635 | -0.005 |
| 4 | 0.914 | +0.002 |
| 5 | 0.643 | -0.001 |
| 6 | 0.559 | +0.002 |
| 7 | 0.639 | +0.001 |
| 8 | 0.560 | -0.011 |
| 9 | 0.513 | +0.011 |
| 10 | 0.611 | +0.036 |
| 11 | 0.834 | +0.004 |
| 12 | 0.679 | +0.005 |

## Observations
- **Effectiveness**: TTA improved the mIoU by about 0.002. While small, it is a "free" improvement at inference time without retraining.
- **Leaderboard**: The score **0.60445** confirms that TTA aids generalization.
- **Class Analysis**: Some classes improved significantly (e.g., Class 10 +3.6%), while others slightly degraded (e.g., Wall -1.4%), possibly due to artifacts at the edges or viewpoint dependence not being perfectly invariant to flipping in some scenes (though indoor scenes are usually quite symmetric in distribution).

# Exp066 Report: Post-Processing (Noise Removal & Hole Filling)

## Experiment Configuration
- **Script**: `src/exp066.py`
- **Method**: Morphological Post-Processing on top of Ensemble + TTA.
- **Base Model**: Best Ensemble (Exp060, 062, 063) with weights `(0.6, 0.2, 0.2)` + TTA (Flip).
- **Post-Processing Steps**:
    1. **Remove Small Objects (Noise Removal)**:
        - Applied to: Books, Chair, Objects, Picture, TV.
        - Threshold: `min_size = 100` pixels.
        - Strategy: Replced with Background/0.
    2. **Fill Holes**:
        - Applied to: Ceiling, Floor, Wall.
        - Kernel: 3x3.
        - Strategy: Morphological Closing (Dilation -> Erosion) to fill small gaps.

## Results Summary
The post-processing steps did not provide a gain and slightly degraded performance, suggesting the model's predictions were already quite spatially coherent or that the heuristic parameters (size 100, kernel 3) were suboptimal.

- **LB Score**: 0.60425 (Lower than Exp065's 0.60445)
- **Validation mIoU (Post-Process)**: 0.6424 (-0.0002 vs Base)
- **Validation mIoU (Base)**: 0.6426

### Impact Analysis (Validation)
| Metric | Base (Ens + TTA) | Post-Process | Difference |
| :--- | :---: | :---: | :---: |
| **mIoU** | 0.64256 | 0.64240 | **-0.00016** |
| **Pixel Acc** | 0.82975 | 0.82969 | -0.00006 |

### Per-Class IoU (Post-Process)
| Class ID | IoU | Change vs Base |
| :---: | :---: | :---: |
| 0 | 0.722 | -0.002 |
| 1 (**Wall**) | 0.353 | +0.000 |
| 2 | 0.689 | -0.000 |
| 3 | 0.635 | +0.000 |
| 4 | 0.914 | -0.000 |
| 5 | 0.643 | +0.000 |
| 6 | 0.559 | -0.000 |
| 7 | 0.639 | +0.000 |
| 8 | 0.560 | +0.000 |
| 9 | 0.513 | +0.000 |
| 10 (**TV**) | 0.611 | +0.000 |
| 11 | 0.834 | -0.000 |
| 12 | 0.679 | -0.000 |

## Observations
- **Negative Impact**: The changes were extremely minimal but consistently negative.
- **Cause**: The "Remove Small Objects" step might have removed valid small instances (e.g., small objects in the distance), or the "Fill Holes" might have over-smoothed boundaries.
- **Conclusion**: Heuristic post-processing is risky without very careful tuning per class. Advanced methods like CRF (Conditional Random Fields) or refining the model training (like Boundary Aware Loss in Exp063) are generally more robust than simple morphological operations.
