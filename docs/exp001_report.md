# Exp001 RGB Only CE Weighted Report

## Experiment Configuration
- **Script**: `src/exp001.py`
- **Model Architecture**: DeepLabV3+ with ResNet50 encoder (Pretrained on ImageNet)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss Function**: Weighted CrossEntropyLoss

### Class Weights
| Class ID | Weight |
| :---: | :---: |
| 0 | 0.0382 |
| 1 | 0.2609 |
| 2 | 0.0962 |
| 3 | 0.0436 |
| 4 | 0.0159 |
| 5 | 0.0104 |
| 6 | 0.0111 |
| 7 | 0.0889 |
| 8 | 0.0585 |
| 9 | 0.0491 |
| 10 | 0.2923 |
| 11 | 0.0063 |
| 12 | 0.0286 |

## Results Summary
The model achieved its best performance at **Epoch 27**.

- **LB Score**: 0.5259
- **Best mIoU**: 0.5600
- **Pixel Accuracy**: 0.7691
- **Validation Loss**: 1.1127

### Per-Class IoU (Epoch 27)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU | Diff vs Baseline (Exp000) |
| :---: | :---: | :---: |
| 0 | 0.644 | +0.008 |
| 1 | 0.371 | +0.070 |
| 2 | 0.623 | -0.007 |
| 3 | 0.536 | +0.007 |
| 4 | 0.845 | -0.006 |
| 5 | 0.545 | -0.018 |
| 6 | 0.467 | -0.016 |
| 7 | 0.580 | +0.005 |
| 8 | 0.540 | -0.008 |
| 9 | 0.380 | -0.002 |
| 10 | 0.404 | -0.062 |
| 11 | 0.764 | -0.027 |
| 12 | 0.581 | -0.042 |

*Positive diff indicates improvement over Exp000.*

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.8178 | 1.3453 | 0.2660 | 0.4745 |
| 5 | 0.4887 | 0.9138 | 0.4774 | 0.7113 |
| 10 | 0.2230 | 0.9755 | 0.5281 | 0.7464 |
| 15 | 0.1370 | 1.0280 | 0.5512 | 0.7619 |
| 20 | 0.1066 | 1.1026 | 0.5528 | 0.7651 |
| 25 | 0.0888 | 1.1391 | 0.5561 | 0.7668 |
| **27** | **0.0842** | **1.1127** | **0.5600** | **0.7691** |
| 30 | 0.0833 | 1.1449 | 0.5599 | 0.7679 |

## Observations
- The model reached peak mIoU at Epoch 27 with 0.5600, which is slightly lower than the baseline of 0.5676.
- **Class Imbalance Impact**:
    - Class 1 (Weight: 0.2609) saw a significant improvement in IoU (+0.070).
    - Class 10 (Weight: 0.2923) strangely saw a significant decrease in IoU (-0.062).
    - Class 4 (Weight: 0.0159) maintained high performance, dropping only slightly.
- **Validation Loss**: The validation loss is consistently higher than the baseline (around 1.1 vs 0.8), likely due to the weighted loss function amplifying errors in weighted classes, or simply the scale of the loss being different.
- **Overfitting**: Train loss decreases steadily, while Valid loss increases after around epoch 5-10, suggesting overfitting to the training data, despite the mIoU slowly improving until Epoch 27.
