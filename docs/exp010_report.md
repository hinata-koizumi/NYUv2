# Exp010 Report: RGB-D Early Fusion

## Experiment Configuration
- **Script**: `src/exp010.py`
- **Model Architecture**: DeepLabV3+ with ResNet50 encoder (Pretrained on ImageNet)
    - **Input Channels**: 4 (RGB + Depth)
    - **Fusion**: Early Fusion (Depth concatenated as 4th channel)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8
- **Loss**: CrossEntropyLoss (No class weights, IGNORE_INDEX=255)
- **Depth Processing**:
    - Clipped to [0.71m, 10.0m]
    - Normalized to [0, 1] relative to min/max range

## Results Summary
The model achieved its best performance at **Epoch 17**.

- **LB Score**: 0.53647
- **Best mIoU**: 0.5771
- **Pixel Accuracy**: 0.7879
- **Validation Loss**: 0.7589

### Per-Class IoU (Epoch 17)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.645 |
| 1 | 0.383 |
| 2 | 0.661 |
| 3 | 0.538 |
| 4 | 0.879 |
| 5 | 0.567 |
| 6 | 0.493 |
| 7 | 0.568 |
| 8 | 0.531 |
| 9 | 0.365 |
| 10 | 0.443 |
| 11 | 0.803 |
| 12 | 0.628 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.6172 | 1.1479 | 0.3198 | 0.6501 |
| 5 | 0.4366 | 0.7740 | 0.5075 | 0.7546 |
| 10 | 0.1994 | 0.7531 | 0.5650 | 0.7754 |
| 15 | 0.1230 | 0.7586 | 0.5703 | 0.7858 |
| **17** | **0.1074** | **0.7589** | **0.5771** | **0.7879** |
| 20 | 0.0944 | 0.7912 | 0.5651 | 0.7840 |
| 25 | 0.0803 | 0.7889 | 0.5735 | 0.7868 |
| 30 | 0.0747 | 0.7912 | 0.5752 | 0.7874 |

## Observations
- **Performance**: The RGB-D early fusion model (LB 0.5365) outperformed the RGB-only baseline (LB 0.5214), confirming the value of depth information.
- **Best Epoch**: Peak performance was reached at Epoch 17, similar to the baseline.
- **Overfitting**: Validation loss and mIoU fluctuated after Epoch 17, with loss slightly increasing, indicating potential overfitting in later epochs similar to the baseline.
- **Class Performance**: Class 4 remains the highest performing class (0.879), further improved from baseline (0.851). Most classes showed improvement.
