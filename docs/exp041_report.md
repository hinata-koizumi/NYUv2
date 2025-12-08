# Exp041 Report: RGB-D Early Fusion + Geometric Augmentation + ResNet101 + Log Depth

## Experiment Configuration
- **Script**: `src/exp041.py`
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
    - Normalized using **Logarithmic Depth** encoding:
        - `depth_log = np.log(depth + eps)`
        - `log_min = np.log(0.71)`
        - `log_max = np.log(10.0)`
        - `depth = (depth_log - log_min) / (log_max - log_min)`
- **Augmentations**:
    - **Library**: Albumentations (Synchronized for RGB, Depth, Label)
    - **Geometric**:
        - Resize (480x640)
        - HorizontalFlip (p=0.5)
        - ShiftScaleRotate (shift=0.05, scale=0.1, rotate=5, p=1.0)
        - RandomCrop (480x640, p=0.5)

## Results Summary
The model achieved its best performance at **Epoch 25**.

- **LB Score**: 0.57845
- **Best mIoU**: 0.6207 (Validation)
- **Pixel Accuracy**: 0.8182 (Validation)
- **Validation Loss**: 0.6752

### Per-Class IoU (Epoch 25)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.711 |
| 1 | 0.326 |
| 2 | 0.663 |
| 3 | 0.595 |
| 4 | 0.895 |
| 5 | 0.625 |
| 6 | 0.552 |
| 7 | 0.629 |
| 8 | 0.562 |
| 9 | 0.472 |
| 10 | 0.561 |
| 11 | 0.818 |
| 12 | 0.658 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.5629 | 1.1374 | 0.3534 | 0.6645 |
| 5 | 0.5265 | 0.7766 | 0.4939 | 0.7554 |
| 10 | 0.2520 | 0.6687 | 0.5841 | 0.8045 |
| 15 | 0.1652 | 0.6601 | 0.6062 | 0.8091 |
| 20 | 0.1230 | 0.6656 | 0.6136 | 0.8153 |
| **25** | **0.1089** | **0.6752** | **0.6207** | **0.8182** |
| 30 | 0.1028 | 0.6901 | 0.6165 | 0.8151 |

## Observations
- **Performance**: LB Score is 0.57845, which is lower than Exp040 (Inverse Depth, LB: 0.58571) and slightly higher than Exp030 (Linear Depth, LB: 0.57291).
- **Validation vs Test**: While the validation mIoU (0.6207) was slightly higher than Exp040 (0.6185), the LB score dropped. This suggests that Inverse Depth (Exp040) might generalize better to the test set or handle the specific depth distribution of the test set better than Log Depth.
- **Depth Encoding Comparison**:
    - **Linear (Exp030)**: LB 0.57291
    - **Inverse (Exp040)**: LB 0.58571 (Best)
    - **Log (Exp041)**: LB 0.57845
- **Conclusion**: Inverse depth encoding appears to be the most effective representation for this dataset and model configuration, likely because it emphasizes near-field details where semantic information is often densest.
