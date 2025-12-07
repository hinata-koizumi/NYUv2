# Exp020 Report: RGB-D Late Fusion + Geometric Augmentation

## Experiment Configuration
- **Script**: `src/exp020.py`
- **Model Architecture**: Late Fusion DeepLab
    - **RGB Branch**: ResNet50 (Pretrained on ImageNet, in_channels=3)
    - **Depth Branch**: ResNet18 (Random Initialization, in_channels=1)
    - **Fusion**: Late Fusion (Concatenation of logits + 1x1 Conv)
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
    - **Library**: Albumentations (Synchronized for RGB, Depth, Label) - Same as Exp011
    - **Geometric**:
        - Resize (480x640)
        - HorizontalFlip (p=0.5)
        - ShiftScaleRotate (shift=0.05, scale=0.1, rotate=5, p=1.0)
        - RandomCrop (480x640, p=0.5)

## Results Summary
The model achieved its best performance at **Epoch 26**.

- **LB Score**: 0.55438
- **Best mIoU**: 0.5881
- **Pixel Accuracy**: 0.8000
- **Validation Loss**: 0.7135

### Per-Class IoU (Epoch 26)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.678 |
| 1 | 0.256 |
| 2 | 0.704 |
| 3 | 0.582 |
| 4 | 0.911 |
| 5 | 0.591 |
| 6 | 0.502 |
| 7 | 0.599 |
| 8 | 0.538 |
| 9 | 0.426 |
| 10 | 0.416 |
| 11 | 0.811 |
| 12 | 0.631 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.7929 | 1.3505 | 0.2308 | 0.6103 |
| 5 | 0.6359 | 0.8232 | 0.4714 | 0.7437 |
| 10 | 0.3147 | 0.7132 | 0.5287 | 0.7819 |
| 15 | 0.2064 | 0.7050 | 0.5714 | 0.7897 |
| 20 | 0.1535 | 0.7053 | 0.5861 | 0.7989 |
| 25 | 0.1321 | 0.7269 | 0.5823 | 0.7966 |
| **26** | **0.1308** | **0.7135** | **0.5881** | **0.8000** |
| 30 | 0.1255 | 0.7087 | 0.5856 | 0.8012 |

## Observations
- **Performance**: LB Score (0.55438) is virtually identical to Exp011 (0.55426). The difference is negligible.
- **Model Architecture**: Late Fusion (separate encoders) did not provide a significant advantage over Early Fusion (ResNet50 with 4 input channels) in this setting. 
- **Complexity**: Late Fusion introduces more parameters and complexity (two backbones), making it computationally heavier than Early Fusion.
- **Class Performance**: Class 4 (Floor?) remains very high at 0.911. Class 1 is notably lower (0.256) compared to Exp011 (0.320).
- **Validation vs Test**: While Validation mIoU was slightly lower than Exp011 (0.5881 vs 0.5945), the Test set performance (LB) was sustained, suggesting robust generalization or that the validation set differences were within margin of error.
