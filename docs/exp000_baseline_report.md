# Exp000 Baseline Report

## Experiment Configuration
- **Script**: `src/exp000_baseline.py`
- **Model Architecture**: DeepLabV3+ with ResNet50 encoder (Pretrained on ImageNet)
- **Input Size**: 480 (H) x 640 (W)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (30 epochs)
- **Batch Size**: 8

## Results Summary
The model achieved its best performance at **Epoch 17**.

- **LB Score**: 0.52136
- **Best mIoU**: 0.5676
- **Pixel Accuracy**: 0.7816
- **Validation Loss**: 0.7991

### Per-Class IoU (Epoch 17)
*Note: Class mapping was not available in the source code.*

| Class ID | IoU |
| :---: | :---: |
| 0 | 0.636 |
| 1 | 0.301 |
| 2 | 0.630 |
| 3 | 0.529 |
| 4 | 0.851 |
| 5 | 0.563 |
| 6 | 0.483 |
| 7 | 0.575 |
| 8 | 0.548 |
| 9 | 0.382 |
| 10 | 0.466 |
| 11 | 0.791 |
| 12 | 0.623 |

## Training Progression
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.6204 | 1.1514 | 0.3342 | 0.6488 |
| 5 | 0.4307 | 0.7921 | 0.5105 | 0.7518 |
| 10 | 0.2048 | 0.7947 | 0.5419 | 0.7627 |
| 15 | 0.1270 | 0.8042 | 0.5616 | 0.7776 |
| **17** | **0.1083** | **0.7991** | **0.5676** | **0.7816** |
| 20 | 0.0970 | 0.8410 | 0.5484 | 0.7763 |
| 25 | 0.0811 | 0.8341 | 0.5598 | 0.7793 |
| 30 | 0.0749 | 0.8392 | 0.5614 | 0.7794 |

## Observations
- The model reached peak mIoU at Epoch 17.
- Validation loss started to plateau and slightly increase after Epoch 17, suggesting potential overfitting, although mIoU remained relatively stable around 0.56.
- Class 4 achieved the highest IoU (0.851), while Class 1 had the lowest (0.301).
