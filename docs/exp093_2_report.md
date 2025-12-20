# Exp093.2 HR + EMA + TTA + OOF Report

## Experiment Configuration
- **Script**: `base_model_093_2.py`
- **Model Architecture**: FPN with ConvNeXt Base encoder (Pretrained on ImageNet)
- **Input Size**: 720 (H) x 960 (W)
- **Crop Size**: 576 x 768
- **Classes**: 13 (NYUv2)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (50 epochs)
- **Batch Size**: 4
- **Loss Function**: CrossEntropyLoss + 0.1 * Depth L1 Loss
- **EMA**: Decay 0.999
- **TTA**: 10 Combinations (Scales: 0.5, 0.75, 1.0, 1.25, 1.5 × Horizontal Flip)

## Results Summary
The model was trained for 50 epochs with EMA and validated using TTA.

- **LB Score**: **0.69475**
- **OOF mIoU (Best T)**: **0.7027**
- **Best Temperature**: 1.0

### TTA Performance (OOF)
| Temperature | mIoU |
| :---: | :---: |
| 0.7 | 0.7019 |
| 0.8 | 0.7023 |
| 0.9 | 0.7026 |
| **1.0** | **0.7027** |

## Training Progression (Fold 0 Example)
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.5271 | 3.9635 | 0.0377 | 0.3056 |
| 10 | 0.1983 | 1.0560 | 0.6215 | 0.7938 |
| 20 | 0.0967 | 0.7915 | 0.6720 | 0.8407 |
| 30 | 0.0769 | 0.7302 | 0.6865 | 0.8507 |
| 40 | 0.0784 | 0.7672 | 0.6997 | 0.8601 |
| **46** | **0.0728** | **0.7731** | **0.7004** | **0.8622** |
| 50 | 0.0721 | 0.7858 | 0.7004 | 0.8621 |

## Observations
- **High Performance**: Achieved ~0.70 mIoU on OOF and ~0.695 on LB, showing strong generalization.
- **TTA Effectiveness**: TTA with temperature scaling showed that standard temperature `T=1.0` was optimal (mIoU 0.7027), though lower temperatures were very close.
- **Convergence**: Model continued to improve slowly until the very end (Epoch 46-50), with EMA likely stabilizing the weights.
- **Depth Multi-task**: The depth loss (`0.1 * L1`) was included, assisting in learning geometry-aware features.
