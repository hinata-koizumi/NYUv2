# Exp093.4 Boundary Loss + Class Weights Report

## Experiment Configuration
- **Script**: `base_model_093_4.py`
- **Model Architecture**: FPN with ConvNeXt Base encoder (Pretrained on ImageNet)
- **Input Size**: 720 (H) x 960 (W)
- **Crop Size**: 576 x 768
- **Classes**: 13 (NYUv2)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (50 epochs)
- **Batch Size**: 4
- **Loss Function**: BoundaryAwareCELoss + 0.1 * Depth L1 Loss
  - **Boundary Weight**: 2.0
  - **Class Weights**: 1.5x for Books (1), Table (9), TV (10)
- **EMA**: Decay 0.999
- **TTA**: 10 Combinations (Scales: 0.5, 0.75, 1.0, 1.25, 1.5 × Horizontal Flip)

## Results Summary
The model was trained for 50 epochs with EMA and validated using TTA.

- **LB Score**: **0.69163**
- **OOF mIoU (Best T)**: **0.7124** (at T=0.7)
- **Best OOF Temperature**: 0.7

### TTA Performance (OOF)
| Temperature | mIoU |
| :---: | :---: |
| **0.7** | **0.7124** |
| 0.8 | 0.7124 |
| 0.9 | 0.7124 |
| 1.0 | 0.7123 |

## Training Progression (Fold 0 Example)
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| ... | ... | ... | ... | ... |
| 48 | 0.0856 | 0.8396 | 0.7092 | 0.8628 |
| 49 | 0.0874 | 0.8427 | 0.7092 | 0.8628 |
| **50** | **0.0862** | **0.8453** | **0.7092** | **0.8628** |

## Observations
- **Overview**: This experiment introduced Boundary Loss and specific Class Weights to target difficult classes (Books, Table, TV).
- **Performance**:
    - **LB**: 0.69163
    - **OOF**: 0.7124, which is a strong result, though the LB is slightly lower than the generated OOF suggests.
- **Loss Strategy**:
    - **Boundary Aware**: Giving higher weight to edges (2.0x) aimed to improve segmentation sharpness.
    - **Class Balancing**: Weighting specific classes (Books, Table, TV) by 1.5x.
