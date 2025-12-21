# Exp093.6 DeepLabV3+ ConvNeXt Report

## Experiment Configuration
- **Script**: `base_model_093_6.py`
- **Model Architecture**: DeepLabV3Plus with ConvNeXt Base encoder (Pretrained on ImageNet)
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

- **LB Score**: **0.69195**
- **OOF mIoU (Best T)**: **0.7127** (at T=0.7)
- **Best OOF Temperature**: 0.7

### TTA Performance (OOF)
| Temperature | mIoU |
| :---: | :---: |
| **0.7** | **0.7127** |
| 0.8 | 0.7126 |
| 0.9 | 0.7125 |
| 1.0 | 0.7124 |

## Observations
- **Overview**: This experiment replaced the FPN decoder with a DeepLabV3+ decoder to better capture multi-scale context using Atrous Spatial Pyramid Pooling (ASPP).
- **Performance**:
    - **LB**: 0.69195
    - **OOF**: 0.7127.
- **Architecture**:
    - **DeepLabV3+**: Theoretically captures sharper object boundaries and context. The performance is competitive with FPN (exp093_4) but slightly lower than the RGB-D approach (exp093_5).
