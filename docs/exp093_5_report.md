# Exp093.5 RGB-D 4-Channel Input Report

## Experiment Configuration
- **Script**: `base_model_093_5.py`
- **Model Architecture**: FPN with ConvNeXt Base encoder (Pretrained on ImageNet)
- **Input Channels**: 4 (RGB + Inverse Depth)
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

- **LB Score**: **0.69343**
- **OOF mIoU (Best T)**: **0.7160** (at T=0.7)
- **Best OOF Temperature**: 0.7

### TTA Performance (OOF)
| Temperature | mIoU |
| :---: | :---: |
| **0.7** | **0.7160** |
| 0.8 | 0.7159 |
| 0.9 | 0.7159 |
| 1.0 | 0.7158 |

## Training Progression (Fold 0 Example)
| Epoch | Train Loss | Valid Loss | Valid mIoU | Pixel Acc |
| :---: | :---: | :---: | :---: | :---: |
| ... | ... | ... | ... | ... |
| 48 | 0.0512 | 0.7306 | 0.7132 | 0.8675 |
| 49 | 0.0503 | 0.7334 | 0.7130 | 0.8675 |
| **50** | **0.0503** | **0.7359** | **0.7128** | **0.8674** |

## Observations
- **Overview**: This experiment utilized a 4-channel input, concatenating the Inverse Depth map with RGB to leverage geometric information directly in the early layers.
- **Performance**:
    - **LB**: 0.69343
    - **OOF**: 0.7160. This is the highest OOF score among the recent experiments.
- **Technique**:
    - **RGB-D Input**: The depth channel helps the model distinguish objects with similar textures but different depths.
    - **Robustness**: The improvement in OOF mIoU (0.7160) suggests that explicit depth input provides a strong signal for segmentation.
