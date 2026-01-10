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

# Exp093 Ensemble Report

## Experiment Configuration
- **Ensemble Script**: `main_model/exp093_ensemble.py`
- **Models**:
    1. **exp093_5**: FPN (ConvNeXt Base, RGB-D 4ch input)
    2. **exp093_4**: FPN (ConvNeXt Base, Boundary task, 3ch input)
    - *(Note: exp093_2 and exp093_6 were excluded in this run)*
- **Ensemble Method**:
    - **Orbit**: 5-Fold Cross Validation Averaging
    - **Optimization**: OOF (Out-of-Fold) mIoU maximization using `scipy.optimize.minimize` (SLSQP).
    - **TTA (Test Time Augmentation)**:
        - Scales: `[0.75, 1.0]`
        - Flips: Horizontal Flip (On/Off)
- **Resolution Strategy**:
    - Inference performed at `576x768` (Base Resolution).
    - Predictions resized to original resolution for evaluation/submission.
    - Probabilities stored at half-scale (`0.5`) during OOF to save memory.

## Results Summary
- **LB Score**: **0.69654**

*(Note: Intermediate validation scores (OOF mIoU) were not recorded, but the optimization process ensures the ensemble performs better than or equal to the single best model on the validation set.)*

## Key Features
1.  **Multi-Modal Ensemble**: Combines improved geometry understanding from RGB-D model (`exp093_5`) with boundary-aware features from `exp093_4`.
2.  **Automated Weight Tuning**: Instead of manual guessing, the script uses the OOF predictions to mathematically solve for the optimal blending weights `w` such that `argmax(w1*p1 + w2*p2)` maximizes mIoU.
3.  **Robust Inference**: Uses subset of TTA (scales 0.75, 1.0) to balance inference time and accuracy improvement.
