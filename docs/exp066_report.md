# Exp066 Report: Post-Processing (Noise Removal & Hole Filling)

## Experiment Configuration
- **Script**: `src/exp066.py`
- **Method**: Morphological Post-Processing on top of Ensemble + TTA.
- **Base Model**: Best Ensemble (Exp060, 062, 063) with weights `(0.6, 0.2, 0.2)` + TTA (Flip).
- **Post-Processing Steps**:
    1. **Remove Small Objects (Noise Removal)**:
        - Applied to: Books, Chair, Objects, Picture, TV.
        - Threshold: `min_size = 100` pixels.
        - Strategy: Replced with Background/0.
    2. **Fill Holes**:
        - Applied to: Ceiling, Floor, Wall.
        - Kernel: 3x3.
        - Strategy: Morphological Closing (Dilation -> Erosion) to fill small gaps.

## Results Summary
The post-processing steps did not provide a gain and slightly degraded performance, suggesting the model's predictions were already quite spatially coherent or that the heuristic parameters (size 100, kernel 3) were suboptimal.

- **LB Score**: 0.60425 (Lower than Exp065's 0.60445)
- **Validation mIoU (Post-Process)**: 0.6424 (-0.0002 vs Base)
- **Validation mIoU (Base)**: 0.6426

### Impact Analysis (Validation)
| Metric | Base (Ens + TTA) | Post-Process | Difference |
| :--- | :---: | :---: | :---: |
| **mIoU** | 0.64256 | 0.64240 | **-0.00016** |
| **Pixel Acc** | 0.82975 | 0.82969 | -0.00006 |

### Per-Class IoU (Post-Process)
| Class ID | IoU | Change vs Base |
| :---: | :---: | :---: |
| 0 | 0.722 | -0.002 |
| 1 (**Wall**) | 0.353 | +0.000 |
| 2 | 0.689 | -0.000 |
| 3 | 0.635 | +0.000 |
| 4 | 0.914 | -0.000 |
| 5 | 0.643 | +0.000 |
| 6 | 0.559 | -0.000 |
| 7 | 0.639 | +0.000 |
| 8 | 0.560 | +0.000 |
| 9 | 0.513 | +0.000 |
| 10 (**TV**) | 0.611 | +0.000 |
| 11 | 0.834 | -0.000 |
| 12 | 0.679 | -0.000 |

## Observations
- **Negative Impact**: The changes were extremely minimal but consistently negative.
- **Cause**: The "Remove Small Objects" step might have removed valid small instances (e.g., small objects in the distance), or the "Fill Holes" might have over-smoothed boundaries.
- **Conclusion**: Heuristic post-processing is risky without very careful tuning per class. Advanced methods like CRF (Conditional Random Fields) or refining the model training (like Boundary Aware Loss in Exp063) are generally more robust than simple morphological operations.
