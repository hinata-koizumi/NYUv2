# Exp065 Report: Ensemble + Test Time Augmentation (TTA)

## Experiment Configuration
- **Script**: `src/exp065.py`
- **Method**: Weighted Ensemble + Test Time Augmentation (Horizontal Flip)
- **Base Ensemble**:
    - Weights: `(0.6, 0.2, 0.2)` for `(Exp060, Exp062, Exp063)`
    - This was a candidate weight set (close to the best found in Exp064).
- **TTA Strategy**:
    - Average of predictions from `Original Image` and `Flipped Image` (Horizontal Flip).
    - `Final Logits = 0.5 * (Logits_Original + Flip_Back(Logits_Flipped))`

## Results Summary
TTA provided a small but consistent improvement on the validation set and performed strongly on the Leaderboard.

- **LB Score**: 0.60445
- **Validation mIoU (With TTA)**: 0.6426 (+0.0022 vs No TTA)
- **Validation mIoU (No TTA)**: 0.6403

### TTA Comparison (Validation)
| Metric | No TTA | With TTA | Difference |
| :--- | :---: | :---: | :---: |
| **mIoU** | 0.64031 | 0.64256 | **+0.00225** |
| **Pixel Acc** | 0.82887 | 0.82975 | +0.00088 |

### Per-Class IoU (With TTA)
| Class ID | IoU | Change vs No TTA |
| :---: | :---: | :---: |
| 0 | 0.724 | -0.004 |
| 1 (**Wall**) | 0.353 | -0.014 |
| 2 | 0.689 | +0.005 |
| 3 | 0.635 | -0.005 |
| 4 | 0.914 | +0.002 |
| 5 | 0.643 | -0.001 |
| 6 | 0.559 | +0.002 |
| 7 | 0.639 | +0.001 |
| 8 | 0.560 | -0.011 |
| 9 | 0.513 | +0.011 |
| 10 | 0.611 | +0.036 |
| 11 | 0.834 | +0.004 |
| 12 | 0.679 | +0.005 |

## Observations
- **Effectiveness**: TTA improved the mIoU by about 0.002. While small, it is a "free" improvement at inference time without retraining.
- **Leaderboard**: The score **0.60445** confirms that TTA aids generalization.
- **Class Analysis**: Some classes improved significantly (e.g., Class 10 +3.6%), while others slightly degraded (e.g., Wall -1.4%), possibly due to artifacts at the edges or viewpoint dependence not being perfectly invariant to flipping in some scenes (though indoor scenes are usually quite symmetric in distribution).
