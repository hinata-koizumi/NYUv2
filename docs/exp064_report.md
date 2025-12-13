# Exp064 Report: Ensemble Search (Exp060 + Exp062 + Exp063)

## Experiment Configuration
- **Script**: `src/exp064.py`
- **Method**: Weighted Ensemble (Grid Search)
- **Models Ensembled**:
    1. **Exp060**: Multi-Task ResNet101 (Baseline)
    2. **Exp062**: Class Weighted Loss
    3. **Exp063**: Boundary Aware Loss
- **Search Strategy**: Evaluated predefined combinations of weights on the validation set to maximize mIoU.

## Results Summary
The ensemble found an optimal combination that significantly outperformed individual models.

- **LB Score**: 0.60274
- **Best mIoU**: 0.6461 (Validation)
- **Best Weights**: `(0.4, 0.3, 0.3)` corresponding to `0.4 * Exp060 + 0.3 * Exp062 + 0.3 * Exp063`

### Per-Class IoU (Best Ensemble)
| Class ID | IoU |
| :---: | :---: |
| 0 | 0.740 |
| 1 (**Wall**) | 0.375 |
| 2 | 0.696 |
| 3 | 0.641 |
| 4 | 0.914 |
| 5 | 0.644 |
| 6 | 0.559 |
| 7 | 0.646 |
| 8 | 0.578 |
| 9 | 0.510 |
| 10 | 0.594 |
| 11 | 0.832 |
| 12 | 0.672 |

## Observations
- **Performance**: The ensemble approach successfully pushed the validation mIoU to **0.6461**, a significant improvement over individual models (typically ~0.62-0.63).
- **LB Correlation**: The improvement translates well to the Leaderboard, breaking the 0.60 barrier with a score of **0.60274**.
- **Contribution**: All three models contribute roughly equally (0.4/0.3/0.3), suggesting that each model captures complementary information (Baseline structure, Minority classes from Exp062, Boundaries from Exp063).
