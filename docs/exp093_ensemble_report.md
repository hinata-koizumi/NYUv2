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
