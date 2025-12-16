# Exp093 Ensemble Report

## Experiment Configuration
- **Inference Script**: `src2/exp093_inference_ensemble.py`
- **Training Script**: `src2/exp093_fpn_convnextb_smartcrop.py`
- **Model Architecture**: FPN with ConvNeXt Base encoder (Pretrained on ImageNet)
- **Input Strategy**: 
    - Resize to 600x800
    - **Smart Crop** to 512x512 (Prob 0.5, targeting small objects: Books, Chair, Objects, Picture, TV)
- **Auxiliary Task**: Depth Estimation (Loss Lambda=0.1)
- **Classes**: 13
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (50 epochs)
- **Batch Size**: 8 (Train), 2 (Inference)
- **Ensemble**: 5-Fold Averaging

## Results Summary
The submission used an ensemble of 5 models trained with 5-fold cross-validation.

- **LB Score**: 0.67273
- **Best mIoU (Fold Avg)**: 0.6920
- **Pixel Accuracy**: 0.8522
- **Validation Loss**: 0.7059

### Per-Class IoU (Fold Avg)
| Class ID | IoU |
| :---: | :---: |
| 0 | 0.7961 |
| 1 | 0.3326 |
| 2 | 0.7456 |
| 3 | 0.7180 |
| 4 | 0.9181 |
| 5 | 0.6855 |
| 6 | 0.6147 |
| 7 | 0.6250 |
| 8 | 0.7221 |
| 9 | 0.5299 |
| 10 | 0.7193 |
| 11 | 0.8401 |
| 12 | 0.7496 |

### Key Features
1.  **Smart Crop**: Training involved looking for specific small object classes and actively cropping around them to improve small object segmentation performance.
2.  **Depth Supervision**: The model was multi-task, predicting depth alongside segmentation to encourage geometric understanding, though only segmentation was used for submission.
3.  **ConvNeXt Base**: A stronger backbone than the baseline ResNet50.

## Observations
- The ensemble achieved a significantly higher score (0.67273) compared to the baseline (0.52136).
- The use of Smart Crop likely contributed to better recognition of under-represented or small classes.
- Multi-task learning with depth likely helped the model generalize better on the geometric structure of indoor scenes.
