import numpy as np
from typing import Tuple


def update_confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int, ignore_index: int, cm: np.ndarray) -> np.ndarray:
    preds = preds.flatten()
    labels = labels.flatten()
    m = labels != ignore_index
    preds = preds[m]
    labels = labels[m]
    cm += np.bincount(num_classes * labels + preds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def compute_metrics(cm: np.ndarray, return_present_only: bool = False) -> Tuple[float, float, np.ndarray]:
    """
    Compute pixel accuracy, mIoU, and per-class IoU from confusion matrix.
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        return_present_only: If True, also return mIoU computed only over classes present in GT
        
    Returns:
        pixel_acc, miou_all, iou_array
        If return_present_only=True: pixel_acc, miou_all, iou_array, miou_present
    """
    pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-10)
    inter = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = inter / union
    
    # All-classes mIoU (includes classes not in GT as NaN)
    miou_all = np.nanmean(iou)
    
    if return_present_only:
        # Present-only mIoU (only classes with GT pixels > 0)
        gt_present = cm.sum(axis=1) > 0
        iou_present = iou[gt_present]
        miou_present = np.nanmean(iou_present) if len(iou_present) > 0 else 0.0
        return pixel_acc, miou_all, iou, miou_present
    
    return pixel_acc, miou_all, iou
