import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

# =========================
# Loss
# =========================
def _flatten_probas_and_targets(
    probas: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # probas: (B,C,H,W) after softmax
    # targets: (B,H,W)
    b, c, h, w = probas.shape
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
    targets = targets.contiguous().view(-1)
    valid = targets != ignore_index
    return probas[valid], targets[valid]


def soft_dice_loss_present_only(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    probas = F.softmax(logits, dim=1)
    probas_flat, targets_flat = _flatten_probas_and_targets(probas, targets, ignore_index)

    if probas_flat.numel() == 0:
        return logits.new_tensor(0.0)

    targets_oh = F.one_hot(targets_flat, num_classes=num_classes).to(dtype=probas_flat.dtype)

    intersection = (probas_flat * targets_oh).sum(dim=0)
    cardinality = probas_flat.sum(dim=0) + targets_oh.sum(dim=0)
    dice = (2.0 * intersection + eps) / (cardinality + eps)

    present = targets_oh.sum(dim=0) > 0
    if not present.any():
        return logits.new_tensor(0.0)

    return 1.0 - dice[present].mean()


class CombinedSegLoss(nn.Module):
    def __init__(self, cfg, class_weights=None):
        """
        Args:
            cfg: Config object
            class_weights: Optional tensor of shape (num_classes,) for class weighting
        """
        super().__init__()
        self.ignore_index = cfg.IGNORE_INDEX
        self.num_classes = cfg.NUM_CLASSES
        self.mode = cfg.SEG_LOSS
        self.dice_weight = float(cfg.DICE_WEIGHT)
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, ignore_index=self.ignore_index)
        if self.mode == "ce":
            return ce
        if self.mode == "ce_dice":
            dice = soft_dice_loss_present_only(
                logits=logits,
                targets=targets,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )
            return ce + self.dice_weight * dice
        raise ValueError(f"Unknown SEG_LOSS: {self.mode}")


class BoundaryAwareCELoss(nn.Module):
    """
    Boundary-aware Cross Entropy (docs/exp09_log.md Exp093.4).
    Weights pixels on label edges by `boundary_weight` (default 2.0).
    Supports optional class weights.
    """
    def __init__(self, num_classes: int, ignore_index: int, class_weights=None, boundary_weight: float = 1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.boundary_weight = float(boundary_weight)
        self.class_weights = class_weights

    @staticmethod
    def _edge_mask_from_labels(y: torch.Tensor, ignore_index: int) -> torch.Tensor:
        """
        y: (B,H,W) long
        returns: (B,H,W) float mask in {0,1}, edges=1
        """
        # Ignore pixels are set to a sentinel that won't create false edges
        y_work = y.clone()
        y_work[y_work == ignore_index] = -1
        # 3x3 neighborhood min/max; if differs -> boundary
        # Use pooling on float
        yf = y_work.unsqueeze(1).to(dtype=torch.float32)
        y_max = torch.nn.functional.max_pool2d(yf, kernel_size=3, stride=1, padding=1)
        y_min = -torch.nn.functional.max_pool2d(-yf, kernel_size=3, stride=1, padding=1)
        edge = (y_max != y_min).to(dtype=torch.float32).squeeze(1)
        # Do not weight ignore pixels
        edge = edge * (y != ignore_index).to(dtype=torch.float32)
        return edge

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-pixel CE
        ce = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B,H,W)
        if self.boundary_weight <= 1.0:
            # mean over non-ignore pixels (CE already 0 on ignore? No, CE sets to 0? It sets arbitrary; so mask)
            m = (targets != self.ignore_index).to(dtype=ce.dtype)
            return (ce * m).sum() / (m.sum() + 1e-6)

        edge = self._edge_mask_from_labels(targets, self.ignore_index)
        pixel_w = 1.0 + (self.boundary_weight - 1.0) * edge
        m = (targets != self.ignore_index).to(dtype=ce.dtype)
        loss = (ce * pixel_w * m).sum() / (m.sum() + 1e-6)
        return loss

# =========================
# Metrics
# =========================
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
