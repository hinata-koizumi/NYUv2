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
    def __init__(self, cfg):
        super().__init__()
        self.ignore_index = cfg.IGNORE_INDEX
        self.num_classes = cfg.NUM_CLASSES
        self.mode = cfg.SEG_LOSS
        self.dice_weight = float(cfg.DICE_WEIGHT)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
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

def compute_metrics(cm: np.ndarray) -> Tuple[float, float, np.ndarray]:
    pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-10)
    inter = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = inter / union
    miou = np.nanmean(iou)
    return pixel_acc, miou, iou
