import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from configs import default as config
except ImportError:
    import configs.default as config


class ModelCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(config.WEIGHTS_LIST).float()
        self.planar_enabled = bool(getattr(config, "PLANAR_HEAD_ENABLE", False))
        self.planar_weight = float(getattr(config, "PLANAR_LOSS_WEIGHT", 0.0))

    def forward(self, logits, targets, planar_logits=None, planar_target=None, planar_valid=None):
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)

        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, ignore_index=255)

        if not self.planar_enabled:
            return ce_loss

        if planar_logits is None or planar_target is None or planar_valid is None:
            return ce_loss

        planar_target = planar_target.to(planar_logits.device)
        planar_valid = planar_valid.to(planar_logits.device)

        planar_loss = F.binary_cross_entropy_with_logits(planar_logits, planar_target, reduction="none")
        denom = planar_valid.sum().clamp_min(1.0)
        planar_loss = (planar_loss * planar_valid).sum() / denom

        return ce_loss + self.planar_weight * planar_loss
