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
        self.confusion_weight = float(getattr(config, "CONFUSION_PENALTY_WEIGHT", 0.0))
        self.confusion_targets = getattr(config, "CONFUSION_PENALTY_TARGET_IDS", [1, 9])
        self.confusion_bad = getattr(config, "CONFUSION_PENALTY_BAD_IDS", [5, 6])

    def forward(self, logits, targets, planar_logits=None, planar_target=None, planar_valid=None):
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)

        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, ignore_index=255)

        conf_loss = None
        if self.confusion_weight > 0:
            valid = targets != 255
            target_mask = valid & torch.zeros_like(targets, dtype=torch.bool)
            for tid in self.confusion_targets:
                target_mask |= (targets == int(tid))

            if target_mask.any():
                probs = torch.softmax(logits, dim=1)
                bad_prob = torch.zeros_like(probs[:, 0])
                for bid in self.confusion_bad:
                    bad_prob = bad_prob + probs[:, int(bid)]
                bad_prob = torch.clamp(bad_prob, 0.0, 1.0 - 1e-6)
                conf_loss = (-torch.log(1.0 - bad_prob))[target_mask].mean()

        total = ce_loss
        if conf_loss is not None:
            total = total + (self.confusion_weight * conf_loss)

        if not self.planar_enabled:
            return total

        if planar_logits is None or planar_target is None or planar_valid is None:
            return total

        planar_target = planar_target.to(planar_logits.device)
        planar_valid = planar_valid.to(planar_logits.device)

        planar_loss = F.binary_cross_entropy_with_logits(planar_logits, planar_target, reduction="none")
        denom = planar_valid.sum().clamp_min(1.0)
        planar_loss = (planar_loss * planar_valid).sum() / denom

        return total + self.planar_weight * planar_loss
