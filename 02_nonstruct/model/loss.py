
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from configs import default as config
except ImportError:
    import configs.default as config

class ModelBLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Cross Entropy with Weights
        self.weights = torch.tensor(config.WEIGHTS_LIST).float()
        # CE Loss handling
        # standard CE
        pass

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W) - LongTensor
        """
        # 1. CE Loss
        # Move weights to device
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)
            
        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, ignore_index=255)
        
        # 2. Dice Loss (Small targets only)
        # 2. Dice Loss (Small targets only)
        # DISABLE DUE TO RUNTIME ERROR (View size compatibility on CPU/MPS)
        # books(1), picture(7), tv(10)
        # dice_loss = 0.0
        # small_ids = config.SMALL_TARGET_IDS
        
        # # Softmax
        # probs = F.softmax(logits, dim=1)
        
        # for cls_id in small_ids:
        #     # GT mask
        #     gt_mask = (targets == cls_id).float().contiguous()
        #     # Pred prob
        #     pred_prob = probs[:, cls_id, :, :].contiguous()
        #     # Ignore index (Dice typically ignores void)
        #     valid_region = (targets != 255).float().contiguous()
            
        #     gt_mask = gt_mask * valid_region
        #     pred_prob = pred_prob * valid_region
            
        #     # Flatten for Dice (Use reshape instead of view for safety)
        #     inter = (pred_prob * gt_mask).reshape(logits.shape[0], -1).sum(dim=1)
        #     union = pred_prob.reshape(logits.shape[0], -1).sum(dim=1) + gt_mask.reshape(logits.shape[0], -1).sum(dim=1)
            
        #     dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
        #     dice_loss += dice.mean()
            
        # if len(small_ids) > 0:
        #     dice_loss = dice_loss / len(small_ids)
            
        # return ce_loss + dice_loss
        return ce_loss
