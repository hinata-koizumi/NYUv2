
import numpy as np
import torch
import torch.nn.functional as F
import os
try:
    from . import config
except ImportError:
    import config

def save_logits(logits, ids, output_dir, file_prefix="test"):
    """
    Save logits: (N, C, 480, 640) float16
    logits: Tensor or Numpy
    ids: list of file_ids corresponding to logits
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
        
    logits = logits.astype(np.float16)
    
    # Save .npy
    np.save(os.path.join(output_dir, f"{file_prefix}_logits.npy"), logits)
    
    # Save IDs
    with open(os.path.join(output_dir, f"{file_prefix}_ids.txt"), "w") as f:
        for x in ids:
            f.write(f"{x}\n")
            
def calculate_metrics(logits, targets, device="cpu"):
    """
    Return dict of metrics.
    logits: (B, C, H, W)
    targets: (B, H, W)
    """
    preds = torch.argmax(logits, dim=1) # (B, H, W)
    
    # Ignore Mask
    valid_mask = (targets != 255)
    
    metrics = {}
    
    # 1. Floor/Wall IoU
    # Floor=4, Wall=11
    # Simple IoU
    def get_iou(pid):
        tp = ((preds == pid) & (targets == pid) & valid_mask).sum().float()
        union = (((preds == pid) | (targets == pid)) & valid_mask).sum().float()
        if union == 0: return float('nan')
        return (tp / union).item()
        
    metrics['iou_floor'] = get_iou(4)
    metrics['iou_wall'] = get_iou(11)
    
    # 2. Objects Ratio
    # Pred Area / GT Area for 'objects'(6)
    # Check "Attractor" behavior.
    
    # Only count valid regions
    p_obj = ((preds == 6) & valid_mask).sum().float()
    t_obj = ((targets == 6) & valid_mask).sum().float()
    
    if t_obj > 0:
        metrics['ratio_objects'] = (p_obj / t_obj).item()
    else:
        # If GT has no objects, but Pred has some -> Infinity/Bad.
        # If both 0 -> 1.0 (Safe)
        if p_obj > 0:
            metrics['ratio_objects'] = 999.0 # flag expansion
        else:
            metrics['ratio_objects'] = 1.0

    # 3. Suck-in Rate: GT(Struct7) -> Pred(Objects)
    # How much of Struct7 pixels became Objects?
    # Struct7 IDs
    s7_ids = torch.tensor(config.STRUCT7_IDS, device=device)
    # mask of pixels that ARE struct7 in GT
    # isin in torch is a bit heavy if not recent.
    # use boolean mask sum
    gt_s7 = torch.zeros_like(targets, dtype=torch.bool)
    for sid in config.STRUCT7_IDS:
        gt_s7 |= (targets == sid)
        
    gt_s7 &= valid_mask
    s7_count = gt_s7.sum().float()
    
    if s7_count > 0:
        # pixels that are GT S7 AND Pred Objects
        suck_mask = gt_s7 & (preds == 6)
        suck_count = suck_mask.sum().float()
        metrics['suck_rate'] = (suck_count / s7_count).item()
    else:
        metrics['suck_rate'] = 0.0

    # 4. Small Target IoU (books, picture, tv)
    metrics['iou_books'] = get_iou(1)
    metrics['iou_picture'] = get_iou(7)
    metrics['iou_tv'] = get_iou(10)
    
    return metrics
    
class MetricAggregator:
    def __init__(self):
        self.sums = {}
        self.counts = {}
        
    def update(self, m_dict):
        for k, v in m_dict.items():
            if np.isnan(v): continue
            self.sums[k] = self.sums.get(k, 0.0) + v
            self.counts[k] = self.counts.get(k, 0) + 1
            
    def compute(self):
        ret = {}
        for k in self.sums:
            if self.counts[k] > 0:
                ret[k] = self.sums[k] / self.counts[k]
            else:
                ret[k] = 0.0
        return ret
