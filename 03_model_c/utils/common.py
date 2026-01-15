import numpy as np
import torch
import os

try:
    from configs import default as config
except ImportError:
    import configs.default as config


def save_logits(logits, ids, output_dir, file_prefix="test"):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()

    logits = logits.astype(np.float16)

    np.save(os.path.join(output_dir, f"{file_prefix}_logits.npy"), logits)

    with open(os.path.join(output_dir, f"{file_prefix}_ids.txt"), "w") as f:
        for x in ids:
            f.write(f"{x}\n")


def calculate_metrics(logits, targets, device="cpu"):
    preds = torch.argmax(logits, dim=1)
    valid_mask = (targets != 255)

    metrics = {}

    def get_iou(pid):
        tp = ((preds == pid) & (targets == pid) & valid_mask).sum().float()
        union = (((preds == pid) | (targets == pid)) & valid_mask).sum().float()
        if union == 0:
            return float("nan")
        return (tp / union).item()

    metrics["iou_floor"] = get_iou(4)
    metrics["iou_wall"] = get_iou(11)

    p_obj = ((preds == 6) & valid_mask).sum().float().item()
    t_obj = ((targets == 6) & valid_mask).sum().float().item()

    metrics["pred_objects_pixels"] = p_obj
    metrics["gt_objects_pixels"] = t_obj
    metrics["valid_pixels"] = valid_mask.sum().float().item()

    gt_s7 = torch.zeros_like(targets, dtype=torch.bool)
    for sid in config.STRUCT7_IDS:
        gt_s7 |= (targets == sid)

    gt_s7 &= valid_mask
    s7_count = gt_s7.sum().float()

    if s7_count > 0:
        suck_mask = gt_s7 & (preds == 6)
        suck_count = suck_mask.sum().float()
        metrics["suck_rate"] = (suck_count / s7_count).item()
    else:
        metrics["suck_rate"] = 0.0

    metrics["iou_books"] = get_iou(1)
    metrics["iou_picture"] = get_iou(7)
    metrics["iou_tv"] = get_iou(10)
    metrics["iou_table"] = get_iou(9)

    return metrics


class MetricAggregator:
    def __init__(self):
        self.sums = {}
        self.counts = {}
        self.total_pred_objects = 0.0
        self.total_gt_objects = 0.0
        self.total_valid_pixels = 0.0

    def update(self, m_dict):
        if "pred_objects_pixels" in m_dict:
            self.total_pred_objects += m_dict["pred_objects_pixels"]
        if "gt_objects_pixels" in m_dict:
            self.total_gt_objects += m_dict["gt_objects_pixels"]
        if "valid_pixels" in m_dict:
            self.total_valid_pixels += m_dict["valid_pixels"]

        for k, v in m_dict.items():
            if k in ["pred_objects_pixels", "gt_objects_pixels", "valid_pixels"]:
                continue
            if np.isnan(v):
                continue
            self.sums[k] = self.sums.get(k, 0.0) + v
            self.counts[k] = self.counts.get(k, 0) + 1

    def compute(self):
        ret = {}
        for k in self.sums:
            if self.counts[k] > 0:
                ret[k] = self.sums[k] / self.counts[k]
            else:
                ret[k] = 0.0

        if self.total_gt_objects > 0:
            ret["ratio_objects_global"] = self.total_pred_objects / self.total_gt_objects
        else:
            ret["ratio_objects_global"] = 999.0 if self.total_pred_objects > 0 else 1.0

        if self.total_valid_pixels > 0:
            ret["pred_objects_percent_global"] = (self.total_pred_objects / self.total_valid_pixels) * 100
        else:
            ret["pred_objects_percent_global"] = 0.0

        return ret
