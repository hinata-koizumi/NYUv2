
import torch
import numpy as np

class IoUMetricFixed:
    def __init__(self, num_classes: int, ignore_index: int):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.view(-1).cpu()
        targets = targets.view(-1).cpu()
        mask = targets != self.ignore_index
        preds = preds[mask]
        targets = targets[mask]
        if preds.numel() == 0:
            return
        k = (targets * self.num_classes + preds).long()
        bins = torch.bincount(k, minlength=self.num_classes ** 2).float()
        self.confusion += bins.view(self.num_classes, self.num_classes)

    def compute(self):
        intersection = torch.diag(self.confusion)
        union = self.confusion.sum(1) + self.confusion.sum(0) - intersection
        
        # FIX: Only compute IoU for classes that exist in Union (Union > 0)
        # Otherwise set to NaN
        iou = intersection / (union + 1e-7)
        
        # Mask out classes with 0 union
        valid_classes = union > 0
        
        if valid_classes.sum() == 0:
            return 0.0, iou
            
        miou = iou[valid_classes].mean().item()
        return miou, iou

def test_metric():
    num_classes = 13
    ignore_index = 255
    metric = IoUMetricFixed(num_classes, ignore_index)

    # Case 1: Perfect match
    preds = torch.tensor([0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 2, 0, 1])
    metric.update(preds, targets)
    miou, iou = metric.compute()
    print(f"Case 1 (Perfect): mIoU={miou:.4f}")
    
    # Case 2: Ignore index
    metric = IoUMetricFixed(num_classes, ignore_index)
    preds = torch.tensor([0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 255, 0, 1]) # 3rd item ignored
    metric.update(preds, targets)
    miou, iou = metric.compute()
    print(f"Case 2 (Ignore): mIoU={miou:.4f}")
    
    # Case 3: Complete mismatch
    metric = IoUMetricFixed(num_classes, ignore_index)
    preds = torch.tensor([0, 0, 0])
    targets = torch.tensor([1, 1, 1])
    metric.update(preds, targets)
    miou, iou = metric.compute()
    print(f"Case 3 (Mismatch): mIoU={miou:.4f}")

if __name__ == "__main__":
    test_metric()
