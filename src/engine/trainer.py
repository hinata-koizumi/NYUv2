import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .inference import Predictor
from ..utils.metrics import update_confusion_matrix, compute_metrics, CombinedSegLoss

def train_one_epoch(model, ema, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for x, y, _meta in tqdm(loader, desc="Train", leave=False):
        # x: (B, 4, H, W), y: (B, H, W)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
        
        if ema is not None:
            ema.update(model)
            
        total_loss += loss.item()
        
    return total_loss / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    """
    Validation with TWO phases:
    1. Validation Loss (Standard forward, no TTA, resize only)
       - Actually, we can skip this if we trust mIoU.
       - But let's compute it for logging.
    2. LB-Compatible Metrics (Using Predictor)
    """
    model.eval()
    
    # 1. Val Loss (Fast approximation using first TTA config or just raw forward)
    # But wait, loader yields 'x' which is already adapted.
    # If we use Predictor, it handles unpadding/resizing correctly.
    # Calculating Loss on Unpadded/Resized-Back outputs vs Original Labels is tricky 
    # if we want exact comparison to training loss (which uses crop).
    # But Validation is on Padded images. 
    # If we calculate loss on Padded Output vs Padded Label (if label is padded?), it's consistent.
    # My Dataset `get_valid_transforms` pads image. Does it pad label?
    # Yes, Albumentations pads both.
    # So we can calculate `val_loss` on the straight output of model(x) vs y.
    
    val_loss = 0.0
    # Use a separate loop or rely on Predictor? 
    # Predictor is heavy (accumulates everything). 
    # For frequent validation, maybe too heavy?
    # But LB compatibility is key.
    # Let's do a lightweight loop for Loss.
    
    for x, y, _meta in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        val_loss += loss.item()
    
    val_loss /= max(1, len(loader))
    
    # 2. Metrics using Predictor (The Gold Standard)
    # We use the Predictor to get (N, H_orig, W_orig, C) LOGITS.
    # Then we compute mIoU against Original Labels.
    # But `loader` yields transformed labels (padded).
    # We need ORIGINAL labels for strict LB mIoU.
    # `loader` yields `meta` which has `file_id`.
    # We can reload simple label? Or we can UNPAD the label from batch y?
    # Unpadding label nearest neighbor is safe.
    
    predictor = Predictor(model, loader, device, cfg)
    # Use default TTA combs (or simplified for speed if desired, but user wants consistency)
    # Using 'valid' mode usually implies NO TTA or LIGHT TTA. 
    # But user said "VAL and SUB same infer function".
    # So we should use cfg.TTA_COMBS? 
    # If TTA is heavy (8x), doing it every epoch is slow.
    # Standard practice: Validate with 1.0 scale only, Test with TTA.
    # Or TTA every 10 epochs.
    # I'll default to 1.0 scale for 'validate' loop speed, 
    # but allow passing tta_combs.
    
    # For now, let's use [(1.0, False)] for speed during training validation.
    # Real LB check should be explicit separate run.
    logits_all = predictor.predict_logits(tta_combs=[(1.0, False)], temperature=1.0)
    
    # Compute Metrics
    # logits_all is (N, H, W, C)
    # We need to match with GT.
    # The loader iterates sequentially.
    # We need GTs. 
    # We can rebuild GTs from loader `y` by unpadding.
    
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    
    # Iterator for GT
    # Reuse loader?
    # Note: Predictor consumed loader.
    # Basic DataLoader is reusable.
    
    idx = 0
    for _x, y, meta in loader:
        # y is (B, H_pad, W_pad)
        y_np = y.numpy()
        B = y_np.shape[0]
        
        for b in range(B):
            # Unpad GT
            pad_h = int(meta["pad_h"][b])
            pad_w = int(meta["pad_w"][b])
            orig_h = int(meta["orig_h"][b])
            orig_w = int(meta["orig_w"][b])
            
            # Label was padded top-left content?
            # get_valid_transforms: PadIfNeeded(..., top_left) -> Content is at (0,0)?
            # Wait, Albumentations `PadIfNeeded` with `position="top_left"` means:
            # "Pad to top left" -> image is pushed to bottom right?
            # Standard: `position='center'` puts it in center.
            # `position='top_left'` usually means "Put image in top-left, pad right/bottom".
            # VERIFY THIS.
            # A.PadIfNeeded source: 
            # if position == 'top_left': pad_top=0, pad_left=0. 
            # So Image is at (0,0). Padding is at (H, W).
            # This matches "Right-Bottom" padding described in plan.
            # Correct.
            
            h_curr, w_curr = y_np[b].shape
            valid_h = h_curr - pad_h
            valid_w = w_curr - pad_w
            
            gt_crop = y_np[b, :valid_h, :valid_w]
            
            # Get prediction from generator
            try:
                logits_item = next(logits_all)
            except StopIteration:
                break
                
            gt_final = cv2.resize(
                gt_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
            
            pred = np.argmax(logits_item, axis=2)
            
            cm = update_confusion_matrix(pred, gt_final, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm)
            idx += 1
            
    pixel_acc, miou, _ = compute_metrics(cm)
    return val_loss, miou, pixel_acc
