import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from .inference import Predictor
from ..utils.metrics import update_confusion_matrix, compute_metrics

# ============================================================================
# Validation Helpers (Commit 4)
# ============================================================================

def unpack_batch(batch):
    if len(batch) == 3:
        x, y, meta = batch
        depth_target = depth_valid = None
    elif len(batch) == 5:
        x, y, meta, depth_target, depth_valid = batch
    else:
        raise ValueError(f"Unexpected batch size from loader: {len(batch)}")
    return x, y, meta, depth_target, depth_valid

def add_depth_aux_loss(loss, depth_pred, depth_target, depth_valid, cfg):
    if (
        (depth_pred is None)
        or (depth_target is None)
        or (depth_valid is None)
        or (cfg is None)
        or (not bool(getattr(cfg, "USE_DEPTH_AUX", False)))
    ):
        return loss

    lambda_ = float(getattr(cfg, "DEPTH_LOSS_LAMBDA", 0.0))
    if lambda_ <= 0.0:
        return loss

    m = (depth_valid > 0.5).float()
    l1 = F.l1_loss(depth_pred, depth_target, reduction="none")
    depth_loss = (l1 * m).sum() / (m.sum() + 1e-6)
    return loss + lambda_ * depth_loss

def prepare_validation_gt(y_batch, meta_batch, cfg):
    """
    Prepare ground truth labels for validation.
    
    Unpads and resizes GT labels to original resolution.
    Padding assumption: top_left position -> content at (0,0), padding at right/bottom
    
    Args:
        y_batch: (B, H_pad, W_pad) numpy array
        meta_batch: Dict with 'pad_h', 'pad_w', 'orig_h', 'orig_w' (each is tensor of size B)
        cfg: Config object
    
    Yields:
        gt_final: (H_orig, W_orig) numpy array for each sample in batch
    """
    y_np = y_batch.numpy()
    B = y_np.shape[0]
    
    for b in range(B):
        # Extract meta for this sample
        pad_h = int(meta_batch["pad_h"][b])
        pad_w = int(meta_batch["pad_w"][b])
        orig_h = int(meta_batch["orig_h"][b])
        orig_w = int(meta_batch["orig_w"][b])
        
        # Unpad: content is at (0,0), padding at right/bottom
        h_curr, w_curr = y_np[b].shape
        valid_h = h_curr - pad_h
        valid_w = w_curr - pad_w
        
        gt_crop = y_np[b, :valid_h, :valid_w]
        
        # Resize to original resolution using INTER_NEAREST (critical for labels)
        gt_final = cv2.resize(gt_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        yield gt_final

def train_one_epoch(model, ema, loader, criterion, optimizer, device, scaler=None, use_amp=False, grad_accum_steps=1, cfg=None):
    model.train()
    total_loss = 0.0
    
    optimizer.zero_grad(set_to_none=True)
    
    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        x, y, _meta, depth_target, depth_valid = unpack_batch(batch)
        # x: (B, 4+, H, W), y: (B, H, W)
        x, y = x.to(device), y.to(device)
        if depth_target is not None:
            depth_target = depth_target.to(device)
        if depth_valid is not None:
            depth_valid = depth_valid.to(device)
        
        # AMP Context
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x)
            if isinstance(out, (tuple, list)):
                logits, depth_pred = out
            else:
                logits, depth_pred = out, None

            loss = criterion(logits, y)
            # Depth aux loss (doc Exp093.*)
            loss = add_depth_aux_loss(loss, depth_pred, depth_target, depth_valid, cfg)
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
        
        # Backward
        if scaler is not None:
             scaler.scale(loss).backward()
        else:
             loss.backward()
             
        # Step
        if (i + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if ema is not None:
                ema.update(model)
            
        total_loss += loss.item() * grad_accum_steps # Scale back for logging if we divided
        
    return total_loss / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    """
    Validation with TWO phases:
    1. Validation Loss (Standard forward, no TTA, resize only)
    2. LB-Compatible Metrics (Using Predictor)
    """
    # --- DEBUG START ---
    print(f"\n[DEBUG] Starting Validation")
    print(f"[DEBUG] len(valid_loader): {len(loader)}")
    if hasattr(loader, 'dataset'):
        print(f"[DEBUG] len(valid_loader.dataset): {len(loader.dataset)}")
    else:
        print(f"[DEBUG] valid_loader has no .dataset attribute")
    
    print(f"[DEBUG] Config VAL limits: MAX_BATCHES={getattr(cfg, 'VAL_MAX_BATCHES', 'N/A')}, VAL_STEPS={getattr(cfg, 'VAL_STEPS', 'N/A')}")
    # --- DEBUG END ---

    model.eval()
    
    # 1. Val Loss (Fast approximation)
    val_loss = 0.0
    for batch in loader:
        x, y, _meta, depth_target, depth_valid = unpack_batch(batch)

        x, y = x.to(device), y.to(device)
        if depth_target is not None:
            depth_target = depth_target.to(device)
        if depth_valid is not None:
            depth_valid = depth_valid.to(device)

        out = model(x)
        if isinstance(out, (tuple, list)):
            logits, depth_pred = out
        else:
            logits, depth_pred = out, None

        loss = criterion(logits, y)
        loss = add_depth_aux_loss(loss, depth_pred, depth_target, depth_valid, cfg)
        val_loss += loss.item()
    
    val_loss /= max(1, len(loader))
    
    # 2. Metrics using Predictor (LB-compatible validation)
    predictor = Predictor(model, loader, device, cfg)
    
    # Use [(1.0, False)] for speed during training validation
    logits_all = predictor.predict_logits(tta_combs=[(1.0, False)], temperature=1.0)
    
    # Compute Metrics
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    idx = 0
    
    # For histogram analysis
    total_pixels = 0
    class_counts = np.zeros(cfg.NUM_CLASSES, dtype=np.int64)

    for batch in loader:
        _x, y, meta, _depth_target, _depth_valid = unpack_batch(batch)
        # Prepare GT labels (unpad + resize to original resolution)
        for gt_final in prepare_validation_gt(y, meta, cfg):
            # Get prediction from generator
            try:
                logits_item = next(logits_all)  # (H, W, C)
            except StopIteration:
                break
            
            pred = np.argmax(logits_item, axis=2)  # (H, W)
            
            # Update confusion matrix
            cm = update_confusion_matrix(pred, gt_final, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm)
            
            # Update histogram stats
            c_ids, c_counts = np.unique(pred, return_counts=True)
            for c_id, c_count in zip(c_ids, c_counts):
                if c_id < cfg.NUM_CLASSES:
                    class_counts[c_id] += c_count
            total_pixels += pred.size
            
            idx += 1
            
    pixel_acc, miou, iou_list = compute_metrics(cm)
    
    # Class-wise IoU dict
    class_iou = {i: iou for i, iou in enumerate(iou_list)}
    
    # --- DEBUG INFO ---
    print("\n[DEBUG] Validation Analysis:")
    
    # 1. Class IoU
    print(f"[DEBUG] Class IoU (0-{cfg.NUM_CLASSES-1}):")
    # Show only first 20 or so if many classes, but NYUv2 has 40. Show all succinctly.
    # Format nicely
    iou_str = ", ".join([f"{i}:{iou:.4f}" for i, iou in enumerate(iou_list)])
    print(f"  {iou_str}")
    
    # 2. Prediction Histogram
    if total_pixels > 0:
        pred_ratios = class_counts / total_pixels
        # Get top 5 classes
        top5_indices = np.argsort(-pred_ratios)[:5]
        print(f"[DEBUG] Top 5 Predicted Classes (ratio):")
        for i in top5_indices:
            print(f"  Class {i}: {pred_ratios[i]:.4f} ({class_counts[i]} pixels)")
            
        # Check for zero/near-zero predictions
        zero_pred_classes = np.where(class_counts == 0)[0]
        print(f"[DEBUG] Classes with ZERO predictions: {len(zero_pred_classes)}/{cfg.NUM_CLASSES}")
        if len(zero_pred_classes) > 0:
            print(f"  IDs: {zero_pred_classes}")
            
    # KPI Logic for optimization check
    zero_iou_classes = [i for i, iou in enumerate(iou_list) if iou < 1e-6]
    
    # Present-only mIoU (Classes that actually exist in GT)
    # cm shape is (num_classes, num_classes). Rows are GT, Cols are Pred.
    # Sum over columns (axis=1) gives GT counts.
    gt_counts = cm.sum(axis=1)
    present_indices = np.where(gt_counts > 0)[0]
    if len(present_indices) > 0:
        present_iou_list = [iou_list[i] for i in present_indices]
        present_miou = np.mean(present_iou_list)
    else:
        present_miou = 0.0
        
    print(f"[DEBUG] KPIs: num_zero_iou_classes={len(zero_iou_classes)}/{cfg.NUM_CLASSES}")
    print(f"[DEBUG] KPIs: present_only_mIoU={present_miou:.4f} (over {len(present_indices)} classes)")
    
    # Class 3 Diagnosis: Top 2 confusion targets
    # cm[3, :] is row 3 (GT=3), showing what it was predicted as.
    if 3 < cfg.NUM_CLASSES:
        gt3_preds = cm[3, :]
        if gt3_preds.sum() > 0:
            top2_pidx = np.argsort(gt3_preds)[-2:][::-1]
            top2_counts = gt3_preds[top2_pidx]
            total_gt3 = gt3_preds.sum()
            msg = f"[DEBUG] Class 3 Diagn: GT(3) -> Pred{list(zip(top2_pidx, top2_counts))} (Total {total_gt3})"
            print(msg)
    # ------------------
    
    return val_loss, miou, pixel_acc, class_iou


@torch.no_grad()
def validate_tta_sweep(model, loader, device, cfg):
    """
    Doc reproduction: sweep temperatures with full TTA (cfg.TTA_COMBS) and pick best mIoU.
    Returns:
      best_temp, best_miou, results_dict
    """
    model.eval()
    results = {}
    best_temp = float(cfg.TEMPERATURES[0]) if len(getattr(cfg, "TEMPERATURES", [])) > 0 else 1.0
    best_miou = -1.0

    for t in cfg.TEMPERATURES:
        predictor = Predictor(model, loader, device, cfg)
        logits_all = predictor.predict_logits(tta_combs=cfg.TTA_COMBS, temperature=float(t))

        cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)

        for batch in loader:
            _x, y, meta, _depth_target, _depth_valid = unpack_batch(batch)

            for gt_final in prepare_validation_gt(y, meta, cfg):
                try:
                    logits_item = next(logits_all)  # (H, W, C)
                except StopIteration:
                    break
                pred = np.argmax(logits_item, axis=2)
                cm = update_confusion_matrix(pred, gt_final, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm)

        _acc, miou, _iou_list = compute_metrics(cm)
        results[float(t)] = float(miou)
        if float(miou) > float(best_miou):
            best_miou = float(miou)
            best_temp = float(t)

    return best_temp, best_miou, results
