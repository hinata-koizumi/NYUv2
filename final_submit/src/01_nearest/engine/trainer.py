import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple

from ..utils.metrics import update_confusion_matrix, compute_metrics


def _unpack_batch(batch, *, context: str):
    if isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            x, y, meta = batch
            depth_target = depth_valid = boundary_target = None
        elif len(batch) == 5:
            x, y, meta, depth_target, depth_valid = batch
            boundary_target = None
        elif len(batch) == 6:
             x, y, meta, depth_target, depth_valid, boundary_target = batch
        else:
            raise ValueError(
                f"Unexpected {context} batch structure: type={type(batch)} "
                f"len={len(batch) if hasattr(batch,'__len__') else 'n/a'}"
            )
    else:
         raise ValueError(f"Unexpected batch type: {type(batch)}")
         
    return x, y, meta, depth_target, depth_valid, boundary_target


def train_one_epoch(
    model,
    ema,
    loader,
    criterion,
    optimizer,
    device: str,
    scaler=None,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    cfg=None,
) -> Tuple[float, float, float]:
    """
    Exp100 Trainer (Corrected).
    Returns: (total_loss, aux_loss, grad_norm)
    """

    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    device_type = "cuda" if str(device) == "cuda" else "cpu"
    use_cuda = device_type == "cuda"
    amp_dtype_name = str(getattr(cfg, "AMP_DTYPE", "bf16")).lower() if cfg is not None else "bf16"
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    use_amp = bool(use_amp) and use_cuda

    total_loss = 0.0
    total_aux = 0.0
    total_grad_norm = 0.0
    denom = max(1, len(loader))

    # Aux loss config
    depth_lambda = float(getattr(cfg, "DEPTH_LOSS_LAMBDA", 0.0)) if cfg is not None else 0.0
    use_depth_aux = bool(getattr(cfg, "USE_DEPTH_AUX", False)) if cfg is not None else False
    
    boundary_weight = float(getattr(cfg, "BOUNDARY_LOSS_WEIGHT", 0.2)) if cfg is not None else 0.2

    is_sam = hasattr(optimizer, "first_step") and hasattr(optimizer, "second_step")

    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        x, y, _meta, depth_target, depth_valid, boundary_target = _unpack_batch(batch, context="train")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if depth_target is not None:
            depth_target = depth_target.to(device, non_blocking=True)
        if depth_valid is not None:
            depth_valid = depth_valid.to(device, non_blocking=True)
        if boundary_target is not None:
            boundary_target = boundary_target.to(device, non_blocking=True)

        # Forward closure
        def _forward_loss():
            with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                out = model(x)
                
                # Unpack Model Output
                # Expecting: seg_logits, [depth_pred], [boundary_pred]
                # It might vary depending on config.
                # Let's handle generic tuple return or specific logic.
                seg_logits = depth_pred = boundary_pred = None
                
                if isinstance(out, (tuple, list)):
                    # Order? seg, depth, boundary?
                    # MetaArch logic:
                    # if depth and boundary: return seg, depth, boundary
                    # if depth only: return seg, depth
                    # if boundary only: return seg, boundary
                    # This is tricky without explicit dict.
                    # Assumption: Seg is always first.
                    seg_logits = out[0]
                    others = out[1:]
                    # Heuristic: 
                    # If 1 channel -> Boundary? No, depth is 1 channel too.
                    # We rely on Config or Model structure?
                    # Let's standardize meta_arch to return a Dict or structured tuple?
                    # For minimal change:
                    # If len(out) == 3 -> seg, depth, boundary (if both enabled)
                    # Use provided config to guess.
                    
                    has_depth_head = use_depth_aux
                    has_bound_head = (boundary_target is not None) # Or checking cfg
                    
                    idx = 0
                    if has_depth_head and idx < len(others):
                         depth_pred = others[idx]
                         idx += 1
                    
                    if has_bound_head and idx < len(others):
                         boundary_pred = others[idx]
                         idx += 1
                else:
                    seg_logits = out

                loss = criterion(seg_logits, y)
                aux_val_tracker = 0.0

                # Optional Depth Aux Loss
                if (
                    use_depth_aux
                    and depth_lambda > 0.0
                    and (depth_pred is not None)
                    and (depth_target is not None)
                    and (depth_valid is not None)
                ):
                    denom_valid = depth_valid.sum().clamp_min(1.0)
                    depth_l1 = (torch.abs(depth_pred - depth_target) * depth_valid).sum() / denom_valid
                    loss = loss + depth_lambda * depth_l1
                    aux_val_tracker += float(depth_l1.item())
                    
                # Optional Boundary Loss
                if (boundary_pred is not None) and (boundary_target is not None):
                    # Mask Valid Pixels (ignore=255)
                    # y is (B, H, W).
                    # boundary_target is (B, 1, H, W).
                    # boundary_pred is (B, 1, H, W).
                    
                    valid_mask_b = (y.unsqueeze(1) != int(cfg.IGNORE_INDEX)).float()
                    
                    # Sigmoid for BCE/Dice
                    b_probs = torch.sigmoid(boundary_pred)
                    
                    # 1. BCE
                    # F.binary_cross_entropy_with_logits is stable
                    bce = F.binary_cross_entropy_with_logits(boundary_pred, boundary_target, reduction='none')
                    bce = (bce * valid_mask_b).sum() / valid_mask_b.sum().clamp_min(1.0)
                    
                    # 2. Dice
                    inter = (b_probs * boundary_target * valid_mask_b).sum()
                    union = ((b_probs + boundary_target) * valid_mask_b).sum()
                    dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
                    
                    b_loss = bce + dice
                    loss = loss + boundary_weight * b_loss
                    aux_val_tracker += float(b_loss.item())

                if grad_accum_steps > 1:
                    loss = loss / float(grad_accum_steps)
            return loss, aux_val_tracker

        # --- 1. Forward & Backward ---
        loss, aux_val = _forward_loss()

        if use_amp:
            if scaler is None:
                # Scaler required for fp16, but for bf16 it might be None.
                # If None, assume bf16/no-scaler logic.
                pass 
            else:
                scaler.scale(loss).backward()
        else:
            loss.backward()

        # --- 2. Optimizer Step (SAM vs Standard) ---
        if is_sam:
            # Clip grad (pre-step)
            clip_norm = float(getattr(cfg, "GRAD_CLIP_NORM", 0.0)) if cfg is not None else 0.0
            
            # Manual unscale for first step/clipping to avoid "double unscale" error in GradScaler
            # because we need to unscale TWICE (for ascent, then for descent).
            if use_amp and scaler is not None:
                scale = scaler.get_scale()
                inv_scale = 1.0 / scale
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            p.grad.data.mul_(inv_scale)
            
            if clip_norm > 0.0:
                 param_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                 total_grad_norm += float(param_norm.item())
            else:
                 # Just calculate norm
                 total_norm = 0.0
                 for p in model.parameters():
                     if p.grad is not None:
                         param_norm = p.grad.data.norm(2)
                         total_norm += param_norm.item() ** 2
                 total_grad_norm += total_norm ** 0.5
            
            # SAM Step 1: Ascent
            optimizer.first_step(zero_grad=True)

            # SAM Step 2: Forward at peak
            loss2, _ = _forward_loss()
            if use_amp and scaler is not None:
                scaler.scale(loss2).backward()
            else:
                loss2.backward()

            # SAM Step 3: Descent (Actual update)
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)

            optimizer.second_step(zero_grad=True)
            
            if use_amp and scaler is not None:
                scaler.update()

            if ema is not None and hasattr(ema, "update"):
                 ema.update(model)

        else:
            # Standard Step
            if (i + 1) % grad_accum_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    
                clip_norm = float(getattr(cfg, "GRAD_CLIP_NORM", 0.0)) if cfg is not None else 0.0
                if clip_norm and clip_norm > 0.0:
                    param_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                    total_grad_norm += float(param_norm.item())
                else:
                    # Just calculate norm
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_grad_norm += total_norm ** 0.5

                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if ema is not None and hasattr(ema, "update"):
                    ema.update(model)

        total_loss += loss.item() * (float(grad_accum_steps) if grad_accum_steps > 1 else 1.0)
        total_aux += aux_val * (float(grad_accum_steps) if grad_accum_steps > 1 else 1.0)

    return total_loss / denom, total_aux / denom, total_grad_norm / denom


@torch.no_grad()
def validate(model, loader, criterion, device: str, cfg):
    model.eval()

    device_type = "cuda" if str(device) == "cuda" else "cpu"
    use_cuda = device_type == "cuda"
    amp_dtype_name = str(getattr(cfg, "AMP_DTYPE", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    use_amp = bool(getattr(cfg, "USE_AMP", False)) and use_cuda

    depth_lambda = float(getattr(cfg, "DEPTH_LOSS_LAMBDA", 0.0))
    use_depth_aux = bool(getattr(cfg, "USE_DEPTH_AUX", False))

    cm = np.zeros((int(cfg.NUM_CLASSES), int(cfg.NUM_CLASSES)), dtype=np.int64)
    total_loss = 0.0
    denom = max(1, len(loader))

    for batch in tqdm(loader, desc="Valid", leave=False):
        x, y, _meta, depth_target, depth_valid, boundary_target = _unpack_batch(batch, context="valid")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if depth_target is not None:
            depth_target = depth_target.to(device, non_blocking=True)
        if depth_valid is not None:
            depth_valid = depth_valid.to(device, non_blocking=True)
        # We don't necessarily compute boundary loss in validation but unpacking must match.

        with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(x)
            if isinstance(out, (tuple, list)):
                # Handle generic tuple output (seg, [depth], [boundary])
                seg_logits = out[0]
                others = out[1:]
                idx = 0
                if use_depth_aux and idx < len(others):
                    depth_pred = others[idx]
                    idx += 1
                else:
                    depth_pred = None
            else:
                seg_logits, depth_pred = out, None

            loss = criterion(seg_logits, y)
            if (
                use_depth_aux
                and depth_lambda > 0.0
                and (depth_pred is not None)
                and (depth_target is not None)
                and (depth_valid is not None)
            ):
                denom_valid = depth_valid.sum().clamp_min(1.0)
                depth_l1 = (torch.abs(depth_pred - depth_target) * depth_valid).sum() / denom_valid
                loss = loss + depth_lambda * depth_l1

        total_loss += float(loss.item())

        pred = torch.argmax(seg_logits, dim=1).detach().cpu().numpy().astype(np.int32)
        gt = y.detach().cpu().numpy().astype(np.int32)
        bsz = int(pred.shape[0])
        for i in range(bsz):
            cm = update_confusion_matrix(pred[i], gt[i], int(cfg.NUM_CLASSES), int(cfg.IGNORE_INDEX), cm)

    pixel_acc, miou, iou_list = compute_metrics(cm)
    class_iou = {i: float(iou) for i, iou in enumerate(iou_list)}
    return float(total_loss / denom), float(miou), float(pixel_acc), class_iou


@torch.no_grad()
def validate_tta_sweep(model, loader, device: str, cfg):
    """
    TTA Sweep (Corrected).
    - Uses 32px Alignment.
    - Uses Bilinear Interpolation for ALL channels (RGB + Depth).
    """
    model.eval()

    device_type = "cuda" if str(device) == "cuda" else "cpu"
    use_cuda = device_type == "cuda"
    amp_dtype_name = str(getattr(cfg, "AMP_DTYPE", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    use_amp = bool(getattr(cfg, "USE_AMP", False)) and use_cuda

    temps = list(getattr(cfg, "TEMPERATURES", [1.0]))
    tta_combs = list(getattr(cfg, "TTA_COMBS", [(1.0, False)]))
    if len(tta_combs) == 0:
        tta_combs = [(1.0, False)]

    def _tta_probs(x: torch.Tensor, temperature: float) -> torch.Tensor:
        base_h, base_w = int(x.shape[2]), int(x.shape[3])
        acc = None
        
        for scale, hflip in tta_combs:
            scale = float(scale)
            hflip = bool(hflip)
            x_aug = x
            
            # --- TTA Scale (Bilinear for Everything) ---
            if scale != 1.0:
                nh = max(32, int(round(base_h * scale / 32.0)) * 32)
                nw = max(32, int(round(base_w * scale / 32.0)) * 32)
                # Unified Bilinear Interpolation (Exp099 Style)
                x_aug = F.interpolate(x_aug, size=(nh, nw), mode="bilinear", align_corners=False)
                
            if hflip:
                x_aug = torch.flip(x_aug, dims=[3])

            with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                out = model(x_aug)
            
            logits = out[0] if isinstance(out, (tuple, list)) else out
            
            if hflip:
                logits = torch.flip(logits, dims=[3])
            
            if scale != 1.0:
                logits = F.interpolate(logits, size=(base_h, base_w), mode="bilinear", align_corners=False)
            
            probs = torch.softmax(logits / float(temperature), dim=1)
            acc = probs if acc is None else (acc + probs)
            
        return acc / float(len(tta_combs))

    results: dict[float, float] = {}
    best_temp = float(temps[0]) if len(temps) else 1.0
    best_miou = -1.0

    for t in temps:
        t = float(t)
        cm = np.zeros((int(cfg.NUM_CLASSES), int(cfg.NUM_CLASSES)), dtype=np.int64)

        for batch in tqdm(loader, desc=f"TTA@T={t:.2f}", leave=False):
            x, y, _meta, _dt, _dv, _bt = _unpack_batch(batch, context="tta")

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            probs = _tta_probs(x, temperature=t)
            pred = torch.argmax(probs, dim=1).detach().cpu().numpy().astype(np.int32)
            gt = y.detach().cpu().numpy().astype(np.int32)
            for i in range(int(pred.shape[0])):
                cm = update_confusion_matrix(pred[i], gt[i], int(cfg.NUM_CLASSES), int(cfg.IGNORE_INDEX), cm)

        _pix, miou, _iou = compute_metrics(cm)
        results[t] = float(miou)
        if float(miou) > best_miou:
            best_miou = float(miou)
            best_temp = float(t)

    return best_temp, best_miou, results