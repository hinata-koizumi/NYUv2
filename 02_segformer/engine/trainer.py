import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from typing import Optional, Any, Tuple, Dict, List
from ..utils.metrics import update_confusion_matrix, compute_metrics


def _unpack_batch(batch, *, context: str):
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        x, y, meta = batch
        depth_target = depth_valid = None
    elif isinstance(batch, (tuple, list)) and len(batch) == 5:
        x, y, meta, depth_target, depth_valid = batch
    else:
        raise ValueError(
            f"Unexpected {context} batch structure: type={type(batch)} "
            f"len={len(batch) if hasattr(batch,'__len__') else 'n/a'}"
        )
    return x, y, meta, depth_target, depth_valid


def train_one_epoch(
    model: torch.nn.Module,
    ema: Optional[Any],
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    cfg: Optional[Any] = None,
) -> float:
    """
    Executes one training epoch.
    Handles Forward/Backward, AMP (bf16/fp16), Gradient Accumulation, and SAM.
    Returns:
        float: Average training loss for the epoch.
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
    denom = max(1, len(loader))

    # Aux loss config
    depth_lambda = float(getattr(cfg, "DEPTH_LOSS_LAMBDA", 0.0)) if cfg is not None else 0.0
    use_depth_aux = bool(getattr(cfg, "USE_DEPTH_AUX", False)) if cfg is not None else False

    is_sam = hasattr(optimizer, "first_step") and hasattr(optimizer, "second_step")

    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        x, y, _meta, depth_target, depth_valid = _unpack_batch(batch, context="train")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if depth_target is not None:
            depth_target = depth_target.to(device, non_blocking=True)
        if depth_valid is not None:
            depth_valid = depth_valid.to(device, non_blocking=True)

        # Forward closure
        def _forward_loss():
            with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                out = model(x)
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    seg_logits, depth_pred = out
                else:
                    seg_logits, depth_pred = out, None

                loss = criterion(seg_logits, y)

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
                
                if grad_accum_steps > 1:
                    loss = loss / float(grad_accum_steps)
            return loss

        # --- 1. Forward & Backward ---
        loss = _forward_loss()

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
                 torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
            # SAM Step 1: Ascent
            optimizer.first_step(zero_grad=True)

            # SAM Step 2: Forward at peak
            loss2 = _forward_loss()
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if ema is not None and hasattr(ema, "update"):
                    ema.update(model)

        total_loss += loss.item() * (float(grad_accum_steps) if grad_accum_steps > 1 else 1.0)

    return total_loss / denom


@torch.no_grad()
def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str, cfg: Any) -> Tuple[float, float, float, Dict[int, float]]:
    """
    Runs validation loop.
    Returns: (val_loss, mIoU, pixel_acc, class_iou_dict)
    """
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
        x, y, _meta, depth_target, depth_valid = _unpack_batch(batch, context="valid")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if depth_target is not None:
            depth_target = depth_target.to(device, non_blocking=True)
        if depth_valid is not None:
            depth_valid = depth_valid.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(x)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                seg_logits, depth_pred = out
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
def validate_tta_sweep(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str, cfg: Any) -> Tuple[float, float, Dict[float, float]]:
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
            x, y, _meta, _dt, _dv = _unpack_batch(batch, context="tta")

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