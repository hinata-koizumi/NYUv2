import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# ============================================================================
# Thin Helper Functions (Commit 3)
# ============================================================================

def get_meta_value(meta, key):
    """
    Extract value from meta dict, handling both tensor and scalar values.
    
    Args:
        meta: Dict with metadata (values can be tensors or scalars)
        key: Key to extract
    
    Returns:
        Scalar value (int or float)
    """
    v = meta[key]
    return v.item() if hasattr(v, 'item') else v


def resize_pred_to_gt(pred, gt_shape):
    """
    Resize prediction to match GT shape using INTER_NEAREST.
    
    Args:
        pred: (H, W) numpy array
        gt_shape: (H_gt, W_gt) tuple
    
    Returns:
        pred_resized: (H_gt, W_gt) numpy array
    
    Note: Uses INTER_NEAREST to preserve label values (critical for segmentation)
    """
    h_gt, w_gt = gt_shape
    return cv2.resize(pred, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)


class Predictor:
    def __init__(self, model, loader, device, cfg, ema_model=None):
        self.model = model
        self.loader = loader
        self.device = device
        self.cfg = cfg
        self.ema_model = None  # Will be set if passed or can be assigned manually

    def _unpad_and_resize(self, logits, meta):
        """
        Args:
            logits: (C, H_pad, W_pad) Tensor (CPU)
            meta: Dict with 'orig_h', 'orig_w', 'pad_h', 'pad_w'
        Returns:
            logits_orig: (C, H_orig, W_orig) Tensor (CPU)
        """
        _, h_pad, w_pad = logits.shape
        
        # Use helper to extract meta values (handles both tensor and scalar)
        orig_h = get_meta_value(meta, "orig_h")
        orig_w = get_meta_value(meta, "orig_w")
        pad_h = get_meta_value(meta, "pad_h")
        pad_w = get_meta_value(meta, "pad_w")
        
        # Unpad: content is at top-left, padding at right/bottom
        valid_h = h_pad - pad_h
        valid_w = w_pad - pad_w
        
        # Crop
        logits_crop = logits[:, :valid_h, :valid_w]
        
        # Resize Back (Bilinear for logits)
        # F.interpolate requires (B, C, H, W)
        lc = logits_crop.unsqueeze(0) # (1, C, H, W)
        lc = F.interpolate(
            lc, 
            size=(orig_h, orig_w), 
            mode="bilinear", 
            align_corners=False
        )
        return lc.squeeze(0)

    @torch.no_grad()
    def predict_logits(self, tta_combs=None, temperature=1.0):
        """
        Returns:
            results: List of (logits, meta) OR dict of results?
            Actually, commonly we want to save directly or return a big array.
            For LB compatibility, we usually iterate and save, or return iterator?
            Given the user wants 'predict_logits(...) -> (N, H, W, C)', I will return a Generator or handle saving internally? 
            "standardized output (N,H,W,C)".
            Warning: Large array in memory? 
            (N, 13, 480, 640) * 4 bytes ~ 15MB/img * 650 images ~ 10GB.
            It's feasible but risky. 
            Better to yield or write to membrane/disk.
            But the plan said "Accumulates ... Returns Standardized Output".
            I will implement it returning a list of numpy arrays, which uses memory but is simplest.
            Or better, return a Float16 Array.
            
            Actually, `make_submission_npy` in 093.5 used `memmap`.
            I should arguably support memmap-ing.
            But simply returning a list of `(logits, meta)` allows the caller (metrics or submitter) to decide.
            Let's return a list for now (safe for Valid set, maybe tight for Test if very large, but NYUv2 keys=654 so it's fine).
        """
        self.model.eval()
        if tta_combs is None:
            tta_combs = self.cfg.TTA_COMBS or [(1.0, False)]
            
        # Loop Batch
        # Loader yields: x (B,4,H,W), y (B,H,W), meta (Dict of lists/tensors)
        # Note: `meta` items are collated into tensors/lists by DataLoader.
        for x, y, meta in tqdm(self.loader, desc="Inference"):
            x = x.to(self.device)
            B = x.shape[0]
            
            # Accumulators for this batch: (B, C, H_pad, W_pad)
            # Use float32 for accumulation
            batch_accum = torch.zeros(
                (B, self.cfg.NUM_CLASSES, x.shape[2], x.shape[3]), 
                device=self.device, 
                dtype=torch.float32
            )
            
            # TTA Loop
            count = 0
            for scale, flip in tta_combs:
                # 1. Apply TTA to Input
                # Resize
                if scale != 1.0:
                    h_new = int(x.shape[2] * scale)
                    w_new = int(x.shape[3] * scale)
                    # Align to 32? 
                    # 093.5: h_new = max(32, (int(h * scale) // 32) * 32)
                    pass 
                    # Wait, Input must be mult of 32 for ConvNeXt?
                    # Yes.
                    h_new = max(32, (h_new // 32) * 32)
                    w_new = max(32, (w_new // 32) * 32)
                    
                    x_aug = F.interpolate(x, size=(h_new, w_new), mode='bilinear', align_corners=False)
                else:
                    x_aug = x
                    
                # Flip
                if flip:
                    x_aug = torch.flip(x_aug, dims=[3]) # Width dim
                
                # 2. Inference
                logits = self.model(x_aug)
                
                # 3. Process Logits
                # Apply Temperature
                logits = logits / temperature
                
                # Inverse TTA (Flip)
                if flip:
                    logits = torch.flip(logits, dims=[3])
                
                # Inverse TTA (Scale) -> Resize back to Input Size (H_pad, W_pad)
                if scale != 1.0:
                    logits = F.interpolate(
                        logits, 
                        size=(x.shape[2], x.shape[3]), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Accumulate
                batch_accum += logits
                count += 1
            
            batch_accum /= max(1, count)
            
            # Unpad & Resize Back (Per image due to potentially different original sizes? 
            # Actually NYUv2 is all 480x640, but code should be generic)
            
            # Move to CPU to save GPU memory before unpad logic loop
            batch_accum = batch_accum.cpu()
            
            # Iterate batch to unpad
            # meta keys: 'orig_h' (Tensor B), ...
            for i in range(B):
                # Extract meta for this item
                item_meta = {k: v[i] for k, v in meta.items() if k in ['orig_h', 'orig_w', 'pad_h', 'pad_w', 'file_id']}
                
                # Unpad/Resize
                l_final = self._unpad_and_resize(batch_accum[i], item_meta)
                
                # l_final is (C, H_orig, W_orig)
                # Transpose to (H, W, C) for standard .npy format
                l_final = l_final.permute(1, 2, 0)
                
                # Yield per sample
                yield l_final.numpy()

    @torch.no_grad()
    def predict_proba_one(self, sample, use_ema=True, amp=True, tta_flip=False) -> np.ndarray:
        """
        Single sample prediction with Flip TTA and Probabilistic Averaging.
        Args:
            sample: Tuple (x, y, meta) from NYUDataset
            use_ema: Whether to use EMA model
            amp: Whether to use AMP
            tta_flip: Whether to use Horizontal Flip TTA
        Returns:
            prob: (C, H_orig, W_orig) float32 numpy array
        """
        # 1. Unpack sample (x is already padded by Dataset)
        x_tensor, _, meta = sample
        
        # Unsqueeze to Batch=1
        x = x_tensor.unsqueeze(0).to(self.device) # (1, C, H_pad, W_pad)
        
        # 2. Select Model
        net = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        net.eval()
        
        # 3. Forward (Normal)
        p = forward_proba(net, x, amp=amp) # (1, C, H_pad, W_pad)
        
        # 4. Flip TTA
        if tta_flip:
            x_f = _flip_x(x)
            p_f = forward_proba(net, x_f, amp=amp)
            p_f = _flip_x(p_f) # Unflip prob
            p = 0.5 * (p + p_f)
            
        # 5. Unpad/Resize to Original Resolution
        # p is (1, C, H_pad, W_pad). 
        # _unpad_and_resize expects (C, H_pad, W_pad) and scalar meta
        p = p.squeeze(0) # (C, Hp, Wp)
        
        # Meta in sample is Tensors if from Loader (Collated)?
        # If sample comes from dataset[i] directly, meta values are python scalars/strings.
        # Check if meta values are tensors or scalars.
        # NYUDataset returns scalars in meta dict.
        # BUT _unpad_and_resize expects tensors because it calls .item()?
        # Let's check _unpad_and_resize.
        # It handles .item(). If it's already scalar, .item() might fail on int/float in python < 3.
        # But in PyTorch, you can't .item() a float.
        # I should check and fix _unpad_and_resize if needed or wrap meta values.
        # Or just handle it here.
        
        # Fix: ensure meta values are usable.
        # If sample came from dataset[i], meta['orig_h'] is int.
        # If I wrap them in simple object with .item() or just modify _unpad_and_resize to handle both.
        # I'll modify _unpad_and_resize slightly to be robust.
        
        prob_orig = self._unpad_and_resize(p, meta)
        
        return prob_orig.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_label_one(self, sample, use_ema=True, amp=True, tta_flip=False) -> np.ndarray:
        prob = self.predict_proba_one(sample, use_ema=use_ema, amp=amp, tta_flip=tta_flip)
        return np.argmax(prob, axis=0).astype(np.uint8)

def _flip_x(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W]
    return torch.flip(x, dims=[-1])  # horizontal flip (width is last dim)

@torch.no_grad()
def forward_proba(model, x, amp: bool) -> torch.Tensor:
    # returns prob: [B, C, H, W]
    if amp:
        with torch.cuda.amp.autocast():
            logits = model(x)
    else:
        logits = model(x)
    prob = torch.softmax(logits, dim=1)
    return prob

