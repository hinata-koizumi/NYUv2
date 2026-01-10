import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Generator, Any


def _meta_scalar(meta: dict, key: str):
    v = meta[key]
    return v.item() if hasattr(v, "item") else v


class Predictor:

    def __init__(self, model, loader, device, cfg):
        self.model = model
        self.loader = loader
        self.device = device
        self.cfg = cfg

        self.num_classes = int(getattr(cfg, "NUM_CLASSES"))
        self.device_type = "cuda" if str(device) == "cuda" else "cpu"
        self.use_amp = bool(getattr(cfg, "USE_AMP", False)) and (self.device_type == "cuda")
        amp_dtype_name = str(getattr(cfg, "AMP_DTYPE", "bf16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

        self.verbose = bool(getattr(cfg, "VERBOSE", False))

    def _unpad_and_resize_logits_batch(self, logits_bchw: torch.Tensor, meta: dict, i: int = 0) -> torch.Tensor:
        """
        Batched version.
        logits_bchw: (B, C, H_pad, W_pad)
        meta: dictionary of lists
        """
        B, C, h_pad, w_pad = logits_bchw.shape
        
        # Assume uniform padding/sizing within batch (verified by transforms)
        # Use first sample metadata for sizing
        orig_h = int(meta["orig_h"][0])
        orig_w = int(meta["orig_w"][0])
        pad_h = int(meta["pad_h"][0])
        pad_w = int(meta["pad_w"][0])

        valid_h = h_pad - pad_h
        valid_w = w_pad - pad_w

        # Slice (B, C, valid_h, valid_w)
        logits_crop = logits_bchw[:, :, :valid_h, :valid_w]

        # Interpolate to (orig_h, orig_w)
        # (B, C, H, W)
        x = F.interpolate(logits_crop, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return x

    def _combine_books_protect_batch(self, branches_logits: list[torch.Tensor], branches_info: list[dict]) -> torch.Tensor:
        """
        Batched Books Protection.
        branches_logits: List of (B, C, H, W) tensors.
        branches_info: List of dicts checks {scale, hflip}.
        """
        # Stack -> (K, B, C, H, W)
        stack = torch.stack(branches_logits, dim=0)
        mean_logits = torch.mean(stack, dim=0) # (B, C, H, W)

        use_protect = bool(getattr(self.cfg, "INFER_TTA_BOOKS_PROTECT", False)) and (len(branches_logits) > 1)
        if not use_protect:
            return mean_logits

        books_id = int(getattr(self.cfg, "CLASS_ID_BOOKS", 1))
        protect_idx = -1
        
        # Find 1.0_noflip
        for k, info in enumerate(branches_info):
             if (abs(info["scale"] - 1.0) < 1e-4) and (not info["hflip"]):
                 protect_idx = k
                 break
        
        if protect_idx >= 0:
            final_logits = mean_logits.clone()
            # Replace books channel for all samples in batch
            final_logits[:, books_id, :, :] = stack[protect_idx, :, books_id, :, :]
            return final_logits
        
        return mean_logits

    @torch.no_grad()
    def predict_logits(self, *, tta_combs=None, temperature: float = 1.0, return_details: bool = False, return_logits: bool = True) -> Generator[Any, None, None]:

        if temperature is None:
            temperature = 1.0
        temperature = float(temperature)
        if not (temperature > 0):
            raise ValueError(f"temperature must be > 0 (got {temperature})")

        if tta_combs is None:
            tta_combs = list(getattr(self.cfg, "TTA_COMBS", [(1.0, False)]))
        if len(tta_combs) == 0:
            tta_combs = [(1.0, False)]

        self.model.eval()

        for batch in tqdm(self.loader, desc="Inference", disable=(not self.verbose)):
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                x, _y, meta = batch[0], batch[1], batch[2]
            else:
                raise ValueError("Unexpected inference batch structure")

            x = x.to(self.device, non_blocking=True)
            # x is (B, C, H, W)
            base_h, base_w = int(x.shape[2]), int(x.shape[3])
            
            # 1. Run TTA Branches (Batch Mode)
            branch_logits = [] # Stores (B, C, H, W) tensors
            branch_info = []

            for scale, hflip in tta_combs:
                scale = float(scale)
                hflip = bool(hflip)

                x_aug = x
                if scale != 1.0:
                    nh = max(32, int(round(base_h * scale / 32.0)) * 32)
                    nw = max(32, int(round(base_w * scale / 32.0)) * 32)
                    x_aug = F.interpolate(x_aug, size=(nh, nw), mode="bilinear", align_corners=False)
                
                if hflip:
                    x_aug = torch.flip(x_aug, dims=[3])

                with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                    out = self.model(x_aug)

                logits = out[0] if isinstance(out, (tuple, list)) else out # (B, C, H_out, W_out)

                if hflip:
                    logits = torch.flip(logits, dims=[3])
                
                if scale != 1.0:
                    logits = F.interpolate(logits, size=(base_h, base_w), mode="bilinear", align_corners=False)
                
                branch_logits.append(logits)
                branch_info.append({"scale": scale, "hflip": hflip})

            # 2. Combine & Protect (Batch Mode) (B, C, H, W)
            final_logits_pad = self._combine_books_protect_batch(branch_logits, branch_info)
            
            # 3. Unpad & Resize (Batch Mode) -> (B, C, orig_h, orig_w)
            # We assume whole batch maps to same orig_size (NYU default 480x640)
            final_logits = self._unpad_and_resize_logits_batch(final_logits_pad, meta)

            # 4. Softmax (Optional) / CPU Transfer
            # If return_logits=True, we yield logits. Else probs.
            if return_logits:
                output_batch = final_logits.float().cpu().numpy() # (B, C, H, W)
            else:
                probs = torch.softmax(final_logits / temperature, dim=1) # (B, C, H, W)
                output_batch = probs.float().cpu().numpy()

            # 5. Yield per-sample
            bsz = x.shape[0]
            for i in range(bsz):
                res = output_batch[i] # (C, H, W)

                if return_details:
                    # Construct meta
                    # meta is dict of lists/tensors
                    sample_meta = {}
                    if meta is not None:
                        for k, v in meta.items():
                            if isinstance(v, (list, tuple)):
                                 try:
                                     sample_meta[k] = v[i]
                                 except IndexError:
                                     sample_meta[k] = v
                            elif isinstance(v, torch.Tensor):
                                 sample_meta[k] = v[i].item()
                            else:
                                 sample_meta[k] = v
                    
                    # Branches: For validation saving (TTA=1), we just use the main result
                    # as the single branch to satisfy consumer expectation.
                    
                    sample_branches = [{"logits": res}] 
                    
                    yield {
                        "merged_probs": res, # CHW
                        "meta": sample_meta,
                        "branches": sample_branches
                    }
                else:
                    yield res