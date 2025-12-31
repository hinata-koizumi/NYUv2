import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def _meta_scalar(meta: dict, key: str):
    """meta[key] が Tensor でも scalar でも int/float を返す。"""
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

    def _unpad_and_resize_logits(self, logits_chw: torch.Tensor, meta_item: dict) -> torch.Tensor:

        _, h_pad, w_pad = logits_chw.shape

        orig_h = int(_meta_scalar(meta_item, "orig_h"))
        orig_w = int(_meta_scalar(meta_item, "orig_w"))
        pad_h = int(_meta_scalar(meta_item, "pad_h"))
        pad_w = int(_meta_scalar(meta_item, "pad_w"))

        valid_h = h_pad - pad_h
        valid_w = w_pad - pad_w

        logits_crop = logits_chw[:, :valid_h, :valid_w]  # (C, H_valid, W_valid)

        # resize logits back to original resolution (bilinear)
        x = logits_crop.unsqueeze(0)  # (1, C, H, W)
        x = F.interpolate(x, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    @torch.no_grad()
    def predict_logits(self, *, tta_combs=None, temperature: float = 1.0):

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
                raise ValueError(
                    f"Unexpected inference batch structure: type={type(batch)} "
                    f"len={len(batch) if hasattr(batch,'__len__') else 'n/a'}"
                )

            x = x.to(self.device, non_blocking=True)

            base_h, base_w = int(x.shape[2]), int(x.shape[3])

            # TTA: average probabilities (not logits) at the padded resolution.
            prob_acc = None
            for scale, hflip in tta_combs:
                scale = float(scale)
                hflip = bool(hflip)

                x_aug = x
                if scale != 1.0:
                    nh = max(1, int(round(base_h * scale)))
                    nw = max(1, int(round(base_w * scale)))
                    x_aug = F.interpolate(x_aug, size=(nh, nw), mode="bilinear", align_corners=False)
                if hflip:
                    x_aug = torch.flip(x_aug, dims=[3])

                # `torch.cuda.amp.autocast` is deprecated; use `torch.amp.autocast(device_type=...)`.
                with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                    out = self.model(x_aug)

                logits = out[0] if isinstance(out, (tuple, list)) else out  # (B, C, H, W)
                if logits.ndim != 4 or int(logits.shape[1]) != self.num_classes:
                    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

                if hflip:
                    logits = torch.flip(logits, dims=[3])
                if scale != 1.0:
                    logits = F.interpolate(logits, size=(base_h, base_w), mode="bilinear", align_corners=False)

                probs = torch.softmax(logits / temperature, dim=1)  # (B, C, H_pad, W_pad)
                prob_acc = probs if prob_acc is None else (prob_acc + probs)

            probs = prob_acc / float(len(tta_combs))
            probs_cpu = probs.float().cpu()  # float32 on CPU for stable postprocess
            bsz = probs_cpu.shape[0]

            for i in range(bsz):
                meta_i = {
                    "orig_h": meta["orig_h"][i],
                    "orig_w": meta["orig_w"][i],
                    "pad_h": meta["pad_h"][i],
                    "pad_w": meta["pad_w"][i],
                }
                if "file_id" in meta:
                    meta_i["file_id"] = meta["file_id"][i]

                probs_i = self._unpad_and_resize_logits(probs_cpu[i], meta_i)  # (C, H, W)
                probs_hw_c = probs_i.permute(1, 2, 0).contiguous().numpy().astype(np.float32)
                yield probs_hw_c
