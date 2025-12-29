import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class SegFPN(nn.Module):
    _printed_4ch_init_msg = False

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        # Log critical config to catch silent bugs (e.g., in_channels=3 instead of 6)
        print(f"SegFPN init: encoder=tu-convnext_base, depth=4, in_channels={in_channels}, classes={num_classes}")
        
        self.net = smp.FPN(
            encoder_name="tu-convnext_base",
            encoder_weights="imagenet",
            in_channels=in_channels,
            encoder_depth=4,
            classes=num_classes,
        )
        self._init_4th_channel_if_needed(in_channels)

    def _init_4th_channel_if_needed(self, in_channels: int) -> None:
        if in_channels <= 3:
            return
        
        # 1. Locate Stem Conv
        conv = None
        enc = self.net.encoder
        
        # Check for ConvNeXt pattern in timm (via smp TimmUniversalEncoder)
        # enc.model is FeatureListNet wrapping the actual ConvNeXt
        # ConvNeXt has stem_0 as the first Conv2d
        if hasattr(enc, "model") and hasattr(enc.model, "stem_0"):
            if isinstance(enc.model.stem_0, nn.Conv2d):
                conv = enc.model.stem_0
        
        # Fallback patterns for other encoders
        if conv is None:
            if hasattr(enc, "_conv_stem") and isinstance(enc._conv_stem, nn.Conv2d):
                conv = enc._conv_stem
            elif hasattr(enc, "stem") and isinstance(enc.stem, nn.Sequential) and len(enc.stem) > 0:
                if isinstance(enc.stem[0], nn.Conv2d):
                    conv = enc.stem[0]
            elif hasattr(enc, "conv1") and isinstance(enc.conv1, nn.Conv2d):
                conv = enc.conv1
        
        if conv is None:
            print(f"⚠️  SegFPN: Could not locate stem conv for {in_channels}ch init. Skipping weight initialization.")
            print(f"    This may affect training if using non-RGB channels (depth, etc.)")
            return

        if conv.weight.shape[1] != in_channels:
             # Already initialized or mismatch?
             # SMP might re-init if we passed in_channels to constructor
             if conv.weight.shape[1] < in_channels:
                 # Should not happen if smp handles it
                 pass
             return

        # 2. Initialize Extra Channels [3:]
        # Strategy: 
        # - 4ch (RGB + InvDepth): ch3 = Mean(RGB)  (matches exp093_5 doc behavior)
        # - 5ch+: Inv/Log (ch 3, 4): Mean(RGB) * 0.1
        # - 6ch+: Mask (ch 5): 0.0
        with torch.no_grad():
            # RGB weights: [Out, 3, k, k]
            rgb_w = conv.weight[:, :3, :, :]
            mean_w = rgb_w.mean(dim=1, keepdim=True) # [Out, 1, k, k]
            
            # 4ch case: initialize the single extra channel with Mean(RGB)
            if in_channels == 4:
                conv.weight[:, 3, :, :] = mean_w.squeeze(1)
            # Inv/Log (Index 3, 4 relative to 0) -> 3:5
            # Inv/Log (Index 3, 4 relative to 0) -> 3:5
            if in_channels >= 5:
                conv.weight[:, 3:5, :, :] = mean_w * 0.1
            
            # Mask (Index 5) -> 5
            if in_channels >= 6:
                conv.weight[:, 5, :, :] = 0.0

        if not SegFPN._printed_4ch_init_msg:
            if in_channels == 4:
                print("SegFPN: initialized 4th channel weights = mean(RGB)")
            else:
                print(f"SegFPN: initialized extra channels (3-{in_channels}) - Inv/Log: Mean*0.1, Mask: 0.0")
            SegFPN._printed_4ch_init_msg = True

    def get_lr_params(self, base_lr, multiplier=5.0):
        """
        Returns parameter groups with boosted LR for Stem (First Conv).
        Result: [
            {'params': [stem_params], 'lr': base_lr * multiplier},
            {'params': [other_params], 'lr': base_lr}
        ]
        """
        # Locate Stem again
        stem_params = []
        enc = self.net.encoder
        if hasattr(enc, "stem") and isinstance(enc.stem[0], nn.Conv2d):
             stem_params = list(enc.stem.parameters())
        elif hasattr(enc, "model") and hasattr(enc.model, "stem_0") and isinstance(enc.model.stem_0, nn.Conv2d):
             # Actually stem_0 and stem_1 (norm)?
             # User said "stem (input vicinity)".
             # Getting enc.model.stem_0 and potentially stem_1 (LN)
             # Let's grab 'stem_0' and 'stem_1' if they exist in enc.model
             stem_params += list(enc.model.stem_0.parameters())
             if hasattr(enc.model, "stem_1"):
                 stem_params += list(enc.model.stem_1.parameters())
        elif hasattr(enc, "conv1") and isinstance(enc.conv1, nn.Conv2d):
             stem_params = list(enc.conv1.parameters())
             
        stem_ids = list(map(id, stem_params))
        other_params = [p for p in self.parameters() if id(p) not in stem_ids]
        
        return [
            {'params': stem_params, 'lr': base_lr * multiplier},
            {'params': other_params, 'lr': base_lr},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits


class MultiTaskFPN(nn.Module):
    """
    Doc reproduction (Exp093.*):
      - Segmentation: FPN(ConvNeXt Base)
      - Auxiliary depth head: L1 loss on normalized depth target (masked)
    """
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        print(f"MultiTaskFPN init: encoder=tu-convnext_base, depth=4, in_channels={in_channels}, classes={num_classes}")
        self.net = smp.FPN(
            encoder_name="tu-convnext_base",
            encoder_weights="imagenet",
            in_channels=in_channels,
            encoder_depth=4,
            classes=num_classes,
        )

        # Depth head operates on decoder features (same spatial as seg head pre-upsample)
        try:
            decoder_channels = int(self.net.decoder.out_channels)
        except Exception:
            # Fallback: infer from segmentation head
            decoder_channels = int(self.net.segmentation_head[0].in_channels)

        self.depth_head = nn.Conv2d(
            in_channels=decoder_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

        # Keep channel init behavior aligned with SegFPN (important for 4ch / 6ch)
        SegFPN._init_4th_channel_if_needed(self, in_channels)  # reuse helper

    def forward(self, x: torch.Tensor):
        feats = self.net.encoder(x)
        dec = self.net.decoder(feats)
        seg_logits = self.net.segmentation_head(dec)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        depth_pred = self.depth_head(dec)
        depth_pred = F.interpolate(depth_pred, size=x.shape[2:], mode="bilinear", align_corners=False)
        return seg_logits, depth_pred


def build_model(cfg) -> nn.Module:
    """
    Factory to match docs/exp09_log.md behavior.
    """
    if bool(getattr(cfg, "USE_DEPTH_AUX", False)) and float(getattr(cfg, "DEPTH_LOSS_LAMBDA", 0.0)) > 0.0:
        return MultiTaskFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS)
    return SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS)
