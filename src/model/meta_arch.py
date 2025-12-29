import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class SegFPN(nn.Module):
    _printed_4ch_init_msg = False

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.net = smp.FPN(
            encoder_name="tu-convnext_base",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        self._init_4th_channel_if_needed(in_channels)

    def _init_4th_channel_if_needed(self, in_channels: int) -> None:
        if in_channels <= 3:
            return
        
        # 1. Locate Stem Conv
        conv = None
        enc = self.net.encoder
        # timm convnext usually has stem[0] or model.stem_0 (via smp)
        if hasattr(enc, "stem") and isinstance(enc.stem[0], nn.Conv2d):
            conv = enc.stem[0]
        elif hasattr(enc, "model") and hasattr(enc.model, "stem_0") and isinstance(enc.model.stem_0, nn.Conv2d):
            conv = enc.model.stem_0
        elif hasattr(enc, "conv1") and isinstance(enc.conv1, nn.Conv2d):
            conv = enc.conv1
        else:
            print("SegFPN: Could not locate stem conv for init.")
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
        # - Inv/Log (ch 3, 4): Mean(RGB) * 0.1
        # - Mask (ch 5): 0.0
        with torch.no_grad():
            # RGB weights: [Out, 3, k, k]
            rgb_w = conv.weight[:, :3, :, :]
            mean_w = rgb_w.mean(dim=1, keepdim=True) # [Out, 1, k, k]
            
            # Inv/Log (Index 3, 4 relative to 0) -> 3:5
            if in_channels >= 5:
                conv.weight[:, 3:5, :, :] = mean_w * 0.1
            
            # Mask (Index 5) -> 5
            if in_channels >= 6:
                conv.weight[:, 5, :, :] = 0.0

        if not SegFPN._printed_4ch_init_msg:
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
        feats = self.net.encoder(x)
        dec = self.net.decoder(feats)
        logits = self.net.segmentation_head(dec)
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits
