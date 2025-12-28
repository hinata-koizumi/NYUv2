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
        if in_channels != 4:
            return

        enc = self.net.encoder
        # timm convnext usually has stem[0]
        if hasattr(enc, "stem") and isinstance(enc.stem[0], nn.Conv2d):
            conv = enc.stem[0]
        elif hasattr(enc, "conv1") and isinstance(enc.conv1, nn.Conv2d):
            conv = enc.conv1
        else:
            return

        if conv.weight.shape[1] != 4:
            return

        with torch.no_grad():
            conv.weight[:, 3, :, :] = conv.weight[:, :3, :, :].mean(dim=1)
        if not SegFPN._printed_4ch_init_msg:
            print("SegFPN: initialized 4th channel weights = mean(RGB)")
            SegFPN._printed_4ch_init_msg = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net.encoder(x)
        dec = self.net.decoder(feats)
        logits = self.net.segmentation_head(dec)
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits
