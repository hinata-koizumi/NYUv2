"""
infer_modelc_test_logits.py

Generate Model C averaged test logits in test_ids.txt order.
Outputs:
- test_logits.npy: float16 (N,13,480,640)
- test_ids.txt: filenames with .png, one per line (exact order used)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = "/root/datasets/NYUv2"
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "03_model_c"))

from configs import default as config
from data.dataset import ModelCDataset
from model.arch import ConvNeXtBaseFPNContext


def _die(msg: str) -> None:
    raise SystemExit(f"[MODEL-C-TEST][FATAL] {msg}")


def load_test_ids(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-ids", default="00_data/ids/test_ids.txt")
    ap.add_argument("--test-image-dir", default="00_data/test/image")
    ap.add_argument("--test-depth-dir", default="00_data/test/depth")
    ap.add_argument("--ckpt-dir", default="03_model_c/output")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    ap.add_argument("--device", default=default_dev)
    ap.add_argument("--out-dir", default="00_data/output/modelc_test")
    args = ap.parse_args()

    test_ids = load_test_ids(args.test_ids)
    if len(test_ids) != 654:
        print(f"[MODEL-C-TEST] warning: expected 654 test ids, got {len(test_ids)}")

    if not os.path.isdir(args.test_image_dir):
        _die(f"Missing test image dir: {args.test_image_dir}")
    if not os.path.isdir(args.test_depth_dir):
        _die(f"Missing test depth dir: {args.test_depth_dir}")

    def norm_id(x: str) -> str:
        s = str(x).strip()
        if s.endswith(".png") or s.endswith(".jpg"):
            return s
        return f"{s}.png"

    img_paths = [os.path.join(args.test_image_dir, norm_id(fid)) for fid in test_ids]
    dep_paths = [os.path.join(args.test_depth_dir, norm_id(fid)) for fid in test_ids]

    ds = ModelCDataset(
        image_paths=np.array(img_paths),
        label_paths=None,
        depth_paths=np.array(dep_paths),
        is_train=False,
        ids=test_ids,
    )
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True)

    folds = [int(x) for x in str(args.folds).split(",") if str(x).strip() != ""]
    if not folds:
        _die("No folds specified")

    os.makedirs(args.out_dir, exist_ok=True)
    out_logits_path = os.path.join(args.out_dir, "test_logits.npy")
    out_acc_path = os.path.join(args.out_dir, "_test_logits_acc_f32.npy")
    out_ids_path = os.path.join(args.out_dir, "test_ids.txt")

    N = len(ds)
    acc = np.lib.format.open_memmap(out_acc_path, mode="w+", dtype=np.float32, shape=(N, 13, 480, 640))
    acc[:] = 0.0

    device = torch.device(args.device)
    print(f"[MODEL-C-TEST] device={device} folds={folds} N={N} bs={int(args.batch_size)}")

    for k in folds:
        ckpt = os.path.join(args.ckpt_dir, f"fold{k}_last.pth")
        if not os.path.exists(ckpt):
            _die(f"Missing checkpoint: {ckpt}")

        model = ConvNeXtBaseFPNContext(
            num_classes=13,
            in_chans=config.IN_CHANS,
            pretrained=False,
            planar_head=config.PLANAR_HEAD_ENABLE,
        ).to(device)
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        model.eval()

        idx0 = 0
        with torch.no_grad():
            for batch in dl:
                xb = batch[0]
                bsz = xb.shape[0]
                xb = xb.to(device)
                output = model(xb)
                logits = output[0] if isinstance(output, (tuple, list)) else output
                if logits.shape[2:] != (480, 640):
                    logits = torch.nn.functional.interpolate(logits, size=(480, 640), mode="bilinear", align_corners=False)
                acc[idx0 : idx0 + bsz] += logits.float().cpu().numpy()
                idx0 += bsz

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[MODEL-C-TEST] fold{k} done")

    acc /= float(len(folds))
    out_f16 = np.lib.format.open_memmap(out_logits_path, mode="w+", dtype=np.float16, shape=(N, 13, 480, 640))
    for i in range(N):
        out_f16[i] = acc[i].astype(np.float16, copy=False)

    with open(out_ids_path, "w", encoding="utf-8") as f:
        for x in test_ids:
            f.write(f"{x}\n")

    print(f"[MODEL-C-TEST] wrote: {os.path.abspath(out_logits_path)}")
    print(f"[MODEL-C-TEST] ids:   {os.path.abspath(out_ids_path)}")


if __name__ == "__main__":
    main()
