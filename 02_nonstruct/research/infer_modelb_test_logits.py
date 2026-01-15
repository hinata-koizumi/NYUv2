"""
infer_modelb_test_logits.py

Generate Model B (02_nonstruct) averaged test logits in *test_ids.txt order*.

Outputs:
- test_logits.npy: float16, (654, 13, 480, 640)
- test_ids.txt: filenames with .png, one per line (exact order used)

This is intended for LB check / submission building.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# NOTE: `02_nonstruct/model.py` shadows `02_nonstruct/model/` directory.
from ..model import ConvNeXtBaseFPN3Ch


CLASS_NAMES = [
    "bed",
    "books",
    "ceiling",
    "chair",
    "floor",
    "furniture",
    "objects",
    "picture",
    "sofa",
    "table",
    "tv",
    "wall",
    "window",
]
NUM_CLASSES = 13


def _die(msg: str) -> None:
    raise SystemExit(f"[MODEL-B-TEST][FATAL] {msg}")


def load_test_ids(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            out.append(t)
    return out


class TestRGBDataset(Dataset):
    def __init__(self, image_dir: str, ids_png: List[str]):
        self.image_dir = image_dir
        self.ids = ids_png
        self.paths = [os.path.join(image_dir, x) for x in ids_png]
        # ImageNet norm (same as 01_nearest / ModelB)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).float()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).float()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Ensure 480x640
        if img.shape[:2] != (480, 640):
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        x = (x - self.mean) / self.std
        return x, self.ids[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-ids", default="00_data/ids/test_ids.txt")
    ap.add_argument("--test-image-dir", default="00_data/test/image")
    ap.add_argument(
        "--ckpt-dir",
        default="00_data/model/02_nonstruct_frozen/ckpts",
    )
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    default_dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--device", default=default_dev)
    ap.add_argument("--out-dir", default="00_data/output/modelb_frozen_test")
    args = ap.parse_args()

    test_ids = load_test_ids(args.test_ids)
    if len(test_ids) != 654:
        print(f"[MODEL-B-TEST] warning: expected 654 test ids, got {len(test_ids)}")

    if not os.path.isdir(args.test_image_dir):
        _die(f"Missing test image dir: {args.test_image_dir}")

    ds = TestRGBDataset(args.test_image_dir, test_ids)
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    folds = [int(x) for x in str(args.folds).split(",") if str(x).strip() != ""]
    if not folds:
        _die("No folds specified")

    os.makedirs(args.out_dir, exist_ok=True)
    out_logits_path = os.path.join(args.out_dir, "test_logits.npy")
    out_acc_path = os.path.join(args.out_dir, "_test_logits_acc_f32.npy")
    out_ids_path = os.path.join(args.out_dir, "test_ids.txt")

    N = len(ds)
    acc = np.lib.format.open_memmap(out_acc_path, mode="w+", dtype=np.float32, shape=(N, NUM_CLASSES, 480, 640))
    acc[:] = 0.0

    device = torch.device(args.device)
    print(f"[MODEL-B-TEST] device={device} folds={folds} N={N} bs={int(args.batch_size)}")

    for k in folds:
        ckpt = os.path.join(args.ckpt_dir, f"best_fold{k}.pth")
        if not os.path.exists(ckpt):
            _die(f"Missing checkpoint: {ckpt}")

        model = ConvNeXtBaseFPN3Ch(num_classes=NUM_CLASSES, pretrained=False).to(device)
        sd = torch.load(ckpt, map_location="cpu")
        # handle common wrappers
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # strip possible prefixes
        if isinstance(sd, dict):
            sd2 = {}
            for kk, vv in sd.items():
                nk = str(kk)
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                sd2[nk] = vv
            sd = sd2
        model.load_state_dict(sd, strict=True)
        model.eval()

        idx0 = 0
        with torch.no_grad():
            for xb, _ids in dl:
                bsz = xb.shape[0]
                xb = xb.to(device)
                logits = model(xb)  # (B,13,480,640)
                if logits.shape[1] != NUM_CLASSES:
                    _die(f"unexpected C: {tuple(logits.shape)}")
                if logits.shape[2:] != (480, 640):
                    logits = torch.nn.functional.interpolate(logits, size=(480, 640), mode="bilinear", align_corners=False)
                acc[idx0 : idx0 + bsz] += logits.float().cpu().numpy()
                idx0 += bsz

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[MODEL-B-TEST] fold{k} done")

    acc /= float(len(folds))
    # write float16 logits
    out_f16 = np.lib.format.open_memmap(out_logits_path, mode="w+", dtype=np.float16, shape=(N, NUM_CLASSES, 480, 640))
    for i in range(N):
        out_f16[i] = acc[i].astype(np.float16, copy=False)

    with open(out_ids_path, "w", encoding="utf-8") as f:
        for x in test_ids:
            f.write(f"{x}\n")

    print(f"[MODEL-B-TEST] wrote: {os.path.abspath(out_logits_path)}")
    print(f"[MODEL-B-TEST] ids:   {os.path.abspath(out_ids_path)}")


if __name__ == "__main__":
    main()

