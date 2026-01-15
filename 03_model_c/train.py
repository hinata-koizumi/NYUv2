import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

sys.path.append("/root/datasets/NYUv2")

from configs import default as config
from data.dataset import ModelCDataset
from model.loss import ModelCLoss
from model.arch import ConvNeXtBaseFPNContext
from utils.common import calculate_metrics, MetricAggregator, save_logits

import importlib
n01 = importlib.import_module("01_nearest.constants")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="Run 1 iter")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def _unpack_batch(batch):
    if config.PLANAR_HEAD_ENABLE:
        x, y, ids, planar_target, planar_valid = batch
    else:
        x, y, ids = batch
        planar_target, planar_valid = None, None
    return x, y, ids, planar_target, planar_valid


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, epoch, device):
    model.train()
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False)

    losses = []

    for batch in pbar:
        x, y, _, planar_target, planar_valid = _unpack_batch(batch)
        x, y = x.to(device).contiguous(), y.to(device).contiguous()
        if planar_target is not None:
            planar_target = planar_target.to(device)
        if planar_valid is not None:
            planar_valid = planar_valid.to(device)

        optimizer.zero_grad()

        dev_type = device.type
        if dev_type == "mps":
            enable_amp = False
        elif dev_type == "cpu":
            enable_amp = True
            dev_type = "cpu"
        else:
            enable_amp = True

        if enable_amp:
            with torch.amp.autocast(device_type=dev_type):
                output = model(x)
                if config.PLANAR_HEAD_ENABLE:
                    logits, planar_logits = output
                    loss = loss_fn(logits, y, planar_logits, planar_target, planar_valid)
                else:
                    logits = output
                    loss = loss_fn(logits, y)
        else:
            output = model(x)
            if config.PLANAR_HEAD_ENABLE:
                logits, planar_logits = output
                loss = loss_fn(logits, y, planar_logits, planar_target, planar_valid)
            else:
                logits = output
                loss = loss_fn(logits, y)

        if scaler is not None and enable_amp and dev_type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())

    return np.mean(losses)


def validate(model, loader, device, output_dir=None, save_preds=False):
    model.eval()
    agg = MetricAggregator()

    all_logits = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            x, y, ids, _, _ = _unpack_batch(batch)
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type=device.type if device.type != "mps" else "cpu"):
                output = model(x)
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output

            if logits.shape[2:] != (480, 640):
                logits = torch.nn.functional.interpolate(
                    logits, size=(480, 640), mode="bilinear", align_corners=False
                )

            m_dict = calculate_metrics(logits, y, device=device)
            agg.update(m_dict)

            if save_preds:
                all_logits.append(logits.float().cpu().numpy())
                all_ids.extend(ids)

    metrics = agg.compute()

    if save_preds and output_dir:
        full_logits = np.concatenate(all_logits, axis=0)
        save_logits(full_logits, all_ids, output_dir, file_prefix="oof_fold_temp")

    return metrics


def main():
    args = get_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Verifying Class Consistency...")
    if config.CLASS_NAMES != n01.CLASS_NAMES:
        print("[FAIL] CLASS_NAMES mismatch!")
        print(f"Local: {config.CLASS_NAMES}")
        print(f"01:    {n01.CLASS_NAMES}")
        raise ValueError("Class Name Mismatch with 01_nearest")

    with open(config.SPLITS_FILE) as f:
        manifest = json.load(f)

    fold_filename = manifest["folds"][args.fold]
    fold_path = os.path.join(os.path.dirname(config.SPLITS_FILE), fold_filename)

    with open(fold_path) as f:
        fold_data = json.load(f)

    train_ids = fold_data["train_ids"]
    val_ids = fold_data["val_ids"]

    img_dir = os.path.join(config.DATA_DIR, "train/image")
    lbl_dir = os.path.join(config.DATA_DIR, "train/label")
    dep_dir = os.path.join(config.DATA_DIR, "train/depth")

    def get_paths(id_list):
        imgs = [os.path.join(img_dir, f"{i}.jpg") for i in id_list]
        lbls = [os.path.join(lbl_dir, f"{i}.png") for i in id_list]
        deps = [os.path.join(dep_dir, f"{i}.png") for i in id_list]
        if not os.path.exists(imgs[0]):
            imgs = [os.path.join(img_dir, f"{i}.png") for i in id_list]
        return np.array(imgs), np.array(lbls), np.array(deps)

    train_imgs, train_lbls, train_deps = get_paths(train_ids)
    val_imgs, val_lbls, val_deps = get_paths(val_ids)

    if args.smoke:
        train_imgs, train_lbls, train_deps = train_imgs[:8], train_lbls[:8], train_deps[:8]
        val_imgs, val_lbls, val_deps = val_imgs[:4], val_lbls[:4], val_deps[:4]

    ds_train = ModelCDataset(train_imgs, train_lbls, train_deps, is_train=True, ids=train_ids)
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ds_val = ModelCDataset(val_imgs, val_lbls, val_deps, is_train=False, ids=val_ids)
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNeXtBaseFPNContext(
        num_classes=13,
        in_chans=config.IN_CHANS,
        pretrained=True,
        planar_head=config.PLANAR_HEAD_ENABLE,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = ModelCLoss()

    for ep in range(args.epochs):
        train_loss = train_one_epoch(model, dl_train, optimizer, scaler, loss_fn, ep, device)
        metrics = validate(model, dl_val, device)
        scheduler.step()

        log_str = f"Ep {ep}: Loss {train_loss:.4f} "
        log_str += f"| F/W: {metrics.get('iou_floor',0):.3f}/{metrics.get('iou_wall',0):.3f} "
        log_str += f"| ObjRat: {metrics.get('ratio_objects_global',0):.2f} "
        log_str += f"| ObjPct: {metrics.get('pred_objects_percent_global',0):.1f}% "
        log_str += f"| Suck: {metrics.get('suck_rate',0):.3f} "
        log_str += f"| Books/Table: {metrics.get('iou_books',0):.3f}/{metrics.get('iou_table',0):.3f}"
        print(log_str)

        if metrics.get("iou_floor", 0) < 0.1 or metrics.get("iou_wall", 0) < 0.1:
            print("SAFETY STOP: Floor/Wall collapsed.")
            break

        if metrics.get("ratio_objects_global", 0) > 1.25:
            print(f"SAFETY STOP: Objects explosion (global ratio={metrics.get('ratio_objects_global',0):.2f}).")
            break

        if metrics.get("pred_objects_percent_global", 0) > 25.0:
            print(f"SAFETY STOP: Objects percentage too high ({metrics.get('pred_objects_percent_global',0):.1f}%).")
            break

        torch.save(
            {
                "epoch": ep,
                "state_dict": model.state_dict(),
                "metrics": metrics,
            },
            os.path.join(config.OUTPUT_DIR, f"fold{args.fold}_last.pth"),
        )

    print("Running Final Inference on Val...")
    validate(model, dl_val, device, output_dir=config.OUTPUT_DIR, save_preds=True)

    src_npy = os.path.join(config.OUTPUT_DIR, "oof_fold_temp_logits.npy")
    src_txt = os.path.join(config.OUTPUT_DIR, "oof_fold_temp_ids.txt")

    dst_npy = os.path.join(config.OUTPUT_DIR, f"oof_fold{args.fold}_logits.npy")
    dst_txt = os.path.join(config.OUTPUT_DIR, f"val_ids_fold{args.fold}.txt")

    if os.path.exists(src_npy):
        saved_ids_arr = np.loadtxt(src_txt, dtype=str)
        if len(saved_ids_arr) != len(val_ids):
            print(f"[WARN] OOF count mismatch: {len(saved_ids_arr)} vs {len(val_ids)}")
        elif not np.array_equal(saved_ids_arr, val_ids):
            print("[WARN] OOF ID Order Mismatch! Check saving logic.")
        else:
            print("[OK] OOF Logits Order Verified (Matches val_ids).")

        os.rename(src_npy, dst_npy)
        os.rename(src_txt, dst_txt)


if __name__ == "__main__":
    main()
