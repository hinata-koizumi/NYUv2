
import os
import random
import numpy as np
import torch
import torch.nn as nn
import json
import csv
import re
import copy
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # True for 4090 optimization user said? But deterministic often needs False. User suggested True for speed. Setting True.
    if torch.cuda.is_available():
         torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id: int) -> None:
    seed = int(np.random.get_state()[1][0]) + int(worker_id)
    np.random.seed(seed)
    random.seed(seed)

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                model_v = msd[k]
                if not torch.is_floating_point(model_v):
                    ema_v.copy_(model_v)
                else:
                    ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)

class CheckpointManager:
    def __init__(self, save_dir: str, top_k: int):
        self.save_dir = save_dir
        self.top_k = int(top_k)
        self.items: List[Tuple[float, str]] = []
        os.makedirs(save_dir, exist_ok=True)
        self._prune_existing()

    def _prune_existing(self) -> None:
        pat = re.compile(r"^model_epoch(\d+)_miou([0-9.]+)\.pth$")
        found: List[Tuple[float, str]] = []

        try:
            files = os.listdir(self.save_dir)
        except FileNotFoundError:
            return

        for fn in files:
            m = pat.match(fn)
            if not m:
                continue
            miou = float(m.group(2))
            path = os.path.join(self.save_dir, fn)
            found.append((miou, path))

        found.sort(key=lambda x: x[0], reverse=True)
        self.items = found[: self.top_k]

        for _miou, p in found[self.top_k :]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    def _atomic_torch_save(self, state_dict: dict, path: str) -> None:
        tmp = path + ".tmp"
        try:
            torch.save(state_dict, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def save(self, model: nn.Module, epoch: int, miou: float) -> None:
        path = os.path.join(self.save_dir, f"model_epoch{epoch}_miou{miou:.4f}.pth")
        self._atomic_torch_save(model.state_dict(), path)

        self.items.append((float(miou), path))
        self.items.sort(key=lambda x: x[0], reverse=True)

        while len(self.items) > self.top_k:
            _, p = self.items.pop()
            if os.path.exists(p):
                os.remove(p)

class Logger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. TensorBoard
        self.tb = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        
        # 2. CSVs
        self.metrics_path = os.path.join(out_dir, "metrics.csv")
        with open(self.metrics_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "lr", "train_loss", "valid_loss", "valid_miou", "valid_pixel_acc"])
            
        self.class_iou_path = os.path.join(out_dir, "classwise_iou.csv")
        # Header will be written on first log
        
        # 3. Artifacts
        self.artifacts_dir = os.path.join(out_dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def log_step(self, step: int, data: dict):
        for k, v in data.items():
            self.tb.add_scalar(k, v, step)
            
    def log_epoch(self, epoch: int, data: dict):
        # Data contains: lr, train_loss, valid_loss, valid_miou, valid_pixel_acc, class_iou (dict)
        # Optional: perf/throughput, gpu/mem
        
        # CSV
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, 
                data.get("lr", 0),
                data.get("train_loss", 0),
                data.get("valid_loss", 0),
                data.get("valid_miou", 0),
                data.get("valid_pixel_acc", 0)
            ])
            
        # TB: Main Metrics
        self.tb.add_scalar("valid/mIoU", data.get("valid_miou", 0), epoch)
        self.tb.add_scalar("valid/loss", data.get("valid_loss", 0), epoch)
        self.tb.add_scalar("train/loss_avg", data.get("train_loss", 0), epoch)
        
        if "valid_miou_ema" in data:
            self.tb.add_scalar("valid/mIoU_ema", data["valid_miou_ema"], epoch)
            
        # TB: LR Groups
        if "lr_stem" in data:
            self.tb.add_scalar("lr/stem", data["lr_stem"], epoch)
        if "lr_base" in data:
            self.tb.add_scalar("lr/base", data["lr_base"], epoch)
            
        # TB: Perf/GPU
        if "throughput" in data:
            self.tb.add_scalar("perf/img_per_sec", data["throughput"], epoch)
        if "gpu_mem" in data:
            self.tb.add_scalar("gpu/mem_alloc_max", data["gpu_mem"], epoch)
            
        # Class-wise CSV
        if "class_iou" in data:
            c_iou = data["class_iou"] # dict {0: val, 1: val...}
            # If first time, write header
            if not os.path.exists(self.class_iou_path) or os.path.getsize(self.class_iou_path) == 0:
                with open(self.class_iou_path, "w", newline="") as f:
                    w = csv.writer(f)
                    header = ["epoch"] + [f"class_{k}" for k in sorted(c_iou.keys())]
                    w.writerow(header)
            
            with open(self.class_iou_path, "a", newline="") as f:
                w = csv.writer(f)
                row = [epoch] + [c_iou[k] for k in sorted(c_iou.keys())]
                w.writerow(row)
                
            # TB for Class-wise
            for k, v in c_iou.items():
                self.tb.add_scalar(f"class_iou/class_{k}", v, epoch)
                
        # Main Metrics TB
        self.tb.add_scalar("valid/mIoU", data.get("valid_miou", 0), epoch)
        self.tb.add_scalar("valid/loss", data.get("valid_loss", 0), epoch)
        self.tb.add_scalar("train/loss_avg", data.get("train_loss", 0), epoch)

    def save_summary(self, data: dict):
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(data, f, indent=2)

    def save_image(self, name: str, img_np: np.ndarray):
        import cv2
        path = os.path.join(self.artifacts_dir, name)
        # img_np assumed RGB HWC
        cv2.imwrite(path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    def close(self):
        self.tb.close()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.stop = False

    def __call__(self, val_score):
        score = val_score
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0

def save_config(cfg, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_resolved.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
