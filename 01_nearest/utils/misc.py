import os
import random
import numpy as np
import torch
import torch.nn as nn
import json
import csv
import re
import copy
from typing import List, Tuple, Dict, Any, Optional
import subprocess
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        return "unknown"


def configure_runtime(cfg) -> None:
    """
    Configure PyTorch runtime flags for performance/reproducibility.
    """
    if cfg is not None and hasattr(cfg, "apply_runtime_settings"):
        try:
            cfg.apply_runtime_settings()
            return
        except Exception:
            pass

    # Fallback defaults
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def worker_init_fn(worker_id: int) -> None:
    # Use torch's initial seed to generate a unique seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ModelEMA:
    """
    Model Exponential Moving Average.
    """
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
                if k in msd:
                    model_v = msd[k]
                    if not torch.is_floating_point(model_v):
                        ema_v.copy_(model_v)
                    else:
                        ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)


class CheckpointManager:
    """
    Manages top-K checkpoints based on mIoU.
    """
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

        # Delete extras
        for _miou, p in found[self.top_k :]:
            try:
                os.remove(p)
            except OSError:
                pass

    def save_artifact(self, state_dict: dict, path: str) -> None:
        """
        Public method to safely save a generic artifact (e.g. model_best.pth) atomically.
        """
        tmp = path + ".tmp"
        try:
            torch.save(state_dict, tmp)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)

    def save(self, model: nn.Module, epoch: int, miou: float) -> None:
        path = os.path.join(self.save_dir, f"model_epoch{epoch}_miou{miou:.4f}.pth")
        self.save_artifact(model.state_dict(), path)

        self.items.append((float(miou), path))
        self.items.sort(key=lambda x: x[0], reverse=True)

        while len(self.items) > self.top_k:
            _, p = self.items.pop()
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def save_last(self, model: nn.Module, optimizer: Any, epoch: int, miou: float):
        """Save latest checkpoint for resuming."""
        path = os.path.join(self.save_dir, "last.pth")
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "miou": miou
        }
        torch.save(state, path)


class Logger:
    """
    Logs metrics to TensorBoard and CSV.
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        self.tb = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        
        self.metrics_path = os.path.join(out_dir, "metrics.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "lr", "train_loss", "valid_loss", "valid_miou", "valid_pixel_acc", "grad_norm", "aux_loss"])
            
        self.class_iou_path = os.path.join(out_dir, "classwise_iou.csv")

    def log_epoch(self, epoch: int, data: dict):
        # CSV
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, 
                data.get("lr", 0),
                data.get("train_loss", 0),
                data.get("valid_loss", 0),
                data.get("valid_miou", 0),
                data.get("valid_pixel_acc", 0),
                data.get("grad_norm", 0),
                data.get("aux_loss", 0),
            ])
            
        # TensorBoard
        self.tb.add_scalar("valid/mIoU", data.get("valid_miou", 0), epoch)
        self.tb.add_scalar("valid/loss", data.get("valid_loss", 0), epoch)
        self.tb.add_scalar("train/loss_avg", data.get("train_loss", 0), epoch)
        
        if "throughput" in data:
            self.tb.add_scalar("perf/img_per_sec", data["throughput"], epoch)
        if "grad_norm" in data:
            self.tb.add_scalar("train/grad_norm", data["grad_norm"], epoch)
        if "aux_loss" in data:
            self.tb.add_scalar("train/aux_loss", data["aux_loss"], epoch)
            
        # Generic fallback for any other keys in data that are scalars
        for k, v in data.items():
            if k not in ["lr", "train_loss", "valid_loss", "valid_miou", "valid_pixel_acc", "class_iou", "throughput", "grad_norm", "aux_loss"]:
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(f"extra/{k}", v, epoch)
            
        # Class-wise
        if "class_iou" in data:
            c_iou = data["class_iou"]
            # Header check
            if not os.path.exists(self.class_iou_path) or os.path.getsize(self.class_iou_path) == 0:
                with open(self.class_iou_path, "w", newline="") as f:
                    w = csv.writer(f)
                    header = ["epoch"] + [f"class_{k}" for k in sorted(c_iou.keys())]
                    w.writerow(header)
            
            with open(self.class_iou_path, "a", newline="") as f:
                w = csv.writer(f)
                row = [epoch] + [c_iou[k] for k in sorted(c_iou.keys())]
                w.writerow(row)

    def save_summary(self, data: dict):
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(data, f, indent=2)

    def close(self):
        self.tb.close()


def save_config(cfg, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_resolved.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)