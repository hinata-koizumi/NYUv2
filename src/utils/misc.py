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

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def init_logger(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "train_log.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss", "valid_loss", "valid_miou", "valid_pixel_acc"])
    return path

def log_metrics(path: str, epoch: int, lr: float, tr_loss: float, va_loss: float, miou: float, acc: float) -> None:
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch, lr, tr_loss, va_loss, miou, acc])

def save_config(cfg, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
