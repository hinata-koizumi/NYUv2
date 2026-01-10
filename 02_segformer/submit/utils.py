"""
Internal utilities for the submit pipeline (Exp100 Final).
Ensures TTA settings match the training configuration.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import fields


def load_cfg_from_fold_dir(fold_dir: str):
    """
    Load `main.configs.base_config.Config` values from `config_resolved.json`.
    """
    from ..configs.base_config import Config

    p = os.path.join(fold_dir, "config_resolved.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing config: {p}")

    with open(p, "r") as f:
        d = json.load(f)

    valid_keys = {f.name for f in fields(Config)}
    overrides = {k: v for k, v in d.items() if k in valid_keys}

    # JSON list -> tuple where appropriate
    if "CROP_SIZE" in overrides and overrides["CROP_SIZE"] is not None:
        overrides["CROP_SIZE"] = tuple(overrides["CROP_SIZE"])
    if "MEAN" in overrides and overrides["MEAN"] is not None:
        overrides["MEAN"] = tuple(overrides["MEAN"])
    if "STD" in overrides and overrides["STD"] is not None:
        overrides["STD"] = tuple(overrides["STD"])
    if "SMALL_OBJ_IDS" in overrides and overrides["SMALL_OBJ_IDS"] is not None:
        overrides["SMALL_OBJ_IDS"] = tuple(overrides["SMALL_OBJ_IDS"])
    if "TTA_COMBS" in overrides and overrides["TTA_COMBS"] is not None:
        overrides["TTA_COMBS"] = tuple(tuple(x) for x in overrides["TTA_COMBS"])
    if "TEMPERATURES" in overrides and overrides["TEMPERATURES"] is not None:
        overrides["TEMPERATURES"] = tuple(float(x) for x in overrides["TEMPERATURES"])

    cfg = Config().with_overrides(**overrides)
    cfg.validate()
    return cfg


def discover_folds(exp_dir: str) -> list[int]:
    """Discover fold indices under exp_dir by matching `fold{idx}` directories."""
    pat = re.compile(r"^fold(\d+)$")
    folds: list[int] = []
    if not os.path.exists(exp_dir):
         raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
         
    for name in sorted(os.listdir(exp_dir)):
        m = pat.match(name)
        if not m:
            continue
        p = os.path.join(exp_dir, name)
        if os.path.isdir(p):
            folds.append(int(m.group(1)))
    if len(folds) == 0:
        raise FileNotFoundError(f"No fold directories found under: {exp_dir}")
    return sorted(folds)


def best_ckpt_path(fold_dir: str) -> str:
    """Exp100 recipe: best ckpt is `model_best.pth` only."""
    p = os.path.join(fold_dir, "model_best.pth")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing weights: {p}")
    return p


def fixed_tta_combs() -> list[tuple[float, bool]]:
    """
    Exp100 TTA Recipe (Matches Config):
      - scales: [1.0, 1.25, 1.5] (Zoom-in logic for small objects)
      - hflip: [False, True]
    """
    scales = [0.75, 1.0]
    return [(float(s), False) for s in scales] + [(float(s), True) for s in scales]


def safe_torch_load(path: str, *, map_location: str = "cpu"):
    """
    Load a state_dict from disk safely.
    """
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions
        return torch.load(path, map_location=map_location)