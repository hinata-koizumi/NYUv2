import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

@dataclass(frozen=True, slots=True)
class Config:

    # --- Experiment ---
    EXP_NAME: str = "exp097_phase1_refactor"
    SEED: int = 42
    N_FOLDS: int = 5

    # --- Paths ---
    DATA_ROOT: str = "data"
    OUTPUT_ROOT: str = "data/output"

    # --- Task ---
    NUM_CLASSES: int = 13
    IGNORE_INDEX: int = 255

    # --- Input / preprocessing (FIXED) ---
    IN_CHANNELS: int = 4

    RESIZE_HEIGHT: int = 720
    RESIZE_WIDTH: int = 960
    CROP_SIZE: Optional[Tuple[int, int]] = (576, 768)  # (H, W)

    # Smart crop (fixed behavior)
    SMART_CROP_PROB: float = 0.7
    # Small object ids used by the smart-crop logic
    SMALL_OBJ_IDS: Tuple[int, ...] = (1, 3, 6, 7, 10)
    # Smart-crop zoom
    SMART_CROP_ZOOM_PROB: float = 0.6
    SMART_CROP_ZOOM_RANGE: Tuple[float, float] = (0.3, 0.6)  # ABSOLUTE RULE: (0.3, 0.6)
    SMART_CROP_ZOOM_ONLY_SMALL: bool = True

    # Copy-Paste augmentation (Mild Setting)
    COPY_PASTE_ENABLE: bool = True
    COPY_PASTE_PROB: float = 0.3
    COPY_PASTE_MAX_OBJS: int = 3
    COPY_PASTE_OBJ_IDS: Tuple[int, ...] = (1, 3, 6, 7, 10)
    COPY_PASTE_BG_IDS: Tuple[int, ...] = (4, 5, 11)
    COPY_PASTE_BG_MIN_COVER: float = 0.5
    COPY_PASTE_MIN_AREA: int = 20
    COPY_PASTE_MAX_AREA: int = 0
    COPY_PASTE_MAX_AREA_RATIO: float = 0.03
    COPY_PASTE_MAX_TRIES: int = 20
    COPY_PASTE_MAX_OBJS_TOTAL: int = 0

    # Depth dropout (Disabled for Phase 1)
    DEPTH_CHANNEL_DROPOUT_PROB: float = 0.0
    DEPTH_COARSE_DROPOUT_PROB: float = 0.0
    DEPTH_COARSE_DROPOUT_MAX_HOLES: int = 4
    DEPTH_COARSE_DROPOUT_MIN_FRAC: float = 0.05
    DEPTH_COARSE_DROPOUT_MAX_FRAC: float = 0.2

    # Depth range (meters)
    DEPTH_MIN: float = 0.6
    DEPTH_MAX: float = 10.0

    # RGB normalization
    MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    STRICT_DEPTH_FOR_TRAIN: bool = True
    SANITY_CHECK_FIRST_N: int = 20

    # --- Train (FIXED) ---
    EPOCHS: int = 50
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    OPTIMIZER: str = "sam_adamw"  # FIXED: SAM
    SAM_RHO: float = 0.02         # FIXED: 0.02
    ETA_MIN: float = 1e-6
    GRAD_ACCUM_STEPS: int = 1
    # Gradient clipping
    GRAD_CLIP_NORM: float = 1.0

    # --- Mixed precision ---
    USE_AMP: bool = True
    AMP_DTYPE: str = "bf16"

    # --- Performance / determinism ---
    DETERMINISTIC: bool = False
    CUDNN_BENCHMARK: bool = True
    ALLOW_TF32: bool = True
    MATMUL_PRECISION: str = "high"
    USE_CHANNELS_LAST: bool = True

    # --- EMA (FIXED) ---
    USE_EMA: bool = True   # Explicit Control
    EMA_DECAY: float = 0.999

    # --- Checkpoints ---
    SAVE_TOP_K: int = 5

    # --- Loss (FIXED) ---
    # Depth auxiliary loss (Disabled)
    USE_DEPTH_AUX: bool = False
    DEPTH_LOSS_LAMBDA: float = 0.0

    # --- TTA (FIXED) ---
    # FIXED: [1.0, 1.25, 1.5] scale combinations
    TTA_COMBS: Tuple[Tuple[float, bool], ...] = (
        (1.0, False), (1.0, True),
        (1.25, False), (1.25, True),
        (1.5, False), (1.5, True),
    )
    # Temperature sweep during validation
    TEMPERATURES: Tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)

    # --- Submit-time ensembling (inference only) ---
    # Use top-K checkpoints *per fold* (by filename mIoU) in addition to model_best.pth.
    # This can improve leaderboard score without retraining, at the cost of inference time.
    # 1 disables (current behavior).
    SUBMIT_CKPT_ENSEMBLE_K: int = 1

    # --- Logging / verbosity (FIXED) ---
    VERBOSE: bool = False
    DEBUG: bool = False
    SAVE_VIS: bool = False

    @property
    def TRAIN_DIR(self) -> str:
        return os.path.join(self.DATA_ROOT, "train")

    @property
    def TEST_DIR(self) -> str:
        return os.path.join(self.DATA_ROOT, "test")

    @property
    def DEVICE(self) -> str:
        try:
            import torch  # local import keeps config importable without torch
        except Exception:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def apply_runtime_settings(self) -> None:
        """
        Apply global torch runtime settings that affect performance / determinism.
        Call this once at the beginning of train/submit scripts.
        """
        import torch  # local import keeps config importable without torch

        torch.backends.cudnn.benchmark = bool(self.CUDNN_BENCHMARK)
        torch.backends.cudnn.deterministic = bool(self.DETERMINISTIC)

        # Matmul precision (PyTorch 2.x)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(self.MATMUL_PRECISION))

        # TF32 (CUDA only)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = bool(self.ALLOW_TF32)
            torch.backends.cudnn.allow_tf32 = bool(self.ALLOW_TF32)

    def validate(self) -> None:
        errors = []

        if int(self.IN_CHANNELS) != 4:
            errors.append(f"IN_CHANNELS must be 4 for this fixed base (got {self.IN_CHANNELS})")

        if self.RESIZE_HEIGHT <= 0 or self.RESIZE_WIDTH <= 0:
            errors.append("RESIZE_HEIGHT/RESIZE_WIDTH must be positive")

        if self.CROP_SIZE is not None:
            ch, cw = self.CROP_SIZE
            if ch <= 0 or cw <= 0:
                errors.append("CROP_SIZE must be positive when set")
            if ch > self.RESIZE_HEIGHT or cw > self.RESIZE_WIDTH:
                errors.append(
                    f"CROP_SIZE {self.CROP_SIZE} must be <= RESIZE (H,W)=({self.RESIZE_HEIGHT},{self.RESIZE_WIDTH})"
                )

        if self.DEPTH_MIN <= 0 or self.DEPTH_MAX <= 0 or self.DEPTH_MIN >= self.DEPTH_MAX:
            errors.append(f"DEPTH_MIN/DEPTH_MAX invalid: min={self.DEPTH_MIN}, max={self.DEPTH_MAX}")

        zoom_p = float(getattr(self, "SMART_CROP_ZOOM_PROB", 0.0))
        if zoom_p < 0.0 or zoom_p > 1.0:
            errors.append(f"SMART_CROP_ZOOM_PROB must be in [0,1] (got {zoom_p})")
        zoom_range = getattr(self, "SMART_CROP_ZOOM_RANGE", (1.0, 1.0))
        if len(zoom_range) != 2:
            errors.append(f"SMART_CROP_ZOOM_RANGE must be a 2-tuple (got {zoom_range})")
        else:
            z0, z1 = float(zoom_range[0]), float(zoom_range[1])
            if z0 <= 0.0 or z1 <= 0.0 or z0 > z1 or z1 > 1.0:
                errors.append(f"SMART_CROP_ZOOM_RANGE invalid: {zoom_range} (expect 0<min<=max<=1)")

        cp_prob = float(getattr(self, "COPY_PASTE_PROB", 0.0))
        if cp_prob < 0.0 or cp_prob > 1.0:
            errors.append(f"COPY_PASTE_PROB must be in [0,1] (got {cp_prob})")
        cp_max = int(getattr(self, "COPY_PASTE_MAX_OBJS", 1))
        if cp_max < 1:
            errors.append(f"COPY_PASTE_MAX_OBJS must be >=1 (got {cp_max})")
        cp_cover = float(getattr(self, "COPY_PASTE_BG_MIN_COVER", 0.5))
        if cp_cover < 0.0 or cp_cover > 1.0:
            errors.append(f"COPY_PASTE_BG_MIN_COVER must be in [0,1] (got {cp_cover})")
        cp_min_area = int(getattr(self, "COPY_PASTE_MIN_AREA", 0))
        if cp_min_area < 0:
            errors.append(f"COPY_PASTE_MIN_AREA must be >=0 (got {cp_min_area})")
        cp_max_area = int(getattr(self, "COPY_PASTE_MAX_AREA", 0))
        if cp_max_area < 0:
            errors.append(f"COPY_PASTE_MAX_AREA must be >=0 (got {cp_max_area})")
        cp_max_ratio = float(getattr(self, "COPY_PASTE_MAX_AREA_RATIO", 0.0))
        if cp_max_ratio < 0.0:
            errors.append(f"COPY_PASTE_MAX_AREA_RATIO must be >=0 (got {cp_max_ratio})")

        depth_drop = float(getattr(self, "DEPTH_CHANNEL_DROPOUT_PROB", 0.0))
        if depth_drop < 0.0 or depth_drop > 1.0:
            errors.append(f"DEPTH_CHANNEL_DROPOUT_PROB must be in [0,1] (got {depth_drop})")
        depth_coarse = float(getattr(self, "DEPTH_COARSE_DROPOUT_PROB", 0.0))
        if depth_coarse < 0.0 or depth_coarse > 1.0:
            errors.append(f"DEPTH_COARSE_DROPOUT_PROB must be in [0,1] (got {depth_coarse})")
        coarse_min = float(getattr(self, "DEPTH_COARSE_DROPOUT_MIN_FRAC", 0.0))
        coarse_max = float(getattr(self, "DEPTH_COARSE_DROPOUT_MAX_FRAC", 0.0))
        if coarse_min < 0.0 or coarse_max < 0.0 or coarse_min > coarse_max or coarse_max > 1.0:
            errors.append(
                f"DEPTH_COARSE_DROPOUT_MIN_FRAC/MAX_FRAC invalid: min={coarse_min}, max={coarse_max}"
            )

        if bool(self.USE_DEPTH_AUX) and float(self.DEPTH_LOSS_LAMBDA) <= 0.0:
            errors.append("USE_DEPTH_AUX=True requires DEPTH_LOSS_LAMBDA > 0")

        if str(self.AMP_DTYPE).lower() not in ("bf16", "fp16"):
            errors.append(f"AMP_DTYPE must be 'bf16' or 'fp16' (got {self.AMP_DTYPE})")

        if str(self.MATMUL_PRECISION) not in ("highest", "high", "medium"):
            errors.append(f"MATMUL_PRECISION must be one of: highest/high/medium (got {self.MATMUL_PRECISION})")

        # Removed scale=0.5 validation check as we are intentionally using larger scales now

        if errors:
            raise ValueError("Invalid Config:\n- " + "\n- ".join(errors))

    def with_overrides(self, **kwargs: Any) -> "Config":
        """
        Return a new Config with the provided field overrides applied.

        This keeps Config immutable (frozen=True) while still allowing controlled
        changes from CLI (e.g., EXP_NAME, N_FOLDS).
        """
        return replace(self, **kwargs)

    def apply_preset(self, preset: str) -> "Config":
        """
        Doc reproduction presets.

        This repository's `base_config.py` is already fixed to the exp093_5 4ch setting.
        We keep the preset hook for CLI compatibility, but only accept known values.
        """
        p = str(preset).strip()
        if p in ("exp093_5", "exp093.5", "exp093_5_rgbd_4ch", "exp097", "exp097_0"):
            # Already matches this fixed base.
            return self
        raise ValueError(f"Unknown preset: {preset!r}. Supported: exp093_5, exp097")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Add derived fields explicitly
        d["TRAIN_DIR"] = self.TRAIN_DIR
        d["TEST_DIR"] = self.TEST_DIR
        d["DEVICE"] = self.DEVICE
        return d
