import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

@dataclass(frozen=True, slots=True)
class Config:

    # --- Experiment ---
    EXP_NAME: str = "exp093_5_convnext_rgbd_4ch_base"
    SEED: int = 42
    N_FOLDS: int = 5

    # --- Paths ---
    DATA_ROOT: str = "data"
    OUTPUT_ROOT: str = "data/output"  # match existing repo outputs under `data/output/...`

    # --- Task ---
    NUM_CLASSES: int = 13
    IGNORE_INDEX: int = 255

    # --- Input / preprocessing (FIXED) ---
    # Fixed to "RGB + InverseDepth" 4ch style reproduction
    IN_CHANNELS: int = 4

    RESIZE_HEIGHT: int = 720
    RESIZE_WIDTH: int = 960
    CROP_SIZE: Optional[Tuple[int, int]] = (576, 768)  # (H, W)

    # Smart crop (fixed behavior)
    SMART_CROP_PROB: float = 0.5
    # Small object ids used by the smart-crop logic
    SMALL_OBJ_IDS: Tuple[int, ...] = (1, 3, 6, 7, 10)

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
    ETA_MIN: float = 1e-6
    GRAD_ACCUM_STEPS: int = 1

    # --- Mixed precision (FIXED) ---
    # Fixed to bf16 (recommended on RTX 4090). If your GPU doesn't support bf16 well,
    # change AMP_DTYPE to "fp16".
    USE_AMP: bool = True
    AMP_DTYPE: str = "bf16"  # "bf16" | "fp16"

    # --- Performance / determinism (FIXED) ---
    # These are applied via apply_runtime_settings().
    DETERMINISTIC: bool = False
    CUDNN_BENCHMARK: bool = True
    ALLOW_TF32: bool = True
    MATMUL_PRECISION: str = "high"  # "highest" | "high" | "medium"
    USE_CHANNELS_LAST: bool = True

    # --- EMA (FIXED) ---
    EMA_DECAY: float = 0.999

    # --- Checkpoints (FIXED) ---
    SAVE_TOP_K: int = 5

    # --- Loss (FIXED) ---
    # Segmentation: CrossEntropy only
    # Depth auxiliary loss (masked L1 on normalized linear depth target)
    USE_DEPTH_AUX: bool = True
    DEPTH_LOSS_LAMBDA: float = 0.1

    # --- TTA (FIXED) ---
    # 10 combinations: scales 0.5,0.75,1.0,1.25,1.5 × {no flip, flip}
    TTA_COMBS: Tuple[Tuple[float, bool], ...] = (
        (0.5, False), (0.5, True),
        (0.75, False), (0.75, True),
        (1.0, False), (1.0, True),
        (1.25, False), (1.25, True),
        (1.5, False), (1.5, True),
    )
    # Temperature sweep during validation
    TEMPERATURES: Tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)

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

        if bool(self.USE_DEPTH_AUX) and float(self.DEPTH_LOSS_LAMBDA) <= 0.0:
            errors.append("USE_DEPTH_AUX=True requires DEPTH_LOSS_LAMBDA > 0")

        if str(self.AMP_DTYPE).lower() not in ("bf16", "fp16"):
            errors.append(f"AMP_DTYPE must be 'bf16' or 'fp16' (got {self.AMP_DTYPE})")

        if str(self.MATMUL_PRECISION) not in ("highest", "high", "medium"):
            errors.append(f"MATMUL_PRECISION must be one of: highest/high/medium (got {self.MATMUL_PRECISION})")

        # Ensure TTA includes 0.5 scale (expected by this base)
        if not any(abs(float(s) - 0.5) < 1e-12 for s, _f in self.TTA_COMBS):
            errors.append("TTA_COMBS must include scale=0.5 for this base")

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
        if p in ("exp093_5", "exp093.5", "exp093_5_rgbd_4ch"):
            # Already matches this fixed base.
            return self
        raise ValueError(f"Unknown preset: {preset!r}. Supported: exp093_5")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Add derived fields explicitly
        d["TRAIN_DIR"] = self.TRAIN_DIR
        d["TEST_DIR"] = self.TEST_DIR
        d["DEVICE"] = self.DEVICE
        return d
