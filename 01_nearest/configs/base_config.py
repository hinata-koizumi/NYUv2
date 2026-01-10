import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple
from ..constants import (
    NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES, CLASS_ID_BOOKS, CLASS_ID_TABLE,
    IN_CHANNELS, RESIZE_HEIGHT, RESIZE_WIDTH, CROP_SIZE,
    SPLIT_MODE, SPLIT_BLOCK_SIZE, TTA_COMBS, DEFAULT_OUTPUT_ROOT
)

@dataclass(frozen=True, slots=True)
class Config:
    """
    Exp100 Final Configuration (Nearest Depth + Zero Init + TTA Fix).
    This configuration is FROZEN for the final training run.
    """

    # --- Experiment ---
    EXP_NAME: str = "exp100_final_01_nearest"
    SEED: int = 42
    N_FOLDS: int = 5

    # --- Paths ---
    DATA_ROOT: str = "00_data"
    OUTPUT_ROOT: str = "00_data/output"
    SPLIT_ROOT: str = "03_ensemble/splits"

    # --- Task ---
    NUM_CLASSES: int = NUM_CLASSES
    IGNORE_INDEX: int = IGNORE_INDEX
    # Standard NYUv2 Class Mapping
    CLASS_NAMES: Tuple[str, ...] = tuple(CLASS_NAMES)
    # Critical Class IDs (User Verification Required: Checked against standard list)
    CLASS_ID_BOOKS: int = CLASS_ID_BOOKS
    CLASS_ID_TABLE: int = CLASS_ID_TABLE

    # --- Input / preprocessing (FIXED) ---
    IN_CHANNELS: int = IN_CHANNELS  # RGB + Depth

    RESIZE_HEIGHT: int = RESIZE_HEIGHT
    RESIZE_WIDTH: int = RESIZE_WIDTH
    
    # Exp-A: Dynamic Resize Lower Bound
    DYNAMIC_RESIZE_MIN_SCALE: float = 0.75
    CROP_SIZE: Optional[Tuple[int, int]] = CROP_SIZE  # (H, W)

    # Smart crop (fixed behavior)
    SMART_CROP_PROB: float = 0.7
    SMART_CROP_TABLE_BONUS_PROB: float = 0.3 # Bonus prob for Table (Prioritize Table)
    # Small object ids used by the smart-crop logic (Added 9=Table)
    SMALL_OBJ_IDS: Tuple[int, ...] = (1, 3, 6, 7, 9, 10)
    
    # Smart-crop zoom (Aggressive Zoom for Small Objects)
    SMART_CROP_ZOOM_PROB: float = 0.6
    SMART_CROP_ZOOM_RANGE: Tuple[float, float] = (0.3, 0.6)  # Fixed rule
    SMART_CROP_ZOOM_ONLY_SMALL: bool = True

    # Copy-Paste augmentation (Mild Setting)
    COPY_PASTE_ENABLE: bool = True
    COPY_PASTE_PROB: float = 0.3
    COPY_PASTE_MAX_OBJS: int = 3
    # Added 1=Books to copy-paste targets
    COPY_PASTE_OBJ_IDS: Tuple[int, ...] = (1, 3, 6, 7)
    ENABLE_BOOKS_IMP: bool = True
    COPY_PASTE_BG_IDS: Tuple[int, ...] = (4, 5, 11)
    COPY_PASTE_BG_MIN_COVER: float = 0.5
    COPY_PASTE_MIN_AREA: int = 20
    COPY_PASTE_MAX_AREA: int = 0
    COPY_PASTE_MAX_AREA_RATIO: float = 0.03
    COPY_PASTE_MAX_TRIES: int = 20
    COPY_PASTE_MAX_OBJS_TOTAL: int = 0

    # Depth dropout (Disabled for Final Run to ensure stability)
    DEPTH_CHANNEL_DROPOUT_PROB: float = 0.0
    DEPTH_COARSE_DROPOUT_PROB: float = 0.0
    DEPTH_COARSE_DROPOUT_MAX_HOLES: int = 4
    DEPTH_COARSE_DROPOUT_MIN_FRAC: float = 0.05
    DEPTH_COARSE_DROPOUT_MAX_FRAC: float = 0.2

    # Depth range (meters)
    DEPTH_MIN: float = 0.6
    DEPTH_MAX: float = 10.0

    # RGB normalization (ImageNet)
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
    
    OPTIMIZER: str = "sam_adamw"  # SAM is essential for generalization
    SAM_RHO: float = 0.02
    
    ETA_MIN: float = 1e-6
    GRAD_ACCUM_STEPS: int = 1
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
    USE_EMA: bool = True
    EMA_DECAY: float = 0.999

    # --- Checkpoints ---
    SAVE_TOP_K: int = 5

    # --- Loss (FIXED) ---
    USE_DEPTH_AUX: bool = True
    DEPTH_LOSS_LAMBDA: float = 1.0

    # --- TTA (FIXED) ---
    # 【Exp100特化設定】
    # 拡大(1.25, 1.5)は精度を下げるため削除。
    # 基本(1.0)と、視野を広げつつノイズを抑える縮小(0.75)のみを採用。
    # 拡大(1.25, 1.5)は精度を下げるため削除。
    # 基本(1.0)と、視野を広げつつノイズを抑える縮小(0.75)のみを採用。
    TTA_COMBS: Tuple[Tuple[float, bool], ...] = tuple([tuple(x) for x in TTA_COMBS])
    TEMPERATURES: Tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)

    # --- Inference Recipe (Phase 0 Locked) ---
    INFER_TTA_ENABLE: bool = True
    INFER_TTA_BOOKS_PROTECT: bool = True
    INFER_TTA_BOOKS_BRANCH: str = "1.0_noflip" # Identification tag for the protective branch

    # --- Submit-time ensembling ---
    SUBMIT_CKPT_ENSEMBLE_K: int = 1

    # --- Validation Strategy ---
    SPLIT_MODE: str = SPLIT_MODE
    SPLIT_BLOCK_SIZE: int = SPLIT_BLOCK_SIZE

    # --- Logging / verbosity ---
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
            import torch
        except Exception:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def apply_runtime_settings(self) -> None:
        """
        Apply global torch runtime settings.
        """
        import torch

        torch.backends.cudnn.benchmark = bool(self.CUDNN_BENCHMARK)
        torch.backends.cudnn.deterministic = bool(self.DETERMINISTIC)

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(self.MATMUL_PRECISION))

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = bool(self.ALLOW_TF32)
            torch.backends.cudnn.allow_tf32 = bool(self.ALLOW_TF32)

    def validate(self) -> None:
        """
        Validates the configuration integrity.
        """
        errors = []

        if int(self.IN_CHANNELS) != 4:
            errors.append(f"IN_CHANNELS must be 4 (got {self.IN_CHANNELS})")

        if self.RESIZE_HEIGHT <= 0 or self.RESIZE_WIDTH <= 0:
            errors.append("RESIZE_HEIGHT/RESIZE_WIDTH must be positive")

        if self.CROP_SIZE is not None:
            ch, cw = self.CROP_SIZE
            if ch > self.RESIZE_HEIGHT or cw > self.RESIZE_WIDTH:
                errors.append(
                    f"CROP_SIZE {self.CROP_SIZE} must be <= RESIZE (H,W)=({self.RESIZE_HEIGHT},{self.RESIZE_WIDTH})"
                )

        if self.DEPTH_MIN >= self.DEPTH_MAX:
            errors.append(f"DEPTH_MIN/DEPTH_MAX invalid: min={self.DEPTH_MIN}, max={self.DEPTH_MAX}")

        # Check Class IDs
        if hasattr(self, "CLASS_NAMES") and self.CLASS_NAMES:
            if self.CLASS_ID_BOOKS < len(self.CLASS_NAMES) and self.CLASS_NAMES[self.CLASS_ID_BOOKS] != "books":
                 errors.append(f"CLASS_ID_BOOKS={self.CLASS_ID_BOOKS} does not map to 'books' (got {self.CLASS_NAMES[self.CLASS_ID_BOOKS]})")
            if self.CLASS_ID_TABLE < len(self.CLASS_NAMES) and self.CLASS_NAMES[self.CLASS_ID_TABLE] != "table":
                 errors.append(f"CLASS_ID_TABLE={self.CLASS_ID_TABLE} does not map to 'table' (got {self.CLASS_NAMES[self.CLASS_ID_TABLE]})")

        # Check Smart Crop Zoom Logic
        zoom_range = getattr(self, "SMART_CROP_ZOOM_RANGE", (1.0, 1.0))
        if len(zoom_range) != 2 or zoom_range[0] > zoom_range[1]:
             errors.append(f"SMART_CROP_ZOOM_RANGE invalid: {zoom_range}")

        if errors:
            raise ValueError("Invalid Config:\n- " + "\n- ".join(errors))

    def with_overrides(self, **kwargs: Any) -> "Config":
        return replace(self, **kwargs)

    def apply_preset(self, preset: str) -> "Config":
        """
        Supports legacy preset names for compatibility, but strictly enforces
        this configuration structure.
        """
        p = str(preset).strip()
        # Accept all previous experiment tags as valid aliases for this codebase
        if p in ("exp093_5", "exp093.5", "exp093_5_rgbd_4ch", "exp097", "exp097_0", "exp100"):
            return self
        raise ValueError(f"Unknown preset: {preset!r}. This codebase is fixed to Exp100.")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["TRAIN_DIR"] = self.TRAIN_DIR
        d["TEST_DIR"] = self.TEST_DIR
        d["DEVICE"] = self.DEVICE
        return d