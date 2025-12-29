import os
import torch

class Config:
    # Doc-aligned default (Exp093.5)
    # - docs/exp09_log.md: "FPN with ConvNeXt Base encoder" + "RGB-D 4ch input"
    # - archive/main1/exp093_ensemble.py expects outputs under:
    #   data/outputs/exp093_5_convnext_rgbd_4ch/fold{}/model_best.pth
    EXP_NAME = "exp093_5_convnext_rgbd_4ch"
    SEED = 42

    # Image sizes
    RESIZE_HEIGHT = 720
    RESIZE_WIDTH = 960
    CROP_SIZE = (576, 768)  # (H, W) for training only

    # Smart crop
    SMART_CROP_PROB = 0.5
    SMALL_OBJ_IDS = [1, 3, 6, 7, 10]

    # Train
    # Doc (Exp093.*) uses 50 epochs baseline
    EPOCHS = 50
    WARMUP_EPOCHS = 5
    BATCH_SIZE = 4
    NUM_WORKERS = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Optimization
    # Keep AMP off by default for doc-aligned reproducibility (can be enabled for speed)
    USE_AMP = False
    GRAD_ACCUM_STEPS = 1    # Effective Batch = 8 * 1 = 8. Increase if need 16/32
    STEM_LR_MULT = 1.0

    # Early stopping / checkpoints
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 1e-4
    MIN_EPOCHS = 20
    SAVE_TOP_K = 5
    SAVE_START_EPOCH = 20

    # LR schedule
    ETA_MIN = 1e-6
    LR_SCHEDULE = "cosine_drop"
    LR_DROP_EPOCH = 40
    LR_DROP_FACTOR = 0.3
    COSINE_RESTART_T0 = 40
    COSINE_RESTART_T_MULT = 1

    # Task
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    IN_CHANNELS = 4  # RGB + InvDepth (doc Exp093.5)
    
    # Input Adapter Mode
    INPUT_MODE = "rgbd_4ch" # "rgb", "rgbd_4ch", "2stream", "rgbd_6ch"

    # Safety checks
    STRICT_DEPTH_FOR_TRAIN = True
    SANITY_CHECK_FIRST_N = 20

    # EMA
    EMA_DECAY = 0.999

    # CV
    N_FOLDS = 5

    # Loss
    SEG_LOSS = "ce_dice"      # "ce" | "ce_dice"
    DICE_WEIGHT = 0.5

    # Depth auxiliary task (doc Exp093.*: + 0.1 * Depth L1 Loss)
    USE_DEPTH_AUX = True
    DEPTH_LOSS_LAMBDA = 0.1

    # Boundary-aware loss (doc Exp093.4)
    USE_BOUNDARY_LOSS = False
    BOUNDARY_WEIGHT = 2.0

    # Optional explicit class weights (doc Exp093.4: Books(1), Table(9), TV(10) -> 1.5x)
    USE_CLASS_WEIGHTS = False
    CLASS_WEIGHTS = None  # e.g. [1.0]*13 with boosts; if None and USE_CLASS_WEIGHTS=True, train_net computes from data

    # Depth input preprocessing
    # Doc Exp093.2/4 used 0.71; Exp093.5 is "inverse depth" input.
    # Keep 0.71 here to match the exp09_log description.
    DEPTH_MIN = 0.71
    DEPTH_MAX = 10.0

    # RGB normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # TTA
    # scale, flip (same as base_model_093_5.py)
    TTA_COMBS = [
        (0.5, False), (0.5, True),
        (0.75, False), (0.75, True),
        (1.0, False),  (1.0, True),
        (1.25, False), (1.25, True),
        (1.5, False),  (1.5, True),
    ]
    TEMPERATURES = [0.7, 0.8, 0.9, 1.0]

    # Training control
    USE_EARLY_STOPPING = False

    DATA_ROOT = "data"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR = os.path.join(DATA_ROOT, "test")

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    @classmethod
    def to_dict(cls) -> dict:
        d = {}
        for k, v in cls.__dict__.items():
            if k.startswith("__") or k in ("DEVICE",):
                continue
            if callable(v) or isinstance(v, (classmethod, staticmethod, type)):
                continue
            d[k] = v
        return d

    def validate(self) -> None:
        """
        Fail-fast config validation (used by main/train_net.py).
        This is intentionally minimal; doc-repro presets will tighten constraints.
        """
        if self.NUM_CLASSES <= 0:
            raise ValueError("NUM_CLASSES must be > 0")
        if self.IN_CHANNELS not in (3, 4, 6):
            raise ValueError(f"Unsupported IN_CHANNELS={self.IN_CHANNELS} (expected 3/4/6)")
        if self.INPUT_MODE == "rgb" and self.IN_CHANNELS != 3:
            raise ValueError("INPUT_MODE=rgb requires IN_CHANNELS=3")
        if self.INPUT_MODE == "rgbd_4ch" and self.IN_CHANNELS != 4:
            raise ValueError("INPUT_MODE=rgbd_4ch requires IN_CHANNELS=4")
        if self.INPUT_MODE == "rgbd_6ch" and self.IN_CHANNELS != 6:
            raise ValueError("INPUT_MODE=rgbd_6ch requires IN_CHANNELS=6")
        if self.RESIZE_HEIGHT <= 0 or self.RESIZE_WIDTH <= 0:
            raise ValueError("RESIZE_HEIGHT/RESIZE_WIDTH must be > 0")
        if self.CROP_SIZE is not None:
            ch, cw = self.CROP_SIZE
            if ch <= 0 or cw <= 0:
                raise ValueError("CROP_SIZE must be positive")
            if ch > self.RESIZE_HEIGHT or cw > self.RESIZE_WIDTH:
                raise ValueError("CROP_SIZE must be <= RESIZE size")

        if bool(getattr(self, "USE_DEPTH_AUX", False)) and float(getattr(self, "DEPTH_LOSS_LAMBDA", 0.0)) <= 0.0:
            raise ValueError("USE_DEPTH_AUX=True requires DEPTH_LOSS_LAMBDA > 0")

        if bool(getattr(self, "USE_BOUNDARY_LOSS", False)) and float(getattr(self, "BOUNDARY_WEIGHT", 1.0)) < 1.0:
            raise ValueError("BOUNDARY_WEIGHT must be >= 1.0")

    def apply_preset(self, name: str) -> None:
        """
        Apply doc-reproduction presets (docs/exp09_log.md).
        Supported:
          - exp093_2: HR + EMA + TTA + OOF (3ch, CE + 0.1*DepthL1)
          - exp093_4: Boundary + Class Weights (3ch, Boundary CE + 0.1*DepthL1)
          - exp093_5: RGB-D 4ch input (4ch, CE + 0.1*DepthL1)
        """
        n = str(name).strip().lower()
        if n in ("exp093_2", "093_2", "0932"):
            self.EXP_NAME = "exp093_2_hr_ema_tta_oof"
            self.IN_CHANNELS = 3
            self.INPUT_MODE = "rgb"
            self.SEG_LOSS = "ce"
            self.USE_DEPTH_AUX = True
            self.DEPTH_LOSS_LAMBDA = 0.1
            self.USE_BOUNDARY_LOSS = False
            self.USE_CLASS_WEIGHTS = False
            self.CLASS_WEIGHTS = None
            self.EPOCHS = 50
            self.BATCH_SIZE = 4
            self.LEARNING_RATE = 1e-4
            self.WEIGHT_DECAY = 1e-4
            self.EMA_DECAY = 0.999
            self.RESIZE_HEIGHT = 720
            self.RESIZE_WIDTH = 960
            self.CROP_SIZE = (576, 768)
            self.DEPTH_MIN = 0.71
            self.DEPTH_MAX = 10.0
            self.TTA_COMBS = [
                (0.5, False), (0.5, True),
                (0.75, False), (0.75, True),
                (1.0, False),  (1.0, True),
                (1.25, False), (1.25, True),
                (1.5, False),  (1.5, True),
            ]
            self.TEMPERATURES = [0.7, 0.8, 0.9, 1.0]
            self.USE_EARLY_STOPPING = False
            self.USE_AMP = False
            self.STEM_LR_MULT = 1.0
        elif n in ("exp093_4", "093_4", "0934"):
            self.EXP_NAME = "exp093_4_boundary_cb"
            self.IN_CHANNELS = 3
            self.INPUT_MODE = "rgb"
            self.SEG_LOSS = "ce"
            self.USE_DEPTH_AUX = True
            self.DEPTH_LOSS_LAMBDA = 0.1
            self.USE_BOUNDARY_LOSS = True
            self.BOUNDARY_WEIGHT = 2.0
            self.USE_CLASS_WEIGHTS = True
            w = [1.0] * 13
            w[1] = 1.5
            w[9] = 1.5
            w[10] = 1.5
            self.CLASS_WEIGHTS = w
            self.EPOCHS = 50
            self.BATCH_SIZE = 4
            self.LEARNING_RATE = 1e-4
            self.WEIGHT_DECAY = 1e-4
            self.EMA_DECAY = 0.999
            self.RESIZE_HEIGHT = 720
            self.RESIZE_WIDTH = 960
            self.CROP_SIZE = (576, 768)
            self.DEPTH_MIN = 0.71
            self.DEPTH_MAX = 10.0
            self.TTA_COMBS = [
                (0.5, False), (0.5, True),
                (0.75, False), (0.75, True),
                (1.0, False),  (1.0, True),
                (1.25, False), (1.25, True),
                (1.5, False),  (1.5, True),
            ]
            self.TEMPERATURES = [0.7, 0.8, 0.9, 1.0]
            self.USE_EARLY_STOPPING = False
            self.USE_AMP = False
            self.STEM_LR_MULT = 1.0
        elif n in ("exp093_5", "093_5", "0935"):
            self.EXP_NAME = "exp093_5_convnext_rgbd_4ch"
            self.IN_CHANNELS = 4
            self.INPUT_MODE = "rgbd_4ch"
            self.SEG_LOSS = "ce"
            self.USE_DEPTH_AUX = True
            self.DEPTH_LOSS_LAMBDA = 0.1
            self.USE_BOUNDARY_LOSS = False
            self.USE_CLASS_WEIGHTS = False
            self.CLASS_WEIGHTS = None
            self.EPOCHS = 50
            self.BATCH_SIZE = 4
            self.LEARNING_RATE = 1e-4
            self.WEIGHT_DECAY = 1e-4
            self.EMA_DECAY = 0.999
            self.RESIZE_HEIGHT = 720
            self.RESIZE_WIDTH = 960
            self.CROP_SIZE = (576, 768)
            self.DEPTH_MIN = 0.71
            self.DEPTH_MAX = 10.0
            self.TTA_COMBS = [
                (0.5, False), (0.5, True),
                (0.75, False), (0.75, True),
                (1.0, False),  (1.0, True),
                (1.25, False), (1.25, True),
                (1.5, False),  (1.5, True),
            ]
            self.TEMPERATURES = [0.7, 0.8, 0.9, 1.0]
            self.USE_EARLY_STOPPING = False
            self.USE_AMP = False
            self.STEM_LR_MULT = 1.0
        else:
            raise ValueError(f"Unknown preset: {name}")

        # Ensure internal consistency after applying.
        self.validate()
