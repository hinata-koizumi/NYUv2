import os
import torch

class Config:
    EXP_NAME = "exp_modular_v1"
    SEED = 42

    # Image sizes
    RESIZE_HEIGHT = 720
    RESIZE_WIDTH = 960
    CROP_SIZE = (576, 768)  # (H, W) for training only

    # Smart crop
    SMART_CROP_PROB = 0.5
    SMALL_OBJ_IDS = [1, 3, 6, 7, 10]

    # Train
    EPOCHS = 180
    WARMUP_EPOCHS = 5
    BATCH_SIZE = 6
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Early stopping / checkpoints
    EARLY_STOPPING_PATIENCE = 30
    EARLY_STOPPING_MIN_DELTA = 0.0003
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
    IN_CHANNELS = 4  # RGB + normalized inv-depth
    
    # Input Adapter Mode
    INPUT_MODE = "rgbd_4ch" # "rgb", "rgbd_4ch", "2stream"

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

    # Depth input preprocessing
    DEPTH_MIN = 0.6
    DEPTH_MAX = 10.0

    # RGB normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # TTA
    # scale, flip (same as base_model_093_5.py)
    TTA_COMBS = [
        (0.75, False), (0.75, True),
        (1.0, False),  (1.0, True),
        (1.25, False), (1.25, True),
        (1.5, False),  (1.5, True),
    ]
    TEMPERATURES = [0.6, 0.7, 0.8, 1.0]

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
