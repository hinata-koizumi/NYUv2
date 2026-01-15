import os

# --- 1) Paths ---
ROOT_DIR = "/root/datasets/NYUv2"
DATA_DIR = os.path.join(ROOT_DIR, "00_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "03_model_c/output")

# IDs files
TRAIN_IDS_FILE = os.path.join(DATA_DIR, "ids/train_ids.txt")
TEST_IDS_FILE = os.path.join(DATA_DIR, "ids/test_ids.txt")
SPLITS_FILE = os.path.join(DATA_DIR, "splits/folds_v1.json")

# Input / Output Specs
RESIZE_HEIGHT = 480
RESIZE_WIDTH = 640
TRAIN_SIZE = (480, 640)

# --- 2) Classes & IDs (Synced with 01_nearest) ---
CLASS_NAMES = [
    "bed",       # 0: Struct
    "books",     # 1: Non-Struct (Target)
    "ceiling",   # 2: Non-Struct (Protect-ish)
    "chair",     # 3: Struct
    "floor",     # 4: Struct (Protect)
    "furniture", # 5: Non-Struct
    "objects",   # 6: Non-Struct (Danger)
    "picture",   # 7: Non-Struct (Target)
    "sofa",      # 8: Struct
    "table",     # 9: Struct (Target)
    "tv",        # 10: Non-Struct (Target)
    "wall",      # 11: Struct (Protect)
    "window"     # 12: Struct
]

CLASS_ID_BOOKS = 1
CLASS_ID_TABLE = 9
CLASS_ID_PICTURE = 7
CLASS_ID_TV = 10

SMALL_TARGET_IDS = [1, 7, 10]
PROTECT_IDS = [4, 11, 2]  # Floor, Wall, Ceiling
STRUCT7_IDS = [0, 3, 4, 8, 9, 11, 12]  # bed, chair, floor, sofa, table, wall, window
NONSTRUCT_IDS = [1, 7, 10, 5, 6, 2]  # books, picture, tv, furniture, objects, ceiling
OBJECTS_ID = 6

# --- 3) Loss Weights ---
# books/table boosted to 1.5 (v0 target)
CLASS_WEIGHTS = {
    0: 0.7,  # bed (Struct7)
    1: 1.5,  # books (Target)
    2: 1.0,  # ceiling (Protect)
    3: 0.7,  # chair (Struct7)
    4: 1.0,  # floor (Protect)
    5: 1.0,  # furniture
    6: 0.8,  # objects (Suppress/Safety)
    7: 1.3,  # picture (Target)
    8: 0.7,  # sofa (Struct7)
    9: 1.5,  # table (Target)
    10: 1.3, # tv (Target)
    11: 1.0, # wall (Protect)
    12: 0.7  # window (Struct7)
}
WEIGHTS_LIST = [CLASS_WEIGHTS[i] for i in range(13)]

# --- 4) Depth Input & Augmentation ---
DEPTH_MIN = 0.71
DEPTH_MAX = 10.0

DEPTH_SCALE_JITTER = 0.10  # +/- 10%
DEPTH_NOISE_STD = 0.01     # meters
DEPTH_DROPOUT_PROB = 0.02
DEPTH_DROPOUT_BLOCKS = 2
DEPTH_DROPOUT_BLOCK_SIZE = (12, 24)  # (min, max) pixels
DEPTH_QUANT_STEP = 0.01    # meters

DEPTH_GRAD_SCALE = 1.0
DEPTH_CURV_SCALE = 1.0

USE_CURVATURE = True
IN_CHANS = 7 if USE_CURVATURE else 6

# --- 5) Depth-Aware Patch Sampling ---
TARGETED_CROP_PROB = 0.60  # 60% targeted, 40% random
TARGETED_BOOKS_PROB = 0.75
TARGETED_SAMPLE_TRIES = 10

# Hard-mask targeted cropping (Ensemble-2 mistakes)
USE_HARD_MASK = True
HARD_MASK_DIR = os.path.join(ROOT_DIR, "00_data/output/model_c_hard_masks")
HARD_CROP_PROB = 0.70
HARD_MASK_BOOKS_PROB = 0.75
HARD_SAMPLE_TRIES = 10

BOOKS_DEPTH_RANGE = (2.0, 5.0)
TABLE_DEPTH_RANGE_NEAR = (1.0, 2.0)
TABLE_DEPTH_RANGE_FAR = (2.0, 3.0)
TABLE_NEAR_PROB = 0.70

# --- 6) Geometric Aug (Image-space) ---
TRAIN_SCALE_MIN = 0.85
TRAIN_SCALE_MAX = 1.15
HFLIP_PROB = 0.5

# --- 7) Planar Auxiliary Head (Optional) ---
PLANAR_HEAD_ENABLE = True
PLANAR_LOSS_WEIGHT = 0.10
PLANAR_CURV_THRESH = 0.02

# --- 8) Confusion Penalty (books/table vs furniture/objects) ---
CONFUSION_PENALTY_WEIGHT = 0.10
CONFUSION_PENALTY_TARGET_IDS = [1, 9]  # books, table
CONFUSION_PENALTY_BAD_IDS = [5, 6]     # furniture, objects

# Misc
SEED = 42
