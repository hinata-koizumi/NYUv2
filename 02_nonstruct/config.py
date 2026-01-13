
import os
import numpy as np

# --- 1) Paths & Contract ---
ROOT_DIR = "/Users/koizumihinata/NYUv2"
DATA_DIR = os.path.join(ROOT_DIR, "00_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "02_nonstruct/output")

# IDs files
TRAIN_IDS_FILE = os.path.join(DATA_DIR, "ids/train_ids.txt")
TEST_IDS_FILE = os.path.join(DATA_DIR, "ids/test_ids.txt")
SPLITS_FILE = os.path.join(DATA_DIR, "splits/folds_v1.json")

# Input / Output Specs
RESIZE_HEIGHT = 480
RESIZE_WIDTH = 640
# Note: Training often uses larger or different sizes, but Output Logits MUST be 480x640.
# The previous model (01_nearest) used RESIZE_HEIGHT=720, WIDTH=960 for training input.
# We should probably follow "Model (Arch 01_nearest reuse)" implies similar input size?
# Request says: "logits save ... 480x640".
# Request says: "Arch ... reuse ... valid/test logits 480x640 bilinear".
# Request doesn't strictly specify Training Resolution, but B is for small objects, so higher res helps.
# I will stick to 01_nearest's training resolution (720x960) or similar (640x640?) to ensure small objects are visible.
# Let's use standard NYUv2 training resolution often used: 480x640 or slightly larger.
# 01_nearest uses 720x960 and crops 576x768.
# I'll stick close to 01_nearest capable settings: Train at 480x640+ (maybe 640 scale).
TRAIN_SIZE = (480, 640) # We'll start simple or allow dynamic.

# --- 2) Classes & IDs Strategy (Synced with 01_nearest) ---
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
    "table",     # 9: Struct
    "tv",        # 10: Non-Struct (Target)
    "wall",      # 11: Struct (Protect)
    "window"     # 12: Struct
]

# Groups
CLASS_ID_BOOKS = 1
CLASS_ID_PICTURE = 7
CLASS_ID_TV = 10
SMALL_TARGET_IDS = [1, 7, 10]

PROTECT_IDS = [4, 11, 2] # Floor, Wall, Ceiling
STRUCT7_IDS = [0, 3, 4, 8, 9, 11, 12] # bed, chair, floor, sofa, table, wall, window
NONSTRUCT_IDS = [1, 7, 10, 5, 6, 2] # books, picture, tv, furniture, objects, ceiling
OBJECTS_ID = 6

# --- 3) Loss Weights (v0 Safety) ---
# Protect (floor/wall/ceiling): 0.8-1.0
# Struct7 (bed/chair...): 0.6-0.8
# Furniture: 1.0
# Objects: 0.8-1.0 (Do NOT raise!)
# Small Target: 1.2-1.6

CLASS_WEIGHTS = {
    0: 0.7,  # bed (Struct7)
    1: 1.5,  # books (Small Target - Boost!)
    2: 1.0,  # ceiling (Protect)
    3: 0.7,  # chair (Struct7)
    4: 1.0,  # floor (Protect)
    5: 1.0,  # furniture
    6: 0.8,  # objects (Suppress/Safety)
    7: 1.4,  # picture (Small Target)
    8: 0.7,  # sofa (Struct7)
    9: 0.7,  # table (Struct7)
    10: 1.4, # tv (Small Target)
    11: 1.0, # wall (Protect)
    12: 0.7  # window (Struct7)
}
# Convert to tensor-ready list
WEIGHTS_LIST = [CLASS_WEIGHTS[i] for i in range(13)]

# --- 4) Sampling Params ---
# p(image) ~ 0.2 + nonstruct_ratio + 2.0 * small_target_ratio
SAMPLE_BASE_PROB = 0.2
SAMPLE_SMALL_FACTOR = 2.0

# Smart Crop
SMART_CROP_PROB_SMALL = 0.7  # 0.6-0.8
SMART_CROP_PROB_NONSTRUCT = 0.25 # 0.2-0.3
SMART_CROP_PROTECT_MIN_RATIO = 0.1 # "protect... minimum X%" (e.g. 10%)

