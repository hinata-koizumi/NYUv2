"""
Centralized constants for the Nearest Final project.
"""

# API / Config Defaults
NUM_CLASSES = 13
CLASS_NAMES = [
    "bed", "books", "ceiling", "chair", "floor", "furniture", 
    "objects", "picture", "sofa", "table", "tv", "wall", "window"
]
CLASS_ID_BOOKS = 1
CLASS_ID_TABLE = 9
IGNORE_INDEX = 255

# Split Configuration
SPLIT_MODE = "group"
SPLIT_BLOCK_SIZE = 50

# Input Dimensions
RESIZE_HEIGHT = 720
RESIZE_WIDTH = 960
CROP_SIZE = (576, 768)
IN_CHANNELS = 4

# TTA Settings (Scales, HFlip)
# Matches training config TTA_COMBS
TTA_COMBS = [
    (0.75, False),
    (0.75, True),
    (1.0, False),
    (1.0, True)
]

# Output Directory Structure
# The root output directory relative to the project root
DEFAULT_OUTPUT_ROOT = "00_data/output/01_nearest"
