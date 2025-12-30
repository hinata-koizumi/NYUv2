"""
Compatibility wrappers for `src.data.*` imports.
"""

from main.data.dataset import NYUDataset  # noqa: F401
from main.data.transforms import (  # noqa: F401
    get_train_transforms,
    get_valid_transforms,
    get_color_transforms,
)
from main.data.adapters import get_adapter  # noqa: F401


