"""
Compatibility wrappers for `src.engine.*` imports.
"""

from main.engine.inference import Predictor  # noqa: F401
from main.engine.trainer import train_one_epoch, validate  # noqa: F401


