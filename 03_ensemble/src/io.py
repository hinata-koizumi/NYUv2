import numpy as np
from pathlib import Path
import json

def load_logits(path, shape=None):
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path)
    # Add mmap support if needed
    raise ValueError(f"Unsupported file format: {path}")

def load_file_ids(path):
    return np.load(path)

def save_logits(logits, path):
    np.save(path, logits)

def save_submission(ids, matches, path):
    # flexible submission saving
    pass
