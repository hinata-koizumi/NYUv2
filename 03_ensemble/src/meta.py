import json
from pathlib import Path
from datetime import datetime

def load_meta(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_meta(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_run_meta(split_id, model_configs, weights):
    return {
        "split_id": split_id,
        "models": model_configs,
        "weights": weights,
        "timestamp": datetime.now().isoformat()
    }
