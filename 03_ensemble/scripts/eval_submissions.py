import argparse
import sys
from pathlib import Path
import json
import numpy as np

sys.path.append(str(Path(__file__).parents[1]))
from src.metrics import calculate_miou
from src.io import load_logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
        
    run_dir = Path(config['output']['run_dir'])
    
    # Load submission
    sub_path = run_dir / "submission.npy"
    if not sub_path.exists():
        print("Submission not found.")
        sys.exit(1)
        
    preds = np.load(sub_path)
    
    # Load GT
    # Needs GT path. 
    # Placeholder
    print("GT evaluation not implemented (requires GT mask path).")

if __name__ == '__main__':
    main()
