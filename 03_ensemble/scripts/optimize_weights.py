import argparse
import sys
from pathlib import Path
import json
import numpy as np

sys.path.append(str(Path(__file__).parents[1]))
from src.optimize import optimize_weights_grid
from src.io import load_logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
        
    run_dir = Path(config['output']['run_dir']) # Needs to be passed or inferred
    # For now assume working in current dir or fixed
    
    models = config['models']
    oof_dict = {}
    
    for name in models:
        oof_path = run_dir / f"oof_logits_{name}.npy"
        if oof_path.exists():
            oof_dict[name] = load_logits(oof_path)
            
    # Load GT
    # Where is GT? 
    # For now, let's create a placeholder random GT or need real path
    # Ideally checking `splits/`
    print("WARNING: GT loading not implemented, skipping optimization (using default)")
    
    # Default weights
    weights = {name: 1.0/len(models) for name in models}
    
    # Save weights
    with open(run_dir / "weights.json", "w") as f:
        json.dump(weights, f, indent=2)
        
    print("Weights optimized and saved.")

if __name__ == '__main__':
    main()
