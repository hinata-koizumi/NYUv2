import argparse
import sys
from pathlib import Path
import json
import numpy as np

sys.path.append(str(Path(__file__).parents[1]))
from src.io import load_logits, save_logits, save_submission

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
        
    run_dir = Path(config['output']['run_dir'])
    
    # Load weights
    with open(run_dir / "weights.json") as f:
        weights = json.load(f)
        
    # Load Test Logits and Ensemble
    ensemble_logits = None
    
    print("Ensembling test logits...")
    for name, w in weights.items():
        if w == 0: continue
        
        cfg = config['models'][name]
        path = Path(cfg['path'])
        
        # Test logits in fold0?
        test_logits = load_logits(path / "fold0" / "test_logits.npy")
        
        if ensemble_logits is None:
            ensemble_logits = np.zeros_like(test_logits, dtype=np.float32)
            
        ensemble_logits += test_logits * w
        
    # Save submission
    test_ids = np.load(Path(config['models'][list(weights.keys())[0]]['path']) / "fold0" / "test_file_ids.npy")
    
    create_submission_file(ensemble_logits, test_ids, run_dir) # Use submit.py logic (stub for now)
    
    # Also save raw submission npy
    save_logits(np.argmax(ensemble_logits, axis=1), run_dir / "submission.npy")
    
    print("Submission created.")

if __name__ == '__main__':
    main()
