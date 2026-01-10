import argparse
import sys
from pathlib import Path
import json
import numpy as np

sys.path.append(str(Path(__file__).parents[1]))
from src.io import load_logits, save_logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
        
    run_dir = Path(config['output']['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True) # Should technically be specific run
    
    # In real flow, we should create a specific run dir YYYYMMDD...
    # For now we dump to root of runs/ or assume it's set
    
    models = config['models']
    
    # Collect all file IDs first to define global order
    # Assuming splits/manifest.json or similar defines the universe, 
    # OR we just concat all validation ids
    
    global_ids = []
    # Implementation detail: We need a canonical order.
    # We can read all val_file_ids from one model (assuming one covers all)
    
    # Let's use the first model to define OOF structure
    first_model_path = Path(list(models.values())[0]['path'])
    
    all_fold_ids = []
    
    for k in range(5):
        fpath = first_model_path / f"fold{k}" / "val_file_ids.npy"
        if fpath.exists():
            ids = np.load(fpath)
            all_fold_ids.append(ids)
            
    global_ids = np.concatenate(all_fold_ids)
    # Sort or keep fold order? usually good to sort to be deterministic
    # But if we want to match alignment, maybe just concat is enough if folds are disjoint partition
    
    # Let's save global IDs
    save_logits(global_ids, run_dir / "oof_file_ids.npy")
    
    # Now build OOF for each model
    for name, cfg in models.items():
        print(f"Building OOF for {name}...")
        path = Path(cfg['path'])
        
        fold_parts = []
        for k in range(5):
            logits = load_logits(path / f"fold{k}" / "val_logits.npy")
            fold_parts.append(logits)
            
        full_oof = np.concatenate(fold_parts, axis=0) # (N, 13, 480, 640)
        
        # Verify alignment?
        # If we just concat, we assume folds are processed in 0..4 order
        
        save_logits(full_oof, run_dir / f"oof_logits_{name}.npy")
        
    print("OOF build complete.")

if __name__ == '__main__':
    main()
