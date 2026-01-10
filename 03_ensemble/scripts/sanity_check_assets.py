import argparse
import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parents[1]))
from src.meta import load_meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    models = config['models']
    
    print("Checking assets...")
    
    first_test_ids = None
    first_model = None

    for name, cfg in models.items():
        print(f"[{name}] Checking...")
        path = Path(cfg['path'])
        
        # Check folders
        if not path.exists():
            print(f"ERROR: {path} does not exist")
            sys.exit(1)

        # Iterate 5 folds
        val_total = 0
        for k in range(5):
            fold_dir = path / f"fold{k}"
            if not fold_dir.exists():
                print(f"ERROR: {fold_dir} missing")
                sys.exit(1)
                
            # Check required files
            if not (fold_dir / "val_logits.npy").exists():
                 print(f"ERROR: val_logits.npy missing in {fold_dir}")
            if not (fold_dir / "val_file_ids.npy").exists():
                 print(f"ERROR: val_file_ids.npy missing in {fold_dir}")
            
            val_ids = np.load(fold_dir / "val_file_ids.npy")
            val_total += len(val_ids)

        # Check Test
        # Assuming test is in fold0 or separate? CONTRACT says "models/outputs/.../fold{k}"
        # "test_logits" usually one common file or per fold?
        # CONTRACT: "outputs/<model_tag>/<exp_name>/fold{k}/... test_logits"
        # Usually test is same for all folds, but CONTRACT says it exists in fold dir.
        # We check fold0 for test
        
        fold0 = path / "fold0"
        if (fold0 / "test_file_ids.npy").exists():
            test_ids = np.load(fold0 / "test_file_ids.npy")
            if first_test_ids is None:
                first_test_ids = test_ids
                first_model = name
            else:
                if not np.array_equal(test_ids, first_test_ids):
                    print(f"ERROR: test_file_ids mismatch between {first_model} and {name}")
                    sys.exit(1)
        else:
             print(f"WARNING: test_file_ids.npy missing in {fold0}")

    print("Sanity check passed!")

if __name__ == '__main__':
    main()
