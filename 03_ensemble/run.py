import argparse
import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Adjust path to find src
sys.path.append(str(Path(__file__).parent))

from src.io import load_logits, save_logits, save_submission  # noqa: E402
from src.optimize import optimize_mean, optimize_grid_w, optimize_books_gate, optimize_temp_calib  # noqa: E402
from src.metrics import calculate_miou  # noqa: E402
from src.meta import save_meta  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description="Ensemble Pipeline Runner")
    parser.add_argument('--preset', required=True, choices=['mean', 'grid_w', 'books_gate', 'temp_calib'])
    parser.add_argument('--convnext_dir', required=True, help="Path to ConvNeXt outputs")
    parser.add_argument('--segformer_dir', required=True, help="Path to SegFormer outputs")
    parser.add_argument('--out_dir', default=None, help="Output directory (default: auto-generated in runs/)")
    
    args = parser.parse_args()
    
    # 1. Setup Output Directory
    if args.out_dir:
        run_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f"runs/run_{timestamp}_{args.preset}")
    
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Starting Run: {args.preset} ===")
    print(f"Output Directory: {run_dir}")
    
    # Define Models
    model_paths = {
        "convnext": Path(args.convnext_dir),
        "segformer": Path(args.segformer_dir)
    }
    
    # 2. Sanity Check & OOF Loading
    print("\n--- 1. Loading & Checking Assets ---")
    oof_dict = {}
    test_logits_dict = {}
    gt_masks = None
    
    # We need GT for optimization. Assuming GT is loadable or part of the OOF generation?
    # For now, let's look for masks in fold directories if available or split manifest
    # Since we don't have explicit GT path argument, we might need to rely on 'val_labels.npy' if it exists
    # Or assume it is passed.
    # The prompt implies: "OOFで最適化" -> We need GT.
    # Let's check one fold directory for labels
    
    # Load OOFs
    first_fold0 = list(model_paths.values())[0] / "fold0"
    if (first_fold0 / "val_labels.npy").exists(): # Assuming labels exist
        print("Found val_labels.npy in fold0.")
        # Load all folds labels
        gt_parts = []
        for k in range(5):
             pk = list(model_paths.values())[0] / f"fold{k}" / "val_labels.npy"
             if pk.exists():
                 gt_parts.append(np.load(pk))
             else:
                 print(f"Missing GT for fold{k}")
        if len(gt_parts) == 5:
            gt_masks = np.concatenate(gt_parts)
    else:
        print("WARNING: GT masks not found. Optimization will use Dummy GT (for Verification).")
        # Dummy GT for structure testing
        total_len = 0
        for k in range(5):
             pk = list(model_paths.values())[0] / f"fold{k}" / "val_file_ids.npy"
             if pk.exists():
                 total_len += len(np.load(pk))
        gt_masks = np.zeros(total_len, dtype=np.uint8)

    for name, path in model_paths.items():
        print(f"Loading {name} from {path}...")
        
        # Load OOF (Concatenate 5 all folds)
        fold_oofs = []
        for k in range(5):
            p = path / f"fold{k}" / "val_logits.npy"
            if not p.exists():
                print(f"ERROR: Missing {p}")
                return
            fold_oofs.append(load_logits(p))
        oof_dict[name] = np.concatenate(fold_oofs, axis=0)
        
        # Load Test (Fold0)
        p_test = path / "fold0" / "test_logits.npy"
        if not p_test.exists():
             print(f"ERROR: Missing {p_test}")
             return
        test_logits_dict[name] = load_logits(p_test)

    # 3. Optimization
    print(f"\n--- 2. Optimizing ({args.preset}) ---")
    weights_or_config = None
    
    if args.preset == 'mean':
        weights_or_config = optimize_mean(oof_dict, gt_masks)
    elif args.preset == 'grid_w':
        weights_or_config = optimize_grid_w(oof_dict, gt_masks)
    elif args.preset == 'books_gate':
        weights_or_config = optimize_books_gate(oof_dict, gt_masks)
    elif args.preset == 'temp_calib':
        weights_or_config = optimize_temp_calib(oof_dict, gt_masks)
        
    # Save weights
    with open(run_dir / "weights.json", "w") as f:
        # Convert numpy types if any
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            raise TypeError
        json.dump(weights_or_config, f, indent=2, default=convert)
        
    print(f"Weights saved: {weights_or_config}")
    
    # 4. Submission
    print("\n--- 3. Generating Submission ---")
    
    # Ensemble Logic
    final_test_logits = np.zeros_like(list(test_logits_dict.values())[0])
    
    if args.preset in ['mean', 'grid_w', 'temp_calib']:
        weights = weights_or_config
        for name, w in weights.items():
            final_test_logits += test_logits_dict[name] * w
            
    elif args.preset == 'books_gate':
        # Apply gating logic
        # {method: books_gate, params: {w_global, w_books}, models: [m1, m2]}
        params = weights_or_config['params']
        m1, m2 = weights_or_config['models'] # Make sure order matches oof_dict keys used in optimize
        
        # Re-resolve order based on dict keys if not robust, but optimize_books_gate returns used models list
        
        w_g = params['w_global']
        w_b = params['w_books']
        BOOKS_IDX = 3
        
        l1 = test_logits_dict[m1]
        l2 = test_logits_dict[m2]
        
        final_test_logits = l1 * w_g + l2 * (1.0 - w_g)
        final_test_logits[:, BOOKS_IDX] = l1[:, BOOKS_IDX] * w_b + l2[:, BOOKS_IDX] * (1.0 - w_b)

    # Save
    save_logits(np.argmax(final_test_logits, axis=1), run_dir / "submission.npy")
    
    # Copy file ids
    # test_ids = np.load(model_paths['convnext'] / "fold0" / "test_file_ids.npy")
    # save_logits(test_ids, run_dir / "test_file_ids.npy")

    print(f"\nSaved submission to {run_dir}/submission.npy")
    
    # 5. Meta
    save_meta({
        "preset": args.preset,
        "args": vars(args),
        "weights": weights_or_config
    }, run_dir / "meta.json")

if __name__ == '__main__':
    main()
