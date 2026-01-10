#!/bin/bash
# Generate OOF (Val) and Test Assets for all 5 folds
# FORCE SPLIT_MODE=group to match training and prevent leakage!

for i in 0 1 2 3 4; do
    echo "========================================"
    # Run OOF Inference (via CLI)
    python3 -m nearest_final infer_oof \
        --exp_name nearest_final \
        --fold $i \
        --batch_mul 1 \
        --split_mode group
        
    if [ $? -ne 0 ]; then
        echo "Generation failed for Fold $i"
        exit 1
    fi
done
echo "All assets generated successfully."
