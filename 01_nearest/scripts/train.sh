#!/bin/bash
# Run remaining folds (1-4) for Step C
# Preset: exp100 (Final)
# Split: Group (Robust)
# Epochs: 50 (Full)

for i in 1 2 3 4; do
    echo "========================================"
    echo "Starting Fold $i"
    echo "========================================"
    python3 -m nearest_final train \
        --preset exp100 \
        --fold $i \
        --set SPLIT_MODE=group \
        --set EPOCHS=50 \
        --set SAVE_TOP_K=5
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "Fold $i failed!"
        exit 1
    fi
done
echo "All folds completed successfully."
