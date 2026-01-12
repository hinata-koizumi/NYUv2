#!/bin/bash
set -e

# Train all 5 folds
for fold in 0 1 2 3 4; do
    echo "Training Fold $fold..."
    python -m 01_nearest train --fold $fold --recipe configs/final_recipe.json
done
