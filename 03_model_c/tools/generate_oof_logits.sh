#!/bin/bash
# Generate OOF logits for 01_nearest and 02_nonstruct
# This will take significant time (training 5 folds for each model)

set -e

ROOT="/root/datasets/NYUv2"
cd "$ROOT"

echo "=== Step 1: Generate 01_nearest OOF logits ==="
echo "This requires training 01_nearest for all 5 folds first."
echo "Then run OOF inference for each fold."

# 01_nearest training (if checkpoints don't exist)
# for fold in 0 1 2 3 4; do
#     echo "Training 01_nearest fold $fold..."
#     cd 01_nearest
#     python3 -m cli train --fold $fold --epochs 50
#     cd ..
# done

# 01_nearest OOF inference (after training)
# for fold in 0 1 2 3 4; do
#     echo "Generating OOF logits for 01_nearest fold $fold..."
#     cd 01_nearest
#     python3 -m research.oneoff.oof_infer \
#         --exp_dir data/output/nearest_final \
#         --fold $fold \
#         --save_dir golden_artifacts/folds/fold$fold
#     cd ..
# done

# Merge 01_nearest OOF logits
# cd 01_nearest
# python3 -m research.oneoff.merge_golden
# cd ..

echo ""
echo "=== Step 2: Generate 02_nonstruct OOF logits ==="
echo "This requires training 02_nonstruct for all 5 folds first."

# 02_nonstruct training (if checkpoints don't exist)
# for fold in 0 1 2 3 4; do
#     echo "Training 02_nonstruct fold $fold..."
#     cd 02_nonstruct
#     python3 train.py --fold $fold --epochs 50 --batch_size 4
#     cd ..
# done

# 02_nonstruct OOF logits are saved automatically during training
# They should be in: 02_nonstruct/output/oof_fold{k}_logits.npy

echo ""
echo "=== Step 3: Copy 02_nonstruct OOF logits to frozen location ==="
mkdir -p 00_data/02_nonstruct_frozen/golden_artifacts/oof
# for fold in 0 1 2 3 4; do
#     if [ -f "02_nonstruct/output/oof_fold${fold}_logits.npy" ]; then
#         cp "02_nonstruct/output/oof_fold${fold}_logits.npy" \
#            "00_data/02_nonstruct_frozen/golden_artifacts/oof/"
#         echo "Copied fold $fold"
#     fi
# done

echo ""
echo "=== Done ==="
echo "After completion, you should have:"
echo "  - 01_nearest/golden_artifacts/oof/oof_logits.npy"
echo "  - 00_data/02_nonstruct_frozen/golden_artifacts/oof/oof_fold{0-4}_logits.npy"
