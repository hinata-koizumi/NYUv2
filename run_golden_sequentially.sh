#!/bin/bash
set -e

echo "Starting Robust Sequential Golden Artifact Generation..."

# Run each fold in a separate process to clear memory/MPS context completely between folds.

echo ">> Processing Fold 0..."
python -m nearest_final generate_golden --fold 0

echo ">> Processing Fold 1..."
python -m nearest_final generate_golden --fold 1

echo ">> Processing Fold 2..."
python -m nearest_final generate_golden --fold 2

echo ">> Processing Fold 3..."
python -m nearest_final generate_golden --fold 3

echo ">> Processing Fold 4..."
python -m nearest_final generate_golden --fold 4

# Merge
echo ">> Merging Artifacts..."
python -m nearest_final merge_golden

echo "All steps completed successfully!"
