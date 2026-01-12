#!/bin/bash
set -e

echo "Generating Golden Artifacts (All Folds)..."
python -m 01_nearest generate-golden --all_folds
