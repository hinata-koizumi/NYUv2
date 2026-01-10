#!/bin/bash
set -e

echo "Starting High-Performance Golden Artifact Generation (RTX 4090 Optimized)..."

# RTX 4090 has 24GB VRAM. 
# Image size 480x640 (or 576x768 crop).
# BF16 mixed precision is enabled by default in config.
# Batch Size 32 should be safe and significantly faster than 4.
# If OOM occurs, try reducing to 16.

BATCH_SIZE=32

echo ">> Processing Fold 0 (BS=${BATCH_SIZE})..."
python -m nearest_final generate_golden --fold 0 --batch_size ${BATCH_SIZE}

echo ">> Processing Fold 1 (BS=${BATCH_SIZE})..."
python -m nearest_final generate_golden --fold 1 --batch_size ${BATCH_SIZE}

echo ">> Processing Fold 2 (BS=${BATCH_SIZE})..."
python -m nearest_final generate_golden --fold 2 --batch_size ${BATCH_SIZE}

echo ">> Processing Fold 3 (BS=${BATCH_SIZE})..."
python -m nearest_final generate_golden --fold 3 --batch_size ${BATCH_SIZE}

echo ">> Processing Fold 4 (BS=${BATCH_SIZE})..."
python -m nearest_final generate_golden --fold 4 --batch_size ${BATCH_SIZE}

# Merge
echo ">> Merging Artifacts..."
python -m nearest_final merge_golden

echo "All steps completed successfully!"
