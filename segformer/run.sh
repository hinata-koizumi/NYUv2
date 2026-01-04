#!/bin/bash
set -euo pipefail

# Exp100 Training Script
# Runs Exp100 (Nearest Depth + Zero Init) for all folds.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Exp100 Defaults
PRESET="exp100"
EXP_NAME="exp100_final_nearest"

# Output directory
OUT_DIR="data/output/${EXP_NAME}"
mkdir -p "${OUT_DIR}"

# Logging
LOG_FILE="${OUT_DIR}/training.log"
echo "Starting training for ${EXP_NAME} (Preset: ${PRESET})..."
echo "Logs: ${LOG_FILE}"

# Run Training
python3 -u -m main train \
  --preset "${PRESET}" \
  --exp_name "${EXP_NAME}" \
  2>&1 | tee "${LOG_FILE}"

echo "Training complete."