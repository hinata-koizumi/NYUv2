#!/bin/bash
set -euo pipefail

# Wrapper to run the current `main/` pipeline from anywhere.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRESET="${PRESET:-exp093_5}"
EXP_NAME="${EXP_NAME:-exp093_5_convnext_rgbd_4ch_repro}"

OUT_DIR="data/output/${EXP_NAME}"
mkdir -p "${OUT_DIR}"

LOG_FILE="${OUT_DIR}/training.log"

python -u -m main train \
  --preset "${PRESET}" \
  --exp_name "${EXP_NAME}" \
  2>&1 | tee "${LOG_FILE}"


