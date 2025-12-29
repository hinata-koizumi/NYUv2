#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Usage:
#   bash scripts/run_all.sh exp=exp_modular_v1 folds=5 tta_flip=1 use_ema=1
# ----------------------------

# defaults
EXP="exp_modular_v1"
FOLDS=5
TTA_FLIP=1
USE_EMA=1

# parse key=value args
for arg in "$@"; do
  case $arg in
    exp=*) EXP="${arg#*=}" ;;
    folds=*) FOLDS="${arg#*=}" ;;
    tta_flip=*) TTA_FLIP="${arg#*=}" ;;
    use_ema=*) USE_EMA="${arg#*=}" ;;
    *) echo "Unknown arg: $arg" ; exit 1 ;;
  esac
done

echo "=== RUN ALL ==="
echo "EXP=$EXP  FOLDS=$FOLDS  TTA_FLIP=$TTA_FLIP  USE_EMA=$USE_EMA"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"

# optional: set deterministic-ish env
export PYTHONUNBUFFERED=1

# 1) Train folds
for ((k=0; k<"$FOLDS"; k++)); do
  echo "=== [TRAIN] fold=$k / $((FOLDS-1)) ==="

  # Matched to your src/train_net.py arguments
  python -m src.train_net \
    --exp_name "$EXP" \
    --fold "$k"

  echo "=== [TRAIN DONE] fold=$k ==="
done

# 2) Make submission (k-fold ensemble + flip TTA)
echo "=== [SUBMISSION] building submission.npy and submission.zip ==="

SUBMIT_ARGS=()
# Matched to your src/make_submission.py arguments
SUBMIT_ARGS+=(--exp_name "$EXP")
SUBMIT_ARGS+=(--folds "$FOLDS")

if [ "$TTA_FLIP" -eq 1 ]; then
  SUBMIT_ARGS+=(--tta_flip)
fi
if [ "$USE_EMA" -eq 1 ]; then
  SUBMIT_ARGS+=(--use_ema)
fi

python -m src.make_submission "${SUBMIT_ARGS[@]}"

echo "=== ALL DONE ==="
echo "Generated: submission.npy and submission.zip"
