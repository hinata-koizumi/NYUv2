#!/bin/bash
set -e

EXP_DIR=${1:-"data/output/nearest_final"}
OUTPUT_NAME=${2:-"submission"}

echo "Generating Submission from $EXP_DIR..."
python -m 01_nearest submit --exp-dir "$EXP_DIR" --out "$OUTPUT_NAME"
