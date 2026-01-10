#!/bin/bash
set -e

echo "Starting Golden Analysis Pipeline..."

echo "1. Merging Artifacts..."
python3 -m nearest_final merge_golden

echo "2. Optimizing Folds..."
python3 -m nearest_final optimize_folds

echo "3. Defining KPIs..."
python3 -m nearest_final define_kpis

echo "Analysis Complete. Check golden_artifacts/."
