#!/bin/bash

# RTX 4090 最適化設定
# - batch-size 8: VRAM 24GBを安全に使う設定（16は大きすぎてOOM発生）
# - num-workers 8: CPUコア数に合わせて調整（16は多すぎてオーバーヘッド発生）
# - encoder-name: 以前指定されていた timm-efficientnet-b5 を採用
# - python -u: ログをバッファせずリアルタイムに表示するオプション
# - PYTORCH_CUDA_ALLOC_CONF: メモリフラグメンテーションを防ぐ設定

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ログファイルのパス
LOG_FILE="data/output_sota/training.log"

# ログディレクトリを作成
mkdir -p data/output_sota

# 標準出力と標準エラー出力をログファイルにリダイレクト（同時にコンソールにも表示）
# --no-resume: 既存のチェックポイントを無視して最初から学習を開始
python -u src/nyuv2_sota_pipeline.py \
    --dataset-root data \
    --output-dir data/output_sota \
    --encoder-name timm-efficientnet-b5 \
    --batch-size 8 \
    --num-workers 8 \
    --epochs 50 \
    --n-splits 5 \
    --no-resume \
    2>&1 | tee "$LOG_FILE"

