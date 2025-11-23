#!/bin/bash
# 提出用ZIPファイルを作成するスクリプト

SUBMISSION_DIR="submission_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SUBMISSION_DIR"

# 1. submission.npyをコピー
cp data/output_sota/submission.npy "$SUBMISSION_DIR/"

# 2. スクリプトをコピー
cp src/nyuv2_sota_pipeline.py "$SUBMISSION_DIR/"

# 3. README.mdをコピー（参考用）
cp README.md "$SUBMISSION_DIR/"

# 4. tar.gzファイルを作成
tar -czf "${SUBMISSION_DIR}.tar.gz" "$SUBMISSION_DIR"

echo "提出用アーカイブファイルを作成しました: ${SUBMISSION_DIR}.tar.gz"
echo "含まれるファイル:"
ls -lh "$SUBMISSION_DIR"
echo ""
echo "ファイルサイズ:"
du -sh "${SUBMISSION_DIR}.tar.gz"
