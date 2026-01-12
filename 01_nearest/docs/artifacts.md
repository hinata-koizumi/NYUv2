# Artifacts & Reproducibility (v1.0 Frozen)

The heavy assets for `01_nearest` are stored in Google Drive to keep the git repository light. To fully reproduce the environment, you must download these assets.

## 1. Storage Location
- **Google Drive Folder**: `NYUv2_assets/01_nearest_v1.0_frozen/`
- **Hosted By**: [Insert Team/User Name]

## 2. Contents
| File | Size | Description |
| :--- | :--- | :--- |
| `test_logits.npy` | ~10GB | Global Test Logits (654 images, 13 classes, 480x640, float16). |
| `ckpts/best_fold*.pth` | ~1GB ea | Model weights for Folds 0-4. |
| `oof_logits.npy` | ~12GB | (Optional) OOF Logits if needed for research. |

## 3. Retrieval (Vast AI / Cloud)
We recommend using `rclone` to retrieve assets.

### Setup
```bash
rclone config
# Create a remote named 'gdrive' connected to the Team Drive.
```

### Download
Execute from the project root:
```bash
# Verify destination exists
mkdir -p 01_nearest/golden_artifacts

# Download Test Logits
rclone copy gdrive:NYUv2_assets/01_nearest_v1.0_frozen/test_logits.npy 01_nearest/golden_artifacts/ --progress

# Download Checkpoints (if retraining/inference needed)
mkdir -p 01_nearest/checkpoints
rclone copy gdrive:NYUv2_assets/01_nearest_v1.0_frozen/ckpts/ 01_nearest/checkpoints/ --progress
```

## 4. Verification
After download, **YOU MUST VERIFY** integrity using the git-tracked manifest.

```bash
# 1. Install dependencies
pip install -r 01_nearest/requirements.txt

# 2. Run Verification Script
python -m 01_nearest.tests.verify_frozen_state
```
*Expected Output*: `--- FROZEN STATE VERIFIED ---`

## 5. Metadata
- **Manifest**: `golden_artifacts/sha256_manifest.txt` (Source of Truth)
- **Lock**: `golden_artifacts/FROZEN.lock`
