# 実装の遍歴（時系列ログ）

## 0) 出発点（方向性の確定）

* **目的**: NYUv2セグメンテーションを **RGB + Depth（逆深度）4ch** で強化したい
* **モデル**: `smp.FPN + tu-convnext_base`
* **追加**:
    * **EMA**: validを安定化
    * **TTA**: 提出時の精度向上を狙う
* **課題意識**: データ少 → **過学習しやすい**
    * 対策: 「学習を止める仕組み（EarlyStopping）」と「評価を安定させる仕組み（EMA）」を入れる方針

---

## 1) Augmentation設計の整理（無効な処理を削った）

### やっていた / 検討していた
* `Resize(720x960)` の後に `PadIfNeeded(CROP_SIZE)` を入れる案が混ざっていた

### 気づき（手戻りポイント）
* `Resize(720x960)` の直後に `PadIfNeeded(CROP_SIZE)` を入れても、**Padされない**（すでに十分大きい）
    * **教訓**: “意味のないAug” が入ると、後で「あれ効いてる？」で迷って手戻りする

### 確定した形
* **Train**: `Resize → HFlip → ShiftScaleRotate(scaleのみ)` + 色変換（RGBのみ）
* **Valid/Test**: `Resize → PadIfNeeded(32倍)`（モデル都合のパディング）

---

## 2) Smart Crop導入（小物体を拾う戦略）

### 追加した理由
* データが少なく、クラス不均衡＆小物体が学習されにくい
    * 対策: `SMALL_OBJ_IDS` を中心に crop する確率を入れて、学習分布を補正

### 重要な設計の分岐（再発しやすい）
* Smart cropは **trainのみ**（labelがある前提）
* `return_raw_for_tta=True` の時は **cropしない**（TTA評価は“画面全体”でやる）

---

## 3) Depth周りの「静かな精度劣化」を潰した（最大の手戻り源）

ここが一番の“同じミスを繰り返しやすい地雷”だったので、段階を分けて潰していった。

### 3-1) 欠損Depthを「黙ってゼロ埋め」してしまう問題

**以前の挙動**
* depthが読めない時、ゼロ配列を返して続行してしまう
    * **結果**: 学習は回るが、4ch目が壊れて **精度が静かに落ちる**（最悪の手戻り）

**対策として追加**
* `STRICT_DEPTH_FOR_TRAIN=True` を導入
    * train/validでは depthが欠けてたら **即エラーで止める**
    * testは欠損あり得るので許容

### 3-2) Albumentationsで valid mask が壊れる問題（255問題）

**何が起きるか**
* `depth_valid` を Albumentations で `"mask"` 扱いにしてるので、Padや境界埋めで **255（IGNORE_INDEX）が混入** し得る
* それを `valid > 0.5` みたいにすると、**255が valid=1 扱い** になってしまう
    * **結果**: depth計算対象が増えて壊れる → 精度が落ちる（これも静かに落ちる）

**対策として確定した処理**
* `valid = ((valid > 0.5) & (valid < 1.5)).astype(np.float32)`
    * “1だけがvalid”になるように **255を排除**

### 3-3) depth_m の範囲保証

* transform後に `depth_m` が 0 などを含むことがあるので、**再clip** を追加
    * **効果**: `inv = 1/depth` の異常値（inf/爆発）を防ぐ

---

## 4) 4ch入力の初期化（ConvNeXtの4ch対応）

### やったこと
* encoderのstem convの 4ch目を `mean(RGB)` で初期化
    * **効果**: ImageNet事前学習との整合を保って、学習初期が安定

### 小さい改善（ログのノイズ削減）
* 以前は foldごと/ロードごとに print が大量に出る
    * 対策: `_printed_4ch_init_msg` で **1回だけ出す** ように変更（見落とし防止）

---

## 5) Diceの設計（present-only）

### 狙い
* 出現してないクラスに対してDiceを計算するとノイズになる
    * 対策: “present-only” で安定化

### 確定
* `CE + DICE_WEIGHT * Dice(present-only)` を採用

---

## 6) TTA & Temperature の扱いが「本番で使われる形」になった

### 以前の手戻りポイント
* foldごとに TTA sweep はしてるのに、提出時に **固定温度（例: 0.7）** をそのまま使ってしまう
    * **状態**: “検証したのに反映されてない”

### 最終形
* `TEMPERATURES=[0.6,0.7,0.8,1.0]` でfoldごとにmIoU計測
* fold平均で最良の温度 `best_temp_global` を決めて **提出推論に反映**
* ※これは厳密なOOF統合ではないが、少なくとも「検証結果を使って提出する」一貫性が出た。

---

# 再発防止のチェックリスト（次から迷わないガードレール）

同じミスを防ぐための実装・レビュー時のチェックリストです。

## Augmentation & Data
- [ ] **Augmentationの順序・意味**: `Resize` 後の `Pad` が無駄になっていないか？意味のないAugは削除したか？
- [ ] **Smart Crop**: Train以外（Valid/Test/TTA）でCropが発火していないか？特に `return_raw` 時はCrop禁止。

## Depth / 4ch Handling
- [ ] **Strict Depth Loading**: 学習時、Depth読み込み失敗をエラーにしているか？（サイレント失敗禁止）
- [ ] **Mask Binarization**: マスク処理で `255` (Ignore) が `1` (Valid) に化けていないか？
    - OK: `((val > 0.5) & (val < 1.5))`
    - NG: `val > 0`
- [ ] **Value Safety**: 逆数 (`1/depth`) を取る前に、分母を安全な範囲にClipしているか？
- [ ] **Initialization**: 4ch目の重みが適切に初期化されているか？（RGB平均など）

## Inference & Submission
- [ ] **Parameter Consistency**: Validationで決めたハイパーパラメータ（Temperature等）が、提出用コードに動的に反映されているか？（手打ち固定値の禁止）
- [ ] **TTA Consistency**: TTA時の入力解像度や処理が、学習時/検証時と論理的に矛盾していないか？

# NYUv2 Segmentation: Implementation Progress Log

This document summarizes **all major implementations, fixes, experiments, and decisions** made during this chat session for the NYUv2 semantic segmentation pipeline.

Repo root: `/root/NYUv2`  
Primary script (current): `final/main.py`

---

## Goals & Constraints

- **Goal**: Improve public leaderboard (LB) performance (target discussed: **0.75**).
- **Dataset**: NYUv2 (RGB, depth, 13-class semantic segmentation).
- **Metric**: mean IoU (mIoU).
- **Submission rule**: submission zip must contain **only** `submission.npy` (implemented).
- **Pretrained weights rule-awareness**:
  - Avoid using task-finetuned SegFormer checkpoints (e.g. ADE20K finetuned) unless rules allow.
  - Prefer generic backbones (e.g. `nvidia/mit-b4`) when using SegFormer.

---

## Deleted / Lost Files (Recovered Functionality)

Deleted during the session (per IDE state):
- `exp090.py`
- `make_fold_ensemble_submission.py`
- `main.py`

Recovery action:
- **Rebuilt the lost `exp090.py`-equivalent pipeline inside `final/main.py`**, including:
  - fold training
  - EMA weights
  - TTA (test-time augmentation)
  - temperature sweep on validation
  - OOF aggregation
  - fold ensemble submission
  - `submission.zip` creation

---

## What `final/main.py` Supports Now (End-to-End)

### Modes
- `--mode train`: trains folds (all folds by default, or subset).
- `--mode submit`: loads trained fold checkpoints, runs fold-ensemble + TTA on test set, saves `submission.npy` and `submission.zip`.
- `--mode train_submit`: runs train then submit (submit picks `exp_dir` default based on `exp_name`).

### Outputs
Under: `data/outputs/<exp_name>/`
- per-fold:
  - `foldK/model_best.pth` (EMA weights)
  - `foldK/model_last.pth` (EMA weights saved every epoch)
  - `foldK/train_log.csv`
  - `foldK/tta_results.json`
  - `foldK/config.json`
- experiment-level:
  - `oof_summary.json`
- submit:
  - `<exp_dir>/ensemble_submit/submission.npy`
  - `<exp_dir>/ensemble_submit/submission.zip` (**zip contains only submission.npy**)

---

## Implementations & Fixes (Chronological, High-Level)

### 1) Depth Cutting (Disable depth loss)
- Added a config switch to **disable depth loss** for segmentation-only training.
- Fixed a DataLoader crash when depth target became `None` by always returning a tensor.

Motivation:
- Multitask depth supervision was suspected to interfere with segmentation early training.
- “Depth cutting” showed clear signs of improvement compared to prior baselines.

### 2) Transform / Augment Alignment and Albumentations Compatibility
- Updated augmentation code to be compatible with **Albumentations v2**:
  - replaced deprecated `ShiftScaleRotate` usage with `A.Affine`
  - updated deprecated `PadIfNeeded` args (`value/mask_value` → `fill/fill_mask`)
- Ensured **validation transform** aligns with train crop/resize:
  - `Resize(720x960) -> CenterCrop(576x768)`
- Ensured normalization consistency:
  - Train/valid: ImageNet mean/std
  - TTA/submit: explicitly normalizes inside the TTA/infer functions

### 3) Reproducible / Robust Path Handling
- Implemented project-root resolution helpers so the script runs from any working directory.
- Added CLI `--data_root` to explicitly point to training data when needed.

### 4) Training Upgrades: Scheduler, Losses, Early Stop
- LR scheduling:
  - Added **OneCycleLR** with `MAX_LR`, using `steps_per_epoch`.
- Loss functions:
  - **CE + Dice** (configurable lambda)
  - optional **Lovasz-Softmax** with ramp-up (`LOVASZ_START_EPOCH`, `LOVASZ_RAMP_EPOCHS`)
  - label smoothing for CE
- Early stopping:
  - `EARLY_STOP_MIN_EPOCHS`, `EARLY_STOP_PATIENCE`, `EARLY_STOP_MIN_DELTA`

### 5) Model-side: Architecture Flexibility + SegFormer Integration
Added architectural options:
- SMP-style: `fpn_mt`, `fpn`, `unetpp`, `deeplabv3p`, `pspnet`
- HF Transformers: `segformer` via wrapper (`SegformerForSemanticSegmentation`)

Important guardrails:
- SegFormer logits are upsampled to match input resolution.
- Warns about likely rule-violating “finetuned” SegFormer model names (ADE20K etc).

### 6) RGBD 4-channel Input
Implemented `USE_DEPTH_AS_INPUT` mode:
- `IN_CHANNELS=4`, depth loaded and concatenated as 4th channel
- **Separated geometric transforms vs color transforms** so depth does not receive RGB-only photometric aug
- Added dedicated `TestImageDataset` to handle test-time depth
- Explicit error: SegFormer does not support 4ch input in this setup

### 7) Mix Augmentations (ClassMix / CutMix)
Implemented training-only mixing:
- `MIX_AUG = none|classmix|cutmix`
- `MIX_PROB`

Key incident/fix:
- A bug caused pretrained encoder weights to be overwritten to `""` via CLI parsing → random init → mIoU ~0.2.
- Fixed CLI parsing to keep `imagenet` by default and only override when user explicitly passes `--encoder_weights`.

### 8) Auxiliary Boundary (Edge) Head
Added optional boundary prediction as an auxiliary task:
- `USE_EDGE_AUX`, `EDGE_LOSS_LAMBDA`, `EDGE_POS_WEIGHT`
- dataset generates `edge_target` from labels
- model returns `(seg, depth, edge)`; training/valid compute BCE loss for edge

### 9) Class-aware Crop (Sampling Rare Classes)
Added class-aware crop sampling:
- `USE_CLASS_AWARE_CROP`, `CLASS_AWARE_ALPHA`, `CLASS_AWARE_MIN_PIXELS`
- dataset precomputes class frequencies; sampling focuses on rarer classes for crop targets

### 10) SAM Optimizer (Sharpness-Aware Minimization)
Implemented SAM compatible with:
- AdamW param groups
- OneCycleLR
- EMA updates
- two-step forward/backward update

SAM bugs found and fixed:
- `AttributeError: 'SAM' object has no attribute 'state'`
  - fixed by keeping SAM perturbations separate from base optimizer state
- `KeyError: 'exp_avg'`
  - fixed by ensuring base optimizer state is initialized appropriately in the SAM step flow

### 11) Efficiency / Usability (Recent)
Improved fold/TTA workflow:
- Removed redundant second pass over validation for confusion matrices:
  - `validate_tta_sweep()` now returns confusion matrices per temperature
  - training reuses them for OOF aggregation
- New CLI flags:
  - `--folds "0,1,2"` (comma-separated; overrides `--fold_only`)
  - `--skip_tta_sweep` (faster training runs; no OOF temp computation)

---

## Key Errors Encountered and Fixes

- `ModuleNotFoundError: albumentations`
  - Added deps to `requirements.txt` and installed.
- `FileNotFoundError: data/train/image`
  - Fixed by robust project-root relative path resolution + `--data_root`.
- DataLoader collate error due to `None` depth target
  - Fixed by always returning a tensor for depth target.
- Albumentations v2 deprecation warnings
  - Fixed by migrating to v2-compatible transforms/args.
- SMP arch + certain encoders channel mismatch (`weight size [0, ...]`)
  - Added guard / guidance; pivoted to supported combos or SegFormer.
- Logging file missing (`train_log.csv`)
  - Made `log_metrics` robust: ensure parent directory exists and initialize headers if missing.
- SAM integration runtime errors (`state`, `exp_avg`)
  - Fixed as described above.

---

## Experiments Snapshot (as seen under `data/outputs/`)

Directories observed:
- `ab_fix_nomix`
- `ab_fix_classmix`
- `ab_edgeaux_fold0`
- `ab_classcrop_fold0`
- `ab_sam_fold0` (SAM rho=0.05)
- `ab_sam_rho002` (SAM rho=0.02, **best among SAM tests**)
- `ab_asam_rho002_fold0` (ASAM rho=0.02, **worse than SAM**)

### SAM vs Baseline (Fold0)
- Baseline `ab_fix_nomix`:
  - best valid mIoU ≈ **0.6941**
  - best TTA mIoU ≈ **0.7022** (temp 1.0)
- SAM rho=0.02 `ab_sam_rho002`:
  - best valid mIoU ≈ **0.7008**
  - best TTA mIoU ≈ **0.7080** (temp 0.7)
- ASAM rho=0.02 `ab_asam_rho002_fold0`:
  - best valid mIoU ≈ **0.6924**
  - best TTA mIoU ≈ **0.7035** (temp 1.0)

Conclusion:
- **SAM(non-adaptive) is a keeper.**
- **ASAM(adaptive) was rejected** due to consistent regression vs SAM.

---

## Current Recommended “Final” Configuration

Adopt:
- `USE_DEPTH_LOSS = False` (depth cutting)
- `USE_SAM = True`, `SAM_RHO = 0.02`, `SAM_ADAPTIVE = False`
- Keep pretrained encoder weights (`ENCODER_WEIGHTS="imagenet"`) unless deliberately testing random init
- Use fold ensemble + TTA; use OOF best temperature from `oof_summary.json` for submit

---

## Canonical Commands (Copy/Paste)

### Train 5 folds with SAM rho=0.02
```bash
python /root/NYUv2/final/main.py \
  --mode train \
  --exp_name sam_rho002_final \
  --use_sam --sam_rho 0.02
```

### Train only specific folds (e.g., fold0 + fold1)
```bash
python /root/NYUv2/final/main.py \
  --mode train \
  --exp_name sam_rho002_partial \
  --use_sam --sam_rho 0.02 \
  --folds "0,1"
```

### Submit (fold-ensemble + TTA), using OOF best temperature automatically
```bash
python /root/NYUv2/final/main.py \
  --mode submit \
  --exp_dir /root/NYUv2/data/outputs/sam_rho002_final
```

### Fast training without TTA sweep (no OOF temp)
```bash
python /root/NYUv2/final/main.py \
  --mode train \
  --exp_name sam_rho002_fast \
  --use_sam --sam_rho 0.02 \
  --skip_tta_sweep
```

---

## Status / Next Actions

- Observation from user: **Fold1 dropped to ~0.68**, so improvements may not be stable across folds.
- Next recommended action (decision point):
  - Run **all 5 folds** with `sam_rho002_final` and check `oof_summary.json`.
  - If OOF mIoU doesn’t move meaningfully, then a larger “architecture/data” shift is required (not just optimizer tweaks).


