## 概要

RGB画像から、画像内の各ピクセルがどのクラスに属するかを予測するセマンティックセグメンテーションタスクです。

### データセット

- データセット: NYUv2 dataset
- 訓練データ: 795枚
- テストデータ: 654枚
- 入力: RGB画像 + 深度マップ（元画像サイズは可変）
- 出力: 13クラスのセグメンテーションマップ
- 評価指標: Mean IoU (Intersection over Union)

#### データセットの詳細（[NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)）

- 画像は屋内シーンを撮影したもので、家具や壁、床などの物体が含まれています
- 各画像に対して13クラスのセグメンテーションラベルが提供されます
- データは以下のディレクトリ構造で提供:

```
data/NYUv2/
├─train/
│  ├─image/
│  ├─depth/
│  └─label/
└─test/
   ├─image/
   └─depth/
```

### タスクの詳細

- 入力のRGB画像と深度マップから、各ピクセルが13クラスのどれに属するかを予測
- 評価はMean IoUを使用
  - 各クラスごとにIoUを計算し、その平均を取る
  - IoUは以下の式で計算: `IoU = TP / (TP + FP + FN)`

### 前処理

- 入力画像は512×512にリサイズ
- ピクセル値は0-1に正規化
- セグメンテーションラベルは0-12の整数値（13クラス）
  - 255はignore index（評価から除外）

### 提出形式

- テスト画像（RGB + Depth）の各ピクセルに対してクラス（0~12）を予測したものをnumpy配列として保存
- ファイル名: `submission.npy`
- 配列の形状: `[テストデータ数, 高さ, 幅]`
- 各ピクセルの値: 0-12の整数（予測クラス）
- 提出時は `submission.npy`、テストに使用した `.pt` 重み、ノートブックを ZIP でまとめる

## 考えられる工夫の例

- 事前学習モデルの fine-tuning
- 損失関数の再設計（クラス不均衡対策）
- データ拡張（RandomResizedCrop / Flip / ColorJitter など）

## 注意点

- 最終的な予測モデルは、配布された訓練データで学習したものを使用すること
- 学習を行わず事前学習済みモデルの知識のみを利用した推論は禁止
  - 例: LLMへ入力して推論結果のみを取得する行為

### 事前学習モデルの利用

許可:

- 構成要素としての事前学習モデルの利用（特徴抽出など）
- 上記構成要素のファインチューニング

禁止:

- タスク解決用の事前学習モデルをそのまま利用すること
  - 例: VQAタスク用モデルをそのまま VQA に使う

## 実験履歴
実験1：
dl_basic_2025_competition_nyuv2_baseline-1.pyとnyuv2_sota_pipeline-1.pyをrun
Fold 1: Best mIoU: 0.6116 (Epoch 49) 
Fold 2: Best mIoU: 0.6120 (Epoch 44) 
Fold 3: Best mIoU: 0.6157 (Epoch 50) 
Fold 4: Best mIoU: 0.6124 (Epoch 50) 
Fold 5: Best mIoU: 0.6102 (Epoch 45)
LBスコア:0.59193

実験2：
dl_basic_2025_competition_nyuv2_baseline-2.py
nyuv2_sota_pipeline-2.py
Fold 1 best mIoU ≈ 0.6318
Fold 2 best mIoU ≈ 0.6204
Fold 3 best mIoU ≈ 0.6367
Fold 4 best mIoU ≈ 0.6180 
Fold 5 best mIoU ≈ 0.6412
LBスコア:0.61031
