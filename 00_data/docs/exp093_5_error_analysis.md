# Exp093.5 (ConvNeXt RGB-D 4ch) エラー分析レポート（統合版）

対象: `exp095_05_repro` を用いて再現した `exp093_5_convnext_rgbd_4ch_repro`（5-fold, 各fold `model_best.pth`）  
目的: 「mIoUが低い」で終わらせず、**何がどこで壊れているか**を構造化し、次の打ち手（Loss/Aug/解像度/Depth運用）に直結させる。

---

## 生成物（分析アーティファクト）

本レポートは、以下に出力された **fold別の分析結果**を集約して書いています。

- **出力先**: `data/analysis/exp093_5_convnext_rgbd_4ch_repro/fold{0..4}/`
- **各foldのファイル**
  - **(1) 正規化混同行列**: `confusion_normalized_gt.png`, `confusion_normalized_gt.csv`, `confusion_top_pairs.json`
  - **(2) 物体サイズ×精度**: `scale_sensitivity_scatter.png`, `instance_iou_vs_area.csv`
  - **(3) Depth依存**: `depth_accuracy_by_bin.png`, `depth_accuracy_by_bin.csv`, `depth_dependency_summary.json`
  - **(4) Worst-20可視化**: `worst_k.csv`, `worst_k/*.png`（2x2: RGB / Depth / GT / Pred）

再実行コマンド例:

```bash
python3 /root/datasets/NYUv2/exp095_05_repro/analyze_valid.py \
  --exp_name exp093_5_convnext_rgbd_4ch_repro --fold 0 \
  --batch_size 2 --num_workers 2 --worst_k 20
```

（最終挙動に寄せたい場合は `--tta` を付与）

---

## 全体サマリ（fold0-4 集計）

- **mIoU**: 平均 **0.6976**（std 0.0060）
- **Pixel Accuracy**: 平均 **0.8597**（std 0.0021）

**クラス別IoU（平均、低い順）**:
- **Books**: 0.3576（std 0.1016, min 0.2298, max 0.5147）
- **Table**: 0.5350（std 0.0373）
- **Objects**: 0.6333（std 0.0070）
- **Picture**: 0.6447（std 0.0355）
- **TV**: 0.6898（std 0.0658）
- **Floor/Wall** は高く安定（Floor ~0.93, Wall ~0.85）

**結論（まず最初に疑うべきボトルネック）**:
- **Books / Table / TV / Picture / Objects が弱点**で、特に **Books が極端に低い**（fold間のばらつきも大きい）
- 小物体・細部が **親カテゴリに吸われる**（後述の混同解析と一致）
- **小さい連結成分ほどIoUがほぼゼロ**（Smart Crop導入の効果が十分に出ていない可能性）
- Depthは有効に見える一方、**遠方帯域で正解率が落ちる**（Depth運用/解像度/学習設計の見直し余地）

---

## 1) 「混同」の構造化：何と何を間違えているのか？

参照: 各foldの `confusion_normalized_gt.png`（GT行で正規化）と `confusion_top_pairs.json`

### fold横断で一貫して強い混同（top-5に入る頻度）

- **Table → Furniture（5/5 fold）**
- **Picture → Objects（5/5 fold）**
- **Books → Furniture（4/5 fold）**
- 次点: **TV → Furniture（2/5 fold）**

### 解釈（境界で起きていること）

- **包含関係/飲み込み**が強い: 小さくて細いカテゴリ（Books, TV, Table など）が **Furniture（親）**に吸われる  
  - 「本棚（Bookshelf）」や「キャビネット（Cabinet）」は13クラスでは `Furniture` に寄るため、実質的に「小物が親に飲まれる」問題として観測される
- **Picture → Objects** は「壁上の平面物体」が Objects 側に寄る（輪郭/境界の曖昧さ、面積の小ささが影響）

### 推奨アクション

- **Boundary-aware系の強化**: 特に **Books/Table/TV vs Furniture** 境界の寄与を上げる（境界画素重み）
- **クラスペア特化Aug**: CutMix/Copy-Paste を「親への吸い込み」が多いクラス中心に（境界を含む貼り付け）
- **Booksの不安定性**が大きいので、Worst-20で「GTノイズ（Booksの塗りが粗い/欠落）」が混ざってないか要確認

---

## 2) 「物体サイズ」と精度の相関：Smart Cropは機能しているか？

参照: 各foldの `scale_sensitivity_scatter.png`, `instance_iou_vs_area.csv`

### 指標の定義（重要）

- GTの各クラスについて **連結成分（connected component）** を取り、その成分と **予測の同一クラス領域（クラス全体のマスク）** の IoU を計算
  - インスタンスセグではないため、**「そのクラスの予測が散っている」ほどIoUが厳しく下がる**（=やや保守的）
  - それでも「面積が小さいとほぼ当たらない」傾向を見るには十分

### 面積bin別のIoU（fold0-4 集計）

**全クラス**
- <=64px: mean 0.0002 / median 0.0000
- 64-256px: mean 0.0046 / median 0.0012
- 256-1kpx: mean 0.0226 / median 0.0075
- 1k-4kpx: mean 0.0773 / median 0.0366
- 4k-16kpx: mean 0.2262 / median 0.1479
- >16kpx: mean 0.4904 / median 0.4704

**小物体クラス（1,3,6,7,10 = Books/Chair/Objects/Picture/TV）**
- <=64px: mean 0.0005 / median 0.0000
- 64-256px: mean 0.0049 / median 0.0016
- …（以降も概ね同傾向）

### 解釈

- **極小（~256px以下）がほぼ壊滅**。Smart Crop を導入していても、現設定では救い切れていない可能性が高い
- 小物体側の改善は、IoUを直接押し上げるだけでなく、(1)の「親に吸われる」混同も抑制しやすい

### 推奨アクション

- **入力解像度アップ / crop設計見直し**（まず最優先）
- **Smart Cropの再調整**: `smart_crop_prob`, cropサイズ、`small_obj_ids` の見直し
- **小物体寄りの損失**（class reweight / focal寄り / boundary寄り）を併用

---

## 3) 「深度（Depth）」依存度：Depthは毒か薬か？

参照: 各foldの `depth_accuracy_by_bin.png`, `depth_accuracy_by_bin.csv`, `depth_dependency_summary.json`

### 距離別エラー率（fold0-4平均）

Depth[m]帯域のピクセル正解率（平均 ± std）:
- 1.10m: 0.8351 ± 0.0095
- 2.65m: 0.8766 ± 0.0101（中距離は比較的良い）
- 7.29m: 0.7779 ± 0.0555
- 9.61m: 0.6858 ± 0.0706（**遠方で顕著に悪化**）

### Depth欠損領域の挙動

今回の train/valid 分割では `missing_depth_total_px` が **全foldで0**（`depth_dependency_summary.json`）  
つまり、**Depth欠損（0mm）周辺の乱れ**は、このデータ分布では定量検証できていません。

### 推奨アクション

- **遠方弱化対策**: 遠方は「小物体 + RGBのテクスチャ弱さ + 深度の粗さ」が重なりやすい  
  - 解像度アップ / 小物体対策がまず効く可能性が高い
- **Depth Dropout（augmentation）**: 欠損が少ないデータでも、Depthノイズ/欠損への頑健性を作る目的で導入価値あり

---

## 4) Worst-20の徹底解剖（定性的敗因）

参照: `data/analysis/exp093_5_convnext_rgbd_4ch_repro/fold*/worst_k/*.png`

各パネルは以下を2x2で並べています:
- 左上: RGB
- 右上: Depth可視化（逆深度っぽい色付け）
- 左下: GT
- 右下: Pred

見るべき観点（チェックリスト）:
- **暗所・照明**: 暗い/白飛びで破綻してないか（ColorJitter強化候補）
- **視点**: 見上げ/見下ろしで一貫して壊れるか（幾何Aug/学習データの偏り）
- **ラベルノイズ**: GTが粗いのにモデルは妥当、が混ざるか（除外/ラベル平滑化/境界重み）

---

## 次の打ち手（優先度）

1. **小物体対策（最優先）**
   - 解像度・crop設計の再検討
   - Smart Cropの頻度/サイズ/対象クラス再調整
2. **吸い込み（Books/Table/TV→Furniture）対策**
   - Boundary-aware loss（該当境界に重み）
   - ペア特化Aug（CutMix/Copy-Paste）
3. **遠方帯域の劣化（Depth含む）対策**
   - Depth Dropout導入（RGB判断力を育てる）
   - 遠方小物体が潰れない解像度・学習戦略へ


