ensemble/ の最終フォルダ構成（これで固定）
ensemble/
  CONTRACT.md
  configs/
    ensemble_recipe.json
  splits/
    manifest.json
    fold0.json
    fold1.json
    fold2.json
    fold3.json
    fold4.json
  src/
    io.py
    align.py
    metrics.py
    optimize.py
    submit.py
    meta.py
  scripts/
    build_global_oof.py
    optimize_weights.py
    make_submission.py
    eval_submissions.py
    sanity_check_assets.py
  runs/
    run_YYYYMMDD_HHMMSS/
      meta.json
      weights.json
      oof_metrics.json
      submission.npy
      submission.zip
      logs.txt

CONTRACT.md（“契約”として固定）

ここに書く内容を固定する（重要）。実装もこの契約に沿わせる。

必須入力（各モデルが提供）

モデル名は convnext / segformer を前提（将来追加可）。

モデルごとに outputs/<model_tag>/<exp_name>/fold{k}/ に以下があること：

val_logits.(npy|mmap) : (N_val_k, 13, 480, 640) float16

val_file_ids.npy : (N_val_k,) 文字列（basename統一）

test_logits.(npy|mmap) : (654, 13, 480, 640) float16

test_file_ids.npy : (654,)（fold間で完全一致）

meta.json : split_id, tta_branches, in_channels, ckpt_type, commit_hash

必須出力（ensembleが生成）

oof_logits_<model>.npy : (795, 13, 480, 640)

oof_file_ids.npy : (795,)（共通）

weights.json : 最適重み

submission.npy / submission.zip

ensemble_recipe.json（これで運用固定）

例（概念）：

{
  "split_id": "group_block50_v1",
  "models": {
    "convnext": {"path": "../outputs/convnext_exp100_group"},
    "segformer": {"path": "../outputs/segformer_mitbX_group"}
  },
  "ensemble": {
    "method": "logits_weighted_mean",
    "optimize_on": "oof",
    "constraints": {"nonneg": true, "sum_to_one": true},
    "class_gating": {
      "enabled": false,
      "books_only": false
    }
  },
  "output": {"run_dir": "runs/"}
}


※ 最初は class_gatingはoff で固定（壊れにくい）。
必要になったら booksだけ on を追加する。

scripts の役割（これで迷わない）
1) sanity_check_assets.py

最初に必ず通す

split_id一致

file_idsのユニーク

test_file_idsがfold間で一致

dtype/shape一致

convnextとsegformerで test_file_ids が一致

2) build_global_oof.py

fold別 val_logits を file_idで並べ替えて global OOF を作る

oof_logits_convnext.npy

oof_logits_segformer.npy

oof_file_ids.npy

oof_metrics_<model>.json（単体OOFもここで出す）

3) optimize_weights.py

OOFで重み最適化（最初はモデル重み2つだけ）

weights.json

oof_metrics.json（アンサンブルOOF）

最初は探索方法を固定：

2モデルなら wを0.00〜1.00で0.01刻みグリッドで十分（速い・堅い）

後で3本以上になったらDirichlet探索を追加

4) make_submission.py

test_logitsを weights.json で合成して

submission.npy

submission.zip

運用手順（毎回これだけ）

python -m ensemble.scripts.sanity_check_assets --config configs/ensemble_recipe.json

python -m ensemble.scripts.build_global_oof --config ...

python -m ensemble.scripts.optimize_weights --config ...

python -m ensemble.scripts.make_submission --config ...

追加の“固定ルール”（事故が消える）

ensemble/は学習コード禁止

生成物は必ず runs/run_YYYYMMDD_HHMMSS/ に吐く（上書きしない）

meta.json に

split_id

各モデルのcommit_hash

weights

生成日時
を書く（再現性が担保される）

これで ensemble/ は完成形になる。
