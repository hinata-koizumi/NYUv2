# Model C (Depth + Context)

Model C v0 targets books/table separation using depth-derived geometry and global context.

## Design Highlights
- Input: RGB + normalized depth + depth gradients (dx, dy) + depth curvature (7ch total).
- Context: FPN decoder + pyramid pooling context head (prioritize global layout).
- Optional auxiliary: planar-ness head from depth curvature (weak supervision).

## Data/Training Focus
- Depth-aware patch sampling (60% targeted, 40% random):
  - books: 2–5m
  - table: 1–3m (prefer 1–2m)
- Loss weights: books/table boosted to 1.5x.
- Depth augmentation: scale jitter, noise, dropout holes, quantization.

## Usage
From `03_model_c`:
```
python train.py --fold 0 --epochs 50 --batch_size 4
```

Config: `03_model_c/configs/default.py`

## Model C-v1 (Hard Mask Prep)
To focus on books/table vs furniture/objects mistakes, build hard masks from frozen Ensemble-2:
```
python tools/build_hard_masks.py
```
This writes `{id}.npz` files under `00_data/output/model_c_hard_masks` by default.
