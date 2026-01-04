"""
Submit pipeline (fixed recipe) for NYUv2 exp093_5 style.

This package is intentionally fixed:
  - per-fold: model_best.pth only
  - global temperature T*: computed via OOF
  - fixed TTA: scales [0.75, 1.0] × hflip {False, True}
  - fold ensemble: uniform mean of probabilities, then argmax
"""


