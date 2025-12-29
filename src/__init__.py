"""
Compatibility package.

The training/inference scripts under `main/` import modules via `src.*`.
This repo's actual implementation lives under `main/`.

By providing `src` as a thin wrapper, we keep scripts stable and avoid
duplicating code or rewriting all imports.
"""


