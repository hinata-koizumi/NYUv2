# Model Card: 01_nearest (Structure Specialist)

## 1. Overview
This model is a strictly specialized ensemble component for NYUv2. 
- **Role**: `Structure Specialist`.
- **Goal**: Maximize performance on structural classes (`bed`, `chair`, `floor`, `sofa`, `table`, `wall`, `window`) and boundary adherence.
- **Base Arc**: ConvNeXt + FPN + `boundary_head`.

## 2. Strengths & Weaknesses
| Type | Description | Metric |
| :--- | :--- | :--- |
| **Strength** | **Structure Precision**: Very few hallucinations of walls/floors. | `struct_precision` > 0.93 |
| **Strength** | **Pillar/Legs**: Explicit table leg supervision via Smart Crop. | `mIoU_table` > 0.55 |
| **Strength** | **Boundaries**: Sharper edges due to Boundary Aux Head. | `mIoU_boundary` > 0.044 |
| **Weakness** | **Books (Far)**: Struggles with small books at distance. | - |
| **Weakness** | **Granularity**: Confusion between `furniture` vs `objects`. | - |

## 3. Ensemble Strategy
- **Recommendation**: Use as the "Base Skeleton".
- **Compositing**:
  - `Struct Classes`: High weight (trust this model).
  - `Non-Struct`: Low weight (defer to Generalist).
  - `Boundary Pixels`: High weight (trust its edge definitions).

## 4. Reproducibility
- **Git Commit**: `af4532dda421cef58e217ac88ebab421d4d8c698`
- **Recipe Hash**: `9b3ddc2f74d4d31938d3afe9a0111b3df390aa1c71bc809a204b8645984733d7`
- **Output Artifacts**: See `golden_artifacts/sha256_manifest.txt`.
