# Evaluation Protocol (E1 Specialist)

This protocol must be strictly followed for any model comparing against or ensembling with `01_nearest`.

## 1. Inference Pipeline
1. **Input**: RGB Image (Variable size, typically ~480x640).
2. **Forward**: Model outputs logits `(13, H_feat, W_feat)`.
3. **Resize**: Bilinear interpolation to `(480, 640)`.
   - *Note*: Do NOT use Nearest Neighbor for logit resizing.
4. **Post-Process**:
   - Apply `T=1.0` (Identity).
   - `Argmax` over 13 classes.

## 2. Ignore Label
- **Index**: `255`
- **Handling**: All metric calculations (IoU, Accuracy, Boundary IoU) must **mask out** pixels where `GT == 255`.
- **Boundary Generation**: 
  - `GT` is first cleaned: `gt[gt==255] = 0` (Safe value).
  - Boundary map `B` is generated from cleaned GT (`dilate != erode`, k=3).
  - Finally, `B` is masked by `GT != 255`.

## 3. Metrics
- **mIoU_struct**: Mean IoU of `[bed, chair, floor, sofa, table, wall, window]`.
- **mIoU_table**: IoU of `table`.
- **mIoU_boundary**: Intersection/Union of predicted boundary vs GT boundary.
