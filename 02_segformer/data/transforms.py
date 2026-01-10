import cv2
import albumentations as A

# --- FIX FOR EXP100 FINAL ---
# Revert depth to "image" (Linear interpolation) for geometric augmentations.
# This prevents jagged artifacts during rotation/scaling.
# Valid mask remains "mask" (Nearest) to keep strict 0/1 boundaries.
ADDITIONAL_TARGETS = {"depth": "image", "depth_valid": "mask"}


def _compat_fill_kwargs(cfg):
    """
    Albumentations API compatibility:
      - newer: fill / fill_mask
      - older: value / mask_value
    """
    # Use IGNORE_INDEX (255) for padding.
    # Dataset.py will convert 255 in Valid Mask to 0 (Invalid), safely handling the border.
    ignore_idx = int(getattr(cfg, "IGNORE_INDEX", 255))
    new = {"fill": 0, "fill_mask": ignore_idx}
    old = {"value": 0, "mask_value": ignore_idx}
    return new, old


def _build_transform(builder_new, builder_old) -> A.BasicTransform:
    try:
        return builder_new()
    except TypeError:
        return builder_old()


def safe_shift_scale_rotate(cfg) -> A.BasicTransform:
    new_fill, old_fill = _compat_fill_kwargs(cfg)

    return _build_transform(
        lambda: A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=0.2,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            interpolation=cv2.INTER_LINEAR, # Explicitly Linear for Image/Depth
            p=0.5,
            **new_fill,
        ),
        lambda: A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=0.2,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            interpolation=cv2.INTER_LINEAR,
            p=0.5,
            **old_fill,
        ),
    )


def safe_pad_if_needed(cfg, min_height: int, min_width: int) -> A.BasicTransform:
    new_fill, old_fill = _compat_fill_kwargs(cfg)

    return _build_transform(
        lambda: A.PadIfNeeded(
            min_height=min_height,
            min_width=min_width,
            border_mode=cv2.BORDER_CONSTANT,
            position="top_left",
            **new_fill,
        ),
        lambda: A.PadIfNeeded(
            min_height=min_height,
            min_width=min_width,
            border_mode=cv2.BORDER_CONSTANT,
            position="top_left",
            **old_fill,
        ),
    )


def get_train_transforms(cfg) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            safe_shift_scale_rotate(cfg),
        ],
        additional_targets=ADDITIONAL_TARGETS,
    )


def get_color_transforms(cfg) -> A.Compose:
    return A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)]
    )


def get_valid_transforms(cfg) -> A.Compose:
    # Scale up to multiple of 32
    h_pad = ((cfg.RESIZE_HEIGHT + 31) // 32) * 32
    w_pad = ((cfg.RESIZE_WIDTH + 31) // 32) * 32

    return A.Compose(
        [
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH, interpolation=cv2.INTER_LINEAR),
            safe_pad_if_needed(cfg, min_height=h_pad, min_width=w_pad),
        ],
        additional_targets=ADDITIONAL_TARGETS,
    )