import cv2
import albumentations as A

# Keep targets consistent across train/valid for RGB-D (+ valid mask) pipelines
ADDITIONAL_TARGETS = {"depth": "image", "depth_valid": "mask"}


def _compat_fill_kwargs(cfg):
    """
    Albumentations API compatibility:
      - newer: fill / fill_mask
      - older: value / mask_value
    """
    new = {"fill": 0, "fill_mask": cfg.IGNORE_INDEX}
    old = {"value": 0, "mask_value": cfg.IGNORE_INDEX}
    return new, old


def _build_transform(builder_new, builder_old) -> A.BasicTransform:
    """Try building with new kwargs; fall back to old kwargs on TypeError."""
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
            p=0.5,
            **new_fill,
        ),
        lambda: A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=0.2,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
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
            position="top_left",  # CRITICAL: padding at right/bottom -> content at top-left
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
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH),
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
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH),
            safe_pad_if_needed(cfg, min_height=h_pad, min_width=w_pad),
        ],
        additional_targets=ADDITIONAL_TARGETS,
    )
