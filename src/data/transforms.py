import cv2
import albumentations as A

def safe_shift_scale_rotate(cfg) -> A.BasicTransform:
    try:
        return A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=0.2,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=cfg.IGNORE_INDEX,
            p=0.5,
        )
    except TypeError:
        return A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=0.2,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=cfg.IGNORE_INDEX,
            p=0.5,
        )

def safe_pad_if_needed(cfg, min_height: int, min_width: int) -> A.BasicTransform:
    try:
        return A.PadIfNeeded(
            min_height=min_height,
            min_width=min_width,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=cfg.IGNORE_INDEX,
            position="top_left", # CRITICAL: We want padding at right/bottom -> content at top-left
        )
    except TypeError:
        return A.PadIfNeeded(
            min_height=min_height,
            min_width=min_width,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=cfg.IGNORE_INDEX,
            position="top_left",
        )

def get_train_transforms(cfg) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH),
            A.HorizontalFlip(p=0.5),
            safe_shift_scale_rotate(cfg),
        ],
        additional_targets={"depth": "image", "depth_valid": "mask"},
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
        additional_targets={"depth": "image", "depth_valid": "mask"},
    )
