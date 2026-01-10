
import argparse
import ast
import re
from typing import Any, get_args, get_origin
from .base_config import Config

def _coerce_scalar(val: str) -> Any:
    """
    Best-effort parsing for CLI overrides.
    """
    s = str(val).strip()
    low = s.lower()
    if low in ("true", "t", "yes", "y", "1", "on"):
        return True
    if low in ("false", "f", "no", "n", "0", "off"):
        return False

    # int / float
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", s) or re.fullmatch(
            r"[+-]?\d+(?:[eE][+-]?\d+)", s
        ):
            return float(s)
    except Exception:
        pass

    # Python literal
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _coerce_to_field_type(field_type: Any, raw: Any) -> Any:
    """
    Coerce override value to Config field type.
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is None and str(field_type).startswith("typing.Optional"):
        pass
    if origin is type(None):
        return None
    
    # Tuples
    if origin is tuple:
        elem_t = args[0] if args else Any
        if isinstance(raw, str):
            if "," in raw and not (raw.strip().startswith(("(", "[", "{"))):
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                if elem_t is float:
                    return tuple(float(p) for p in parts)
                return tuple(int(p) for p in parts)
            raw2 = _coerce_scalar(raw)
        else:
            raw2 = raw
        
        if isinstance(raw2, (list, tuple)):
            return tuple(_coerce_to_field_type(elem_t, x) for x in raw2)
        return (_coerce_to_field_type(elem_t, raw2),)

    # Scalars
    if field_type is bool:
        if isinstance(raw, bool): return raw
        return bool(_coerce_scalar(str(raw)))
    if field_type is int:
        if isinstance(raw, int) and not isinstance(raw, bool): return raw
        return int(_coerce_scalar(str(raw)))
    if field_type is float:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool): return float(raw)
        return float(_coerce_scalar(str(raw)))
    if field_type is str:
        return str(raw)

    if isinstance(raw, str):
        return _coerce_scalar(raw)
    return raw


def _parse_set_overrides(pairs: list[str], cfg) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    fields = getattr(cfg, "__dataclass_fields__", {})
    if pairs:
        for item in pairs:
            if "=" not in str(item):
                raise ValueError(f"Invalid --set {item!r}. Expected KEY=VALUE.")
            k, v = str(item).split("=", 1)
            key = k.strip()
            if key not in fields:
                raise ValueError(f"Unknown config field in --set: {key!r}")
            raw = v.strip()
            field_t = fields[key].type
            overrides[key] = _coerce_to_field_type(field_t, raw)
    return overrides


def build_config_from_args(preset: str | None = None, exp_name: str | None = None, set_pairs: list[str] | None = None) -> Config:
    cfg = Config()
    if preset:
        cfg = cfg.apply_preset(preset)
    if exp_name:
        cfg = cfg.with_overrides(EXP_NAME=str(exp_name))
    
    overrides = _parse_set_overrides(set_pairs, cfg)
    if overrides:
        cfg = cfg.with_overrides(**overrides)
        
    cfg.validate()
    return cfg
