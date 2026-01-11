import argparse
import ast
import json
import os
import sys
import warnings
from typing import Any, get_args, get_origin

warnings.filterwarnings("ignore", category=UserWarning)

from .configs.base_config import Config
from .engine.train_task import run_train
from .engine.infer_task import run_infer

def _coerce_scalar(val: str) -> Any:
    s = str(val).strip()
    low = s.lower()
    if low in ("true", "t", "yes", "y", "1", "on"):
        return True
    if low in ("false", "f", "no", "n", "0", "off"):
        return False
    try:
        import re
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", s):
            return float(s)
    except Exception:
        pass
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

def _load_recipe(recipe_path: str, cfg: Config) -> Config:
    with open(recipe_path, "r") as f:
        recipe = json.load(f)
    
    overrides = {}
    
    # Map Recipe Keys to Config Keys
    if "lr" in recipe: overrides["LEARNING_RATE"] = float(recipe["lr"])
    if "image_size_train" in recipe:
        overrides["RESIZE_HEIGHT"] = recipe["image_size_train"][0]
        overrides["RESIZE_WIDTH"] = recipe["image_size_train"][1]
    if "crop_size" in recipe:
        overrides["CROP_SIZE"] = tuple(recipe["crop_size"])
    if "in_channels" in recipe: overrides["IN_CHANNELS"] = int(recipe["in_channels"])
    if "seed" in recipe: overrides["SEED"] = int(recipe["seed"])
    
    if "tta" in recipe:
        scales = recipe["tta"].get("scales", [1.0])
        hflip = recipe["tta"].get("hflip", False)
        combs = []
        for s in scales:
            combs.append((float(s), False))
            if hflip:
                combs.append((float(s), True))
        overrides["TTA_COMBS"] = tuple(combs)
        
    if "books_protect" in recipe:
        val = recipe["books_protect"]
        if isinstance(val, bool):
             overrides["INFER_TTA_BOOKS_PROTECT"] = val
        elif isinstance(val, dict):
             overrides["INFER_TTA_BOOKS_PROTECT"] = val.get("enabled", True)
             
    if "SPLIT_MODE" in recipe: overrides["SPLIT_MODE"] = recipe["SPLIT_MODE"]
    if "CKPT_TYPE" in recipe: overrides["CKPT_SELECT"] = recipe["CKPT_TYPE"]
    
    return cfg.with_overrides(**overrides)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m 01_nearest", description="NYUv2 Nearest Final CLI")
    p.add_argument("--frozen", action="store_true", help="Force frozen mode (block train)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1. Train
    tr = sub.add_parser("train", help="Train Exp100 model")
    tr.add_argument("--fold", type=int, default=None, help="Run specific fold (0-4). If None, run all.")
    tr.add_argument("--recipe", type=str, default=None, help="Path to json recipe")
    tr.add_argument("--exp_name", type=str, default="nearest_final", help="Experiment name")
    tr.add_argument("--limit", type=int, default=0, help="Limit training data (debug)")
    tr.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override Config fields (repeatable). Example: --set LEARNING_RATE=2e-4",
    )
    
    # 2. Infer
    inf = sub.add_parser("infer", help="Generate OOF/Test logits")
    inf.add_argument("--fold", type=int, required=True, help="Fold index")
    inf.add_argument("--split", type=str, default=None, help="Split mode (e.g. block50)")
    inf.add_argument("--mode", type=str, choices=["oof", "test", "both"], default="oof", help="Inference mode")
    inf.add_argument("--save_logits", action="store_true", help="Save logits to disk")
    inf.add_argument("--exp_name", type=str, default="nearest_final", help="Experiment name")
    inf.add_argument("--limit", type=int, default=0, help="Limit data (debug)")
    
    # 3. Make Submission
    sbm = sub.add_parser("make-submission", help="Generate submission from test logits")
    sbm.add_argument("--exp_dir", type=str, required=True, help="Experiment directory containing folds")
    sbm.add_argument("--folds", type=int, nargs="+", default=[0,1,2,3,4], help="Folds to include")
    sbm.add_argument("--out", type=str, default="submission", help="Output filename base")
    
    # 4. Generate Golden (Pass-through for verify)
    gen = sub.add_parser("generate-golden", help="Generate Golden Artifacts")
    gen.add_argument("--sanity", action="store_true")
    gen.add_argument("--all_folds", action="store_true")
    gen.add_argument("--fold", type=int, default=0)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Base Config
    cfg = Config()
    
    # --- Train ---
    if args.cmd == "train":
        # Freeze Check
        base = os.path.dirname(__file__)
        lock_golden = os.path.join(base, "golden_artifacts/FROZEN.lock")
        lock_root = os.path.join(base, "../FROZEN.lock") # Root logic
        # Assuming run from root, os.path.join("FROZEN.lock") is simpler, 
        # but let's be robust relative to module.
        
        # Best way to find project root from module is relative
        # 01_nearest/cli.py -> ../FROZEN.lock
        lock_alias = os.path.abspath(os.path.join(base, "..", "FROZEN.lock"))
        
        if args.frozen or os.path.exists(lock_golden) or os.path.exists(lock_alias):
            print("FROZEN. use v1.0 tag")
            sys.exit(1)

        if args.recipe:
            print(f"Loading recipe from {args.recipe}")
            cfg = _load_recipe(args.recipe, cfg)
        
        cfg = cfg.with_overrides(EXP_NAME=args.exp_name)
        
        if args.set:
             overrides = _parse_set_overrides(args.set, cfg)
             if overrides:
                 cfg = cfg.with_overrides(**overrides)

        run_train(cfg, args.fold, limit=args.limit)
        return

    # --- Infer ---
    if args.cmd == "infer":
        from .constants import DEFAULT_OUTPUT_ROOT
        # Default to 00_data/output to match base_config if not overridden
        output_root = "00_data/output" 
        exp_dir = os.path.join(output_root, args.exp_name)
        
        run_infer(
            exp_dir=exp_dir,
            fold=args.fold,
            split_mode=args.split,
            save_logits=args.save_logits,
            limit=args.limit
        )
        return

    # --- Make Submission ---
    if args.cmd == "make-submission":
        from .submit.make_submission import run_make_submission
        run_make_submission(exp_dir=args.exp_dir, output_name=args.out)
        return
        
    # --- Generate Golden ---
    if args.cmd == "generate-golden":
        from .submit.generate_golden import run_generate_golden
        run_generate_golden(sanity=args.sanity, all_folds=args.all_folds, fold=args.fold)
        return

if __name__ == "__main__":
    main()