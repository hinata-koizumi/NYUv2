"""
One-command submit pipeline (fixed recipe):
  1) compute global temperature T* via OOF (unless --temp is provided)
  2) run fold ensemble inference and write submission.npy (+ submission.zip)
"""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.submit",
        description="One-command pipeline: OOF global temperature -> test ensemble -> submission.npy",
    )
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--oof_summary", type=str, default=None)
    p.add_argument("--recompute_oof", action="store_true")
    p.add_argument("--temps", type=str, default="0.7,0.8,0.9,1.0")
    p.add_argument("--temp", type=float, default=None, help="Override global temperature and skip OOF.")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--batch_mul_oof", type=int, default=1)
    p.add_argument("--batch_mul_test", type=int, default=2)
    p.add_argument("--folds", type=str, default=None)
    p.add_argument("--no_progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    from main.submit import ensemble, oof_temp

    args = build_parser().parse_args(argv)
    exp_dir = os.path.abspath(os.path.expanduser(str(args.exp_dir)))
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"--exp_dir not found: {exp_dir}")

    oof_summary = (
        os.path.abspath(os.path.expanduser(str(args.oof_summary)))
        if args.oof_summary is not None
        else os.path.join(exp_dir, "oof_summary.json")
    )

    if args.temp is None:
        need_oof = bool(args.recompute_oof) or (not os.path.exists(oof_summary))
        if need_oof:
            oof_argv: list[str] = ["--exp_dir", exp_dir, "--out_json", oof_summary, "--temps", str(args.temps)]
            if args.data_root is not None:
                oof_argv += ["--data_root", str(args.data_root)]
            oof_argv += ["--batch_mul", str(int(args.batch_mul_oof))]
            oof_temp.main(oof_argv)

    ens_argv: list[str] = ["--exp_dir", exp_dir, "--oof_summary", oof_summary, "--batch_mul", str(int(args.batch_mul_test))]
    if args.out_dir is not None:
        ens_argv += ["--out_dir", str(args.out_dir)]
    if args.folds is not None:
        ens_argv += ["--folds", str(args.folds)]
    if args.temp is not None:
        ens_argv += ["--temp", str(float(args.temp))]
    if args.data_root is not None:
        ens_argv += ["--data_root", str(args.data_root)]
    if bool(args.no_progress):
        ens_argv += ["--no_progress"]

    ensemble.main(ens_argv)


if __name__ == "__main__":
    main()


