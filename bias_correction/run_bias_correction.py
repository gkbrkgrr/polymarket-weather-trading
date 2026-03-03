#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from bias_correction.config import build_config
from bias_correction.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe two-stage MOS bias-correction pipeline for daily station Tmax predictions."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path")
    parser.add_argument("--predictions_root", type=str, default=None)
    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--backfill_start", type=str, default=None)
    parser.add_argument("--backfill_end", type=str, default=None)
    parser.add_argument("--rolling_window_days", type=int, default=None)
    parser.add_argument("--ewma_halflife_days", type=float, default=None)
    parser.add_argument("--min_history", type=int, default=None)
    parser.add_argument(
        "--model_strategy",
        type=str,
        default=None,
        choices=["single_residual_with_model_feature", "per_model"],
    )
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model directories to process, e.g. xgb,xgb_opt",
    )
    parser.add_argument("--output_suffix", type=str, default=None)
    parser.add_argument("--obs_source_path", type=str, default=None)
    parser.add_argument("--obs_source_dsn", type=str, default=None)
    parser.add_argument("--intermediate_dir", type=str, default=None)
    parser.add_argument("--artifacts_root", type=str, default=None)
    parser.add_argument("--log_level", type=str, default=None)
    parser.add_argument(
        "--reuse_intermediate",
        dest="reuse_intermediate",
        action="store_true",
        help="Reuse cached intermediate parquet tables when available",
    )
    parser.add_argument(
        "--no_reuse_intermediate",
        dest="reuse_intermediate",
        action="store_false",
        help="Force rebuilding intermediate tables",
    )
    parser.set_defaults(reuse_intermediate=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--strict_schema", action="store_true")
    parser.add_argument(
        "--no_overwrite_output",
        action="store_true",
        help="Do not overwrite existing output parquet files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    cli_overrides = {
        "predictions_root": args.predictions_root,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "backfill_start": args.backfill_start,
        "backfill_end": args.backfill_end,
        "rolling_window_days": args.rolling_window_days,
        "ewma_halflife_days": args.ewma_halflife_days,
        "min_history": args.min_history,
        "model_strategy": args.model_strategy,
        "n_jobs": args.n_jobs,
        "models": args.models,
        "output_suffix": args.output_suffix,
        "obs_source_path": args.obs_source_path,
        "obs_source_dsn": args.obs_source_dsn,
        "intermediate_dir": args.intermediate_dir,
        "artifacts_root": args.artifacts_root,
        "log_level": args.log_level,
        "reuse_intermediate": args.reuse_intermediate,
        "dry_run": bool(args.dry_run) if args.dry_run else None,
        "strict_schema": bool(args.strict_schema) if args.strict_schema else None,
        "overwrite_output": False if args.no_overwrite_output else None,
    }

    cfg = build_config(
        config_path=args.config,
        cli_overrides=cli_overrides,
        repo_root=repo_root,
    )
    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
