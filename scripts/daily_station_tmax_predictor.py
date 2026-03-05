#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINT_ROOT = REPO_ROOT / "data" / "point_data" / "gfs" / "raw"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "ml_predictions"

CORE_FEATURE_COLUMNS = [
    "tmax_raw_c",
    "tcc_mean_pct",
    "tcc_max_pct",
    "ws10_mean_mps",
    "ws10_max_mps",
    "td2m_mean_c",
    "td2m_at_tmax_c",
    "rh2m_at_tmax_c",
    "t2m_diurnal_range",
    "ssrd_day_total",
    "tp_day_total",
    "day_of_year_sin",
    "day_of_year_cos",
    "month",
    "station_lat",
    "station_lon",
    "station_elev_m",
]

GLOBAL_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + [
    "city_name",
]

CITY_EXTENDED_FEATURE_COLUMNS = [
    "tmax_raw_c",
    "tcc_mean_pct",
    "tcc_max_pct",
    "tcc_at_tmax_pct",
    "ws10_mean_mps",
    "ws10_max_mps",
    "ws10_at_tmax_mps",
    "td2m_mean_c",
    "td2m_at_tmax_c",
    "rh2m_at_tmax_c",
    "t2m_diurnal_range",
    "ssrd_day_total",
    "tp_day_total",
    "day_of_year_sin",
    "day_of_year_cos",
    "month",
    "lead_time_hours",
    "issue_hour_utc_sin",
    "issue_hour_utc_cos",
    "station_lat",
    "station_lon",
    "station_elev_m",
]

MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "xgb": {
        "kind": "global",
        "model_path": REPO_ROOT / "models" / "tmax_mos_xgb_v1" / "model.joblib",
        "feature_columns": GLOBAL_FEATURE_COLUMNS,
    },
    "xgb_opt": {
        "kind": "global",
        "model_path": REPO_ROOT / "models" / "tmax_mos_xgb_optuna_v1" / "model.joblib",
        "feature_columns": GLOBAL_FEATURE_COLUMNS,
    },
    "xgb_opt_v1_100": {
        "kind": "global",
        "model_path": REPO_ROOT / "models" / "xgb_opt_v1_100" / "model.joblib",
        "feature_columns": GLOBAL_FEATURE_COLUMNS,
    },
    "xgb_opt_v2_100": {
        "kind": "global",
        "model_path": REPO_ROOT / "models" / "xgb_opt_v2_100" / "model.joblib",
        "feature_columns": GLOBAL_FEATURE_COLUMNS,
    },
    "city_extended": {
        "kind": "per_city",
        "model_root": REPO_ROOT / "models" / "city_extended_100",
        "feature_columns": CITY_EXTENDED_FEATURE_COLUMNS,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load daily GFS feature parquet for a cycle, apply a trained MOS model, "
            "and write per-city daily Tmax predictions."
        )
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--cycle", required=True, help="Cycle in YYYYMMDDHH")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output parquet files if they already exist.",
    )
    return parser.parse_args(argv)


def parse_cycle(cycle: str) -> datetime:
    if not re.fullmatch(r"\d{10}", cycle):
        raise SystemExit(f"--cycle must be YYYYMMDDHH, got {cycle!r}")
    return datetime.strptime(cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)


def resolve_source_parquet(cycle: str) -> Path:
    matches = sorted(DEFAULT_POINT_ROOT.glob(f"gfs_{cycle}_*.parquet"))
    if not matches:
        raise SystemExit(
            f"No source parquet found for cycle={cycle} under {DEFAULT_POINT_ROOT}"
        )
    if len(matches) > 1:
        joined = "\n".join(f"- {p}" for p in matches)
        raise SystemExit(
            "Multiple source parquet files found for one cycle; expected exactly one:\n"
            f"{joined}"
        )
    return matches[0]


def validate_input_schema(df: pd.DataFrame, feature_columns: list[str]) -> None:
    required_columns = sorted(set(feature_columns + ["target_date_local", "city_name"]))
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise SystemExit(
            "Input parquet is missing required columns: " + ", ".join(missing)
        )


def validate_cycle_alignment(df: pd.DataFrame, cycle_dt: datetime) -> None:
    if "issue_time_utc" not in df.columns:
        return
    issue_ts = pd.to_datetime(df["issue_time_utc"], utc=True, errors="coerce")
    non_null = issue_ts.dropna()
    if non_null.empty:
        return
    bad = non_null[non_null != pd.Timestamp(cycle_dt)]
    if not bad.empty:
        unique_bad = sorted({ts.isoformat() for ts in bad.tolist()})
        raise SystemExit(
            "Input parquet issue_time_utc does not match --cycle. "
            f"Expected {cycle_dt.isoformat()}, found: {', '.join(unique_bad)}"
        )


def build_global_forecast(
    *,
    df: pd.DataFrame,
    pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    work = df.copy()
    validate_input_schema(work, feature_columns)

    for col in feature_columns:
        if col == "city_name":
            work[col] = work[col].astype("string")
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce").astype(float)
    work["target_date_local"] = pd.to_datetime(
        work["target_date_local"], errors="coerce"
    ).dt.date

    required_columns = sorted(set(feature_columns + ["target_date_local"]))
    work = work.dropna(subset=required_columns).copy()
    if work.empty:
        raise SystemExit("No rows remain after cleaning required input columns.")

    x = work[feature_columns]
    predicted_error = np.asarray(pipeline.predict(x), dtype=float)
    work["Forecast"] = (
        work["tmax_raw_c"].to_numpy(dtype=float) + predicted_error
    ).round(3)
    return work


def build_per_city_forecast(
    *,
    df: pd.DataFrame,
    model_root: Path,
    feature_columns: list[str],
) -> pd.DataFrame:
    work = df.copy()
    validate_input_schema(work, feature_columns)
    work["city_name"] = work["city_name"].astype("string")
    for col in feature_columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype(float)
    work["target_date_local"] = pd.to_datetime(
        work["target_date_local"], errors="coerce"
    ).dt.date

    required_columns = sorted(set(feature_columns + ["target_date_local", "city_name"]))
    work = work.dropna(subset=required_columns).copy()
    if work.empty:
        raise SystemExit("No rows remain after cleaning required input columns.")

    model_cache: dict[str, object] = {}
    frames: list[pd.DataFrame] = []
    missing_models: list[str] = []
    for city_name, city_df in work.groupby("city_name", sort=True):
        city_token = str(city_name)
        model_path = model_root / city_token / "model.joblib"
        if not model_path.exists():
            missing_models.append(city_token)
            continue
        if city_token not in model_cache:
            model_cache[city_token] = joblib.load(model_path)
        model = model_cache[city_token]

        city_pred = city_df.copy()
        x = city_pred[feature_columns]
        predicted_error = np.asarray(model.predict(x), dtype=float)
        city_pred["Forecast"] = (
            city_pred["tmax_raw_c"].to_numpy(dtype=float) + predicted_error
        ).round(3)
        frames.append(city_pred)

    if missing_models:
        uniq = ", ".join(sorted(set(missing_models)))
        raise SystemExit(
            "Missing city-specific model files under "
            f"{model_root}: {uniq}"
        )

    if not frames:
        raise SystemExit("No predictions were produced for any city.")
    return pd.concat(frames, ignore_index=True)


def issue_time_token_yyyymmddhh(series: pd.Series) -> str:
    issue_ts = pd.to_datetime(series, utc=True, errors="coerce").dropna()
    if issue_ts.empty:
        raise ValueError("Cannot build filename token: issue_time_utc is empty/invalid.")
    unique_issue = sorted(set(issue_ts.tolist()))
    if len(unique_issue) != 1:
        raise ValueError(
            "Expected a single issue_time_utc per output file; got multiple values: "
            + ", ".join(ts.isoformat() for ts in unique_issue)
        )
    # Requested format: yyyymmddhh
    return unique_issue[0].strftime("%Y%m%d%H")


def write_city_outputs(
    *,
    df: pd.DataFrame,
    model_name: str,
    overwrite: bool,
) -> list[Path]:
    outputs: list[Path] = []
    for city_name, city_df in df.groupby("city_name", sort=True):
        city_df = city_df.sort_values("target_date_local", kind="mergesort")
        if city_df.empty:
            continue
        token = issue_time_token_yyyymmddhh(city_df["issue_time_utc"])

        out_dir = DEFAULT_OUTPUT_ROOT / model_name / str(city_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir
            / f"{model_name}_daily_tmax_predictions_{token}.parquet"
        )
        if out_path.exists() and not overwrite:
            raise SystemExit(
                f"Output already exists: {out_path}. Re-run with --overwrite to replace."
            )

        tmp_path = out_path.with_suffix(".parquet.part")
        if tmp_path.exists():
            tmp_path.unlink()
        city_df.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)
        outputs.append(out_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cycle_dt = parse_cycle(args.cycle)
    model_cfg = MODEL_CONFIGS[args.model]
    feature_columns = list(model_cfg["feature_columns"])

    src_path = resolve_source_parquet(args.cycle)
    df_src = pd.read_parquet(src_path)
    validate_cycle_alignment(df_src, cycle_dt)

    if model_cfg["kind"] == "global":
        model_path = Path(model_cfg["model_path"])
        if not model_path.exists():
            raise SystemExit(f"Model file not found: {model_path}")
        pipeline = joblib.load(model_path)
        df_pred = build_global_forecast(
            df=df_src,
            pipeline=pipeline,
            feature_columns=feature_columns,
        )
    elif model_cfg["kind"] == "per_city":
        model_root = Path(model_cfg["model_root"])
        if not model_root.exists():
            raise SystemExit(f"Model directory not found: {model_root}")
        df_pred = build_per_city_forecast(
            df=df_src,
            model_root=model_root,
            feature_columns=feature_columns,
        )
    else:
        raise SystemExit(f"Unsupported model kind: {model_cfg['kind']!r}")

    outputs = write_city_outputs(
        df=df_pred,
        model_name=args.model,
        overwrite=bool(args.overwrite),
    )

    print(f"Source: {src_path}")
    print(f"Rows predicted: {len(df_pred)}")
    print(f"City files written: {len(outputs)}")
    for out in outputs:
        print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
