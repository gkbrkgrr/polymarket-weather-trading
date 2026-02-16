#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependencies. Install with: pip install numpy pandas pyarrow"
    ) from exc

try:
    import joblib
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'joblib'. Install with: pip install joblib"
    ) from exc

try:
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'scikit-learn'. Install with: pip install scikit-learn"
    ) from exc

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'xgboost'. Install with: pip install xgboost"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "train_data" / "gfs"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "tmax_mos_xgb_v1"

FEATURE_COLUMNS = [
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
    "city_name",
]

NUMERIC_COLUMNS = [
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

CITY_COLUMN = "city_name"
DATE_COLUMN = "target_date_local"
OBS_COLUMN = "tmax_obs_c"
RAW_COLUMN = "tmax_raw_c"
TARGET_COLUMN = "bias_target"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost MOS correction model for daily Tmax."
    )
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--valid_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=8000)
    parser.add_argument("--early_stopping_rounds", type=int, default=200)
    return parser.parse_args(argv)


def collect_parquet_files(data_root: Path) -> list[Path]:
    files = sorted(data_root.glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under: {data_root}")
    return files


def load_all_parquet(files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in files:
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def build_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = pred - actual
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    return {"rmse": rmse, "mae": mae, "bias": bias}


def percentile_cutoff_date(unique_dates: np.ndarray, valid_frac: float) -> pd.Timestamp:
    if unique_dates.size < 2:
        raise SystemExit("Need at least 2 unique target dates for train/validation split.")
    cutoff_quantile = 1.0 - valid_frac
    date_ns = unique_dates.astype("datetime64[ns]").astype(np.int64)
    try:
        cutoff_ns = int(np.quantile(date_ns, cutoff_quantile, method="nearest"))
    except TypeError:
        cutoff_ns = int(np.quantile(date_ns, cutoff_quantile, interpolation="nearest"))
    cutoff_date = pd.to_datetime(cutoff_ns).normalize()
    if cutoff_date <= pd.Timestamp(unique_dates.min()):
        cutoff_date = pd.to_datetime(unique_dates[1]).normalize()
    if cutoff_date > pd.Timestamp(unique_dates.max()):
        cutoff_date = pd.to_datetime(unique_dates[-1]).normalize()
    return cutoff_date


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (0.0 < args.valid_frac < 1.0):
        raise SystemExit("--valid_frac must be in (0, 1).")
    if args.n_estimators < 1:
        raise SystemExit("--n_estimators must be >= 1.")
    if args.early_stopping_rounds < 1:
        raise SystemExit("--early_stopping_rounds must be >= 1.")

    np.random.seed(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    files = collect_parquet_files(data_root)
    print(f"Found {len(files)} parquet files under {data_root}")

    df = load_all_parquet(files)
    print(f"Loaded {len(df)} rows before cleaning")

    required_columns = set(FEATURE_COLUMNS + [OBS_COLUMN, DATE_COLUMN])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise SystemExit(
            "Missing required columns in train parquet files: "
            + ", ".join(missing_columns)
        )

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce").dt.normalize()
    df[CITY_COLUMN] = df[CITY_COLUMN].astype("string")
    for col in NUMERIC_COLUMNS + [OBS_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    rows_before_drop = len(df)
    drop_required = FEATURE_COLUMNS + [OBS_COLUMN, DATE_COLUMN]
    df = df.dropna(subset=drop_required).copy()
    rows_removed = rows_before_drop - len(df)
    print(f"Removed {rows_removed} rows with missing required predictors/target/date")
    print(f"Remaining rows after cleaning: {len(df)}")

    if df.empty:
        raise SystemExit("No rows remaining after cleaning.")

    df[TARGET_COLUMN] = df[OBS_COLUMN] - df[RAW_COLUMN]
    df = df.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)

    unique_dates = np.sort(df[DATE_COLUMN].dropna().unique())
    cutoff_date = percentile_cutoff_date(unique_dates=unique_dates, valid_frac=args.valid_frac)
    train_mask = df[DATE_COLUMN] < cutoff_date
    valid_mask = df[DATE_COLUMN] >= cutoff_date

    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    if train_df.empty or valid_df.empty:
        raise SystemExit(
            f"Train/valid split failed with cutoff_date={cutoff_date.date()}: "
            f"train_rows={len(train_df)}, valid_rows={len(valid_df)}"
        )

    print(
        "Split by target_date_local: "
        f"cutoff_date={cutoff_date.date()} train_rows={len(train_df)} valid_rows={len(valid_df)}"
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET_COLUMN].to_numpy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("city", build_ohe(), [CITY_COLUMN]),
            ("num", "passthrough", NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    n_jobs = os.cpu_count() or -1
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": n_jobs,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "n_estimators": int(args.n_estimators),
        "random_state": int(args.seed),
    }

    fit_params = inspect.signature(XGBRegressor.fit).parameters
    init_params = inspect.signature(XGBRegressor.__init__).parameters

    fit_uses_early_stopping = "early_stopping_rounds" in fit_params
    init_uses_early_stopping = "early_stopping_rounds" in init_params

    if init_uses_early_stopping:
        xgb_params["early_stopping_rounds"] = int(args.early_stopping_rounds)

    model = XGBRegressor(**xgb_params)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    preprocessor_for_eval = clone(preprocessor)
    preprocessor_for_eval.fit(X_train)
    X_valid_eval = preprocessor_for_eval.transform(X_valid)

    fit_kwargs = {
        "model__eval_set": [(X_valid_eval, y_valid)],
        "model__verbose": False,
    }
    if fit_uses_early_stopping and not init_uses_early_stopping:
        fit_kwargs["model__early_stopping_rounds"] = int(args.early_stopping_rounds)

    pipeline.fit(X_train, y_train, **fit_kwargs)

    y_pred_valid = pipeline.predict(X_valid)
    valid_df = valid_df.copy()
    valid_df["tmax_pred_c"] = valid_df[RAW_COLUMN].to_numpy() + y_pred_valid

    baseline_metrics = compute_metrics(
        pred=valid_df[RAW_COLUMN].to_numpy(),
        actual=valid_df[OBS_COLUMN].to_numpy(),
    )
    corrected_metrics = compute_metrics(
        pred=valid_df["tmax_pred_c"].to_numpy(),
        actual=valid_df[OBS_COLUMN].to_numpy(),
    )

    print("\nValidation metrics (overall)")
    print(
        "Baseline raw GFS: "
        f"RMSE={baseline_metrics['rmse']:.4f} "
        f"MAE={baseline_metrics['mae']:.4f} "
        f"Bias={baseline_metrics['bias']:.4f}"
    )
    print(
        "Corrected MOS:    "
        f"RMSE={corrected_metrics['rmse']:.4f} "
        f"MAE={corrected_metrics['mae']:.4f} "
        f"Bias={corrected_metrics['bias']:.4f}"
    )

    per_city_records: list[dict[str, float | str | int]] = []
    for city_name, city_df in valid_df.groupby(CITY_COLUMN, sort=True):
        city_baseline = compute_metrics(
            pred=city_df[RAW_COLUMN].to_numpy(),
            actual=city_df[OBS_COLUMN].to_numpy(),
        )
        city_corrected = compute_metrics(
            pred=city_df["tmax_pred_c"].to_numpy(),
            actual=city_df[OBS_COLUMN].to_numpy(),
        )
        per_city_records.append(
            {
                "city_name": str(city_name),
                "rows": int(len(city_df)),
                "baseline_rmse": city_baseline["rmse"],
                "baseline_mae": city_baseline["mae"],
                "baseline_bias": city_baseline["bias"],
                "corrected_rmse": city_corrected["rmse"],
                "corrected_mae": city_corrected["mae"],
                "corrected_bias": city_corrected["bias"],
            }
        )

    per_city_df = pd.DataFrame(per_city_records).sort_values("city_name").reset_index(drop=True)
    print("\nValidation metrics by city")
    if per_city_df.empty:
        print("No validation rows by city.")
    else:
        print(per_city_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    joblib.dump(pipeline, model_path)

    metrics_payload = {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "cutoff_date": cutoff_date.date().isoformat(),
        "rows_loaded_before_cleaning": int(rows_before_drop),
        "rows_removed_missing_required": int(rows_removed),
        "overall_metrics": {
            "baseline_raw_gfs": baseline_metrics,
            "corrected_mos": corrected_metrics,
        },
        "per_city_metrics": per_city_records,
        "hyperparameters_used": {
            **xgb_params,
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "valid_frac": float(args.valid_frac),
        },
    }
    best_iteration = getattr(pipeline.named_steps["model"], "best_iteration", None)
    if best_iteration is not None:
        metrics_payload["best_iteration"] = int(best_iteration)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True)

    print(f"\nSaved model pipeline: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
