#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependencies. Install with: pip install numpy pandas pyarrow"
    ) from exc

try:
    import pyarrow.dataset as ds
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'pyarrow'. Install with: pip install pyarrow"
    ) from exc

try:
    import joblib
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'joblib'. Install with: pip install joblib"
    ) from exc

try:
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
DEFAULT_STATIONS_CSV = REPO_ROOT / "locations.csv"
DEFAULT_MODELS_ROOT = REPO_ROOT / "models"
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "retrain_tmax_mos_100.log"

CITY_COLUMN = "city_name"
DATE_COLUMN = "target_date_local"
OBS_COLUMN = "tmax_obs_c"
RAW_COLUMN = "tmax_raw_c"
TARGET_COLUMN = "mos_target_delta_c"

GLOBAL_FEATURE_COLUMNS = [
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

GLOBAL_NUMERIC_COLUMNS = [
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

DEFAULT_PROFILES = ("city_extended", "xgb_opt_v1", "xgb_opt_v2")


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    kind: str
    source_path: Path
    output_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain selected MOS models on 100% of available training data using "
            "frozen best hyperparameters and no early stopping."
        )
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=list(DEFAULT_PROFILES),
        default=list(DEFAULT_PROFILES),
        help="Model families to retrain.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--stations-csv", default=str(DEFAULT_STATIONS_CSV))
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_PATH))
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first profile/station failure.",
    )
    return parser.parse_args(argv)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("retrain_tmax_mos_100")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def build_profile_specs(models_root: Path) -> dict[str, ProfileSpec]:
    return {
        "city_extended": ProfileSpec(
            name="city_extended",
            kind="per_station",
            source_path=models_root / "tmax_mos_xgb_optuna_extended_per_station",
            output_path=models_root / "city_extended_100",
        ),
        "xgb_opt_v1": ProfileSpec(
            name="xgb_opt_v1",
            kind="global",
            source_path=models_root / "tmax_mos_xgb_optuna_v1" / "best_params.json",
            output_path=models_root / "xgb_opt_v1_100",
        ),
        "xgb_opt_v2": ProfileSpec(
            name="xgb_opt_v2",
            kind="global",
            source_path=models_root / "tmax_mos_xgb_optuna_v2" / "best_params.json",
            output_path=models_root / "xgb_opt_v2_100",
        ),
    }


def build_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_global_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("city", build_ohe(), [CITY_COLUMN]),
            ("num", "passthrough", GLOBAL_NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_dataset_columns(dataset_root: Path, required_columns: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(str(dataset_root), format="parquet")
    available = set(dataset.schema.names)
    missing = sorted(set(required_columns) - available)
    if missing:
        raise ValueError(
            f"Missing required columns in dataset schema at {dataset_root}: " + ", ".join(missing)
        )
    table = dataset.to_table(columns=required_columns)
    return table.to_pandas()


def load_station_names(stations_csv: Path) -> list[str]:
    if not stations_csv.exists():
        raise FileNotFoundError(f"Stations CSV not found: {stations_csv}")

    df = pd.read_csv(stations_csv)
    cols = {c.lower(): c for c in df.columns}
    if "name" in cols:
        name_col = cols["name"]
    elif "city_name" in cols:
        name_col = cols["city_name"]
    else:
        raise ValueError(
            f"Unsupported station schema in {stations_csv}. Expected 'name' or 'city_name'."
        )

    names: list[str] = []
    seen: set[str] = set()
    for value in df[name_col].tolist():
        name = str(value).strip()
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    if not names:
        raise ValueError(f"No station names found in {stations_csv}")
    return names


def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = pred - actual
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def drop_early_stopping(xgb_params: dict[str, Any]) -> dict[str, Any]:
    params = dict(xgb_params)
    params.pop("early_stopping_rounds", None)
    return params


def normalize_global_df(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    required_columns = GLOBAL_FEATURE_COLUMNS + [OBS_COLUMN, DATE_COLUMN]

    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    out[CITY_COLUMN] = out[CITY_COLUMN].astype("string")
    for col in GLOBAL_NUMERIC_COLUMNS + [OBS_COLUMN]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    rows_before = len(out)
    out = out.dropna(subset=required_columns).copy()
    rows_dropped = rows_before - len(out)
    if out.empty:
        raise ValueError("No rows remaining after cleaning.")

    out[TARGET_COLUMN] = out[OBS_COLUMN] - out[RAW_COLUMN]
    out = out.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    return out, rows_dropped


def normalize_station_df(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    required_columns = feature_columns + [OBS_COLUMN, DATE_COLUMN]

    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    for col in feature_columns + [OBS_COLUMN]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    rows_before = len(out)
    out = out.dropna(subset=required_columns).copy()
    rows_dropped = rows_before - len(out)
    if out.empty:
        raise ValueError("No rows remaining after cleaning.")

    out[TARGET_COLUMN] = out[OBS_COLUMN] - out[RAW_COLUMN]
    out = out.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    return out, rows_dropped


def compute_global_per_city_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for city_name, city_df in df.groupby(CITY_COLUMN, sort=True):
        baseline = compute_metrics(
            pred=city_df[RAW_COLUMN].to_numpy(),
            actual=city_df[OBS_COLUMN].to_numpy(),
        )
        corrected = compute_metrics(
            pred=city_df["tmax_pred_c"].to_numpy(),
            actual=city_df[OBS_COLUMN].to_numpy(),
        )
        rows.append(
            {
                "city_name": str(city_name),
                "rows": int(len(city_df)),
                "baseline_rmse": baseline["rmse"],
                "baseline_mae": baseline["mae"],
                "baseline_bias": baseline["bias"],
                "corrected_rmse": corrected["rmse"],
                "corrected_mae": corrected["mae"],
                "corrected_bias": corrected["bias"],
            }
        )
    rows.sort(key=lambda x: x["city_name"])
    return rows


def retrain_global_profile(
    *,
    spec: ProfileSpec,
    data_root: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not spec.source_path.exists():
        raise FileNotFoundError(f"Source best_params not found: {spec.source_path}")

    src_params = read_json(spec.source_path)
    if "xgb_params_final" not in src_params:
        raise ValueError(f"Missing xgb_params_final in {spec.source_path}")
    xgb_params = drop_early_stopping(dict(src_params["xgb_params_final"]))

    required_columns = GLOBAL_FEATURE_COLUMNS + [OBS_COLUMN, DATE_COLUMN]
    raw_df = load_dataset_columns(data_root, required_columns)
    rows_loaded = int(len(raw_df))
    df, rows_dropped = normalize_global_df(raw_df)

    X = df[GLOBAL_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].to_numpy()

    preprocessor = build_global_preprocessor()
    model = XGBRegressor(**xgb_params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    logger.info("[%s] fitting on 100%% data (rows=%d)", spec.name, len(df))
    pipeline.fit(X, y)

    y_delta_pred = pipeline.predict(X)
    pred_corrected = df[RAW_COLUMN].to_numpy() + y_delta_pred
    baseline = compute_metrics(df[RAW_COLUMN].to_numpy(), df[OBS_COLUMN].to_numpy())
    corrected = compute_metrics(pred_corrected, df[OBS_COLUMN].to_numpy())

    eval_df = df.copy()
    eval_df["tmax_pred_c"] = pred_corrected
    per_city = compute_global_per_city_metrics(eval_df)

    spec.output_path.mkdir(parents=True, exist_ok=True)
    model_path = spec.output_path / "model.joblib"
    best_params_path = spec.output_path / "best_params.json"
    metrics_path = spec.output_path / "metrics.json"

    joblib.dump(pipeline, model_path)
    write_json(
        best_params_path,
        {
            "profile": spec.name,
            "source_best_params_path": str(spec.source_path),
            "retrain_mode": "full_data_no_early_stopping",
            "feature_columns": GLOBAL_FEATURE_COLUMNS,
            "xgb_params_final": xgb_params,
        },
    )
    write_json(
        metrics_path,
        {
            "profile": spec.name,
            "retrain_mode": "full_data_no_early_stopping",
            "rows": {
                "loaded_before_cleaning": rows_loaded,
                "dropped_missing_required": int(rows_dropped),
                "used_for_training": int(len(df)),
            },
            "train_metrics_full_data": {
                "baseline_raw_gfs": baseline,
                "corrected_mos": corrected,
            },
            "per_city_metrics_full_data": per_city,
            "source_best_params_path": str(spec.source_path),
        },
    )

    logger.info("[%s] wrote model: %s", spec.name, model_path)
    return {
        "profile": spec.name,
        "rows_used": int(len(df)),
        "baseline_rmse": baseline["rmse"],
        "corrected_rmse": corrected["rmse"],
        "output_dir": str(spec.output_path),
    }


def retrain_city_extended_profile(
    *,
    spec: ProfileSpec,
    data_root: Path,
    stations_csv: Path,
    fail_fast: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not spec.source_path.exists():
        raise FileNotFoundError(f"Source profile directory not found: {spec.source_path}")

    station_names = load_station_names(stations_csv)
    spec.output_path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for station_name in station_names:
        try:
            src_best_params = spec.source_path / station_name / "best_params.json"
            if not src_best_params.exists():
                raise FileNotFoundError(f"Missing source best_params: {src_best_params}")

            src_payload = read_json(src_best_params)
            feature_columns = src_payload.get("feature_columns")
            if not isinstance(feature_columns, list) or not feature_columns:
                raise ValueError(f"Invalid feature_columns in {src_best_params}")
            if "xgb_params_final" not in src_payload:
                raise ValueError(f"Missing xgb_params_final in {src_best_params}")

            xgb_params = drop_early_stopping(dict(src_payload["xgb_params_final"]))
            station_data_root = data_root / station_name
            if not station_data_root.exists():
                raise FileNotFoundError(f"Station data directory missing: {station_data_root}")

            required_columns = list(feature_columns) + [OBS_COLUMN, DATE_COLUMN]
            raw_df = load_dataset_columns(station_data_root, required_columns)
            rows_loaded = int(len(raw_df))
            df, rows_dropped = normalize_station_df(raw_df, list(feature_columns))

            X = df[list(feature_columns)]
            y = df[TARGET_COLUMN].to_numpy()
            model = XGBRegressor(**xgb_params)

            logger.info("[city_extended/%s] fitting on 100%% data (rows=%d)", station_name, len(df))
            model.fit(X, y)

            y_delta_pred = model.predict(X)
            pred_corrected = df[RAW_COLUMN].to_numpy() + y_delta_pred
            baseline = compute_metrics(df[RAW_COLUMN].to_numpy(), df[OBS_COLUMN].to_numpy())
            corrected = compute_metrics(pred_corrected, df[OBS_COLUMN].to_numpy())

            station_out = spec.output_path / station_name
            station_out.mkdir(parents=True, exist_ok=True)
            model_path = station_out / "model.joblib"
            best_params_path = station_out / "best_params.json"
            metrics_path = station_out / "metrics.json"

            joblib.dump(model, model_path)
            write_json(
                best_params_path,
                {
                    "profile": spec.name,
                    "station_name": station_name,
                    "source_best_params_path": str(src_best_params),
                    "retrain_mode": "full_data_no_early_stopping",
                    "feature_columns": feature_columns,
                    "xgb_params_final": xgb_params,
                },
            )
            write_json(
                metrics_path,
                {
                    "profile": spec.name,
                    "station_name": station_name,
                    "retrain_mode": "full_data_no_early_stopping",
                    "rows": {
                        "loaded_before_cleaning": rows_loaded,
                        "dropped_missing_required": int(rows_dropped),
                        "used_for_training": int(len(df)),
                    },
                    "train_metrics_full_data": {
                        "baseline_raw_gfs": baseline,
                        "corrected_mos": corrected,
                    },
                    "source_best_params_path": str(src_best_params),
                },
            )

            results.append(
                {
                    "station_name": station_name,
                    "rows_used": int(len(df)),
                    "baseline_rmse": baseline["rmse"],
                    "corrected_rmse": corrected["rmse"],
                    "output_dir": str(station_out),
                }
            )
            logger.info("[city_extended/%s] wrote model: %s", station_name, model_path)
        except Exception as exc:
            failures.append({"station_name": station_name, "error": str(exc)})
            logger.exception("[city_extended/%s] retrain failed: %s", station_name, exc)
            if fail_fast:
                raise

    summary = {
        "profile": spec.name,
        "retrain_mode": "full_data_no_early_stopping",
        "n_stations_requested": int(len(station_names)),
        "n_success": int(len(results)),
        "n_failed": int(len(failures)),
        "results": results,
        "failures": failures,
    }
    write_json(spec.output_path / "run_summary.json", summary)

    if failures:
        raise RuntimeError(
            f"city_extended retrain had {len(failures)} failed stations; see run_summary.json"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root).expanduser().resolve()
    stations_csv = Path(args.stations_csv).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    log_path = Path(args.log_file).expanduser().resolve()

    logger = setup_logging(log_path)
    profile_specs = build_profile_specs(models_root)
    selected_profiles = list(args.profiles)
    logger.info("Selected profiles: %s", ", ".join(selected_profiles))
    logger.info("data_root=%s", data_root)
    logger.info("models_root=%s", models_root)
    logger.info("stations_csv=%s", stations_csv)

    overall_results: list[dict[str, Any]] = []
    overall_failures: list[dict[str, str]] = []

    for profile_name in selected_profiles:
        spec = profile_specs[profile_name]
        try:
            if spec.kind == "global":
                result = retrain_global_profile(
                    spec=spec,
                    data_root=data_root,
                    logger=logger,
                )
            elif spec.kind == "per_station":
                result = retrain_city_extended_profile(
                    spec=spec,
                    data_root=data_root,
                    stations_csv=stations_csv,
                    fail_fast=bool(args.fail_fast),
                    logger=logger,
                )
            else:
                raise ValueError(f"Unsupported profile kind: {spec.kind}")
            overall_results.append(result)
        except Exception as exc:
            overall_failures.append({"profile": profile_name, "error": str(exc)})
            logger.exception("[%s] profile retrain failed: %s", profile_name, exc)
            if args.fail_fast:
                break

    summary_payload = {
        "selected_profiles": selected_profiles,
        "results": overall_results,
        "failures": overall_failures,
        "retrain_mode": "full_data_no_early_stopping",
    }
    summary_path = models_root / "retrain_tmax_mos_100_summary.json"
    write_json(summary_path, summary_payload)
    logger.info("Wrote overall summary: %s", summary_path)

    if overall_failures:
        logger.error(
            "Completed with failures: %d profile(s) failed.",
            len(overall_failures),
        )
        return 1

    logger.info("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
