#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import re
import sys
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
    import optuna
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'optuna'. Install with: pip install optuna"
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
DEFAULT_LOG_DIR = REPO_ROOT / "logs"

FEATURE_SETS: dict[str, list[str]] = {
    "core": [
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
    ],
    "extended": [
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
}

DATE_COLUMN = "target_date_local"
OBS_COLUMN = "tmax_obs_c"
RAW_COLUMN = "tmax_raw_c"
TARGET_COLUMN = "mos_target_delta_c"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train per-station Optuna-tuned XGBoost MOS models."
    )
    parser.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SETS))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--stations-csv", default=str(DEFAULT_STATIONS_CSV))
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--optuna-jobs", type=int, default=8)
    parser.add_argument("--xgb-threads-per-trial", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=20000)
    parser.add_argument("--timeout-minutes", type=int, default=0)
    parser.add_argument("--study-name-prefix", default="tmax_mos_xgb_optuna_per_station")
    parser.add_argument("--log-file", default="")
    return parser.parse_args(argv)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_tmax_mos_xgb_optuna_per_station")
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


def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    err = pred - actual
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return {"rmse": rmse, "mae": mae, "bias": bias}


def sanitize_for_study_name(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "station"


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
            f"Unsupported station schema in {stations_csv}. "
            "Expected 'name' or 'city_name' column."
        )

    names: list[str] = []
    seen: set[str] = set()
    for raw in df[name_col].tolist():
        name = str(raw).strip()
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)

    if not names:
        raise ValueError(f"No station names found in {stations_csv}")
    return names


def collect_station_parquet_files(data_root: Path, station_name: str) -> list[Path]:
    station_dir = data_root / station_name
    if not station_dir.exists():
        raise FileNotFoundError(f"Station directory not found: {station_dir}")
    files = sorted(station_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for station {station_name} under {station_dir}")
    return files


def load_station_dataset(
    station_dir: Path,
    required_columns: list[str],
) -> pd.DataFrame:
    dataset = ds.dataset(str(station_dir), format="parquet")
    available_columns = set(dataset.schema.names)
    missing_columns = sorted(set(required_columns) - available_columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in station parquet schema ({station_dir}): "
            + ", ".join(missing_columns)
        )
    table = dataset.to_table(columns=required_columns)
    return table.to_pandas()


def split_by_unique_dates(df: pd.DataFrame, valid_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, int]:
    unique_dates = np.sort(df[DATE_COLUMN].dropna().unique())
    n_dates = int(unique_dates.size)
    if n_dates < 2:
        raise ValueError("Need at least 2 unique target dates for train/validation split.")

    cutoff_index = int(math.floor((1.0 - valid_frac) * n_dates))
    cutoff_index = min(max(cutoff_index, 0), n_dates - 1)
    cutoff_date = pd.to_datetime(unique_dates[cutoff_index]).normalize()

    train_df = df.loc[df[DATE_COLUMN] < cutoff_date].copy()
    valid_df = df.loc[df[DATE_COLUMN] >= cutoff_date].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(
            f"Train/valid split failed with cutoff_date={cutoff_date.date()}: "
            f"train_rows={len(train_df)}, valid_rows={len(valid_df)}"
        )
    return train_df, valid_df, cutoff_date, n_dates


def normalize_and_clean(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    required_columns = feature_columns + [OBS_COLUMN, DATE_COLUMN]

    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    for col in feature_columns + [OBS_COLUMN]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    rows_before_drop = len(out)
    out = out.dropna(subset=required_columns).copy()
    dropped_rows = rows_before_drop - len(out)
    if out.empty:
        raise ValueError("No rows remaining after dropping missing required fields.")

    out[TARGET_COLUMN] = out[OBS_COLUMN] - out[RAW_COLUMN]
    out = out.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    return out, dropped_rows


def xgb_early_stopping_support() -> tuple[bool, bool]:
    fit_params = inspect.signature(XGBRegressor.fit).parameters
    init_params = inspect.signature(XGBRegressor.__init__).parameters
    fit_uses_early_stopping = "early_stopping_rounds" in fit_params
    init_uses_early_stopping = "early_stopping_rounds" in init_params
    return fit_uses_early_stopping, init_uses_early_stopping


def sample_xgb_params(
    trial: optuna.Trial,
    *,
    seed: int,
    n_estimators: int,
    xgb_threads_per_trial: int,
) -> tuple[dict[str, Any], int]:
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 200, 800)

    params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": int(n_estimators),
        "random_state": int(seed),
        "n_jobs": int(xgb_threads_per_trial),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "grow_policy": grow_policy,
    }
    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 0, 512)
    else:
        params["max_leaves"] = 0
    return params, early_stopping_rounds


def params_from_best_trial(
    best_params: dict[str, Any],
    *,
    seed: int,
    n_estimators: int,
    xgb_threads_per_trial: int,
) -> tuple[dict[str, Any], int]:
    early_stopping_rounds = int(best_params["early_stopping_rounds"])
    grow_policy = str(best_params["grow_policy"])

    params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": int(n_estimators),
        "random_state": int(seed),
        "n_jobs": int(xgb_threads_per_trial),
        "max_depth": int(best_params["max_depth"]),
        "min_child_weight": float(best_params["min_child_weight"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "gamma": float(best_params["gamma"]),
        "reg_lambda": float(best_params["reg_lambda"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "learning_rate": float(best_params["learning_rate"]),
        "max_delta_step": int(best_params["max_delta_step"]),
        "grow_policy": grow_policy,
    }
    if grow_policy == "lossguide":
        params["max_leaves"] = int(best_params.get("max_leaves", 0))
    else:
        params["max_leaves"] = 0
    return params, early_stopping_rounds


def fit_model_with_eval(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    xgb_params: dict[str, Any],
    early_stopping_rounds: int,
) -> XGBRegressor:
    model = XGBRegressor(**xgb_params)
    fit_uses_early_stopping, init_uses_early_stopping = xgb_early_stopping_support()

    if init_uses_early_stopping:
        model.set_params(early_stopping_rounds=int(early_stopping_rounds))

    fit_kwargs: dict[str, Any] = {
        "eval_set": [(X_valid, y_valid)],
        "verbose": False,
    }
    if fit_uses_early_stopping and not init_uses_early_stopping:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    model.fit(X_train, y_train, **fit_kwargs)
    return model


class OptunaObjective:
    def __init__(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        raw_valid: np.ndarray,
        obs_valid: np.ndarray,
        seed: int,
        n_estimators: int,
        xgb_threads_per_trial: int,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.raw_valid = raw_valid
        self.obs_valid = obs_valid
        self.seed = seed
        self.n_estimators = n_estimators
        self.xgb_threads_per_trial = xgb_threads_per_trial

    def __call__(self, trial: optuna.Trial) -> float:
        xgb_params, early_stopping_rounds = sample_xgb_params(
            trial,
            seed=self.seed,
            n_estimators=self.n_estimators,
            xgb_threads_per_trial=self.xgb_threads_per_trial,
        )
        model = fit_model_with_eval(
            self.X_train,
            self.y_train,
            self.X_valid,
            self.y_valid,
            xgb_params=xgb_params,
            early_stopping_rounds=early_stopping_rounds,
        )
        y_pred_valid = model.predict(self.X_valid)
        corrected_pred = self.raw_valid + y_pred_valid
        rmse = compute_metrics(pred=corrected_pred, actual=self.obs_valid)["rmse"]

        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            trial.set_user_attr("best_iteration", int(best_iteration))
        trial.report(rmse, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return rmse


def train_single_station(
    *,
    station_name: str,
    station_data_dir: Path,
    station_out_dir: Path,
    feature_set_name: str,
    feature_columns: list[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, Any]:
    required_columns = feature_columns + [OBS_COLUMN, DATE_COLUMN]
    raw_df = load_station_dataset(station_data_dir, required_columns)
    rows_loaded = int(len(raw_df))
    df, rows_dropped_missing = normalize_and_clean(raw_df, feature_columns=feature_columns)

    train_df, valid_df, cutoff_date, n_unique_dates = split_by_unique_dates(
        df=df,
        valid_frac=float(args.valid_frac),
    )

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN].to_numpy()
    raw_valid = valid_df[RAW_COLUMN].to_numpy()
    obs_valid = valid_df[OBS_COLUMN].to_numpy()

    baseline_metrics = compute_metrics(pred=raw_valid, actual=obs_valid)
    logger.info(
        "[%s] Baseline validation metrics (raw forecast): RMSE=%.6f MAE=%.6f Bias=%.6f",
        station_name,
        baseline_metrics["rmse"],
        baseline_metrics["mae"],
        baseline_metrics["bias"],
    )

    station_out_dir.mkdir(parents=True, exist_ok=True)
    study_db_path = station_out_dir / "optuna_study.db"
    storage = f"sqlite:///{study_db_path.as_posix()}"
    station_token = sanitize_for_study_name(station_name)
    study_name = f"{args.study_name_prefix}_{feature_set_name}_{station_token}"

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=20,
        n_warmup_steps=0,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    logger.info(
        "[%s] Optuna study ready: name=%s storage=%s existing_trials=%d",
        station_name,
        study_name,
        storage,
        len(study.trials),
    )

    objective = OptunaObjective(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        raw_valid=raw_valid,
        obs_valid=obs_valid,
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        xgb_threads_per_trial=int(args.xgb_threads_per_trial),
    )
    timeout_seconds = None if int(args.timeout_minutes) == 0 else int(args.timeout_minutes) * 60

    logger.info("[%s] Starting Optuna optimization", station_name)
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=timeout_seconds,
        n_jobs=int(args.optuna_jobs),
        gc_after_trial=True,
        show_progress_bar=False,
    )

    trials_csv_path = station_out_dir / "optuna_trials.csv"
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv_path, index=False)
    logger.info("[%s] Saved trial history: %s", station_name, trials_csv_path)

    best_trial = study.best_trial
    logger.info(
        "[%s] Best trial: number=%d corrected_rmse=%.6f params=%s",
        station_name,
        best_trial.number,
        float(best_trial.value),
        json.dumps(best_trial.params, sort_keys=True),
    )

    best_xgb_params, best_early_stopping_rounds = params_from_best_trial(
        best_trial.params,
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        xgb_threads_per_trial=int(args.xgb_threads_per_trial),
    )

    logger.info("[%s] Training final model with best params", station_name)
    final_model = fit_model_with_eval(
        X_train,
        y_train,
        X_valid,
        y_valid,
        xgb_params=best_xgb_params,
        early_stopping_rounds=best_early_stopping_rounds,
    )
    final_y_pred_valid = final_model.predict(X_valid)
    corrected_pred = raw_valid + final_y_pred_valid
    corrected_metrics = compute_metrics(pred=corrected_pred, actual=obs_valid)
    logger.info(
        "[%s] Final corrected validation metrics: RMSE=%.6f MAE=%.6f Bias=%.6f",
        station_name,
        corrected_metrics["rmse"],
        corrected_metrics["mae"],
        corrected_metrics["bias"],
    )

    model_path = station_out_dir / "model.joblib"
    best_params_path = station_out_dir / "best_params.json"
    metrics_path = station_out_dir / "metrics.json"
    joblib.dump(final_model, model_path)

    best_params_payload: dict[str, Any] = {
        "station_name": station_name,
        "feature_set": feature_set_name,
        "feature_columns": feature_columns,
        "study_name": study_name,
        "best_trial_number": int(best_trial.number),
        "best_corrected_rmse": float(best_trial.value),
        "search_params": best_trial.params,
        "xgb_params_final": best_xgb_params,
        "early_stopping_rounds": int(best_early_stopping_rounds),
        "n_estimators": int(args.n_estimators),
    }
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(best_params_payload, f, indent=2, sort_keys=True)

    best_iteration = getattr(final_model, "best_iteration", None)
    metrics_payload: dict[str, Any] = {
        "station_name": station_name,
        "feature_set": feature_set_name,
        "feature_columns": feature_columns,
        "split": {
            "valid_frac": float(args.valid_frac),
            "n_unique_dates": int(n_unique_dates),
            "cutoff_date": cutoff_date.date().isoformat(),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
        },
        "rows": {
            "loaded_before_cleaning": int(rows_loaded),
            "dropped_missing_required": int(rows_dropped_missing),
            "remaining_after_cleaning": int(len(df)),
        },
        "overall_metrics_valid": {
            "baseline_raw_gfs": baseline_metrics,
            "corrected_mos": corrected_metrics,
        },
        "optuna": {
            "study_name": study_name,
            "storage": storage,
            "total_trials": int(len(study.trials)),
            "best_trial_number": int(best_trial.number),
            "best_corrected_rmse": float(best_trial.value),
        },
        "training": {
            "seed": int(args.seed),
            "optuna_jobs": int(args.optuna_jobs),
            "xgb_threads_per_trial": int(args.xgb_threads_per_trial),
            "n_trials_requested": int(args.n_trials),
            "timeout_minutes": int(args.timeout_minutes),
            "best_params": best_trial.params,
            "xgb_params_final": best_xgb_params,
            "early_stopping_rounds": int(best_early_stopping_rounds),
            "n_estimators": int(args.n_estimators),
        },
    }
    if best_iteration is not None:
        metrics_payload["training"]["best_iteration"] = int(best_iteration)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True)

    logger.info("[%s] Saved model: %s", station_name, model_path)
    logger.info("[%s] Saved best params: %s", station_name, best_params_path)
    logger.info("[%s] Saved metrics: %s", station_name, metrics_path)
    logger.info("[%s] Saved Optuna study DB: %s", station_name, study_db_path)

    return {
        "station_name": station_name,
        "rows_loaded_before_cleaning": int(rows_loaded),
        "rows_after_cleaning": int(len(df)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "baseline_rmse": baseline_metrics["rmse"],
        "corrected_rmse": corrected_metrics["rmse"],
        "artifact_dir": str(station_out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (0.0 < float(args.valid_frac) < 1.0):
        raise SystemExit("--valid-frac must be in (0, 1).")
    if int(args.n_trials) < 1:
        raise SystemExit("--n-trials must be >= 1.")
    if int(args.optuna_jobs) < 1:
        raise SystemExit("--optuna-jobs must be >= 1.")
    if int(args.xgb_threads_per_trial) < 1:
        raise SystemExit("--xgb-threads-per-trial must be >= 1.")
    if int(args.n_estimators) < 1:
        raise SystemExit("--n-estimators must be >= 1.")
    if int(args.timeout_minutes) < 0:
        raise SystemExit("--timeout-minutes must be >= 0.")

    feature_set_name = str(args.feature_set)
    feature_columns = FEATURE_SETS[feature_set_name]

    data_root = Path(args.data_root).expanduser().resolve()
    stations_csv = Path(args.stations_csv).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    out_dir = models_root / f"tmax_mos_xgb_optuna_{feature_set_name}_per_station"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
    else:
        log_path = DEFAULT_LOG_DIR / f"train_tmax_mos_xgb_optuna_{feature_set_name}_per_station.log"

    logger = setup_logging(log_path)
    station_names = load_station_names(stations_csv)
    logger.info("Starting per-station Optuna MOS training")
    logger.info("feature_set=%s num_features=%d", feature_set_name, len(feature_columns))
    logger.info("data_root=%s", data_root)
    logger.info("stations_csv=%s", stations_csv)
    logger.info("models_root=%s", models_root)
    logger.info("out_dir=%s", out_dir)
    logger.info(
        "valid_frac=%.3f n_trials=%d optuna_jobs=%d xgb_threads_per_trial=%d n_estimators=%d",
        float(args.valid_frac),
        int(args.n_trials),
        int(args.optuna_jobs),
        int(args.xgb_threads_per_trial),
        int(args.n_estimators),
    )
    logger.info("Stations to train (%d): %s", len(station_names), ", ".join(station_names))

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for station_name in station_names:
        try:
            station_files = collect_station_parquet_files(data_root, station_name)
            station_data_dir = (data_root / station_name).resolve()
            station_out_dir = (out_dir / station_name).resolve()
            logger.info(
                "[%s] Found %d parquet files under %s",
                station_name,
                len(station_files),
                station_data_dir,
            )
            station_result = train_single_station(
                station_name=station_name,
                station_data_dir=station_data_dir,
                station_out_dir=station_out_dir,
                feature_set_name=feature_set_name,
                feature_columns=feature_columns,
                args=args,
                logger=logger,
            )
            results.append(station_result)
        except Exception as exc:
            failures.append({"station_name": station_name, "error": str(exc)})
            logger.exception("[%s] Station training failed: %s", station_name, exc)

    summary = {
        "feature_set": feature_set_name,
        "feature_columns": feature_columns,
        "stations_csv": str(stations_csv),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "n_stations_requested": int(len(station_names)),
        "n_success": int(len(results)),
        "n_failed": int(len(failures)),
        "results": results,
        "failures": failures,
    }
    summary_path = out_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info("Saved run summary: %s", summary_path)

    if failures:
        logger.error(
            "Completed with failures: %d/%d stations failed",
            len(failures),
            len(station_names),
        )
        return 1

    logger.info("Completed successfully: %d/%d stations trained", len(results), len(station_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
