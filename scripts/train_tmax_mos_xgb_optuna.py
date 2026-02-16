#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
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
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "tmax_mos_xgb_optuna_v1"
DEFAULT_LOG_PATH = REPO_ROOT / "train_optuna.log"

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
TARGET_COLUMN = "mos_target_delta_c"
REQUIRED_COLUMNS = sorted(set(FEATURE_COLUMNS + [OBS_COLUMN, DATE_COLUMN]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna tuning + final XGBoost MOS training for daily Tmax."
    )
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--valid_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--optuna_jobs", type=int, default=8)
    parser.add_argument("--xgb_threads_per_trial", type=int, default=4)
    parser.add_argument("--n_estimators", type=int, default=20000)
    parser.add_argument("--timeout_minutes", type=int, default=0)
    parser.add_argument("--study_name", default="tmax_mos_xgb_optuna_v1")
    return parser.parse_args(argv)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_tmax_mos_xgb_optuna")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
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


def build_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("city", build_ohe(), [CITY_COLUMN]),
            ("num", "passthrough", NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = pred - actual
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    return {"rmse": rmse, "mae": mae, "bias": bias}


def collect_parquet_files(data_root: Path) -> list[Path]:
    files = sorted(data_root.glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under: {data_root}")
    return files


def load_dataset(data_root: Path, logger: logging.Logger) -> pd.DataFrame:
    parquet_files = collect_parquet_files(data_root)
    logger.info("Found %d parquet files under %s", len(parquet_files), data_root)

    dataset = ds.dataset(str(data_root), format="parquet")
    available_columns = set(dataset.schema.names)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - available_columns)
    if missing_columns:
        raise SystemExit(
            "Missing required columns in parquet schema: " + ", ".join(missing_columns)
        )

    table = dataset.to_table(columns=REQUIRED_COLUMNS)
    df = table.to_pandas()
    logger.info("Loaded %d rows from parquet dataset", len(df))
    return df


def split_by_unique_dates(
    df: pd.DataFrame, valid_frac: float, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, int]:
    unique_dates = np.sort(df[DATE_COLUMN].dropna().unique())
    n_dates = int(unique_dates.size)
    if n_dates < 2:
        raise SystemExit("Need at least 2 unique target dates for train/validation split.")

    cutoff_index = int(math.floor((1.0 - valid_frac) * n_dates))
    cutoff_index = min(max(cutoff_index, 0), n_dates - 1)
    cutoff_date = pd.to_datetime(unique_dates[cutoff_index]).normalize()

    train_df = df.loc[df[DATE_COLUMN] < cutoff_date].copy()
    valid_df = df.loc[df[DATE_COLUMN] >= cutoff_date].copy()
    if train_df.empty or valid_df.empty:
        raise SystemExit(
            f"Train/valid split failed with cutoff_date={cutoff_date.date()}: "
            f"train_rows={len(train_df)}, valid_rows={len(valid_df)}"
        )

    logger.info(
        "Split by target_date_local: unique_dates=%d cutoff_index=%d cutoff_date=%s "
        "train_rows=%d valid_rows=%d",
        n_dates,
        cutoff_index,
        cutoff_date.date().isoformat(),
        len(train_df),
        len(valid_df),
    )
    return train_df, valid_df, cutoff_date, n_dates


def normalize_and_clean(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce").dt.normalize()
    df[CITY_COLUMN] = df[CITY_COLUMN].astype("string")
    for col in NUMERIC_COLUMNS + [OBS_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    rows_before_drop = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    dropped_rows = rows_before_drop - len(df)
    logger.info(
        "Dropped %d rows with missing required predictor/target/date fields",
        dropped_rows,
    )
    if df.empty:
        raise SystemExit("No rows remaining after cleaning.")

    df[TARGET_COLUMN] = df[OBS_COLUMN] - df[RAW_COLUMN]
    df = df.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    logger.info("Rows remaining after cleaning: %d", len(df))
    return df, dropped_rows


def xgb_early_stopping_support() -> tuple[bool, bool]:
    fit_params = inspect.signature(XGBRegressor.fit).parameters
    init_params = inspect.signature(XGBRegressor.__init__).parameters
    fit_uses_early_stopping = "early_stopping_rounds" in fit_params
    init_uses_early_stopping = "early_stopping_rounds" in init_params
    return fit_uses_early_stopping, init_uses_early_stopping


def sample_xgb_params(
    trial: optuna.Trial,
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


def fit_pipeline_with_eval(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    xgb_params: dict[str, Any],
    early_stopping_rounds: int,
) -> Pipeline:
    preprocessor = build_preprocessor()
    model = XGBRegressor(**xgb_params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    fit_uses_early_stopping, init_uses_early_stopping = xgb_early_stopping_support()
    if init_uses_early_stopping:
        pipeline.named_steps["model"].set_params(
            early_stopping_rounds=int(early_stopping_rounds)
        )

    preprocessor_for_eval = clone(preprocessor)
    preprocessor_for_eval.fit(X_train)
    X_valid_eval = preprocessor_for_eval.transform(X_valid)

    fit_kwargs: dict[str, Any] = {
        "model__eval_set": [(X_valid_eval, y_valid)],
        "model__verbose": False,
    }
    if fit_uses_early_stopping and not init_uses_early_stopping:
        fit_kwargs["model__early_stopping_rounds"] = int(early_stopping_rounds)

    pipeline.fit(X_train, y_train, **fit_kwargs)
    return pipeline


class OptunaObjective:
    def __init__(
        self,
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
            trial=trial,
            seed=self.seed,
            n_estimators=self.n_estimators,
            xgb_threads_per_trial=self.xgb_threads_per_trial,
        )
        pipeline = fit_pipeline_with_eval(
            X_train=self.X_train,
            y_train=self.y_train,
            X_valid=self.X_valid,
            y_valid=self.y_valid,
            xgb_params=xgb_params,
            early_stopping_rounds=early_stopping_rounds,
        )
        y_pred_valid = pipeline.predict(self.X_valid)
        corrected_pred = self.raw_valid + y_pred_valid
        rmse = compute_metrics(pred=corrected_pred, actual=self.obs_valid)["rmse"]

        model = pipeline.named_steps["model"]
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            trial.set_user_attr("best_iteration", int(best_iteration))
        trial.report(rmse, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return rmse


def compute_per_city_metrics(valid_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for city_name, city_df in valid_df.groupby(CITY_COLUMN, sort=True):
        baseline = compute_metrics(
            pred=city_df[RAW_COLUMN].to_numpy(),
            actual=city_df[OBS_COLUMN].to_numpy(),
        )
        corrected = compute_metrics(
            pred=city_df["tmax_pred_c"].to_numpy(),
            actual=city_df[OBS_COLUMN].to_numpy(),
        )
        records.append(
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
    records.sort(key=lambda row: row["city_name"])
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (0.0 < args.valid_frac < 1.0):
        raise SystemExit("--valid_frac must be in (0, 1).")
    if args.n_trials < 1:
        raise SystemExit("--n_trials must be >= 1.")
    if args.optuna_jobs < 1:
        raise SystemExit("--optuna_jobs must be >= 1.")
    if args.xgb_threads_per_trial < 1:
        raise SystemExit("--xgb_threads_per_trial must be >= 1.")
    if args.n_estimators < 1:
        raise SystemExit("--n_estimators must be >= 1.")
    if args.timeout_minutes < 0:
        raise SystemExit("--timeout_minutes must be >= 0.")

    logger = setup_logging(DEFAULT_LOG_PATH)
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Optuna MOS training")
    logger.info("data_root=%s out_dir=%s", data_root, out_dir)
    logger.info(
        "valid_frac=%.3f n_trials=%d optuna_jobs=%d xgb_threads_per_trial=%d n_estimators=%d",
        args.valid_frac,
        args.n_trials,
        args.optuna_jobs,
        args.xgb_threads_per_trial,
        args.n_estimators,
    )

    df_raw = load_dataset(data_root=data_root, logger=logger)
    rows_loaded = int(len(df_raw))
    df, rows_dropped_missing = normalize_and_clean(df_raw, logger=logger)
    train_df, valid_df, cutoff_date, n_unique_dates = split_by_unique_dates(
        df=df,
        valid_frac=float(args.valid_frac),
        logger=logger,
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET_COLUMN].to_numpy()
    raw_valid = valid_df[RAW_COLUMN].to_numpy()
    obs_valid = valid_df[OBS_COLUMN].to_numpy()

    baseline_metrics = compute_metrics(pred=raw_valid, actual=obs_valid)
    logger.info(
        "Baseline validation metrics (raw forecast): RMSE=%.6f MAE=%.6f Bias=%.6f",
        baseline_metrics["rmse"],
        baseline_metrics["mae"],
        baseline_metrics["bias"],
    )

    study_db_path = out_dir / "optuna_study.db"
    storage = f"sqlite:///{study_db_path.as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0, interval_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    logger.info(
        "Optuna study ready: name=%s storage=%s existing_trials=%d",
        args.study_name,
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
    timeout_seconds = None if args.timeout_minutes == 0 else int(args.timeout_minutes * 60)

    logger.info("Starting Optuna optimization")
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=timeout_seconds,
        n_jobs=int(args.optuna_jobs),
        gc_after_trial=True,
        show_progress_bar=False,
    )

    trials_csv_path = out_dir / "optuna_trials.csv"
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv_path, index=False)
    logger.info("Saved trial history: %s", trials_csv_path)

    best_trial = study.best_trial
    logger.info(
        "Best trial: number=%d corrected_rmse=%.6f params=%s",
        best_trial.number,
        float(best_trial.value),
        json.dumps(best_trial.params, sort_keys=True),
    )

    best_xgb_params, best_early_stopping_rounds = params_from_best_trial(
        best_params=best_trial.params,
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        xgb_threads_per_trial=int(args.xgb_threads_per_trial),
    )

    logger.info("Training final model with best params")
    final_pipeline = fit_pipeline_with_eval(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        xgb_params=best_xgb_params,
        early_stopping_rounds=best_early_stopping_rounds,
    )
    final_y_pred_valid = final_pipeline.predict(X_valid)
    valid_eval_df = valid_df.copy()
    valid_eval_df["tmax_pred_c"] = valid_eval_df[RAW_COLUMN].to_numpy() + final_y_pred_valid

    corrected_metrics = compute_metrics(
        pred=valid_eval_df["tmax_pred_c"].to_numpy(),
        actual=valid_eval_df[OBS_COLUMN].to_numpy(),
    )
    logger.info(
        "Final corrected validation metrics: RMSE=%.6f MAE=%.6f Bias=%.6f",
        corrected_metrics["rmse"],
        corrected_metrics["mae"],
        corrected_metrics["bias"],
    )

    per_city_records = compute_per_city_metrics(valid_eval_df)
    per_city_df = pd.DataFrame(per_city_records)
    if per_city_df.empty:
        logger.info("Per-city metrics: no validation rows")
    else:
        logger.info("Per-city metrics:\n%s", per_city_df.to_string(index=False))

    model_path = out_dir / "model.joblib"
    best_params_path = out_dir / "best_params.json"
    metrics_path = out_dir / "metrics.json"
    joblib.dump(final_pipeline, model_path)

    best_params_payload: dict[str, Any] = {
        "study_name": args.study_name,
        "best_trial_number": int(best_trial.number),
        "best_corrected_rmse": float(best_trial.value),
        "search_params": best_trial.params,
        "xgb_params_final": best_xgb_params,
        "early_stopping_rounds": int(best_early_stopping_rounds),
        "n_estimators": int(args.n_estimators),
    }
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(best_params_payload, f, indent=2, sort_keys=True)

    best_iteration = getattr(final_pipeline.named_steps["model"], "best_iteration", None)
    metrics_payload: dict[str, Any] = {
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
        "per_city_metrics_valid": per_city_records,
        "optuna": {
            "study_name": args.study_name,
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

    logger.info("Saved model pipeline: %s", model_path)
    logger.info("Saved best params: %s", best_params_path)
    logger.info("Saved metrics: %s", metrics_path)
    logger.info("Saved Optuna study DB: %s", study_db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
