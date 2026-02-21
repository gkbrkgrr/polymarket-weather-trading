from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "tmax_stageA",
    "tmax_pred",
    "bias_ewma",
    "lead_hours",
    "cycle",
    "doy_sin",
    "doy_cos",
]

BASE_CATEGORICAL_FEATURES = ["station_name", "city"]


def _import_ml_dependencies() -> dict[str, Any]:
    try:
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "Stage-B residual model requires: xgboost, scikit-learn, joblib. "
            "Install them before running non-dry mode."
        ) from exc

    return {
        "joblib": joblib,
        "ColumnTransformer": ColumnTransformer,
        "OneHotEncoder": OneHotEncoder,
        "XGBRegressor": XGBRegressor,
    }


def _build_ohe(OneHotEncoder):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    err = pred - actual
    return {
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def _choose_validation_cutoff(dates: pd.Series) -> pd.Timestamp:
    unique_dates = np.sort(pd.to_datetime(dates, errors="coerce").dropna().unique())
    if unique_dates.size < 2:
        raise ValueError("Need at least 2 distinct target_date values for Stage-B split")

    fixed_cutoff = pd.Timestamp("2025-11-01")
    if pd.Timestamp(unique_dates.min()) <= fixed_cutoff <= pd.Timestamp(unique_dates.max()):
        return fixed_cutoff

    idx = max(1, int(round(unique_dates.size * 0.83)))
    idx = min(idx, unique_dates.size - 1)
    return pd.Timestamp(unique_dates[idx])


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    target_dt = pd.to_datetime(out["target_date"], errors="coerce")
    doy = target_dt.dt.dayofyear.astype(float)
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    return out


def prepare_stage_b_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_calendar_features(df)
    for col in ("model_name", "station_name", "city"):
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


@dataclass(slots=True)
class ResidualModelBundle:
    include_model_feature: bool
    categorical_features: list[str]
    numeric_features: list[str]
    preprocessor: Any
    model: Any
    validation_metrics: dict[str, Any]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.categorical_features + self.numeric_features].copy()
        transformed = self.preprocessor.transform(x)
        return np.asarray(self.model.predict(transformed), dtype=float)


@dataclass(slots=True)
class ResidualModelManager:
    strategy: str
    bundles: dict[str, ResidualModelBundle]

    def predict(self, df: pd.DataFrame, logger=None) -> np.ndarray:
        if df.empty:
            return np.array([], dtype=float)

        pred = np.zeros(len(df), dtype=float)
        if self.strategy == "single_residual_with_model_feature":
            bundle = self.bundles["__all__"]
            pred[:] = bundle.predict(df)
            return pred

        for model_name, idx in df.groupby("model_name", sort=False).groups.items():
            bundle = self.bundles.get(model_name)
            if bundle is None:
                if logger is not None:
                    logger.warning(
                        "No Stage-B model for model_name=%s; residual correction defaults to 0 for %d rows",
                        model_name,
                        len(idx),
                    )
                pred[list(idx)] = 0.0
                continue
            pred[list(idx)] = bundle.predict(df.loc[idx])
        return pred

    def save(self, output_path: Path) -> None:
        deps = _import_ml_dependencies()
        deps["joblib"].dump(self, output_path)


def _fit_single_bundle(
    *,
    df_train: pd.DataFrame,
    include_model_feature: bool,
    n_jobs: int,
) -> ResidualModelBundle:
    deps = _import_ml_dependencies()
    ColumnTransformer = deps["ColumnTransformer"]
    OneHotEncoder = deps["OneHotEncoder"]
    XGBRegressor = deps["XGBRegressor"]

    categorical = list(BASE_CATEGORICAL_FEATURES)
    # Single-model strategy is preferred for robustness with sparse station/lead history.
    # model_name as a categorical feature lets one residual learner share strength across models.
    if include_model_feature:
        categorical.append("model_name")

    usable = _add_calendar_features(df_train)
    for col in categorical:
        usable[col] = usable[col].astype("string")

    required_cols = categorical + NUMERIC_FEATURES + ["r2_target", "target_date", "tmax_obs", "tmax_stageA"]
    usable = usable.dropna(subset=required_cols).copy()
    if usable.empty:
        raise ValueError("No rows remain for Stage-B after dropping missing feature/target columns")

    cutoff = _choose_validation_cutoff(usable["target_date"])
    train_mask = pd.to_datetime(usable["target_date"]).dt.normalize() < cutoff
    valid_mask = ~train_mask

    fit_df = usable.loc[train_mask].copy()
    valid_df = usable.loc[valid_mask].copy()
    if fit_df.empty or valid_df.empty:
        raise ValueError(
            f"Stage-B train/valid split failed with cutoff={cutoff.date()} "
            f"train_rows={len(fit_df)} valid_rows={len(valid_df)}"
        )

    features = categorical + NUMERIC_FEATURES
    x_train = fit_df[features]
    y_train = fit_df["r2_target"].to_numpy(dtype=float)
    x_valid = valid_df[features]
    y_valid = valid_df["r2_target"].to_numpy(dtype=float)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", _build_ohe(OneHotEncoder), categorical),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    x_train_t = preprocessor.fit_transform(x_train)
    x_valid_t = preprocessor.transform(x_valid)

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 8,
        "reg_lambda": 2.0,
        "n_estimators": 2000,
        "n_jobs": int(n_jobs),
        "random_state": 42,
    }

    fit_sig = inspect.signature(XGBRegressor.fit).parameters
    init_sig = inspect.signature(XGBRegressor.__init__).parameters
    init_supports_es = "early_stopping_rounds" in init_sig
    fit_supports_es = "early_stopping_rounds" in fit_sig

    if init_supports_es:
        xgb_params["early_stopping_rounds"] = 100

    model = XGBRegressor(**xgb_params)
    fit_kwargs: dict[str, Any] = {
        "X": x_train_t,
        "y": y_train,
        "eval_set": [(x_valid_t, y_valid)],
        "verbose": False,
    }
    if fit_supports_es and not init_supports_es:
        fit_kwargs["early_stopping_rounds"] = 100
    model.fit(**fit_kwargs)

    r2_hat_valid = np.asarray(model.predict(x_valid_t), dtype=float)
    corrected_valid = valid_df["tmax_stageA"].to_numpy(dtype=float) + r2_hat_valid
    obs_valid = valid_df["tmax_obs"].to_numpy(dtype=float)

    metrics = {
        "split": {
            "cutoff_date": cutoff.date().isoformat(),
            "train_rows": int(len(fit_df)),
            "valid_rows": int(len(valid_df)),
        },
        "stage_a_baseline": _metrics(
            pred=valid_df["tmax_stageA"].to_numpy(dtype=float),
            actual=obs_valid,
        ),
        "stage_b_corrected": _metrics(pred=corrected_valid, actual=obs_valid),
    }

    return ResidualModelBundle(
        include_model_feature=include_model_feature,
        categorical_features=categorical,
        numeric_features=list(NUMERIC_FEATURES),
        preprocessor=preprocessor,
        model=model,
        validation_metrics=metrics,
    )


def train_residual_models(
    *,
    train_df: pd.DataFrame,
    strategy: str,
    n_jobs: int,
    logger,
) -> tuple[ResidualModelManager, dict[str, Any]]:
    if train_df.empty:
        raise ValueError("Training dataframe is empty for Stage-B")

    metrics: dict[str, Any] = {
        "strategy": strategy,
        "models": {},
    }

    if strategy == "single_residual_with_model_feature":
        bundle = _fit_single_bundle(
            df_train=train_df,
            include_model_feature=True,
            n_jobs=n_jobs,
        )
        manager = ResidualModelManager(strategy=strategy, bundles={"__all__": bundle})
        metrics["models"]["__all__"] = bundle.validation_metrics
        logger.info(
            "Stage-B trained (single model) valid RMSE %.4f -> %.4f",
            bundle.validation_metrics["stage_a_baseline"]["rmse"],
            bundle.validation_metrics["stage_b_corrected"]["rmse"],
        )
        return manager, metrics

    bundles: dict[str, ResidualModelBundle] = {}
    for model_name in sorted(train_df["model_name"].astype("string").dropna().unique()):
        subset = train_df[train_df["model_name"] == model_name].copy()
        if len(subset) < 50:
            logger.warning(
                "Skipping Stage-B model_name=%s due to insufficient rows (%d)",
                model_name,
                len(subset),
            )
            continue
        bundle = _fit_single_bundle(
            df_train=subset,
            include_model_feature=False,
            n_jobs=n_jobs,
        )
        bundles[str(model_name)] = bundle
        metrics["models"][str(model_name)] = bundle.validation_metrics
        logger.info(
            "Stage-B trained model_name=%s valid RMSE %.4f -> %.4f",
            model_name,
            bundle.validation_metrics["stage_a_baseline"]["rmse"],
            bundle.validation_metrics["stage_b_corrected"]["rmse"],
        )

    if not bundles:
        raise ValueError("No per-model Stage-B models were trained")

    manager = ResidualModelManager(strategy=strategy, bundles=bundles)
    return manager, metrics


def save_stage_b_artifacts(
    *,
    manager: ResidualModelManager,
    metrics: dict[str, Any],
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "stage_b_residual_model.joblib"
    metrics_path = artifact_dir / "stage_b_metrics.json"

    manager.save(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
