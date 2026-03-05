#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg
import pyarrow.dataset as ds
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TRAIN_DATA_ROOT = REPO_ROOT / "data" / "train_data" / "gfs"
DEFAULT_PREDICTIONS_ROOT = REPO_ROOT / "data" / "ml_predictions"
DEFAULT_MODELS_ROOT = REPO_ROOT / "models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "probability_backtest"

LONDON_TZ = "Europe/London"
STATION_DEFAULT = "London"
RESIDUAL_YEARS = (2021, 2022, 2023, 2024, 2025)
CLIMO_START_DATE = pd.Timestamp("2000-01-01")
CLIMO_END_EXCLUSIVE = pd.Timestamp("2026-01-01")
PROB_EPS = 1e-6

CITY_COLUMN = "city_name"
DATE_COLUMN = "target_date_local"
ISSUE_COLUMN = "issue_time_utc"
LEAD_COLUMN = "lead_time_hours"
OBS_COLUMN = "tmax_obs_c"
RAW_COLUMN = "tmax_raw_c"
TARGET_COLUMN = "mos_target_delta_c"
KEY_COLUMNS = [ISSUE_COLUMN, DATE_COLUMN, LEAD_COLUMN]

CALIB_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, math.inf]
CALIB_LABELS = ["0-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20+"]

STRIKE_RE = re.compile(r"-(?P<strike>neg-\d+|\d+)c$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sprint 1 probability backtest for Polymarket London temperature markets.\n"
            "Uses out-of-sample year-holdout predictions for residual CDF fitting."
        )
    )
    parser.add_argument("--train-data-root", type=Path, default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--master-dsn", type=str, default=None)
    parser.add_argument("--station", type=str, default=STATION_DEFAULT)
    parser.add_argument(
        "--residual-years",
        type=int,
        nargs="+",
        default=list(RESIDUAL_YEARS),
        help="Years used for OOF residual fitting (default: 2021 2022 2023 2024 2025).",
    )
    parser.add_argument(
        "--eval-year",
        type=int,
        default=2026,
        help="Evaluation market year (default: 2026).",
    )
    return parser.parse_args(argv)


def build_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_global_preprocessor(numeric_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("city", build_ohe(), [CITY_COLUMN]),
            ("num", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_strike_from_slug(slug: str) -> int | None:
    match = STRIKE_RE.search(str(slug))
    if match is None:
        return None
    token = match.group("strike")
    if token.startswith("neg-"):
        return -int(token.split("-", 1)[1])
    return int(token)


def round_to_market_integer(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.floor(arr + 0.5).astype(int)


def clip_probs(p: np.ndarray | pd.Series, eps: float = PROB_EPS) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = clip_probs(p)
    return float(-(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log1p(-p_arr)).mean())


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    return float(np.mean(np.square(p_arr - y_arr)))


def load_training_data(
    *,
    data_root: Path,
    required_columns: list[str],
) -> pd.DataFrame:
    dataset = ds.dataset(str(data_root), format="parquet")
    available = set(dataset.schema.names)
    missing = sorted(set(required_columns) - available)
    if missing:
        raise SystemExit(
            f"Training dataset is missing required columns: {', '.join(missing)}"
        )
    table = dataset.to_table(columns=required_columns)
    df = table.to_pandas()
    return df


def normalize_training_data(
    df: pd.DataFrame,
    *,
    numeric_columns: list[str],
) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    out[ISSUE_COLUMN] = pd.to_datetime(out[ISSUE_COLUMN], utc=True, errors="coerce")
    out[CITY_COLUMN] = out[CITY_COLUMN].astype("string")
    for col in numeric_columns + [OBS_COLUMN]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out = out.dropna(subset=[CITY_COLUMN, DATE_COLUMN, ISSUE_COLUMN]).copy()
    out = out.sort_values([CITY_COLUMN, DATE_COLUMN, ISSUE_COLUMN, LEAD_COLUMN], kind="mergesort")
    out = out.drop_duplicates(subset=[CITY_COLUMN] + KEY_COLUMNS, keep="last")
    out["year"] = out[DATE_COLUMN].dt.year
    return out


def fit_predict_oof_city_extended(
    *,
    london_df: pd.DataFrame,
    years: list[int],
    feature_columns: list[str],
    xgb_params: dict,
) -> pd.DataFrame:
    required = feature_columns + [OBS_COLUMN, RAW_COLUMN] + KEY_COLUMNS
    work = london_df.dropna(subset=required).copy()
    work[TARGET_COLUMN] = work[OBS_COLUMN] - work[RAW_COLUMN]

    outputs: list[pd.DataFrame] = []
    for holdout_year in years:
        train = work.loc[work["year"] != holdout_year]
        holdout = work.loc[work["year"] == holdout_year]
        if holdout.empty:
            continue
        if train.empty:
            raise SystemExit(f"City model fold year={holdout_year} has empty training data.")

        model = XGBRegressor(**xgb_params)
        model.fit(train[feature_columns], train[TARGET_COLUMN].to_numpy())
        pred_temp = holdout[RAW_COLUMN].to_numpy(dtype=float) + model.predict(holdout[feature_columns])

        part = holdout[KEY_COLUMNS + [OBS_COLUMN]].copy()
        part["pred_city_extended"] = pred_temp
        outputs.append(part)

    if not outputs:
        raise SystemExit("City model OOF prediction set is empty.")
    out = pd.concat(outputs, ignore_index=True)
    out = out.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    return out


def fit_predict_oof_global(
    *,
    all_df: pd.DataFrame,
    station: str,
    years: list[int],
    feature_columns: list[str],
    numeric_columns: list[str],
    xgb_params: dict,
    pred_col: str,
) -> pd.DataFrame:
    required = feature_columns + [OBS_COLUMN, RAW_COLUMN] + KEY_COLUMNS
    work = all_df.dropna(subset=required).copy()
    work[TARGET_COLUMN] = work[OBS_COLUMN] - work[RAW_COLUMN]
    london_holdout_all = work.loc[work[CITY_COLUMN] == station].copy()
    if london_holdout_all.empty:
        raise SystemExit(f"No London rows available for global model holdout station={station!r}.")

    outputs: list[pd.DataFrame] = []
    for holdout_year in years:
        train = work.loc[work["year"] != holdout_year]
        holdout = london_holdout_all.loc[london_holdout_all["year"] == holdout_year]
        if holdout.empty:
            continue
        if train.empty:
            raise SystemExit(f"Global model fold year={holdout_year} has empty training data.")

        preprocessor = build_global_preprocessor(numeric_columns=numeric_columns)
        model = XGBRegressor(**xgb_params)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(train[feature_columns], train[TARGET_COLUMN].to_numpy())
        pred_temp = holdout[RAW_COLUMN].to_numpy(dtype=float) + pipe.predict(holdout[feature_columns])

        part = holdout[KEY_COLUMNS + [OBS_COLUMN]].copy()
        part[pred_col] = pred_temp
        outputs.append(part)

    if not outputs:
        raise SystemExit(f"Global model OOF prediction set is empty for {pred_col}.")
    out = pd.concat(outputs, ignore_index=True)
    out = out.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    return out


def select_latest_cycle_before_local_midnight(
    df: pd.DataFrame,
    *,
    day_column: str,
    issue_column: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    day_start_local = pd.to_datetime(work[day_column]).dt.tz_localize(LONDON_TZ)
    cutoff_utc = day_start_local.dt.tz_convert("UTC")
    work = work.loc[work[issue_column] < cutoff_utc].copy()
    if work.empty:
        return work
    work = work.sort_values([day_column, issue_column], kind="mergesort")
    work = work.drop_duplicates(subset=[day_column], keep="last")
    return work


def build_residual_series_from_oof(
    *,
    city_oof: pd.DataFrame,
    g1_oof: pd.DataFrame,
    g2_oof: pd.DataFrame,
) -> pd.DataFrame:
    merge_cols = KEY_COLUMNS + [OBS_COLUMN]
    merged = city_oof.merge(g1_oof, on=merge_cols, how="inner").merge(g2_oof, on=merge_cols, how="inner")
    if merged.empty:
        raise SystemExit("No overlapping OOF rows across all three models.")

    preds = merged[["pred_city_extended", "pred_xgb_opt_v1_100", "pred_xgb_opt_v2_100"]].to_numpy(dtype=float)
    merged["t_hat_median"] = np.median(preds, axis=1)
    merged["target_date_local"] = pd.to_datetime(merged[DATE_COLUMN], errors="coerce").dt.normalize()
    merged["issue_time_utc"] = pd.to_datetime(merged[ISSUE_COLUMN], utc=True, errors="coerce")
    merged = merged.dropna(subset=[DATE_COLUMN, ISSUE_COLUMN, OBS_COLUMN, "t_hat_median"]).copy()

    selected = select_latest_cycle_before_local_midnight(
        merged,
        day_column=DATE_COLUMN,
        issue_column=ISSUE_COLUMN,
    )
    if selected.empty:
        raise SystemExit("No OOF rows survived decision-cycle filtering.")

    selected["residual"] = selected[OBS_COLUMN] - selected["t_hat_median"]
    selected["t_round_obs"] = round_to_market_integer(selected[OBS_COLUMN])
    return selected


class EmpiricalResidualCdf:
    def __init__(self, residuals: Iterable[float]) -> None:
        arr = np.asarray(list(residuals), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("Residual array is empty.")
        self._sorted = np.sort(arr)
        self._n = float(arr.size)

    @property
    def size(self) -> int:
        return int(self._n)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self._sorted, x, side="right")
        return idx / self._n

    def interval_prob(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        cdf_upper = self.cdf(upper)
        cdf_lower = self.cdf(lower)
        return cdf_upper - cdf_lower


def load_prediction_rows(
    *,
    predictions_root: Path,
    model_name: str,
    station: str,
) -> pd.DataFrame:
    city_dir = predictions_root / model_name / station
    if not city_dir.exists():
        raise SystemExit(f"Prediction directory not found: {city_dir}")
    files = sorted(city_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No prediction parquet files found under: {city_dir}")

    parts: list[pd.DataFrame] = []
    for path in files:
        part = pd.read_parquet(path, columns=[ISSUE_COLUMN, DATE_COLUMN, LEAD_COLUMN, "Forecast"])
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out[ISSUE_COLUMN] = pd.to_datetime(out[ISSUE_COLUMN], utc=True, errors="coerce")
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    out[LEAD_COLUMN] = pd.to_numeric(out[LEAD_COLUMN], errors="coerce")
    out["Forecast"] = pd.to_numeric(out["Forecast"], errors="coerce")
    out = out.dropna(subset=[ISSUE_COLUMN, DATE_COLUMN, LEAD_COLUMN, "Forecast"]).copy()
    out = out.sort_values([DATE_COLUMN, ISSUE_COLUMN, LEAD_COLUMN], kind="mergesort")
    out = out.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    return out


def fetch_dataframe(
    conn: psycopg.Connection,
    query: str,
    params: dict | None = None,
) -> pd.DataFrame:
    with conn.cursor() as cur:
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)
        rows = cur.fetchall()
        columns = [desc.name for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def build_eval_daily_forecast(
    *,
    predictions_root: Path,
    station: str,
) -> pd.DataFrame:
    m1 = load_prediction_rows(
        predictions_root=predictions_root,
        model_name="city_extended",
        station=station,
    ).rename(columns={"Forecast": "pred_city_extended"})
    m2 = load_prediction_rows(
        predictions_root=predictions_root,
        model_name="xgb_opt_v1_100",
        station=station,
    ).rename(columns={"Forecast": "pred_xgb_opt_v1_100"})
    m3 = load_prediction_rows(
        predictions_root=predictions_root,
        model_name="xgb_opt_v2_100",
        station=station,
    ).rename(columns={"Forecast": "pred_xgb_opt_v2_100"})

    m1_sel = select_latest_cycle_before_local_midnight(m1, day_column=DATE_COLUMN, issue_column=ISSUE_COLUMN)
    m2_sel = select_latest_cycle_before_local_midnight(m2, day_column=DATE_COLUMN, issue_column=ISSUE_COLUMN)
    m3_sel = select_latest_cycle_before_local_midnight(m3, day_column=DATE_COLUMN, issue_column=ISSUE_COLUMN)

    daily = (
        m1_sel[[DATE_COLUMN, "pred_city_extended"]]
        .merge(m2_sel[[DATE_COLUMN, "pred_xgb_opt_v1_100"]], on=DATE_COLUMN, how="inner")
        .merge(m3_sel[[DATE_COLUMN, "pred_xgb_opt_v2_100"]], on=DATE_COLUMN, how="inner")
    )
    if daily.empty:
        raise SystemExit("No overlapping daily decision-cycle forecasts across the three models.")

    pred_array = daily[["pred_city_extended", "pred_xgb_opt_v1_100", "pred_xgb_opt_v2_100"]].to_numpy(dtype=float)
    daily["t_hat_median"] = np.median(pred_array, axis=1)
    daily = daily.sort_values(DATE_COLUMN, kind="mergesort")
    return daily


def fetch_daily_tmax_from_db(
    *,
    conn: psycopg.Connection,
    station: str,
    start: str,
    end_exclusive: str,
) -> pd.DataFrame:
    query = """
        SELECT
            observed_at_local::date AS day_local,
            MAX(COALESCE(temperature_c::double precision, ((temperature_f - 32.0) * (5.0 / 9.0)))) AS tmax_c
        FROM station_observations
        WHERE station = %(station)s
          AND observed_at_local >= %(start)s::timestamp
          AND observed_at_local < %(end_exclusive)s::timestamp
        GROUP BY 1
        ORDER BY 1
    """
    df = fetch_dataframe(
        conn,
        query,
        params={"station": station, "start": start, "end_exclusive": end_exclusive},
    )
    df["day_local"] = pd.to_datetime(df["day_local"], errors="coerce").dt.normalize()
    df["tmax_c"] = pd.to_numeric(df["tmax_c"], errors="coerce")
    df = df.dropna(subset=["day_local", "tmax_c"]).copy()
    df["t_round_obs"] = round_to_market_integer(df["tmax_c"])
    return df


def build_climatology_probabilities(daily_tmax: pd.DataFrame) -> pd.Series:
    if daily_tmax.empty:
        raise SystemExit("Climatology daily Tmax series is empty.")
    freq = daily_tmax["t_round_obs"].value_counts(normalize=True).sort_index()
    freq.index = freq.index.astype(int)
    return freq


def fetch_resolved_london_markets(
    *,
    conn: psycopg.Connection,
) -> pd.DataFrame:
    query = """
        SELECT
            market_id,
            slug,
            status,
            raw ->> 'endDate' AS end_date_utc,
            raw ->> 'outcomePrices' AS outcome_prices
        FROM markets
        WHERE status = 'resolved'
          AND slug ILIKE 'highest-temperature-in-london-on-%'
        ORDER BY market_id::bigint ASC
    """
    df = fetch_dataframe(conn, query)
    if df.empty:
        raise SystemExit("No resolved London markets found.")

    df["strike"] = df["slug"].map(parse_strike_from_slug)
    df = df.dropna(subset=["strike"]).copy()
    df["strike"] = df["strike"].astype(int)

    df["end_date_utc"] = pd.to_datetime(df["end_date_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["end_date_utc"]).copy()
    df["market_day_local"] = (
        df["end_date_utc"]
        .dt.tz_convert(LONDON_TZ)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    def parse_yes_prob(text: str) -> float | None:
        try:
            values = json.loads(text)
            if not isinstance(values, list) or len(values) < 1:
                return None
            return float(values[0])
        except (TypeError, json.JSONDecodeError, ValueError):
            return None

    df["yes_prob_resolved"] = df["outcome_prices"].map(parse_yes_prob)
    df = df.dropna(subset=["yes_prob_resolved"]).copy()
    df["y_market"] = (df["yes_prob_resolved"] >= 0.5).astype(int)
    return df


def attach_market_outcomes_from_observations(
    *,
    markets: pd.DataFrame,
    eval_daily_obs: pd.DataFrame,
) -> pd.DataFrame:
    obs = eval_daily_obs[["day_local", "t_round_obs"]].rename(columns={"day_local": "market_day_local"})
    merged = markets.merge(obs, on="market_day_local", how="inner")
    if merged.empty:
        raise SystemExit("No market rows matched observed daily Tmax dates.")
    merged["y"] = (merged["t_round_obs"] == merged["strike"]).astype(int)
    return merged


def compute_probabilities(
    *,
    df: pd.DataFrame,
    cdf: EmpiricalResidualCdf,
    climo_probs: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    lower = out["strike"].to_numpy(dtype=float) - 0.5 - out["t_hat_median"].to_numpy(dtype=float)
    upper = out["strike"].to_numpy(dtype=float) + 0.5 - out["t_hat_median"].to_numpy(dtype=float)
    out["p_model_residual"] = cdf.interval_prob(lower, upper)
    out["p_climo"] = out["strike"].map(climo_probs).fillna(0.0).astype(float)
    return out


def compute_metric_table(
    *,
    df: pd.DataFrame,
    p_col_model: str,
    p_col_climo: str,
) -> pd.DataFrame:
    y = df["y"].to_numpy(dtype=float)
    table = pd.DataFrame(
        [
            {
                "model": "model_residual",
                "log_loss": binary_log_loss(y, df[p_col_model].to_numpy(dtype=float)),
                "brier_score": brier_score(y, df[p_col_model].to_numpy(dtype=float)),
            },
            {
                "model": "climatology",
                "log_loss": binary_log_loss(y, df[p_col_climo].to_numpy(dtype=float)),
                "brier_score": brier_score(y, df[p_col_climo].to_numpy(dtype=float)),
            },
        ]
    )
    return table


def select_top_n_per_day(
    *,
    df: pd.DataFrame,
    n: int,
    prob_col: str,
) -> pd.DataFrame:
    ranked = df.sort_values(["market_day_local", prob_col, "strike"], ascending=[True, False, True], kind="mergesort")
    top = ranked.groupby("market_day_local", sort=False).head(n).copy()
    return top


def compute_topn_tables(
    *,
    df: pd.DataFrame,
    n: int,
    p_col_model: str,
    p_col_climo: str,
) -> pd.DataFrame:
    top = select_top_n_per_day(df=df, n=n, prob_col=p_col_model)
    y = top["y"].to_numpy(dtype=float)
    if n == 1:
        ll_col = "log_loss_top1"
        br_col = "brier_top1"
    else:
        ll_col = "log_loss_top3"
        br_col = "brier_top3"

    table = pd.DataFrame(
        [
            {
                "model": "model_residual",
                ll_col: binary_log_loss(y, top[p_col_model].to_numpy(dtype=float)),
                br_col: brier_score(y, top[p_col_model].to_numpy(dtype=float)),
            },
            {
                "model": "climatology",
                ll_col: binary_log_loss(y, top[p_col_climo].to_numpy(dtype=float)),
                br_col: brier_score(y, top[p_col_climo].to_numpy(dtype=float)),
            },
        ]
    )
    return table


def compute_calibration_table(df: pd.DataFrame, *, prob_col: str) -> pd.DataFrame:
    out = df.copy()
    out["bin"] = pd.cut(
        out[prob_col],
        bins=CALIB_BINS,
        labels=CALIB_LABELS,
        include_lowest=True,
        right=False,
    )
    grouped = (
        out.groupby("bin", observed=False)
        .agg(
            predicted_mean=(prob_col, "mean"),
            observed_frequency=("y", "mean"),
            count=("y", "size"),
        )
        .reset_index()
    )
    return grouped


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False))


def save_outputs(
    *,
    output_dir: Path,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    calibration: pd.DataFrame,
    eval_detail: pd.DataFrame,
    residual_daily: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table1.to_csv(output_dir / "table1_model_vs_climatology.csv", index=False)
    table2.to_csv(output_dir / "table2_top1.csv", index=False)
    table3.to_csv(output_dir / "table3_top3.csv", index=False)
    calibration.to_csv(output_dir / "calibration_table.csv", index=False)
    eval_detail.to_parquet(output_dir / "market_level_probabilities.parquet", index=False)
    eval_detail.to_csv(output_dir / "market_level_probabilities.csv", index=False)
    residual_daily.to_parquet(output_dir / "residual_daily_oof.parquet", index=False)
    residual_daily.to_csv(output_dir / "residual_daily_oof.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    residual_years = sorted(set(int(y) for y in args.residual_years))
    if not residual_years:
        raise SystemExit("At least one --residual-years value is required.")

    # Model metadata from frozen 100% retrained artifacts (for hyperparameters and feature sets).
    city_meta = read_json(args.models_root / "city_extended_100" / args.station / "best_params.json")
    g1_meta = read_json(args.models_root / "xgb_opt_v1_100" / "best_params.json")
    g2_meta = read_json(args.models_root / "xgb_opt_v2_100" / "best_params.json")

    city_features = list(city_meta["feature_columns"])
    g1_features = list(g1_meta["feature_columns"])
    g2_features = list(g2_meta["feature_columns"])
    g1_numeric = [c for c in g1_features if c != CITY_COLUMN]
    g2_numeric = [c for c in g2_features if c != CITY_COLUMN]

    required_columns = sorted(
        set(
            city_features
            + g1_features
            + g2_features
            + [CITY_COLUMN, DATE_COLUMN, ISSUE_COLUMN, LEAD_COLUMN, OBS_COLUMN]
        )
    )
    numeric_columns = sorted(set(c for c in required_columns if c != CITY_COLUMN and c not in {DATE_COLUMN, ISSUE_COLUMN}))

    print("Loading training dataset...")
    raw_train = load_training_data(
        data_root=args.train_data_root,
        required_columns=required_columns,
    )
    train = normalize_training_data(raw_train, numeric_columns=numeric_columns)
    train = train.loc[train["year"].isin(residual_years)].copy()
    if train.empty:
        raise SystemExit("Training dataset is empty after residual year filtering.")

    london = train.loc[train[CITY_COLUMN] == args.station].copy()
    if london.empty:
        raise SystemExit(f"No London rows found in training data for station={args.station!r}.")
    print(
        f"Training rows (all cities): {len(train):,} | "
        f"London rows: {len(london):,} | residual years: {residual_years}"
    )

    print("Building OOF predictions for city_extended...")
    city_oof = fit_predict_oof_city_extended(
        london_df=london,
        years=residual_years,
        feature_columns=city_features,
        xgb_params=dict(city_meta["xgb_params_final"]),
    )
    print(f"city_extended OOF rows: {len(city_oof):,}")

    print("Building OOF predictions for xgb_opt_v1_100...")
    g1_oof = fit_predict_oof_global(
        all_df=train,
        station=args.station,
        years=residual_years,
        feature_columns=g1_features,
        numeric_columns=g1_numeric,
        xgb_params=dict(g1_meta["xgb_params_final"]),
        pred_col="pred_xgb_opt_v1_100",
    )
    print(f"xgb_opt_v1_100 OOF rows: {len(g1_oof):,}")

    print("Building OOF predictions for xgb_opt_v2_100...")
    g2_oof = fit_predict_oof_global(
        all_df=train,
        station=args.station,
        years=residual_years,
        feature_columns=g2_features,
        numeric_columns=g2_numeric,
        xgb_params=dict(g2_meta["xgb_params_final"]),
        pred_col="pred_xgb_opt_v2_100",
    )
    print(f"xgb_opt_v2_100 OOF rows: {len(g2_oof):,}")

    print("Fitting empirical residual CDF from daily decision-cycle OOF errors...")
    residual_daily = build_residual_series_from_oof(
        city_oof=city_oof,
        g1_oof=g1_oof,
        g2_oof=g2_oof,
    )
    residual_values = residual_daily["residual"].to_numpy(dtype=float)
    residual_cdf = EmpiricalResidualCdf(residual_values)
    print(
        f"Residual sample size (daily, OOF): {residual_cdf.size:,} "
        f"| period: {residual_daily[DATE_COLUMN].min().date()} to {residual_daily[DATE_COLUMN].max().date()}"
    )

    print("Loading decision-cycle forecasts for evaluation year...")
    eval_daily_forecast = build_eval_daily_forecast(
        predictions_root=args.predictions_root,
        station=args.station,
    )
    eval_daily_forecast = eval_daily_forecast.loc[
        eval_daily_forecast[DATE_COLUMN].dt.year == int(args.eval_year)
    ].copy()
    if eval_daily_forecast.empty:
        raise SystemExit(f"No evaluation forecasts found for eval year {args.eval_year}.")
    print(
        f"Evaluation forecast days: {len(eval_daily_forecast):,} "
        f"| period: {eval_daily_forecast[DATE_COLUMN].min().date()} to {eval_daily_forecast[DATE_COLUMN].max().date()}"
    )

    from master_db import resolve_master_postgres_dsn

    dsn = resolve_master_postgres_dsn(explicit_dsn=args.master_dsn)
    with psycopg.connect(dsn) as conn:
        print("Fetching climatology and resolved markets from master_db...")
        climo_daily = fetch_daily_tmax_from_db(
            conn=conn,
            station=args.station,
            start=CLIMO_START_DATE.strftime("%Y-%m-%d"),
            end_exclusive=CLIMO_END_EXCLUSIVE.strftime("%Y-%m-%d"),
        )
        climo_probs = build_climatology_probabilities(climo_daily)

        markets = fetch_resolved_london_markets(conn=conn)
        eval_obs = fetch_daily_tmax_from_db(
            conn=conn,
            station=args.station,
            start=f"{int(args.eval_year)}-01-01",
            end_exclusive=f"{int(args.eval_year) + 1}-01-01",
        )

    market_eval = attach_market_outcomes_from_observations(markets=markets, eval_daily_obs=eval_obs)
    market_eval = market_eval.loc[market_eval["market_day_local"].dt.year == int(args.eval_year)].copy()
    if market_eval.empty:
        raise SystemExit(f"No resolved exact-strike markets found for eval year {args.eval_year}.")

    eval_df = market_eval.merge(
        eval_daily_forecast[[DATE_COLUMN, "t_hat_median"]],
        left_on="market_day_local",
        right_on=DATE_COLUMN,
        how="inner",
    )
    if eval_df.empty:
        raise SystemExit("No market rows matched available daily decision-cycle forecasts.")
    eval_df = eval_df.drop(columns=[DATE_COLUMN])

    eval_df = compute_probabilities(
        df=eval_df,
        cdf=residual_cdf,
        climo_probs=climo_probs,
    )

    table1 = compute_metric_table(
        df=eval_df,
        p_col_model="p_model_residual",
        p_col_climo="p_climo",
    )
    table2 = compute_topn_tables(
        df=eval_df,
        n=1,
        p_col_model="p_model_residual",
        p_col_climo="p_climo",
    )
    table3 = compute_topn_tables(
        df=eval_df,
        n=3,
        p_col_model="p_model_residual",
        p_col_climo="p_climo",
    )
    calibration = compute_calibration_table(eval_df, prob_col="p_model_residual")

    output_eval = eval_df[
        [
            "market_id",
            "slug",
            "market_day_local",
            "strike",
            "t_round_obs",
            "y",
            "y_market",
            "t_hat_median",
            "p_model_residual",
            "p_climo",
        ]
    ].sort_values(["market_day_local", "strike"], kind="mergesort")

    save_outputs(
        output_dir=args.output_dir,
        table1=table1,
        table2=table2,
        table3=table3,
        calibration=calibration,
        eval_detail=output_eval,
        residual_daily=residual_daily,
    )

    mismatch_count = int((output_eval["y"] != output_eval["y_market"]).sum())
    print(f"\nOutcome consistency check (obs-derived y vs market yes/no): mismatches={mismatch_count}")
    print_table("Table 1 - Model vs Climatology", table1)
    print_table("Table 2 - Top-1 Evaluation", table2)
    print_table("Table 3 - Top-3 Evaluation", table3)
    print_table("Calibration Table", calibration)

    print(f"\nArtifacts written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
