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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "probability_backtest_multi"
DEFAULT_LOCATIONS_PATH = REPO_ROOT / "locations.csv"

RESIDUAL_YEARS = (2021, 2022, 2023, 2024, 2025)
CITY_COLUMN = "city_name"
DATE_COLUMN = "target_date_local"
ISSUE_COLUMN = "issue_time_utc"
LEAD_COLUMN = "lead_time_hours"
OBS_COLUMN = "tmax_obs_c"
RAW_COLUMN = "tmax_raw_c"
TARGET_COLUMN = "mos_target_delta_c"
KEY_COLUMNS = [ISSUE_COLUMN, DATE_COLUMN, LEAD_COLUMN]

SLUG_PREFIX_RE = re.compile(r"^highest-temperature-in-([a-z0-9-]+)-on-")
SLUG_EXACT_C_RE = re.compile(r"-(neg-\d+|\d+)c$")
SLUG_RANGE_F_RE = re.compile(r"-(\d+)-(\d+)f$")
SLUG_BELOW_F_RE = re.compile(r"-(\d+)forbelow$")
SLUG_ABOVE_F_RE = re.compile(r"-(\d+)forhigher$")
SLUG_EXACT_F_RE = re.compile(r"-(\d+)f$")
SLUG_SUFFIX_RE = re.compile(r"-(?:neg-\d+|\d+)c$|-(?:\d+-\d+f|\d+forbelow|\d+forhigher|\d+f)$")

PROB_EPS = 1e-6

TOKEN_TO_STATION = {
    "ankara": "Ankara",
    "atlanta": "Atlanta",
    "buenos-aires": "BuenosAires",
    "chicago": "Chicago",
    "dallas": "Dallas",
    "london": "London",
    "miami": "Miami",
    "nyc": "NYC",
    "paris": "Paris",
    "sao-paulo": "SaoPaulo",
    "seattle": "Seattle",
    "seoul": "Seoul",
    "toronto": "Toronto",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-station market-level probabilities for Sprint-2-scale.")
    parser.add_argument("--train-data-root", type=Path, default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--locations-path", type=Path, default=DEFAULT_LOCATIONS_PATH)
    parser.add_argument("--master-dsn", type=str, default=None)
    parser.add_argument(
        "--residual-years",
        type=int,
        nargs="+",
        default=list(RESIDUAL_YEARS),
    )
    parser.add_argument(
        "--stations",
        type=str,
        default="",
        help="Optional comma-separated station list (defaults to all detected stations).",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="resolved",
        help="Market status filter (default: resolved).",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_dataframe(conn: psycopg.Connection, query: str, params: dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)
        rows = cur.fetchall()
        columns = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=columns)


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


def parse_strike_token_c(token: str) -> int:
    if token.startswith("neg-"):
        return -int(token.split("-", 1)[1])
    return int(token)


def c_to_f(c: np.ndarray | pd.Series | float) -> np.ndarray:
    return (np.asarray(c, dtype=float) * 9.0 / 5.0) + 32.0


def f_to_c(f: np.ndarray | pd.Series | float) -> np.ndarray:
    return (np.asarray(f, dtype=float) - 32.0) * (5.0 / 9.0)


def round_to_market_integer_c(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.floor(arr + 0.5).astype(int)


def round_to_market_integer_f(values_f: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values_f, dtype=float)
    return np.floor(arr + 0.5).astype(int)


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
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        cdf_upper = np.where(np.isfinite(upper), self.cdf(upper), 1.0)
        cdf_lower = np.where(np.isfinite(lower), self.cdf(lower), 0.0)
        return cdf_upper - cdf_lower


def load_station_timezones(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    if "name" not in df.columns or "timezone" not in df.columns:
        raise SystemExit(f"locations file missing required columns: {path}")
    out = {}
    for row in df.itertuples(index=False):
        station = str(row.name).strip()
        tz = str(row.timezone).strip()
        if station and tz:
            out[station] = tz
    return out


def detect_stations(predictions_root: Path, models_root: Path) -> list[str]:
    pred_dir = predictions_root / "city_extended"
    if not pred_dir.exists():
        raise SystemExit(f"Predictions directory missing: {pred_dir}")

    station_dirs = sorted(d.name for d in pred_dir.iterdir() if d.is_dir())
    stations = []
    for s in station_dirs:
        if (models_root / "city_extended_100" / s / "best_params.json").exists():
            stations.append(s)
    if not stations:
        raise SystemExit("No stations detected with both predictions and city model metadata.")
    return stations


def filter_stations(stations: list[str], stations_arg: str) -> list[str]:
    if not stations_arg.strip():
        return stations
    requested = [x.strip() for x in stations_arg.split(",") if x.strip()]
    selected = [s for s in stations if s in requested]
    missing = sorted(set(requested) - set(selected))
    if missing:
        print(f"Warning: requested stations not found: {', '.join(missing)}")
    if not selected:
        raise SystemExit("No stations selected.")
    return selected


def load_training_data(*, data_root: Path, required_columns: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(str(data_root), format="parquet")
    available = set(dataset.schema.names)
    missing = sorted(set(required_columns) - available)
    if missing:
        raise SystemExit(f"Training dataset missing required columns: {', '.join(missing)}")
    table = dataset.to_table(columns=required_columns)
    return table.to_pandas()


def normalize_training_data(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
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


def fit_predict_oof_city(
    *,
    station_df: pd.DataFrame,
    years: list[int],
    feature_columns: list[str],
    xgb_params: dict,
) -> pd.DataFrame:
    required = feature_columns + [OBS_COLUMN, RAW_COLUMN] + KEY_COLUMNS
    work = station_df.dropna(subset=required).copy()
    work[TARGET_COLUMN] = work[OBS_COLUMN] - work[RAW_COLUMN]

    outputs: list[pd.DataFrame] = []
    for holdout_year in years:
        train = work.loc[work["year"] != holdout_year]
        holdout = work.loc[work["year"] == holdout_year]
        if holdout.empty:
            continue
        if train.empty:
            raise SystemExit(f"City model fold has empty train set for holdout year {holdout_year}.")

        model = XGBRegressor(**xgb_params)
        model.fit(train[feature_columns], train[TARGET_COLUMN].to_numpy())
        pred_temp = holdout[RAW_COLUMN].to_numpy(dtype=float) + model.predict(holdout[feature_columns])

        part = holdout[[CITY_COLUMN] + KEY_COLUMNS + [OBS_COLUMN]].copy()
        part["pred_city_extended"] = pred_temp
        outputs.append(part)

    if not outputs:
        raise SystemExit("City model OOF outputs are empty.")
    out = pd.concat(outputs, ignore_index=True)
    out = out.drop_duplicates(subset=[CITY_COLUMN] + KEY_COLUMNS, keep="last")
    return out


def fit_predict_oof_global_all_stations(
    *,
    all_df: pd.DataFrame,
    years: list[int],
    feature_columns: list[str],
    numeric_columns: list[str],
    xgb_params: dict,
    pred_col: str,
) -> pd.DataFrame:
    required = feature_columns + [OBS_COLUMN, RAW_COLUMN] + KEY_COLUMNS
    work = all_df.dropna(subset=required).copy()
    work[TARGET_COLUMN] = work[OBS_COLUMN] - work[RAW_COLUMN]

    outputs: list[pd.DataFrame] = []
    for holdout_year in years:
        train = work.loc[work["year"] != holdout_year]
        holdout = work.loc[work["year"] == holdout_year]
        if holdout.empty:
            continue
        if train.empty:
            raise SystemExit(f"Global model fold has empty train set for holdout year {holdout_year}.")

        preprocessor = build_global_preprocessor(numeric_columns=numeric_columns)
        model = XGBRegressor(**xgb_params)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(train[feature_columns], train[TARGET_COLUMN].to_numpy())
        pred_temp = holdout[RAW_COLUMN].to_numpy(dtype=float) + pipe.predict(holdout[feature_columns])

        part = holdout[[CITY_COLUMN] + KEY_COLUMNS + [OBS_COLUMN]].copy()
        part[pred_col] = pred_temp
        outputs.append(part)

    if not outputs:
        raise SystemExit(f"Global model OOF outputs are empty for {pred_col}.")
    out = pd.concat(outputs, ignore_index=True)
    out = out.drop_duplicates(subset=[CITY_COLUMN] + KEY_COLUMNS, keep="last")
    return out


def select_latest_cycle_before_local_midnight(df: pd.DataFrame, *, day_column: str, issue_column: str, timezone: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    day_start_local = pd.to_datetime(work[day_column], errors="coerce").dt.tz_localize(timezone)
    cutoff_utc = day_start_local.dt.tz_convert("UTC")
    work = work.loc[work[issue_column] < cutoff_utc].copy()
    if work.empty:
        return work
    work = work.sort_values([day_column, issue_column], kind="mergesort")
    work = work.drop_duplicates(subset=[day_column], keep="last")
    return work


def load_prediction_rows(*, predictions_root: Path, model_name: str, station: str) -> pd.DataFrame:
    station_dir = predictions_root / model_name / station
    if not station_dir.exists():
        raise SystemExit(f"Prediction directory not found: {station_dir}")
    files = sorted(station_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No prediction parquet files in: {station_dir}")

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


def build_eval_daily_forecast(*, predictions_root: Path, station: str, timezone: str) -> pd.DataFrame:
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

    m1_sel = select_latest_cycle_before_local_midnight(m1, day_column=DATE_COLUMN, issue_column=ISSUE_COLUMN, timezone=timezone)
    m2_sel = select_latest_cycle_before_local_midnight(m2, day_column=DATE_COLUMN, issue_column=ISSUE_COLUMN, timezone=timezone)
    m3_sel = select_latest_cycle_before_local_midnight(m3, day_column=DATE_COLUMN, issue_column=ISSUE_COLUMN, timezone=timezone)

    daily = (
        m1_sel[[DATE_COLUMN, ISSUE_COLUMN, "pred_city_extended"]]
        .rename(columns={ISSUE_COLUMN: "execution_time_utc"})
        .merge(m2_sel[[DATE_COLUMN, "pred_xgb_opt_v1_100"]], on=DATE_COLUMN, how="inner")
        .merge(m3_sel[[DATE_COLUMN, "pred_xgb_opt_v2_100"]], on=DATE_COLUMN, how="inner")
    )
    if daily.empty:
        return daily

    preds = daily[["pred_city_extended", "pred_xgb_opt_v1_100", "pred_xgb_opt_v2_100"]].to_numpy(dtype=float)
    daily["t_hat_median"] = np.median(preds, axis=1)
    daily = daily.sort_values(DATE_COLUMN, kind="mergesort")
    return daily


def fetch_daily_tmax(
    *,
    conn: psycopg.Connection,
    stations: list[str],
    start: str,
    end_exclusive: str,
) -> pd.DataFrame:
    query = """
        SELECT
            station,
            observed_at_local::date AS day_local,
            MAX(COALESCE(temperature_c::double precision, ((temperature_f - 32.0) * (5.0 / 9.0)))) AS tmax_c
        FROM station_observations
        WHERE station = ANY(%(stations)s)
          AND observed_at_local >= %(start)s::timestamp
          AND observed_at_local < %(end_exclusive)s::timestamp
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    df = fetch_dataframe(conn, query, params={"stations": stations, "start": start, "end_exclusive": end_exclusive})
    if df.empty:
        return df
    df["day_local"] = pd.to_datetime(df["day_local"], errors="coerce").dt.normalize()
    df["tmax_c"] = pd.to_numeric(df["tmax_c"], errors="coerce")
    df = df.dropna(subset=["station", "day_local", "tmax_c"]).copy()
    df["t_round_obs_c"] = round_to_market_integer_c(df["tmax_c"].to_numpy(dtype=float))
    df["t_round_obs_f"] = round_to_market_integer_f(c_to_f(df["tmax_c"].to_numpy(dtype=float)))
    return df


def fetch_markets(*, conn: psycopg.Connection, status: str) -> pd.DataFrame:
    query = """
        SELECT
            market_id::text AS market_id,
            slug,
            status,
            raw ->> 'endDate' AS end_date_utc,
            raw ->> 'outcomePrices' AS outcome_prices
        FROM markets
        WHERE status = %(status)s
          AND slug ILIKE 'highest-temperature-in-%%'
        ORDER BY market_id::bigint ASC
    """
    out = fetch_dataframe(conn, query, params={"status": status})
    if out.empty:
        return out
    out["slug"] = out["slug"].astype("string")
    return out


def parse_market_slug(slug: str) -> dict | None:
    m = SLUG_PREFIX_RE.search(str(slug))
    if not m:
        return None
    token = m.group(1)
    station = TOKEN_TO_STATION.get(token)
    if station is None:
        return None

    text = str(slug)
    m_c = SLUG_EXACT_C_RE.search(text)
    if m_c:
        strike_c = parse_strike_token_c(m_c.group(1))
        return {
            "station": station,
            "station_token": token,
            "event_kind": "exact_c",
            "strike_k": int(strike_c),
            "c_strike": int(strike_c),
            "f_low": np.nan,
            "f_high": np.nan,
            "f_thresh": np.nan,
            "lower_c": float(strike_c) - 0.5,
            "upper_c": float(strike_c) + 0.5,
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_r = SLUG_RANGE_F_RE.search(text)
    if m_r:
        f_low = int(m_r.group(1))
        f_high = int(m_r.group(2))
        if f_low > f_high:
            f_low, f_high = f_high, f_low
        mid_c = f_to_c((f_low + f_high) / 2.0)
        strike_k = int(round_to_market_integer_c(np.array([mid_c]))[0])
        return {
            "station": station,
            "station_token": token,
            "event_kind": "range_f",
            "strike_k": strike_k,
            "c_strike": np.nan,
            "f_low": float(f_low),
            "f_high": float(f_high),
            "f_thresh": np.nan,
            "lower_c": float(f_to_c(f_low - 0.5)),
            "upper_c": float(f_to_c(f_high + 0.5)),
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_below = SLUG_BELOW_F_RE.search(text)
    if m_below:
        f_th = int(m_below.group(1))
        strike_k = int(round_to_market_integer_c(np.array([f_to_c(f_th)]))[0])
        return {
            "station": station,
            "station_token": token,
            "event_kind": "below_f",
            "strike_k": strike_k,
            "c_strike": np.nan,
            "f_low": np.nan,
            "f_high": np.nan,
            "f_thresh": float(f_th),
            "lower_c": -math.inf,
            "upper_c": float(f_to_c(f_th + 0.5)),
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_above = SLUG_ABOVE_F_RE.search(text)
    if m_above:
        f_th = int(m_above.group(1))
        strike_k = int(round_to_market_integer_c(np.array([f_to_c(f_th)]))[0])
        return {
            "station": station,
            "station_token": token,
            "event_kind": "above_f",
            "strike_k": strike_k,
            "c_strike": np.nan,
            "f_low": np.nan,
            "f_high": np.nan,
            "f_thresh": float(f_th),
            "lower_c": float(f_to_c(f_th - 0.5)),
            "upper_c": math.inf,
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_f = SLUG_EXACT_F_RE.search(text)
    if m_f:
        f_strike = int(m_f.group(1))
        strike_k = int(round_to_market_integer_c(np.array([f_to_c(f_strike)]))[0])
        return {
            "station": station,
            "station_token": token,
            "event_kind": "exact_f",
            "strike_k": strike_k,
            "c_strike": np.nan,
            "f_low": float(f_strike),
            "f_high": float(f_strike),
            "f_thresh": np.nan,
            "lower_c": float(f_to_c(f_strike - 0.5)),
            "upper_c": float(f_to_c(f_strike + 0.5)),
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    return None


def parse_yes_prob(outcome_prices: str) -> float | None:
    try:
        values = json.loads(outcome_prices)
        if not isinstance(values, list) or len(values) < 1:
            return None
        return float(values[0])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def build_market_frame(raw_markets: pd.DataFrame, station_timezones: dict[str, str]) -> pd.DataFrame:
    parsed = raw_markets["slug"].map(parse_market_slug)
    keep = parsed.notna()
    out = raw_markets.loc[keep].copy()
    parsed_df = pd.DataFrame(parsed.loc[keep].tolist(), index=out.index)
    out = pd.concat([out, parsed_df], axis=1)

    out["end_date_utc"] = pd.to_datetime(out["end_date_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["end_date_utc"]).copy()

    out["market_day_local"] = pd.NaT
    for station, grp_idx in out.groupby("station").groups.items():
        tz = station_timezones.get(station)
        if tz is None:
            continue
        local_day = out.loc[grp_idx, "end_date_utc"].dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
        out.loc[grp_idx, "market_day_local"] = local_day.to_numpy()

    out["market_day_local"] = pd.to_datetime(out["market_day_local"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["station", "market_day_local", "strike_k", "event_key"]).copy()
    out["strike_k"] = pd.to_numeric(out["strike_k"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["strike_k"]).copy()
    out["strike_k"] = out["strike_k"].astype(int)

    out["yes_prob_resolved"] = out["outcome_prices"].map(parse_yes_prob)
    out["y_market"] = np.where(out["yes_prob_resolved"].notna(), (out["yes_prob_resolved"] >= 0.5).astype(int), pd.NA)
    return out


def attach_observed_outcomes(markets: pd.DataFrame, obs_daily: pd.DataFrame) -> pd.DataFrame:
    obs = obs_daily.rename(columns={"day_local": "market_day_local"})
    merged = markets.merge(
        obs[["station", "market_day_local", "tmax_c", "t_round_obs_c", "t_round_obs_f"]],
        on=["station", "market_day_local"],
        how="inner",
    )
    if merged.empty:
        return merged

    y = np.zeros(len(merged), dtype=int)
    kind = merged["event_kind"].astype(str)

    mask_c = kind == "exact_c"
    if mask_c.any():
        c_strike = pd.to_numeric(merged.loc[mask_c, "c_strike"], errors="coerce").to_numpy(dtype=float)
        obs_c = merged.loc[mask_c, "t_round_obs_c"].to_numpy(dtype=float)
        y[mask_c.to_numpy()] = (obs_c == c_strike).astype(int)

    mask_exact_f = kind == "exact_f"
    if mask_exact_f.any():
        f_low = pd.to_numeric(merged.loc[mask_exact_f, "f_low"], errors="coerce").to_numpy(dtype=float)
        obs_f = merged.loc[mask_exact_f, "t_round_obs_f"].to_numpy(dtype=float)
        y[mask_exact_f.to_numpy()] = (obs_f == f_low).astype(int)

    mask_range_f = kind == "range_f"
    if mask_range_f.any():
        f_low = pd.to_numeric(merged.loc[mask_range_f, "f_low"], errors="coerce").to_numpy(dtype=float)
        f_high = pd.to_numeric(merged.loc[mask_range_f, "f_high"], errors="coerce").to_numpy(dtype=float)
        obs_f = merged.loc[mask_range_f, "t_round_obs_f"].to_numpy(dtype=float)
        y[mask_range_f.to_numpy()] = ((obs_f >= f_low) & (obs_f <= f_high)).astype(int)

    mask_below = kind == "below_f"
    if mask_below.any():
        th = pd.to_numeric(merged.loc[mask_below, "f_thresh"], errors="coerce").to_numpy(dtype=float)
        obs_f = merged.loc[mask_below, "t_round_obs_f"].to_numpy(dtype=float)
        y[mask_below.to_numpy()] = (obs_f <= th).astype(int)

    mask_above = kind == "above_f"
    if mask_above.any():
        th = pd.to_numeric(merged.loc[mask_above, "f_thresh"], errors="coerce").to_numpy(dtype=float)
        obs_f = merged.loc[mask_above, "t_round_obs_f"].to_numpy(dtype=float)
        y[mask_above.to_numpy()] = (obs_f >= th).astype(int)

    merged["y"] = y
    return merged


def build_market_probabilities_for_station(
    *,
    station: str,
    station_tz: str,
    train: pd.DataFrame,
    residual_years: list[int],
    city_meta: dict,
    g1_oof_all: pd.DataFrame,
    g2_oof_all: pd.DataFrame,
    predictions_root: Path,
    station_markets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_train = train.loc[(train[CITY_COLUMN] == station) & (train["year"].isin(residual_years))].copy()
    if station_train.empty:
        raise SystemExit(f"No training rows for station {station}.")

    city_oof = fit_predict_oof_city(
        station_df=station_train,
        years=residual_years,
        feature_columns=list(city_meta["feature_columns"]),
        xgb_params=dict(city_meta["xgb_params_final"]),
    )

    g1_station = g1_oof_all.loc[g1_oof_all[CITY_COLUMN] == station].copy()
    g2_station = g2_oof_all.loc[g2_oof_all[CITY_COLUMN] == station].copy()
    merge_cols = [CITY_COLUMN] + KEY_COLUMNS + [OBS_COLUMN]
    merged = city_oof.merge(g1_station, on=merge_cols, how="inner").merge(g2_station, on=merge_cols, how="inner")
    if merged.empty:
        raise SystemExit(f"No overlapping OOF rows across all three models for station {station}.")

    preds = merged[["pred_city_extended", "pred_xgb_opt_v1_100", "pred_xgb_opt_v2_100"]].to_numpy(dtype=float)
    merged["t_hat_median"] = np.median(preds, axis=1)
    merged = merged.dropna(subset=[DATE_COLUMN, ISSUE_COLUMN, OBS_COLUMN, "t_hat_median"]).copy()
    selected = select_latest_cycle_before_local_midnight(
        merged,
        day_column=DATE_COLUMN,
        issue_column=ISSUE_COLUMN,
        timezone=station_tz,
    )
    if selected.empty:
        raise SystemExit(f"No OOF rows survive decision-cycle filter for station {station}.")
    selected["residual"] = selected[OBS_COLUMN] - selected["t_hat_median"]
    residual_cdf = EmpiricalResidualCdf(selected["residual"].to_numpy(dtype=float))

    eval_daily = build_eval_daily_forecast(
        predictions_root=predictions_root,
        station=station,
        timezone=station_tz,
    )
    if eval_daily.empty:
        return pd.DataFrame(), selected

    eval_df = station_markets.merge(
        eval_daily[[DATE_COLUMN, "execution_time_utc", "t_hat_median"]],
        left_on="market_day_local",
        right_on=DATE_COLUMN,
        how="inner",
    )
    if eval_df.empty:
        return pd.DataFrame(), selected

    lower = pd.to_numeric(eval_df["lower_c"], errors="coerce").to_numpy(dtype=float) - eval_df["t_hat_median"].to_numpy(dtype=float)
    upper = pd.to_numeric(eval_df["upper_c"], errors="coerce").to_numpy(dtype=float) - eval_df["t_hat_median"].to_numpy(dtype=float)
    eval_df["p_model"] = residual_cdf.interval_prob(lower=lower, upper=upper)
    eval_df["p_model"] = np.clip(eval_df["p_model"].to_numpy(dtype=float), PROB_EPS, 1.0 - PROB_EPS)

    mode_df = (
        eval_df.sort_values(["market_day_local", "event_key", "p_model", "strike_k"], ascending=[True, True, False, True], kind="mergesort")
        .drop_duplicates(subset=["market_day_local", "event_key"], keep="first")
        [["market_day_local", "event_key", "strike_k"]]
        .rename(columns={"strike_k": "mode_k"})
    )
    eval_df = eval_df.merge(mode_df, on=["market_day_local", "event_key"], how="left")
    eval_df["station_name"] = station
    eval_df["date"] = eval_df["market_day_local"]
    return eval_df, selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    residual_years = sorted(set(int(y) for y in args.residual_years))
    if not residual_years:
        raise SystemExit("At least one residual year is required.")

    station_timezones = load_station_timezones(args.locations_path)
    stations = detect_stations(args.predictions_root, args.models_root)
    stations = filter_stations(stations, args.stations)
    missing_tz = [s for s in stations if s not in station_timezones]
    if missing_tz:
        raise SystemExit(f"Missing timezone mapping for stations: {', '.join(missing_tz)}")

    city_meta_map = {}
    all_city_features: set[str] = set()
    for station in stations:
        meta = read_json(args.models_root / "city_extended_100" / station / "best_params.json")
        city_meta_map[station] = meta
        all_city_features.update(meta["feature_columns"])

    g1_meta = read_json(args.models_root / "xgb_opt_v1_100" / "best_params.json")
    g2_meta = read_json(args.models_root / "xgb_opt_v2_100" / "best_params.json")
    g1_features = list(g1_meta["feature_columns"])
    g2_features = list(g2_meta["feature_columns"])
    g1_numeric = [c for c in g1_features if c != CITY_COLUMN]
    g2_numeric = [c for c in g2_features if c != CITY_COLUMN]

    required_columns = sorted(
        set(all_city_features)
        | set(g1_features)
        | set(g2_features)
        | {CITY_COLUMN, DATE_COLUMN, ISSUE_COLUMN, LEAD_COLUMN, OBS_COLUMN}
    )
    numeric_columns = sorted(set(c for c in required_columns if c not in {CITY_COLUMN, DATE_COLUMN, ISSUE_COLUMN}))

    print(f"Stations selected: {len(stations)} -> {', '.join(stations)}")
    print("Loading training data...")
    raw_train = load_training_data(data_root=args.train_data_root, required_columns=required_columns)
    train = normalize_training_data(raw_train, numeric_columns=numeric_columns)
    train = train.loc[train[CITY_COLUMN].isin(stations)].copy()
    train = train.loc[train["year"].isin(residual_years)].copy()
    if train.empty:
        raise SystemExit("Training data empty after station/year filtering.")
    print(f"Training rows in residual years: {len(train):,}")

    print("Building global OOF predictions (xgb_opt_v1_100)...")
    g1_oof_all = fit_predict_oof_global_all_stations(
        all_df=train,
        years=residual_years,
        feature_columns=g1_features,
        numeric_columns=g1_numeric,
        xgb_params=dict(g1_meta["xgb_params_final"]),
        pred_col="pred_xgb_opt_v1_100",
    )
    print(f"Global OOF v1 rows: {len(g1_oof_all):,}")

    print("Building global OOF predictions (xgb_opt_v2_100)...")
    g2_oof_all = fit_predict_oof_global_all_stations(
        all_df=train,
        years=residual_years,
        feature_columns=g2_features,
        numeric_columns=g2_numeric,
        xgb_params=dict(g2_meta["xgb_params_final"]),
        pred_col="pred_xgb_opt_v2_100",
    )
    print(f"Global OOF v2 rows: {len(g2_oof_all):,}")

    from master_db import resolve_master_postgres_dsn

    dsn = resolve_master_postgres_dsn(explicit_dsn=args.master_dsn)
    with psycopg.connect(dsn) as conn:
        raw_markets = fetch_markets(conn=conn, status=args.status)
        if raw_markets.empty:
            raise SystemExit(f"No markets returned for status={args.status}.")
        markets = build_market_frame(raw_markets, station_timezones=station_timezones)
        markets = markets.loc[markets["station"].isin(stations)].copy()
        if markets.empty:
            raise SystemExit("No parseable station markets after filtering.")

        min_day = markets["market_day_local"].min()
        max_day = markets["market_day_local"].max()
        obs_daily = fetch_daily_tmax(
            conn=conn,
            stations=stations,
            start=(min_day - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            end_exclusive=(max_day + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        )
        if obs_daily.empty:
            raise SystemExit("No daily observations returned for selected station/day window.")

    markets_obs = attach_observed_outcomes(markets, obs_daily)
    if markets_obs.empty:
        raise SystemExit("No market rows matched observed outcomes.")

    probs_parts: list[pd.DataFrame] = []
    residual_parts: list[pd.DataFrame] = []
    for station in stations:
        station_tz = station_timezones[station]
        station_markets = markets_obs.loc[markets_obs["station"] == station].copy()
        if station_markets.empty:
            continue
        print(f"Building station probabilities: {station}")
        station_probs, station_residual = build_market_probabilities_for_station(
            station=station,
            station_tz=station_tz,
            train=train,
            residual_years=residual_years,
            city_meta=city_meta_map[station],
            g1_oof_all=g1_oof_all,
            g2_oof_all=g2_oof_all,
            predictions_root=args.predictions_root,
            station_markets=station_markets,
        )
        if not station_probs.empty:
            probs_parts.append(station_probs)
        if not station_residual.empty:
            station_residual = station_residual.copy()
            station_residual["station_name"] = station
            residual_parts.append(station_residual)

    if not probs_parts:
        raise SystemExit("No station probability rows were produced.")

    out = pd.concat(probs_parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["market_day_local"] = pd.to_datetime(out["market_day_local"], errors="coerce").dt.normalize()
    out["execution_time_utc"] = pd.to_datetime(out["execution_time_utc"], utc=True, errors="coerce")
    out["strike_k"] = pd.to_numeric(out["strike_k"], errors="coerce").astype("Int64")
    out["mode_k"] = pd.to_numeric(out["mode_k"], errors="coerce").astype("Int64")
    out["y"] = pd.to_numeric(out["y"], errors="coerce").astype("Int64")
    out["y_market"] = pd.to_numeric(out["y_market"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["station_name", "slug", "date", "strike_k", "mode_k", "p_model", "y", "execution_time_utc"]).copy()
    out["strike_k"] = out["strike_k"].astype(int)
    out["mode_k"] = out["mode_k"].astype(int)
    out["y"] = out["y"].astype(int)

    out = out.sort_values(["station_name", "date", "event_key", "strike_k", "market_id"], kind="mergesort")
    out = out.drop_duplicates(subset=["station_name", "date", "event_key", "slug"], keep="last")

    output_cols = [
        "market_id",
        "slug",
        "station_name",
        "date",
        "market_day_local",
        "event_key",
        "event_kind",
        "strike_k",
        "mode_k",
        "tmax_c",
        "t_round_obs_c",
        "t_round_obs_f",
        "y",
        "y_market",
        "p_model",
        "t_hat_median",
        "execution_time_utc",
        "lower_c",
        "upper_c",
    ]
    out_final = out[output_cols].copy()
    out_final = out_final.rename(columns={"t_hat_median": "T_hat"})

    residual_out = pd.concat(residual_parts, ignore_index=True) if residual_parts else pd.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_final.to_parquet(args.output_dir / "market_level_probabilities.parquet", index=False)
    out_final.to_csv(args.output_dir / "market_level_probabilities.csv", index=False)
    if not residual_out.empty:
        residual_out.to_parquet(args.output_dir / "residual_daily_oof.parquet", index=False)
        residual_out.to_csv(args.output_dir / "residual_daily_oof.csv", index=False)

    station_counts = (
        out_final.groupby("station_name", as_index=False)
        .agg(rows=("slug", "size"), days=("date", "nunique"))
        .sort_values("station_name", kind="mergesort")
    )
    print("\nStation probability coverage")
    print(station_counts.to_string(index=False))
    print(f"\nTotal rows: {len(out_final):,}")
    print(f"Output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
