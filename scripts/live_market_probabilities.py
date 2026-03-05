#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PREDICTIONS_ROOT = REPO_ROOT / "data" / "ml_predictions"
DEFAULT_RESIDUALS_PATH = REPO_ROOT / "reports" / "probability_backtest_multi" / "residual_daily_oof.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "live_probabilities"
DEFAULT_LOCATIONS_PATH = REPO_ROOT / "locations.csv"
DEFAULT_MANIFEST_NAME = "latest_manifest.json"
DEFAULT_CALIBRATION_VERSION = "residual_oof_v1"

CITY_COLUMN = "city_name"
DATE_COLUMN = "target_date_local"
ISSUE_COLUMN = "issue_time_utc"
LEAD_COLUMN = "lead_time_hours"
PROB_EPS = 1e-6

DEFAULT_MODELS = ("city_extended", "xgb_opt_v1_100", "xgb_opt_v2_100")

SLUG_PREFIX_RE = re.compile(r"^highest-temperature-in-([a-z0-9-]+)-on-")
SLUG_EXACT_C_RE = re.compile(r"-(neg-\d+|\d+)c$")
SLUG_BELOW_C_RE = re.compile(r"-(neg-\d+|\d+)corbelow$")
SLUG_ABOVE_C_RE = re.compile(r"-(neg-\d+|\d+)corhigher$")
SLUG_RANGE_F_RE = re.compile(r"-(\d+)-(\d+)f$")
SLUG_BELOW_F_RE = re.compile(r"-(\d+)forbelow$")
SLUG_ABOVE_F_RE = re.compile(r"-(\d+)forhigher$")
SLUG_EXACT_F_RE = re.compile(r"-(\d+)f$")
SLUG_SUFFIX_RE = re.compile(
    r"-(?:neg-\d+|\d+)c(?:orbelow|orhigher)?$|-(?:\d+-\d+f|\d+forbelow|\d+forhigher|\d+f)$"
)
CYCLE_TOKEN_RE = re.compile(r"^\d{10}$")
CALIBRATION_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build market_level_probabilities for active Polymarket weather markets "
            "from latest daily station Tmax predictions."
        )
    )
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--residuals-path", type=Path, default=DEFAULT_RESIDUALS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--locations-path", type=Path, default=DEFAULT_LOCATIONS_PATH)
    parser.add_argument("--master-dsn", type=str, default=None)
    parser.add_argument(
        "--calibration-version",
        type=str,
        default=DEFAULT_CALIBRATION_VERSION,
        help=(
            "Calibration snapshot version key. Same cycle + same calibration version must be deterministic. "
            f"(default: {DEFAULT_CALIBRATION_VERSION})"
        ),
    )
    parser.add_argument(
        "--cycle",
        type=str,
        default="",
        help=(
            "Optional cycle token YYYYMMDDHH. "
            "If omitted, cycle is derived from max execution_time_utc in produced rows."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model names under predictions root (default: {','.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--statuses",
        type=str,
        default="active",
        help="Comma-separated market statuses to include from master_db.markets (default: active).",
    )
    parser.add_argument(
        "--stations",
        type=str,
        default="",
        help="Optional comma-separated station list (default: auto from residuals + predictions intersection).",
    )
    parser.add_argument(
        "--min-residual-history",
        type=int,
        default=100,
        help="Minimum residual history rows required per station (default: 100).",
    )
    return parser.parse_args(argv)


def parse_cycle_token(value: str) -> str:
    text = str(value).strip()
    if CYCLE_TOKEN_RE.fullmatch(text) is None:
        raise SystemExit(f"--cycle must be YYYYMMDDHH, got {value!r}")
    return text


def parse_calibration_version(value: str) -> str:
    text = str(value).strip()
    if CALIBRATION_VERSION_RE.fullmatch(text) is None:
        raise SystemExit(
            "--calibration-version must match "
            f"{CALIBRATION_VERSION_RE.pattern!r}, got {value!r}"
        )
    return text


def fetch_dataframe(conn: psycopg.Connection, query: str, params: dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)
        rows = cur.fetchall()
        columns = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=columns)


def normalize_station_key(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def load_station_timezones(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    if "name" not in df.columns or "timezone" not in df.columns:
        raise SystemExit(f"locations.csv is missing required columns: {path}")
    out: dict[str, str] = {}
    for row in df.itertuples(index=False):
        station = str(getattr(row, "name", "")).strip()
        tz = str(getattr(row, "timezone", "")).strip()
        if station and tz:
            out[station] = tz
    return out


def parse_model_names(text: str) -> list[str]:
    models = [m.strip() for m in str(text).split(",") if m.strip()]
    if not models:
        raise SystemExit("At least one model must be provided via --models.")
    return models


def parse_statuses(text: str) -> list[str]:
    statuses = [s.strip() for s in str(text).split(",") if s.strip()]
    if not statuses:
        raise SystemExit("At least one status must be provided via --statuses.")
    return statuses


def detect_prediction_stations(predictions_root: Path, model_names: list[str]) -> set[str]:
    station_sets: list[set[str]] = []
    for model_name in model_names:
        model_dir = predictions_root / model_name
        if not model_dir.exists():
            raise SystemExit(f"Prediction model directory not found: {model_dir}")
        stations = {d.name for d in model_dir.iterdir() if d.is_dir()}
        if not stations:
            raise SystemExit(f"No station directories found under: {model_dir}")
        station_sets.append(stations)

    intersection = set.intersection(*station_sets) if station_sets else set()
    if not intersection:
        raise SystemExit("No common stations were found across all requested model directories.")
    return intersection


def load_residual_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Residual history file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    station_col = None
    for c in ("station_name", "station", "city_name", "city"):
        if c in df.columns:
            station_col = c
            break
    if station_col is None:
        raise SystemExit("Residual history file must include one of: station_name, station, city_name, city")

    residual_col = None
    for c in ("residual", "residual_c"):
        if c in df.columns:
            residual_col = c
            break
    if residual_col is None:
        raise SystemExit("Residual history file must include residual or residual_c column.")

    out = pd.DataFrame(
        {
            "station_name": df[station_col].astype("string").str.strip(),
            "residual": pd.to_numeric(df[residual_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["station_name", "residual"]).copy()
    out = out.loc[out["station_name"] != ""].copy()
    out = out.drop_duplicates(keep="last")
    return out


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


def build_station_cdfs(residual_history: pd.DataFrame, *, min_history: int) -> dict[str, EmpiricalResidualCdf]:
    out: dict[str, EmpiricalResidualCdf] = {}
    for station, grp in residual_history.groupby("station_name", sort=True):
        vals = grp["residual"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < int(min_history):
            continue
        out[str(station)] = EmpiricalResidualCdf(vals)
    return out


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


def parse_market_slug(slug: str, station_by_key: dict[str, str]) -> dict | None:
    text = str(slug or "")
    m = SLUG_PREFIX_RE.search(text)
    if not m:
        return None

    token_key = normalize_station_key(m.group(1))
    station = station_by_key.get(token_key)
    if station is None:
        return None

    m_exact_c = SLUG_EXACT_C_RE.search(text)
    if m_exact_c:
        strike_c = parse_strike_token_c(m_exact_c.group(1))
        return {
            "station": station,
            "event_kind": "exact_c",
            "strike_k": int(strike_c),
            "lower_c": float(strike_c) - 0.5,
            "upper_c": float(strike_c) + 0.5,
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_below_c = SLUG_BELOW_C_RE.search(text)
    if m_below_c:
        strike_c = parse_strike_token_c(m_below_c.group(1))
        return {
            "station": station,
            "event_kind": "below_c",
            "strike_k": int(strike_c),
            "lower_c": -math.inf,
            "upper_c": float(strike_c) + 0.5,
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_above_c = SLUG_ABOVE_C_RE.search(text)
    if m_above_c:
        strike_c = parse_strike_token_c(m_above_c.group(1))
        return {
            "station": station,
            "event_kind": "above_c",
            "strike_k": int(strike_c),
            "lower_c": float(strike_c) - 0.5,
            "upper_c": math.inf,
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_range_f = SLUG_RANGE_F_RE.search(text)
    if m_range_f:
        f_low = int(m_range_f.group(1))
        f_high = int(m_range_f.group(2))
        if f_low > f_high:
            f_low, f_high = f_high, f_low
        mid_c = f_to_c((f_low + f_high) / 2.0)
        strike_k = int(round_to_market_integer_c(np.array([mid_c]))[0])
        return {
            "station": station,
            "event_kind": "range_f",
            "strike_k": strike_k,
            "lower_c": float(f_to_c(f_low - 0.5)),
            "upper_c": float(f_to_c(f_high + 0.5)),
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_below_f = SLUG_BELOW_F_RE.search(text)
    if m_below_f:
        f_th = int(m_below_f.group(1))
        strike_k = int(round_to_market_integer_c(np.array([f_to_c(f_th)]))[0])
        return {
            "station": station,
            "event_kind": "below_f",
            "strike_k": strike_k,
            "lower_c": -math.inf,
            "upper_c": float(f_to_c(f_th + 0.5)),
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_above_f = SLUG_ABOVE_F_RE.search(text)
    if m_above_f:
        f_th = int(m_above_f.group(1))
        strike_k = int(round_to_market_integer_c(np.array([f_to_c(f_th)]))[0])
        return {
            "station": station,
            "event_kind": "above_f",
            "strike_k": strike_k,
            "lower_c": float(f_to_c(f_th - 0.5)),
            "upper_c": math.inf,
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    m_exact_f = SLUG_EXACT_F_RE.search(text)
    if m_exact_f:
        f_strike = int(m_exact_f.group(1))
        strike_k = int(round_to_market_integer_c(np.array([f_to_c(f_strike)]))[0])
        return {
            "station": station,
            "event_kind": "exact_f",
            "strike_k": strike_k,
            "lower_c": float(f_to_c(f_strike - 0.5)),
            "upper_c": float(f_to_c(f_strike + 0.5)),
            "event_key": SLUG_SUFFIX_RE.sub("", text),
        }

    return None


def fetch_markets(*, conn: psycopg.Connection, statuses: list[str]) -> pd.DataFrame:
    query = """
        SELECT
            market_id::text AS market_id,
            slug,
            status,
            raw ->> 'endDate' AS end_date_utc
        FROM markets
        WHERE status = ANY(%(statuses)s::text[])
          AND slug ILIKE 'highest-temperature-in-%%'
        ORDER BY market_id::bigint ASC
    """
    out = fetch_dataframe(conn, query, params={"statuses": statuses})
    if out.empty:
        return out
    out["slug"] = out["slug"].astype("string")
    out["status"] = out["status"].astype("string")
    return out


def build_market_frame(raw_markets: pd.DataFrame, *, station_timezones: dict[str, str], station_by_key: dict[str, str]) -> pd.DataFrame:
    parsed = raw_markets["slug"].map(lambda slug: parse_market_slug(str(slug), station_by_key=station_by_key))
    keep = parsed.notna()
    out = raw_markets.loc[keep].copy()
    if out.empty:
        return out

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
    return out


def load_prediction_rows(*, predictions_root: Path, model_name: str, station: str) -> pd.DataFrame:
    station_dir = predictions_root / model_name / station
    if not station_dir.exists():
        return pd.DataFrame(columns=[ISSUE_COLUMN, DATE_COLUMN, LEAD_COLUMN, "Forecast"])

    files = sorted(station_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=[ISSUE_COLUMN, DATE_COLUMN, LEAD_COLUMN, "Forecast"])

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
    out = out.drop_duplicates(subset=[ISSUE_COLUMN, DATE_COLUMN, LEAD_COLUMN], keep="last")
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


def build_eval_daily_forecast(*, predictions_root: Path, station: str, timezone: str, model_names: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    pred_cols: list[str] = []

    for model_name in model_names:
        rows = load_prediction_rows(predictions_root=predictions_root, model_name=model_name, station=station)
        if rows.empty:
            return pd.DataFrame()
        selected = select_latest_cycle_before_local_midnight(
            rows,
            day_column=DATE_COLUMN,
            issue_column=ISSUE_COLUMN,
            timezone=timezone,
        )
        if selected.empty:
            return pd.DataFrame()

        pred_col = f"pred_{model_name}"
        pred_cols.append(pred_col)
        if merged is None:
            merged = selected[[DATE_COLUMN, ISSUE_COLUMN, "Forecast"]].rename(
                columns={ISSUE_COLUMN: "execution_time_utc", "Forecast": pred_col}
            )
        else:
            merged = merged.merge(
                selected[[DATE_COLUMN, "Forecast"]].rename(columns={"Forecast": pred_col}),
                on=DATE_COLUMN,
                how="inner",
            )
        if merged.empty:
            return pd.DataFrame()

    if merged is None or merged.empty:
        return pd.DataFrame()

    preds = merged[pred_cols].to_numpy(dtype=float)
    merged["t_hat_median"] = np.median(preds, axis=1)
    merged = merged.sort_values(DATE_COLUMN, kind="mergesort")
    return merged


def build_station_probabilities(
    *,
    station: str,
    station_tz: str,
    station_markets: pd.DataFrame,
    predictions_root: Path,
    model_names: list[str],
    residual_cdf: EmpiricalResidualCdf,
) -> pd.DataFrame:
    eval_daily = build_eval_daily_forecast(
        predictions_root=predictions_root,
        station=station,
        timezone=station_tz,
        model_names=model_names,
    )
    if eval_daily.empty:
        return pd.DataFrame()

    eval_df = station_markets.merge(
        eval_daily[[DATE_COLUMN, "execution_time_utc", "t_hat_median"]],
        left_on="market_day_local",
        right_on=DATE_COLUMN,
        how="inner",
    )
    if eval_df.empty:
        return pd.DataFrame()

    lower = pd.to_numeric(eval_df["lower_c"], errors="coerce").to_numpy(dtype=float) - eval_df["t_hat_median"].to_numpy(dtype=float)
    upper = pd.to_numeric(eval_df["upper_c"], errors="coerce").to_numpy(dtype=float) - eval_df["t_hat_median"].to_numpy(dtype=float)
    eval_df["p_model"] = residual_cdf.interval_prob(lower=lower, upper=upper)
    eval_df["p_model"] = np.clip(eval_df["p_model"].to_numpy(dtype=float), PROB_EPS, 1.0 - PROB_EPS)

    mode_df = (
        eval_df.sort_values(
            ["market_day_local", "event_key", "p_model", "strike_k"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["market_day_local", "event_key"], keep="first")
        [["market_day_local", "event_key", "strike_k"]]
        .rename(columns={"strike_k": "mode_k"})
    )
    eval_df = eval_df.merge(mode_df, on=["market_day_local", "event_key"], how="left")
    eval_df["station_name"] = station
    eval_df["date"] = eval_df["market_day_local"]
    return eval_df


def parse_station_filter(text: str) -> set[str]:
    if not str(text).strip():
        return set()
    return {normalize_station_key(x) for x in str(text).split(",") if x.strip()}


def _tmp_path_for(path: Path) -> Path:
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    tmp_path.replace(path)


def derive_cycle_from_execution_times(series: pd.Series) -> str:
    exec_ts = pd.to_datetime(series, utc=True, errors="coerce").dropna()
    if exec_ts.empty:
        raise SystemExit("Cannot derive cycle token: execution_time_utc is empty.")
    return exec_ts.max().strftime("%Y%m%d%H")


def canonicalize_residual_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["station_name"] = out["station_name"].astype("string").str.strip()
    out["residual"] = pd.to_numeric(out["residual"], errors="coerce")
    out = out.dropna(subset=["station_name", "residual"]).copy()
    out = out.loc[out["station_name"] != ""].copy()
    out = out.sort_values(["station_name", "residual"], kind="mergesort").reset_index(drop=True)
    return out


def load_or_create_calibration_snapshot(
    *,
    output_dir: Path,
    calibration_version: str,
    source_residuals_path: Path,
) -> tuple[pd.DataFrame, Path, dict]:
    calib_root = output_dir / "calibrations" / calibration_version
    snapshot_path = calib_root / "residual_history.parquet"
    manifest_path = calib_root / "manifest.json"

    if snapshot_path.exists():
        frozen = canonicalize_residual_history(load_residual_history(snapshot_path))
        if frozen.empty:
            raise SystemExit(f"Frozen calibration snapshot is empty: {snapshot_path}")

        frozen_manifest = {}
        if manifest_path.exists():
            frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(frozen_manifest, dict):
                frozen_manifest = {}
        if not frozen_manifest:
            frozen_manifest = {
                "schema_version": 1,
                "calibration_version": calibration_version,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "row_count": int(len(frozen)),
                "source_residuals_path": None,
                "snapshot_file": "residual_history.parquet",
                "snapshot_sha256": sha256_file(snapshot_path),
                "status": "frozen",
            }
            atomic_write_json(manifest_path, frozen_manifest)
        return frozen, snapshot_path, frozen_manifest

    source = canonicalize_residual_history(load_residual_history(source_residuals_path))
    if source.empty:
        raise SystemExit(f"Calibration source residual history is empty: {source_residuals_path}")

    atomic_write_parquet(source, snapshot_path)
    frozen_manifest = {
        "schema_version": 1,
        "calibration_version": calibration_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(source)),
        "source_residuals_path": str(source_residuals_path.resolve()),
        "snapshot_file": "residual_history.parquet",
        "snapshot_sha256": sha256_file(snapshot_path),
        "status": "frozen",
    }
    atomic_write_json(manifest_path, frozen_manifest)
    return source, snapshot_path, frozen_manifest


def write_station_outputs(stations_root: Path, all_rows: pd.DataFrame) -> None:
    for station, grp in all_rows.groupby("station_name", sort=True):
        st_dir = stations_root / str(station)
        st_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(grp, st_dir / "market_level_probabilities.parquet")
        atomic_write_csv(grp, st_dir / "market_level_probabilities.csv")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_names = parse_model_names(args.models)
    statuses = parse_statuses(args.statuses)
    calibration_version = parse_calibration_version(args.calibration_version)

    if not args.predictions_root.exists():
        raise SystemExit(f"Predictions root not found: {args.predictions_root}")

    station_timezones = load_station_timezones(args.locations_path)
    prediction_stations = detect_prediction_stations(args.predictions_root, model_names=model_names)
    residual_history, calibration_snapshot_path, calibration_manifest = load_or_create_calibration_snapshot(
        output_dir=args.output_dir,
        calibration_version=calibration_version,
        source_residuals_path=args.residuals_path,
    )
    residual_cdfs = build_station_cdfs(residual_history, min_history=int(args.min_residual_history))
    if not residual_cdfs:
        raise SystemExit("No stations have enough residual history to build CDFs.")

    pred_by_key = {normalize_station_key(s): s for s in sorted(prediction_stations)}
    cdf_by_key = {normalize_station_key(s): s for s in sorted(residual_cdfs)}
    common_keys = sorted(set(pred_by_key) & set(cdf_by_key))
    if not common_keys:
        raise SystemExit("No stations overlap between predictions and residual history.")

    requested_keys = parse_station_filter(args.stations)
    if requested_keys:
        selected_keys = sorted(set(common_keys) & requested_keys)
        missing = sorted(requested_keys - set(common_keys))
        if missing:
            print("Warning: requested stations missing from residual/prediction overlap:", ", ".join(missing))
        if not selected_keys:
            raise SystemExit("No stations selected after applying --stations filter.")
    else:
        selected_keys = common_keys

    station_by_key = {k: pred_by_key[k] for k in selected_keys}
    selected_stations = [station_by_key[k] for k in selected_keys]
    missing_tz = [s for s in selected_stations if s not in station_timezones]
    if missing_tz:
        raise SystemExit(f"Missing timezone mapping for stations in locations.csv: {', '.join(missing_tz)}")

    print(f"Stations selected: {len(selected_stations)} -> {', '.join(selected_stations)}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Statuses: {', '.join(statuses)}")
    print(f"Calibration version: {calibration_version}")
    print(f"Frozen calibration snapshot: {calibration_snapshot_path}")

    from master_db import resolve_master_postgres_dsn

    dsn = resolve_master_postgres_dsn(explicit_dsn=args.master_dsn)
    with psycopg.connect(dsn) as conn:
        raw_markets = fetch_markets(conn=conn, statuses=statuses)
    if raw_markets.empty:
        raise SystemExit(f"No markets found for statuses={statuses}.")

    markets = build_market_frame(
        raw_markets,
        station_timezones=station_timezones,
        station_by_key=station_by_key,
    )
    if markets.empty:
        raise SystemExit("No parseable station markets after slug parsing and timezone mapping.")

    markets = markets.loc[markets["station"].isin(selected_stations)].copy()
    if markets.empty:
        raise SystemExit("No markets left after station filtering.")

    probs_parts: list[pd.DataFrame] = []
    for station in selected_stations:
        station_markets = markets.loc[markets["station"] == station].copy()
        if station_markets.empty:
            continue

        cdf_station_name = cdf_by_key[normalize_station_key(station)]
        station_probs = build_station_probabilities(
            station=station,
            station_tz=station_timezones[station],
            station_markets=station_markets,
            predictions_root=args.predictions_root,
            model_names=model_names,
            residual_cdf=residual_cdfs[cdf_station_name],
        )
        if not station_probs.empty:
            probs_parts.append(station_probs)

    if not probs_parts:
        raise SystemExit("No station probability rows were produced.")

    out = pd.concat(probs_parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["market_day_local"] = pd.to_datetime(out["market_day_local"], errors="coerce").dt.normalize()
    out["execution_time_utc"] = pd.to_datetime(out["execution_time_utc"], utc=True, errors="coerce")
    out["strike_k"] = pd.to_numeric(out["strike_k"], errors="coerce").astype("Int64")
    out["mode_k"] = pd.to_numeric(out["mode_k"], errors="coerce").astype("Int64")
    out = out.dropna(
        subset=[
            "market_id",
            "slug",
            "status",
            "station_name",
            "date",
            "market_day_local",
            "event_key",
            "event_kind",
            "strike_k",
            "mode_k",
            "p_model",
            "execution_time_utc",
        ]
    ).copy()
    out["strike_k"] = out["strike_k"].astype(int)
    out["mode_k"] = out["mode_k"].astype(int)
    out = out.sort_values(["station_name", "date", "event_key", "strike_k", "market_id"], kind="mergesort")
    out = out.drop_duplicates(subset=["market_id"], keep="last")

    output_cols = [
        "market_id",
        "slug",
        "status",
        "station_name",
        "date",
        "market_day_local",
        "event_key",
        "event_kind",
        "strike_k",
        "mode_k",
        "p_model",
        "t_hat_median",
        "execution_time_utc",
        "lower_c",
        "upper_c",
        "end_date_utc",
    ]
    out_final = out[output_cols].copy().rename(columns={"t_hat_median": "T_hat"})

    station_counts = (
        out_final.groupby("station_name", as_index=False)
        .agg(rows=("slug", "size"), days=("date", "nunique"))
        .sort_values("station_name", kind="mergesort")
    )
    station_day_strike_rows = int(
        out_final[["station_name", "date", "strike_k"]].drop_duplicates().shape[0]
    )

    cycle_token = parse_cycle_token(args.cycle) if args.cycle else derive_cycle_from_execution_times(out_final["execution_time_utc"])
    generated_at_utc = datetime.now(timezone.utc).isoformat()

    cycles_root = args.output_dir / "cycles"
    final_cycle_root = cycles_root / cycle_token / calibration_version
    tmp_cycle_root = cycles_root / cycle_token / f".{calibration_version}.{uuid.uuid4().hex}.tmp"
    if tmp_cycle_root.exists():
        shutil.rmtree(tmp_cycle_root, ignore_errors=True)
    tmp_cycle_root.mkdir(parents=True, exist_ok=True)

    cycle_parquet = tmp_cycle_root / "market_level_probabilities.parquet"
    cycle_csv = tmp_cycle_root / "market_level_probabilities.csv"
    cycle_stations_root = tmp_cycle_root / "stations"
    cycle_manifest_path = tmp_cycle_root / "manifest.json"

    atomic_write_parquet(out_final, cycle_parquet)
    atomic_write_csv(out_final, cycle_csv)
    write_station_outputs(cycle_stations_root, out_final)
    new_probabilities_sha256 = sha256_file(cycle_parquet)

    cycle_manifest = {
        "schema_version": 1,
        "artifact_type": "live_market_probabilities",
        "status": "success",
        "cycle": cycle_token,
        "calibration_version": calibration_version,
        "generated_at_utc": generated_at_utc,
        "row_count": int(len(out_final)),
        "station_day_strike_row_count": station_day_strike_rows,
        "station_count": int(out_final["station_name"].nunique()),
        "date_min": str(out_final["date"].min().date()) if not out_final.empty else None,
        "date_max": str(out_final["date"].max().date()) if not out_final.empty else None,
        "models": list(model_names),
        "statuses": list(statuses),
        "min_residual_history": int(args.min_residual_history),
        "predictions_root": str(args.predictions_root.resolve()),
        "residuals_path": str(calibration_snapshot_path.resolve()),
        "calibration_manifest_file": str((Path("calibrations") / calibration_version / "manifest.json").as_posix()),
        "calibration_snapshot_sha256": calibration_manifest.get("snapshot_sha256"),
        "probabilities_sha256": new_probabilities_sha256,
        "probabilities_file": str((Path("cycles") / cycle_token / calibration_version / "market_level_probabilities.parquet").as_posix()),
        "probabilities_csv": str((Path("cycles") / cycle_token / calibration_version / "market_level_probabilities.csv").as_posix()),
        "stations_root": str((Path("cycles") / cycle_token / calibration_version / "stations").as_posix()),
    }
    atomic_write_json(cycle_manifest_path, cycle_manifest)

    existing_cycle_parquet = final_cycle_root / "market_level_probabilities.parquet"
    existing_cycle_manifest = final_cycle_root / "manifest.json"
    if existing_cycle_parquet.exists():
        existing_hash = sha256_file(existing_cycle_parquet)
        new_hash = new_probabilities_sha256
        if existing_hash != new_hash:
            shutil.rmtree(tmp_cycle_root, ignore_errors=True)
            raise SystemExit(
                "Determinism check failed for cycle+calibration_version: "
                f"cycle={cycle_token} calibration_version={calibration_version} "
                f"existing_sha256={existing_hash} new_sha256={new_hash}"
            )
        # Keep existing immutable cycle artifact when deterministic match is confirmed.
        shutil.rmtree(tmp_cycle_root, ignore_errors=True)
        if existing_cycle_manifest.exists():
            loaded_manifest = json.loads(existing_cycle_manifest.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                cycle_manifest = loaded_manifest
    else:
        final_cycle_root.parent.mkdir(parents=True, exist_ok=True)
        tmp_cycle_root.replace(final_cycle_root)

    # Keep a convenience "latest snapshot" copy for legacy consumers.
    atomic_write_parquet(out_final, args.output_dir / "market_level_probabilities.parquet")
    atomic_write_csv(out_final, args.output_dir / "market_level_probabilities.csv")

    latest_manifest = dict(cycle_manifest)
    latest_manifest["cycle_manifest_file"] = str((Path("cycles") / cycle_token / calibration_version / "manifest.json").as_posix())
    atomic_write_json(args.output_dir / DEFAULT_MANIFEST_NAME, latest_manifest)

    print("\nStation probability coverage")
    print(station_counts.to_string(index=False))
    print(f"\nCycle: {cycle_token}")
    print(f"Calibration version: {calibration_version}")
    if cycle_manifest.get("probabilities_sha256") is not None:
        print(f"Probabilities sha256: {cycle_manifest.get('probabilities_sha256')}")
    print(f"Latest manifest: {args.output_dir / DEFAULT_MANIFEST_NAME}")
    print(f"\nStation/day/strike rows: {station_day_strike_rows:,}")
    print(f"\nTotal rows: {len(out_final):,}")
    print(f"Output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
