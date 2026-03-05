#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROB_PATH = REPO_ROOT / "reports" / "probability_backtest" / "market_level_probabilities.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "trading_backtest_multi"
DEFAULT_PREDICTIONS_ROOT = REPO_ROOT / "data" / "ml_predictions"
LONDON_TZ = "Europe/London"
INITIAL_NAV = 10_000.0

STRIKE_SUFFIX_RE = re.compile(r"-(?:neg-\d+|\d+)c$|-(?:\d+-\d+f|\d+forbelow|\d+forhigher|\d+f)$")
STATION_FROM_SLUG_RE = re.compile(r"^highest-temperature-in-([a-z0-9-]+)-on-")
GENERIC_DIR_NAMES = {
    "reports",
    "report",
    "data",
    "probabilitybacktest",
    "backtest",
    "output",
    "outputs",
}

LONDON_KEY = "london"
NYC_KEY = "nyc"
BUENOS_AIRES_KEY = "buenosaires"


def parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint-2-scale multi-station NO-only trading backtest.")
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--decision-cycle-model", type=str, default="city_extended")
    parser.add_argument("--master-dsn", type=str, default=None)
    parser.add_argument("--stations", type=str, default="", help='Optional comma-separated station list.')
    parser.add_argument("--nav-mode", choices=["per-station", "shared"], default="per-station")

    parser.add_argument("--trade-lookback-hours-primary", type=float, default=24.0)
    parser.add_argument("--trade-lookback-hours-fallback", type=float, default=48.0)
    parser.add_argument("--yes-fallback-slippage-buffer", type=float, default=0.01)

    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--tail-prob-threshold", type=float, default=0.12)
    parser.add_argument("--max-no-price", type=float, default=0.92)
    parser.add_argument("--max-trades-per-day", type=int, default=2)

    parser.add_argument("--stake-pct", type=float, default=0.005)
    parser.add_argument("--stake-cap", type=float, default=50.0)
    parser.add_argument("--max-daily-risk-pct", type=float, default=0.02)

    # Station-specific Sprint-2-scale patch controls.
    parser.add_argument("--london-min-edge-per-risk", type=float, default=0.02)
    parser.add_argument("--london-max-price", type=float, default=0.99)

    parser.add_argument("--nyc-max-trade-age-hours", type=float, default=12.0)
    parser.add_argument("--buenosaires-max-trade-age-hours", type=float, default=12.0)
    parser.add_argument("--nyc-allow-yes-fallback", type=parse_bool, default=False)
    parser.add_argument("--buenosaires-allow-yes-fallback", type=parse_bool, default=False)
    parser.add_argument("--nyc-yes-fallback-slippage-buffer", type=float, default=0.02)
    parser.add_argument("--buenosaires-yes-fallback-slippage-buffer", type=float, default=0.02)
    parser.add_argument("--nyc-edge-threshold", type=float, default=0.03)
    parser.add_argument("--buenosaires-edge-threshold", type=float, default=0.03)
    return parser.parse_args(argv)


def fetch_dataframe(conn: psycopg.Connection, query: str, params: dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def normalize_station_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def derive_event_key(slug: str) -> str:
    return STRIKE_SUFFIX_RE.sub("", str(slug))


def extract_station_from_slug(slug: str) -> str | None:
    match = STATION_FROM_SLUG_RE.search(str(slug).strip().lower())
    if not match:
        return None
    return match.group(1)


def derive_station_from_source_path(source_path: str) -> str | None:
    p = Path(str(source_path))
    parents = [p.parent.name, p.parent.parent.name]
    for cand in parents:
        key = normalize_station_key(cand)
        if key and key not in GENERIC_DIR_NAMES:
            return cand

    stem = p.stem.lower()
    stem = re.sub(r"market[_-]?level[_-]?probabilities", "", stem).strip("-_ ")
    if normalize_station_key(stem):
        return stem
    return None


def read_probability_files(path: Path) -> pd.DataFrame:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        patterns = [
            "**/market_level_probabilities.parquet",
            "**/market_level_probabilities.csv",
            "**/*market_level_probabilities*.parquet",
            "**/*market_level_probabilities*.csv",
        ]
        files = []
        seen: set[str] = set()
        for pat in patterns:
            for f in sorted(path.glob(pat)):
                key = str(f.resolve())
                if key in seen:
                    continue
                seen.add(key)
                files.append(f)
    else:
        raise SystemExit(f"Probabilities path not found: {path}")

    if not files:
        raise SystemExit(f"No market_level_probabilities files found under: {path}")

    parts: list[pd.DataFrame] = []
    for f in files:
        if f.suffix.lower() == ".parquet":
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f)
        df["__source_path"] = str(f)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def standardize_probabilities(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()

    if "slug" not in out.columns:
        raise SystemExit("Probability input must contain slug column.")
    out["slug"] = out["slug"].astype("string")

    station_col = None
    for c in ["station_name", "station", "city_name", "city"]:
        if c in out.columns:
            station_col = c
            break

    if station_col is not None:
        out["station"] = out[station_col].astype("string").str.strip()
    else:
        out["station"] = out["slug"].map(extract_station_from_slug)
        src_station = pd.Series(out["__source_path"].astype(str)).map(derive_station_from_source_path)
        out["station"] = out["station"].fillna(src_station)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    elif "market_day_local" in out.columns:
        out["date"] = pd.to_datetime(out["market_day_local"], errors="coerce").dt.normalize()
    else:
        raise SystemExit("Probability input must include date or market_day_local.")

    if "strike_k" in out.columns:
        out["strike_k"] = pd.to_numeric(out["strike_k"], errors="coerce").astype("Int64")
    elif "strike" in out.columns:
        out["strike_k"] = pd.to_numeric(out["strike"], errors="coerce").astype("Int64")
    else:
        raise SystemExit("Probability input must include strike_k or strike.")

    if "p_model" in out.columns:
        out["p_model"] = pd.to_numeric(out["p_model"], errors="coerce")
    elif "p_model_residual" in out.columns:
        out["p_model"] = pd.to_numeric(out["p_model_residual"], errors="coerce")
    else:
        raise SystemExit("Probability input must include p_model or p_model_residual.")

    if "mode_k" in out.columns:
        out["mode_k"] = pd.to_numeric(out["mode_k"], errors="coerce").astype("Int64")
    else:
        out["mode_k"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    if "y" in out.columns:
        out["y"] = pd.to_numeric(out["y"], errors="coerce").astype("Int64")
    elif "t_round_obs" in out.columns:
        t_round = pd.to_numeric(out["t_round_obs"], errors="coerce")
        out["y"] = (t_round == pd.to_numeric(out["strike_k"], errors="coerce")).astype("Int64")
    else:
        raise SystemExit("Probability input must include y or t_round_obs.")

    out["market_id"] = out["market_id"].astype("string") if "market_id" in out.columns else pd.Series([pd.NA] * len(out), dtype="string")
    out["event_key"] = out["slug"].map(derive_event_key)

    out["execution_time_local_raw"] = pd.Series([pd.NA] * len(out), dtype="object")
    out["execution_time_utc_raw"] = pd.Series([pd.NA] * len(out), dtype="object")
    for c in ["execution_time_local", "decision_cycle_time_local"]:
        if c in out.columns:
            out["execution_time_local_raw"] = out[c]
            break
    for c in ["execution_time_utc", "decision_cycle_time_utc"]:
        if c in out.columns:
            out["execution_time_utc_raw"] = out[c]
            break

    out = out.dropna(subset=["station", "date", "slug", "strike_k", "p_model", "y"]).copy()
    out["station"] = out["station"].astype(str).str.strip()
    out["station_key"] = out["station"].map(normalize_station_key)
    out = out.loc[out["station_key"] != ""].copy()
    out["strike_k"] = out["strike_k"].astype(int)
    out["y"] = out["y"].astype(int)

    # Derive mode_k when missing using max p_model within (station, date, event_key).
    missing_mode = out["mode_k"].isna()
    if missing_mode.any():
        mode_df = (
            out.sort_values(["station_key", "date", "event_key", "p_model", "strike_k"], ascending=[True, True, True, False, True], kind="mergesort")
            .drop_duplicates(subset=["station_key", "date", "event_key"], keep="first")
            [["station_key", "date", "event_key", "strike_k"]]
            .rename(columns={"strike_k": "mode_k_derived"})
        )
        out = out.merge(mode_df, on=["station_key", "date", "event_key"], how="left")
        out.loc[missing_mode, "mode_k"] = out.loc[missing_mode, "mode_k_derived"]
        out = out.drop(columns=["mode_k_derived"])
    out["mode_k"] = pd.to_numeric(out["mode_k"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["mode_k"]).copy()
    out["mode_k"] = out["mode_k"].astype(int)

    out = out.sort_values(["station_key", "date", "event_key", "strike_k", "slug"], kind="mergesort")
    out = out.drop_duplicates(subset=["station_key", "date", "event_key", "strike_k", "slug"], keep="last")
    return out


def filter_stations(prob: pd.DataFrame, stations_arg: str) -> tuple[pd.DataFrame, list[str]]:
    detected = (
        prob[["station", "station_key"]]
        .drop_duplicates()
        .sort_values(["station_key", "station"], kind="mergesort")
    )

    if not stations_arg.strip():
        selected_keys = detected["station_key"].tolist()
    else:
        requested = [s.strip() for s in stations_arg.split(",") if s.strip()]
        req_keys = [normalize_station_key(x) for x in requested]
        selected_keys = sorted(set(req_keys) & set(detected["station_key"].tolist()))
        missing = sorted(set(req_keys) - set(detected["station_key"].tolist()))
        if missing:
            print(f"Warning: requested stations not found in probabilities: {', '.join(missing)}")

    if not selected_keys:
        raise SystemExit("No stations selected after filtering.")

    out = prob.loc[prob["station_key"].isin(selected_keys)].copy()
    names = (
        out[["station", "station_key"]]
        .drop_duplicates()
        .sort_values(["station_key", "station"], kind="mergesort")
    )
    stations = names["station"].tolist()
    return out, stations


def resolve_market_ids_from_slug(conn: psycopg.Connection, prob: pd.DataFrame) -> pd.DataFrame:
    out = prob.copy()
    need = out["market_id"].isna()
    if not need.any():
        out["market_id"] = out["market_id"].astype(str)
        return out

    slugs = sorted(out.loc[need, "slug"].dropna().astype(str).unique().tolist())
    if not slugs:
        out["market_id"] = out["market_id"].astype("string")
        return out

    q = """
        SELECT market_id::text AS market_id, slug
        FROM markets
        WHERE slug = ANY(%(slugs)s)
    """
    m = fetch_dataframe(conn, q, params={"slugs": slugs})
    if m.empty:
        out["market_id"] = out["market_id"].astype("string")
        return out

    out = out.merge(m, on="slug", how="left", suffixes=("", "_map"))
    out["market_id"] = out["market_id"].fillna(out["market_id_map"])
    out = out.drop(columns=["market_id_map"])
    out["market_id"] = out["market_id"].astype("string")
    return out


def parse_execution_local_to_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(LONDON_TZ).dt.tz_convert("UTC")
    return parsed.dt.tz_convert("UTC")


def derive_execution_times(
    *,
    prob: pd.DataFrame,
    predictions_root: Path,
    decision_cycle_model: str,
) -> pd.DataFrame:
    base = prob[["station", "station_key", "date", "event_key"]].drop_duplicates().copy()

    # 1) Use directly provided execution times if present.
    local_raw = prob["execution_time_local_raw"] if "execution_time_local_raw" in prob.columns else pd.Series([pd.NA] * len(prob))
    utc_raw = prob["execution_time_utc_raw"] if "execution_time_utc_raw" in prob.columns else pd.Series([pd.NA] * len(prob))

    has_local = local_raw.notna().any()
    has_utc = utc_raw.notna().any()
    if has_local or has_utc:
        tmp = prob[["station", "station_key", "date", "event_key"]].copy()
        if has_local:
            tmp["execution_time_utc"] = parse_execution_local_to_utc(local_raw)
        else:
            tmp["execution_time_utc"] = pd.to_datetime(utc_raw, utc=True, errors="coerce")
        out = (
            tmp.dropna(subset=["execution_time_utc"])
            .sort_values(["station_key", "date", "event_key", "execution_time_utc"], kind="mergesort")
            .drop_duplicates(subset=["station_key", "date", "event_key"], keep="last")
        )
        out = base.merge(out[["station_key", "date", "event_key", "execution_time_utc"]], on=["station_key", "date", "event_key"], how="left")
    else:
        # 2) Derive from predictions via Sprint-1 decision-cycle rule.
        model_dir = predictions_root / decision_cycle_model
        if not model_dir.exists():
            raise SystemExit(f"Decision-cycle model directory not found: {model_dir}")

        dir_map: dict[str, str] = {}
        for d in sorted(model_dir.iterdir()):
            if d.is_dir():
                dir_map[normalize_station_key(d.name)] = d.name

        out_parts: list[pd.DataFrame] = []
        for station_key, need in base.groupby("station_key", sort=True):
            station_dir_name = dir_map.get(station_key)
            if station_dir_name is None:
                continue
            station_dir = model_dir / station_dir_name
            files = sorted(station_dir.glob("*.parquet"))
            if not files:
                continue

            parts: list[pd.DataFrame] = []
            for f in files:
                part = pd.read_parquet(f, columns=["issue_time_utc", "target_date_local"])
                parts.append(part)
            pred = pd.concat(parts, ignore_index=True)
            pred["issue_time_utc"] = pd.to_datetime(pred["issue_time_utc"], utc=True, errors="coerce")
            pred["date"] = pd.to_datetime(pred["target_date_local"], errors="coerce").dt.normalize()
            pred = pred.dropna(subset=["issue_time_utc", "date"]).copy()
            pred = pred.sort_values(["date", "issue_time_utc"], kind="mergesort").drop_duplicates(
                subset=["date", "issue_time_utc"], keep="last"
            )

            need_days = need[["date"]].drop_duplicates().copy()
            need_days["day_start_utc"] = pd.to_datetime(need_days["date"]).dt.tz_localize(LONDON_TZ).dt.tz_convert("UTC")
            cand = pred.merge(need_days, on="date", how="inner")
            cand = cand.loc[cand["issue_time_utc"] < cand["day_start_utc"]].copy()
            if cand.empty:
                continue
            day_sel = (
                cand.sort_values(["date", "issue_time_utc"], kind="mergesort")
                .drop_duplicates(subset=["date"], keep="last")
                .rename(columns={"issue_time_utc": "execution_time_utc"})
            )

            station_exec = need[["station", "station_key", "date", "event_key"]].merge(
                day_sel[["date", "execution_time_utc"]], on="date", how="left"
            )
            out_parts.append(station_exec)

        if out_parts:
            out = pd.concat(out_parts, ignore_index=True)
            out = base.merge(out[["station_key", "date", "event_key", "execution_time_utc"]], on=["station_key", "date", "event_key"], how="left")
        else:
            out = base.copy()
            out["execution_time_utc"] = pd.NaT

    out["execution_time_utc"] = pd.to_datetime(out["execution_time_utc"], utc=True, errors="coerce")
    out["execution_time_local"] = out["execution_time_utc"].dt.tz_convert(LONDON_TZ)
    return out


def load_trade_prices(
    *,
    conn: psycopg.Connection,
    execution_market: pd.DataFrame,
    lookback_primary_hours: float,
    lookback_fallback_hours: float,
) -> pd.DataFrame:
    market_ids = sorted(execution_market["market_id"].dropna().astype(str).unique().tolist())
    if not market_ids:
        return pd.DataFrame(columns=[
            "market_id",
            "yes_trade_price",
            "no_trade_price",
            "yes_trade_ts",
            "no_trade_ts",
            "yes_age_hours",
            "no_age_hours",
            "yes_lookback",
            "no_lookback",
        ])

    trade_q = """
        SELECT
            t.trade_id,
            t.market_id::text AS market_id,
            t.ts,
            t.outcome_index,
            t.price::double precision AS price,
            m.event_start_time
        FROM trades t
        JOIN markets m ON m.market_id = t.market_id
        WHERE t.market_id = ANY(%(market_ids)s)
          AND t.outcome_index IN (0, 1)
    """
    trades = fetch_dataframe(conn, trade_q, params={"market_ids": market_ids})
    if trades.empty:
        return pd.DataFrame(columns=[
            "market_id",
            "yes_trade_price",
            "no_trade_price",
            "yes_trade_ts",
            "no_trade_ts",
            "yes_age_hours",
            "no_age_hours",
            "yes_lookback",
            "no_lookback",
        ])

    trades["market_id"] = trades["market_id"].astype(str)
    trades["ts"] = pd.to_datetime(trades["ts"], utc=True, errors="coerce")
    trades["event_start_time"] = pd.to_datetime(trades["event_start_time"], utc=True, errors="coerce")
    trades["outcome_index"] = pd.to_numeric(trades["outcome_index"], errors="coerce").astype("Int64")
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades = trades.dropna(subset=["market_id", "ts", "outcome_index", "price"]).copy()
    trades["outcome_index"] = trades["outcome_index"].astype(int)

    exec_tbl = execution_market[["market_id", "execution_time_utc"]].drop_duplicates().copy()
    exec_tbl["market_id"] = exec_tbl["market_id"].astype(str)
    exec_tbl["execution_time_utc"] = pd.to_datetime(exec_tbl["execution_time_utc"], utc=True, errors="coerce")

    trades = trades.merge(exec_tbl, on="market_id", how="inner")
    fallback_td = pd.Timedelta(hours=float(lookback_fallback_hours))
    primary_td = pd.Timedelta(hours=float(lookback_primary_hours))

    trades["window_start_utc"] = trades["execution_time_utc"] - fallback_td
    trades["primary_start_utc"] = trades["execution_time_utc"] - primary_td

    # strict pre-execution, optional market-open lower bound
    mask_time = (trades["ts"] < trades["execution_time_utc"]) & (trades["ts"] >= trades["window_start_utc"])
    mask_open = trades["event_start_time"].isna() | (trades["ts"] >= trades["event_start_time"])
    trades = trades.loc[mask_time & mask_open].copy()
    if trades.empty:
        return pd.DataFrame(columns=[
            "market_id",
            "yes_trade_price",
            "no_trade_price",
            "yes_trade_ts",
            "no_trade_ts",
            "yes_age_hours",
            "no_age_hours",
            "yes_lookback",
            "no_lookback",
        ])

    trades["in_primary_window"] = trades["ts"] >= trades["primary_start_utc"]
    trades["lookback_used_hours"] = np.where(trades["in_primary_window"], float(lookback_primary_hours), float(lookback_fallback_hours))
    trades["age_hours"] = (trades["execution_time_utc"] - trades["ts"]).dt.total_seconds() / 3600.0

    last = trades.sort_values(
        ["market_id", "outcome_index", "in_primary_window", "ts"],
        ascending=[True, True, False, False],
        kind="mergesort",
    ).drop_duplicates(subset=["market_id", "outcome_index"], keep="first")

    price_p = last.pivot(index="market_id", columns="outcome_index", values="price")
    ts_p = last.pivot(index="market_id", columns="outcome_index", values="ts")
    age_p = last.pivot(index="market_id", columns="outcome_index", values="age_hours")
    look_p = last.pivot(index="market_id", columns="outcome_index", values="lookback_used_hours")

    out = pd.DataFrame({"market_id": price_p.index.astype(str)})
    out["yes_trade_price"] = price_p.get(0, np.nan).to_numpy(dtype=float)
    out["no_trade_price"] = price_p.get(1, np.nan).to_numpy(dtype=float)
    out["yes_trade_ts"] = ts_p.get(0, pd.NaT).to_numpy()
    out["no_trade_ts"] = ts_p.get(1, pd.NaT).to_numpy()
    out["yes_age_hours"] = age_p.get(0, np.nan).to_numpy(dtype=float)
    out["no_age_hours"] = age_p.get(1, np.nan).to_numpy(dtype=float)
    out["yes_lookback"] = look_p.get(0, np.nan).to_numpy(dtype=float)
    out["no_lookback"] = look_p.get(1, np.nan).to_numpy(dtype=float)

    return out[[
        "market_id",
        "yes_trade_price",
        "no_trade_price",
        "yes_trade_ts",
        "no_trade_ts",
        "yes_age_hours",
        "no_age_hours",
        "yes_lookback",
        "no_lookback",
    ]]


def apply_filters_and_select(
    universe: pd.DataFrame,
    *,
    tail_prob_threshold: float,
    edge_threshold: float,
    max_no_price: float,
    max_trades_per_day: int,
    yes_fallback_slippage_buffer: float,
    london_min_edge_per_risk: float,
    london_max_price: float,
    nyc_max_trade_age_hours: float,
    buenosaires_max_trade_age_hours: float,
    nyc_allow_yes_fallback: bool,
    buenosaires_allow_yes_fallback: bool,
    nyc_yes_fallback_slippage_buffer: float,
    buenosaires_yes_fallback_slippage_buffer: float,
    nyc_edge_threshold: float,
    buenosaires_edge_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = universe.copy()

    is_london = df["station_key"] == LONDON_KEY
    is_nyc = df["station_key"] == NYC_KEY
    is_buenosaires = df["station_key"] == BUENOS_AIRES_KEY
    is_target_noisy = is_nyc | is_buenosaires

    allow_yes_fallback = np.ones(len(df), dtype=bool)
    allow_yes_fallback[is_nyc.to_numpy()] = bool(nyc_allow_yes_fallback)
    allow_yes_fallback[is_buenosaires.to_numpy()] = bool(buenosaires_allow_yes_fallback)
    df["allow_yes_fallback"] = allow_yes_fallback

    yes_slippage = np.full(len(df), float(yes_fallback_slippage_buffer), dtype=float)
    yes_slippage[is_nyc.to_numpy()] = float(nyc_yes_fallback_slippage_buffer)
    yes_slippage[is_buenosaires.to_numpy()] = float(buenosaires_yes_fallback_slippage_buffer)
    df["yes_fallback_slippage_buffer_effective"] = yes_slippage

    has_no_trade = df["no_trade_price"].notna()
    has_yes_trade = df["yes_trade_price"].notna()
    can_use_yes_fallback = (~has_no_trade) & has_yes_trade & df["allow_yes_fallback"]

    df["price"] = np.nan
    df["price_source"] = pd.Series([None] * len(df), dtype="object")
    df["selected_trade_timestamp_utc"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    df["selected_trade_age_hours"] = np.nan
    df["lookback_used_hours"] = np.nan

    df.loc[has_no_trade, "price"] = pd.to_numeric(df.loc[has_no_trade, "no_trade_price"], errors="coerce")
    df.loc[has_no_trade, "price_source"] = "NO_trade"
    df.loc[has_no_trade, "selected_trade_timestamp_utc"] = df.loc[has_no_trade, "no_trade_ts"]
    df.loc[has_no_trade, "selected_trade_age_hours"] = pd.to_numeric(df.loc[has_no_trade, "no_age_hours"], errors="coerce")
    df.loc[has_no_trade, "lookback_used_hours"] = pd.to_numeric(df.loc[has_no_trade, "no_lookback"], errors="coerce")

    if can_use_yes_fallback.any():
        yes_price = pd.to_numeric(df.loc[can_use_yes_fallback, "yes_trade_price"], errors="coerce").to_numpy(dtype=float)
        yes_slip = df.loc[can_use_yes_fallback, "yes_fallback_slippage_buffer_effective"].to_numpy(dtype=float)
        fallback_price = np.minimum(1.0, 1.0 - yes_price + yes_slip)
        df.loc[can_use_yes_fallback, "price"] = fallback_price
        df.loc[can_use_yes_fallback, "price_source"] = "YES_fallback"
        df.loc[can_use_yes_fallback, "selected_trade_timestamp_utc"] = df.loc[can_use_yes_fallback, "yes_trade_ts"]
        df.loc[can_use_yes_fallback, "selected_trade_age_hours"] = pd.to_numeric(df.loc[can_use_yes_fallback, "yes_age_hours"], errors="coerce")
        df.loc[can_use_yes_fallback, "lookback_used_hours"] = pd.to_numeric(df.loc[can_use_yes_fallback, "yes_lookback"], errors="coerce")

    df["station_overrides_applied"] = ""
    df.loc[is_london, "station_overrides_applied"] = "london_edge_per_risk"
    df.loc[is_nyc, "station_overrides_applied"] = (
        "nyc_trade_age,nyc_edge_threshold,"
        + ("nyc_yes_fallback" if nyc_allow_yes_fallback else "nyc_no_yes_fallback")
    )
    df.loc[is_buenosaires, "station_overrides_applied"] = (
        "buenosaires_trade_age,buenosaires_edge_threshold,"
        + ("buenosaires_yes_fallback" if buenosaires_allow_yes_fallback else "buenosaires_no_yes_fallback")
    )

    df["NO_true"] = 1.0 - pd.to_numeric(df["p_model"], errors="coerce")
    df["edge"] = df["NO_true"] - pd.to_numeric(df["price"], errors="coerce")
    df["edge_per_risk"] = np.where(pd.to_numeric(df["price"], errors="coerce") > 0.0, df["edge"] / df["price"], np.nan)

    df["pass_mode_distance"] = (df["strike_k"] - df["mode_k"]).abs() >= 2
    df["pass_p_model"] = df["p_model"] <= float(tail_prob_threshold)
    df["pass_candidate"] = df["pass_mode_distance"] & df["pass_p_model"]

    df["pass_price_available"] = df["pass_candidate"] & df["price"].notna()

    df["removed_by_no_trade_only"] = (
        df["pass_candidate"]
        & is_target_noisy
        & ~df["allow_yes_fallback"]
        & ~has_no_trade
        & has_yes_trade
    )

    trade_age_max = np.full(len(df), np.inf, dtype=float)
    trade_age_max[is_nyc.to_numpy()] = float(nyc_max_trade_age_hours)
    trade_age_max[is_buenosaires.to_numpy()] = float(buenosaires_max_trade_age_hours)
    df["max_trade_age_hours_effective"] = trade_age_max
    df["pass_trade_age"] = df["pass_price_available"] & (pd.to_numeric(df["selected_trade_age_hours"], errors="coerce") <= trade_age_max)
    df["removed_by_trade_age"] = df["pass_price_available"] & ~df["pass_trade_age"] & is_target_noisy

    edge_threshold_effective = np.full(len(df), float(edge_threshold), dtype=float)
    edge_threshold_effective[is_nyc.to_numpy()] = float(nyc_edge_threshold)
    edge_threshold_effective[is_buenosaires.to_numpy()] = float(buenosaires_edge_threshold)
    df["edge_threshold_effective"] = edge_threshold_effective
    df["pass_edge"] = df["pass_trade_age"] & (df["edge"] >= edge_threshold_effective)
    df["removed_by_station_edge_threshold"] = df["pass_trade_age"] & ~df["pass_edge"] & is_target_noisy

    df["pass_edge_per_risk"] = df["pass_edge"]
    london_mask = is_london & df["pass_edge"]
    if london_mask.any():
        df.loc[london_mask, "pass_edge_per_risk"] = df.loc[london_mask, "edge_per_risk"] >= float(london_min_edge_per_risk)
    df["removed_by_edge_per_risk"] = is_london & df["pass_edge"] & ~df["pass_edge_per_risk"]

    pass_price_cap_other = (~is_london) & df["pass_edge_per_risk"] & (df["price"] <= float(max_no_price))
    pass_price_cap_london = is_london & df["pass_edge_per_risk"] & (df["price"] <= float(london_max_price))
    df["pass_price_cap"] = pass_price_cap_other | pass_price_cap_london

    top_pool = df.loc[df["pass_price_cap"]].copy()
    if not top_pool.empty:
        ranked = top_pool.sort_values(
            ["station_key", "date", "event_key", "edge", "strike_k"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        ranked["rank_top"] = ranked.groupby(["station_key", "date", "event_key"], sort=False).cumcount() + 1
        top_idx = ranked.index[ranked["rank_top"] <= int(max_trades_per_day)]
    else:
        top_idx = pd.Index([])

    df["pass_top2"] = df.index.isin(top_idx)

    df["skipped_reason"] = "selected"
    df.loc[~df["pass_mode_distance"], "skipped_reason"] = "mode_distance"
    df.loc[df["pass_mode_distance"] & ~df["pass_p_model"], "skipped_reason"] = "p_model"
    df.loc[df["pass_candidate"] & ~df["pass_price_available"], "skipped_reason"] = "price_unavailable"
    df.loc[df["removed_by_no_trade_only"], "skipped_reason"] = "no_trade_only_rule"
    df.loc[df["removed_by_trade_age"], "skipped_reason"] = "trade_age"
    df.loc[df["pass_trade_age"] & ~df["pass_edge"], "skipped_reason"] = "edge_too_low"
    df.loc[df["removed_by_station_edge_threshold"], "skipped_reason"] = "station_edge_threshold"
    df.loc[df["removed_by_edge_per_risk"], "skipped_reason"] = "edge_per_risk"
    df.loc[df["pass_edge_per_risk"] & ~df["pass_price_cap"], "skipped_reason"] = "price_too_high"
    df.loc[df["pass_price_cap"] & ~df["pass_top2"], "skipped_reason"] = "not_top2"

    selected = df.loc[df["pass_top2"]].copy()

    station_filter_breakdown = (
        df.groupby(["station", "station_key"], as_index=False)
        .agg(
            candidates_total=("station_key", "size"),
            after_mode_distance=("pass_mode_distance", "sum"),
            after_p_model=("pass_candidate", "sum"),
            after_price_available=("pass_price_available", "sum"),
            after_edge=("pass_edge", "sum"),
            after_price_cap=("pass_price_cap", "sum"),
            after_top2=("pass_top2", "sum"),
            removed_by_trade_age=("removed_by_trade_age", "sum"),
            removed_by_no_trade_only=("removed_by_no_trade_only", "sum"),
            removed_by_edge_per_risk=("removed_by_edge_per_risk", "sum"),
            removed_by_station_edge_threshold=("removed_by_station_edge_threshold", "sum"),
        )
        .sort_values(["station_key", "station"], kind="mergesort")
        .reset_index(drop=True)
    )

    return selected, df, station_filter_breakdown


def run_station_backtest(
    selected: pd.DataFrame,
    *,
    all_days: pd.Series,
    initial_nav: float,
    stake_pct: float,
    stake_cap: float,
    max_daily_risk_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nav = float(initial_nav)
    selected = selected.sort_values(["date", "edge", "strike_k"], ascending=[True, False, True], kind="mergesort")
    by_day = {d: g.copy() for d, g in selected.groupby("date", sort=True)}

    logs: list[pd.DataFrame] = []
    daily_rows: list[dict] = []

    for day in sorted(pd.to_datetime(all_days).dt.normalize().unique()):
        day = pd.Timestamp(day)
        nav_start = nav
        stake = min(nav_start * float(stake_pct), float(stake_cap))
        risk_budget = nav_start * float(max_daily_risk_pct)

        d = by_day.get(day)
        if d is None or d.empty:
            daily_rows.append(
                {
                    "date": day,
                    "nav_start": nav_start,
                    "daily_risk_budget": risk_budget,
                    "daily_exposure": 0.0,
                    "day_pnl": 0.0,
                    "trades": 0,
                    "nav_end": nav_start,
                }
            )
            continue

        w = d.copy()
        w["size"] = stake
        w["exposure"] = w["price"] * w["size"]
        w["cum_exposure"] = w["exposure"].cumsum()
        exec_df = w.loc[w["cum_exposure"] <= risk_budget].copy()

        if exec_df.empty:
            daily_rows.append(
                {
                    "date": day,
                    "nav_start": nav_start,
                    "daily_risk_budget": risk_budget,
                    "daily_exposure": 0.0,
                    "day_pnl": 0.0,
                    "trades": 0,
                    "nav_end": nav_start,
                }
            )
            continue

        exec_df["pnl"] = np.where(
            exec_df["y"] == 0,
            (1.0 - exec_df["price"]) * exec_df["size"],
            -exec_df["price"] * exec_df["size"],
        )
        day_pnl = float(exec_df["pnl"].sum())
        day_exposure = float(exec_df["exposure"].sum())
        nav = nav_start + day_pnl

        logs.append(exec_df)
        daily_rows.append(
            {
                "date": day,
                "nav_start": nav_start,
                "daily_risk_budget": risk_budget,
                "daily_exposure": day_exposure,
                "day_pnl": day_pnl,
                "trades": int(len(exec_df)),
                "nav_end": nav,
            }
        )

    log_df = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=list(selected.columns) + ["size", "exposure", "cum_exposure", "pnl"])
    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty:
        daily_df = pd.DataFrame(columns=["date", "nav_start", "daily_risk_budget", "daily_exposure", "day_pnl", "trades", "nav_end", "daily_return"])
    else:
        daily_df["daily_return"] = np.where(daily_df["nav_start"] > 0, daily_df["day_pnl"] / daily_df["nav_start"], 0.0)
    return log_df, daily_df


def run_backtest_per_station(
    selected: pd.DataFrame,
    base_universe: pd.DataFrame,
    *,
    initial_nav: float,
    stake_pct: float,
    stake_cap: float,
    max_daily_risk_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []

    for station_key, g in base_universe.groupby("station_key", sort=True):
        station_name = g["station"].iloc[0]
        all_days = g["date"].drop_duplicates().sort_values()
        sel = selected.loc[selected["station_key"] == station_key].copy()

        tlog, dd = run_station_backtest(
            sel,
            all_days=all_days,
            initial_nav=initial_nav,
            stake_pct=stake_pct,
            stake_cap=stake_cap,
            max_daily_risk_pct=max_daily_risk_pct,
        )
        if not tlog.empty:
            trade_parts.append(tlog)
        dd["station"] = station_name
        dd["station_key"] = station_key
        daily_parts.append(dd)

    trade_log = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(columns=list(selected.columns) + ["size", "exposure", "cum_exposure", "pnl", "station", "station_key"])
    daily_log = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame(columns=["date", "nav_start", "daily_risk_budget", "daily_exposure", "day_pnl", "trades", "nav_end", "daily_return", "station", "station_key"])
    return trade_log, daily_log


def run_backtest_shared_nav(
    selected: pd.DataFrame,
    base_universe: pd.DataFrame,
    *,
    initial_nav: float,
    stake_pct: float,
    stake_cap: float,
    max_daily_risk_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # shared NAV, but daily risk cap is applied per station from shared NAV at day start
    nav = float(initial_nav)
    selected = selected.sort_values(["date", "station_key", "edge", "strike_k"], ascending=[True, True, False, True], kind="mergesort")
    by_day = {d: g.copy() for d, g in selected.groupby("date", sort=True)}

    all_days = sorted(pd.to_datetime(base_universe["date"]).dt.normalize().unique())
    trade_logs: list[pd.DataFrame] = []
    daily_rows: list[dict] = []

    station_daily_rows: list[dict] = []
    stations = base_universe[["station", "station_key"]].drop_duplicates().sort_values("station_key")

    for day in all_days:
        day = pd.Timestamp(day)
        nav_start = nav
        stake = min(nav_start * float(stake_pct), float(stake_cap))
        risk_per_station = nav_start * float(max_daily_risk_pct)

        day_sel = by_day.get(day)
        if day_sel is None or day_sel.empty:
            day_pnl = 0.0
            day_exposure = 0.0
            day_trades = 0
            nav_end = nav_start
            for row in stations.itertuples(index=False):
                station_daily_rows.append({"date": day, "station": row.station, "station_key": row.station_key, "day_pnl": 0.0})
        else:
            w = day_sel.copy()
            w["size"] = stake
            w["exposure"] = w["price"] * w["size"]
            w = w.sort_values(["station_key", "edge", "strike_k"], ascending=[True, False, True], kind="mergesort")
            w["cum_exposure_station"] = w.groupby("station_key", sort=False)["exposure"].cumsum()
            exec_df = w.loc[w["cum_exposure_station"] <= risk_per_station].copy()

            if not exec_df.empty:
                exec_df["pnl"] = np.where(
                    exec_df["y"] == 0,
                    (1.0 - exec_df["price"]) * exec_df["size"],
                    -exec_df["price"] * exec_df["size"],
                )
                trade_logs.append(exec_df)

            pnl_by_station = exec_df.groupby("station_key", as_index=False)["pnl"].sum() if not exec_df.empty else pd.DataFrame(columns=["station_key", "pnl"])
            pnl_map = dict(zip(pnl_by_station["station_key"], pnl_by_station["pnl"]))
            for row in stations.itertuples(index=False):
                station_daily_rows.append(
                    {
                        "date": day,
                        "station": row.station,
                        "station_key": row.station_key,
                        "day_pnl": float(pnl_map.get(row.station_key, 0.0)),
                    }
                )

            day_pnl = float(exec_df["pnl"].sum()) if not exec_df.empty else 0.0
            day_exposure = float(exec_df["exposure"].sum()) if not exec_df.empty else 0.0
            day_trades = int(len(exec_df)) if not exec_df.empty else 0
            nav_end = nav_start + day_pnl

        nav = nav_end
        daily_rows.append(
            {
                "date": day,
                "nav_start": nav_start,
                "daily_risk_budget": risk_per_station * float(len(stations)),
                "daily_exposure": day_exposure,
                "day_pnl": day_pnl,
                "trades": day_trades,
                "nav_end": nav_end,
            }
        )

    trade_log = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame(columns=list(selected.columns) + ["size", "exposure", "cum_exposure_station", "pnl"])
    daily_log = pd.DataFrame(daily_rows)
    if daily_log.empty:
        daily_log = pd.DataFrame(columns=["date", "nav_start", "daily_risk_budget", "daily_exposure", "day_pnl", "trades", "nav_end", "daily_return"])
    else:
        daily_log["daily_return"] = np.where(daily_log["nav_start"] > 0, daily_log["day_pnl"] / daily_log["nav_start"], 0.0)

    station_daily = pd.DataFrame(station_daily_rows)
    return trade_log, daily_log, station_daily


def compute_performance_metrics(log_df: pd.DataFrame, daily_df: pd.DataFrame, nav_base: float) -> dict[str, float]:
    total_trades = int(len(log_df))
    coverage_days = int(log_df["date"].nunique()) if not log_df.empty else 0
    total_pnl = float(log_df["pnl"].sum()) if not log_df.empty else 0.0
    wins = int((log_df["pnl"] > 0).sum()) if not log_df.empty else 0
    win_rate = float(wins) / float(total_trades) if total_trades > 0 else 0.0

    if total_trades > 0:
        gross_profit = float(log_df.loc[log_df["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-log_df.loc[log_df["pnl"] < 0, "pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf
    else:
        profit_factor = 0.0

    if daily_df.empty:
        max_drawdown = 0.0
        sharpe = 0.0
    else:
        nav = daily_df["nav_end"].astype(float)
        roll_max = nav.cummax()
        drawdown = (nav / roll_max) - 1.0
        max_drawdown = float(-drawdown.min())

        returns = daily_df["daily_return"].astype(float)
        vol = float(returns.std(ddof=0))
        sharpe = float(returns.mean()) / vol * math.sqrt(252.0) if vol > 0 else 0.0

    return {
        "total_trades": total_trades,
        "coverage_days": coverage_days,
        "total_pnl": total_pnl,
        "roi": total_pnl / float(nav_base) if nav_base > 0 else 0.0,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
    }


def build_station_summary(
    *,
    trade_log: pd.DataFrame,
    base_universe: pd.DataFrame,
    station_filter_breakdown: pd.DataFrame,
    initial_nav: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for row in station_filter_breakdown.itertuples(index=False):
        station = row.station
        station_key = row.station_key

        t = trade_log.loc[trade_log["station_key"] == station_key].copy()
        # Per-station risk metrics from station-specific daily pnl over station's calendar.
        station_days = sorted(pd.to_datetime(base_universe.loc[base_universe["station_key"] == station_key, "date"]).dt.normalize().unique())
        if station_days:
            day_df = pd.DataFrame({"date": station_days})
            pnl = t.groupby("date", as_index=False)["pnl"].sum() if not t.empty else pd.DataFrame(columns=["date", "pnl"])
            day_df = day_df.merge(pnl, on="date", how="left")
            day_df["pnl"] = day_df["pnl"].fillna(0.0)
            day_df = day_df.sort_values("date", kind="mergesort")
            day_df["nav_start"] = float(initial_nav) + day_df["pnl"].cumsum().shift(fill_value=0.0)
            day_df["nav_end"] = day_df["nav_start"] + day_df["pnl"]
            day_df["daily_return"] = np.where(day_df["nav_start"] > 0, day_df["pnl"] / day_df["nav_start"], 0.0)
        else:
            day_df = pd.DataFrame(columns=["date", "nav_start", "nav_end", "daily_return"])

        perf = compute_performance_metrics(t, day_df, nav_base=float(initial_nav))

        total_candidate = int(row.candidates_total)
        priced_candidate = int(row.after_price_available)
        trade_count = int(len(t))

        pct_priced = (priced_candidate / total_candidate) if total_candidate > 0 else 0.0
        pct_traded = (trade_count / priced_candidate) if priced_candidate > 0 else 0.0

        rows.append(
            {
                "station": station,
                "total_trades": perf["total_trades"],
                "coverage_days": perf["coverage_days"],
                "total_pnl": perf["total_pnl"],
                "roi": perf["roi"],
                "win_rate": perf["win_rate"],
                "profit_factor": perf["profit_factor"],
                "max_drawdown": perf["max_drawdown"],
                "sharpe_ratio": perf["sharpe_ratio"],
                "total_candidate_strikes": total_candidate,
                "priced_candidate_strikes": priced_candidate,
                "trade_count": trade_count,
                "pct_priced": pct_priced,
                "pct_traded": pct_traded,
            }
        )

    return pd.DataFrame(rows).sort_values("station", kind="mergesort").reset_index(drop=True)


def format_trade_log(trade_log: pd.DataFrame) -> pd.DataFrame:
    out = trade_log.copy()
    out = out.loc[:, ~out.columns.duplicated(keep="first")].copy()
    needed = [
        "station",
        "date",
        "strike_k",
        "mode_k",
        "p_model",
        "edge",
        "edge_per_risk",
        "size",
        "price",
        "pnl",
        "execution_time_local",
        "selected_trade_timestamp_utc",
        "selected_trade_age_hours",
        "price_source",
        "lookback_used_hours",
        "station_overrides_applied",
        "skipped_reason",
        "station_key",
        "execution_time_utc",
    ]
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["execution_time_local"] = pd.to_datetime(out["execution_time_local"], errors="coerce")
    if out["execution_time_local"].dt.tz is None:
        out["execution_time_local"] = out["execution_time_local"].dt.tz_localize(LONDON_TZ)
    out["selected_trade_timestamp_utc"] = pd.to_datetime(out["selected_trade_timestamp_utc"], utc=True, errors="coerce")
    out["execution_time_utc"] = pd.to_datetime(out["execution_time_utc"], utc=True, errors="coerce")

    for c in ["strike_k", "mode_k"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ["p_model", "edge", "edge_per_risk", "size", "price", "pnl", "selected_trade_age_hours", "lookback_used_hours"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["price_source"] = out["price_source"].astype("string").fillna("")
    out["station_overrides_applied"] = out["station_overrides_applied"].astype("string").fillna("")
    out["skipped_reason"] = out["skipped_reason"].astype("string").fillna("executed")

    out = out.dropna(subset=["station", "date", "strike_k", "mode_k", "p_model", "edge", "size", "price", "pnl"]).copy()
    out["strike_k"] = out["strike_k"].astype(int)
    out["mode_k"] = out["mode_k"].astype(int)

    # strict validation
    bad = out.loc[out["selected_trade_timestamp_utc"] >= out["execution_time_utc"]]
    if not bad.empty:
        raise SystemExit("Validation failed: found trades with selected_trade_timestamp_utc >= execution_time_utc")

    return out[[
        "station",
        "date",
        "strike_k",
        "mode_k",
        "p_model",
        "edge",
        "edge_per_risk",
        "size",
        "price",
        "pnl",
        "execution_time_local",
        "selected_trade_timestamp_utc",
        "selected_trade_age_hours",
        "price_source",
        "lookback_used_hours",
        "station_overrides_applied",
        "skipped_reason",
        "station_key",
        "execution_time_utc",
    ]]


def choose_zero_trade_reason(row: pd.Series) -> str:
    if row["after_p_model"] > 0 and row["after_price_available"] == 0:
        if row.get("removed_by_no_trade_only", 0) > 0:
            return "no_trade_only_rule"
        return "price_unavailable"
    if row.get("removed_by_trade_age", 0) > 0 and row["after_price_available"] > 0 and row["after_edge"] == 0:
        return "trade_age_filter"
    if row["after_p_model"] == 0:
        return "probability_or_mode_filters"
    if row.get("removed_by_station_edge_threshold", 0) > 0 and row["after_edge"] == 0:
        return "station_edge_threshold"
    if row.get("removed_by_edge_per_risk", 0) > 0 and row["after_edge"] > 0 and row["after_price_cap"] == 0:
        return "edge_per_risk_filter"
    if row["after_edge"] == 0 and row["after_price_available"] > 0:
        return "edge_filter"
    if row["after_price_cap"] == 0 and row["after_edge"] > 0:
        return "price_cap_filter"
    return "other_filters"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from master_db import resolve_master_postgres_dsn

    if float(args.trade_lookback_hours_primary) <= 0 or float(args.trade_lookback_hours_fallback) <= 0:
        raise SystemExit("trade lookback hours must be positive")
    if float(args.trade_lookback_hours_fallback) < float(args.trade_lookback_hours_primary):
        raise SystemExit("fallback lookback must be >= primary lookback")
    if float(args.yes_fallback_slippage_buffer) < 0:
        raise SystemExit("yes-fallback-slippage-buffer must be >= 0")
    if float(args.london_min_edge_per_risk) < 0:
        raise SystemExit("london-min-edge-per-risk must be >= 0")
    if float(args.london_max_price) <= 0 or float(args.london_max_price) > 1.0:
        raise SystemExit("london-max-price must be in (0, 1]")
    if float(args.nyc_max_trade_age_hours) <= 0 or float(args.buenosaires_max_trade_age_hours) <= 0:
        raise SystemExit("station max trade age hours must be > 0")
    if float(args.nyc_yes_fallback_slippage_buffer) < 0 or float(args.buenosaires_yes_fallback_slippage_buffer) < 0:
        raise SystemExit("station yes fallback slippage buffers must be >= 0")
    if float(args.nyc_edge_threshold) < 0 or float(args.buenosaires_edge_threshold) < 0:
        raise SystemExit("station edge thresholds must be >= 0")

    raw_prob = read_probability_files(args.probabilities)
    prob = standardize_probabilities(raw_prob)
    prob, station_names = filter_stations(prob, args.stations)

    dsn = resolve_master_postgres_dsn(explicit_dsn=args.master_dsn)
    with psycopg.connect(dsn) as conn:
        prob = resolve_market_ids_from_slug(conn, prob)

        execution_times = derive_execution_times(
            prob=prob,
            predictions_root=args.predictions_root,
            decision_cycle_model=args.decision_cycle_model,
        )
        execution_market = prob[["market_id", "station", "station_key", "date", "event_key"]].drop_duplicates().merge(
            execution_times[["station_key", "date", "event_key", "execution_time_utc", "execution_time_local"]],
            on=["station_key", "date", "event_key"],
            how="left",
        )

        price_tbl = load_trade_prices(
            conn=conn,
            execution_market=execution_market,
            lookback_primary_hours=float(args.trade_lookback_hours_primary),
            lookback_fallback_hours=float(args.trade_lookback_hours_fallback),
        )

    universe = (
        prob.merge(
            execution_times[["station_key", "date", "event_key", "execution_time_utc", "execution_time_local"]],
            on=["station_key", "date", "event_key"],
            how="left",
        )
        .merge(price_tbl, on="market_id", how="left")
    )

    selected, candidate_diag, station_filter_breakdown = apply_filters_and_select(
        universe,
        tail_prob_threshold=float(args.tail_prob_threshold),
        edge_threshold=float(args.edge_threshold),
        max_no_price=float(args.max_no_price),
        max_trades_per_day=int(args.max_trades_per_day),
        yes_fallback_slippage_buffer=float(args.yes_fallback_slippage_buffer),
        london_min_edge_per_risk=float(args.london_min_edge_per_risk),
        london_max_price=float(args.london_max_price),
        nyc_max_trade_age_hours=float(args.nyc_max_trade_age_hours),
        buenosaires_max_trade_age_hours=float(args.buenosaires_max_trade_age_hours),
        nyc_allow_yes_fallback=bool(args.nyc_allow_yes_fallback),
        buenosaires_allow_yes_fallback=bool(args.buenosaires_allow_yes_fallback),
        nyc_yes_fallback_slippage_buffer=float(args.nyc_yes_fallback_slippage_buffer),
        buenosaires_yes_fallback_slippage_buffer=float(args.buenosaires_yes_fallback_slippage_buffer),
        nyc_edge_threshold=float(args.nyc_edge_threshold),
        buenosaires_edge_threshold=float(args.buenosaires_edge_threshold),
    )

    if args.nav_mode == "per-station":
        trade_log_raw, station_daily = run_backtest_per_station(
            selected,
            universe,
            initial_nav=INITIAL_NAV,
            stake_pct=float(args.stake_pct),
            stake_cap=float(args.stake_cap),
            max_daily_risk_pct=float(args.max_daily_risk_pct),
        )

        # overall daily from station daily pnl sum
        all_dates = sorted(pd.to_datetime(universe["date"]).dt.normalize().unique())
        day = pd.DataFrame({"date": all_dates})
        pnl = station_daily.groupby("date", as_index=False)["day_pnl"].sum() if not station_daily.empty else pd.DataFrame(columns=["date", "day_pnl"])
        day = day.merge(pnl, on="date", how="left")
        day["day_pnl"] = day["day_pnl"].fillna(0.0)
        overall_nav_base = INITIAL_NAV * float(len(station_names))
        day = day.sort_values("date", kind="mergesort")
        day["nav_start"] = overall_nav_base + day["day_pnl"].cumsum().shift(fill_value=0.0)
        day["nav_end"] = day["nav_start"] + day["day_pnl"]
        day["daily_return"] = np.where(day["nav_start"] > 0, day["day_pnl"] / day["nav_start"], 0.0)
        overall_daily = day.rename(columns={"day_pnl": "day_pnl", "trades": "trades"})
    else:
        trade_log_raw, overall_daily, station_daily_pnl = run_backtest_shared_nav(
            selected,
            universe,
            initial_nav=INITIAL_NAV,
            stake_pct=float(args.stake_pct),
            stake_cap=float(args.stake_cap),
            max_daily_risk_pct=float(args.max_daily_risk_pct),
        )
        # station_daily for per-station metrics in shared mode
        all_station_days = universe[["station", "station_key", "date"]].drop_duplicates().copy()
        station_daily = all_station_days.merge(station_daily_pnl, on=["station", "station_key", "date"], how="left")
        station_daily["day_pnl"] = station_daily["day_pnl"].fillna(0.0)

    if not trade_log_raw.empty:
        trade_log_raw["skipped_reason"] = "executed"
    trade_log = format_trade_log(trade_log_raw)

    station_summary = build_station_summary(
        trade_log=trade_log,
        base_universe=universe,
        station_filter_breakdown=station_filter_breakdown,
        initial_nav=INITIAL_NAV,
    )

    if args.nav_mode == "shared":
        overall_nav_base = INITIAL_NAV
    else:
        overall_nav_base = INITIAL_NAV * float(len(station_names))

    overall_metrics = compute_performance_metrics(trade_log, overall_daily, nav_base=overall_nav_base)
    total_candidate = int(station_filter_breakdown["candidates_total"].sum()) if not station_filter_breakdown.empty else 0
    priced_candidate = int(station_filter_breakdown["after_price_available"].sum()) if not station_filter_breakdown.empty else 0

    overall_row = {
        **overall_metrics,
        "priced_candidate_strikes": priced_candidate,
        "total_candidate_strikes": total_candidate,
        "pct_priced": (priced_candidate / total_candidate) if total_candidate > 0 else 0.0,
    }
    overall_df = pd.DataFrame([overall_row])

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_df.to_csv(out_dir / "overall_performance_summary.csv", index=False)
    station_summary.to_csv(out_dir / "station_performance_summary.csv", index=False)
    station_filter_breakdown[[
        "station",
        "candidates_total",
        "after_mode_distance",
        "after_p_model",
        "after_price_available",
        "after_edge",
        "after_price_cap",
        "after_top2",
        "removed_by_trade_age",
        "removed_by_no_trade_only",
        "removed_by_edge_per_risk",
        "removed_by_station_edge_threshold",
    ]].to_csv(out_dir / "station_filter_breakdown.csv", index=False)

    trade_log.to_parquet(out_dir / "trade_log.parquet", index=False)
    trade_log.to_csv(out_dir / "trade_log.csv", index=False)

    candidate_diag.to_parquet(out_dir / "candidate_diagnostics.parquet", index=False)
    candidate_diag.to_csv(out_dir / "candidate_diagnostics.csv", index=False)

    # stdout summary
    trades_per_station = station_summary[["station", "total_trades"]].copy()
    min_trades = int(trades_per_station["total_trades"].min()) if not trades_per_station.empty else 0
    med_trades = float(trades_per_station["total_trades"].median()) if not trades_per_station.empty else 0.0
    max_trades = int(trades_per_station["total_trades"].max()) if not trades_per_station.empty else 0

    zero_trade = station_summary.loc[station_summary["total_trades"] == 0, ["station"]].copy()
    if not zero_trade.empty:
        z = zero_trade.merge(station_filter_breakdown, on="station", how="left")
        z["primary_failure_reason"] = z.apply(choose_zero_trade_reason, axis=1)
    else:
        z = pd.DataFrame(columns=["station", "primary_failure_reason"])

    print(f"Stations detected: {len(station_names)}")
    print(f"Total trades overall: {int(overall_metrics['total_trades'])}")
    print(f"Trades per station (min/median/max): {min_trades}/{med_trades:.2f}/{max_trades}")
    if z.empty:
        print("Stations with 0 trades: none")
    else:
        print("Stations with 0 trades:")
        print(z[["station", "primary_failure_reason"]].to_string(index=False))

    print("\nOverall performance")
    print(overall_df.to_string(index=False))

    target_stations = ["London", "NYC", "BuenosAires"]
    focus_metrics = station_summary.loc[station_summary["station"].isin(target_stations), [
        "station",
        "total_trades",
        "coverage_days",
        "total_pnl",
        "win_rate",
        "profit_factor",
    ]].copy()
    focus_skips = station_filter_breakdown.loc[station_filter_breakdown["station"].isin(target_stations), [
        "station",
        "removed_by_trade_age",
        "removed_by_no_trade_only",
        "removed_by_edge_per_risk",
        "removed_by_station_edge_threshold",
    ]].copy()
    focus = focus_metrics.merge(focus_skips, on="station", how="outer").sort_values("station", kind="mergesort")
    if not focus.empty:
        print("\nLondon/NYC/BuenosAires focus")
        print(focus.to_string(index=False))

    print("\nOutput dir:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
