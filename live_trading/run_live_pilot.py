#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_trading import db as dbmod
from live_trading.execution import DummyExecutionClient, RealExecutionClient
from live_trading.forecast_progression import apply_pending_progression_updates, attach_progression_features
from live_trading.policy import PolicyContext, apply_policy
from live_trading.pricing import compute_pricing_decision
from live_trading.reporting import generate_daily_report
from live_trading.state import PilotStateStore
from live_trading.telegram_notify import TelegramNotifier
from live_trading.utils_time import (
    load_station_timezones,
    normalize_station_key,
    passes_decision_cutoff,
    station_timezone,
    to_yyyymmdd,
    today_local,
    utc_now,
)


STRIKE_SUFFIX_RE = re.compile(r"-(?:neg-\d+|\d+)c(?:orbelow|orhigher)?$|-(?:\d+-\d+f|\d+forbelow|\d+forhigher|\d+f)$")
CYCLE_TOKEN_RE = re.compile(r"^\d{10}$")
SLUG_PREFIX_RE = re.compile(r"^highest-temperature-in-([a-z0-9-]+)-on-")
SLUG_EXACT_C_RE = re.compile(r"-(neg-\d+|\d+)c$")
SLUG_BELOW_C_RE = re.compile(r"-(neg-\d+|\d+)corbelow$")
SLUG_ABOVE_C_RE = re.compile(r"-(neg-\d+|\d+)corhigher$")
SLUG_RANGE_F_RE = re.compile(r"-(\d+)-(\d+)f$")
SLUG_BELOW_F_RE = re.compile(r"-(\d+)forbelow$")
SLUG_ABOVE_F_RE = re.compile(r"-(\d+)forhigher$")
SLUG_EXACT_F_RE = re.compile(r"-(\d+)f$")
GENERIC_DIR_NAMES = {
    "reports",
    "report",
    "data",
    "probabilitybacktest",
    "backtest",
    "output",
    "outputs",
}
ORDER_KEY_RETENTION_HOURS = 24.0
POSITION_EPS = 1e-9

DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "paper",
    "db_dsn": "postgresql://archive_user:password@127.0.0.1:5432/master_db",
    "probabilities_path": "reports/live_probabilities",
    "output_dir": "live_trading",
    "nav_usd": 10000,
    "stations_allowlist": ["Atlanta", "Dallas", "Toronto", "Ankara", "Seattle", "SaoPaulo", "Seoul", "Chicago"],
    "stations_watchlist": ["London", "Paris", "Miami"],
    "market_types": ["highest_temperature"],
    "timezones": {"default": "Europe/London", "stations": {}},
    "mode_distance_min": 2,
    "p_model_max": 0.12,
    "edge_threshold": 0.02,
    "max_no_price": 0.92,
    "max_spread": 0.05,
    "price_drift_tolerance": 0.01,
    "max_probability_age_days": 2,
    "max_snapshot_age_minutes": 30,
    "snapshot_quote_lookback_per_outcome": 12,
    "pause_on_stale_probabilities": True,
    "pause_on_stale_snapshots": True,
    "health_gate_alert_cooldown_minutes": 30,
    "top_n_per_event_day": 2,
    "use_progression_confidence": True,
    "progression_enable_gate": True,
    "progression_min_cycles_seen": 3,
    "progression_min_consecutive_candidate_cycles": 2,
    "progression_enable_negative_veto": True,
    "progression_negative_edge_trend_threshold": -0.01,
    "progression_min_mode_consistency_ratio": 0.40,
    "progression_negative_p_model_trend_threshold": 0.01,
    "progression_weight_consecutive": 0.30,
    "progression_weight_candidate_ratio": 0.20,
    "progression_weight_edge_trend": 0.20,
    "progression_weight_mode_consistency": 0.15,
    "progression_weight_low_p_model": 0.10,
    "progression_weight_low_edge_volatility": 0.05,
    "progression_edge_trend_cap": 0.05,
    "progression_enable_size_multiplier": True,
    "progression_min_size_multiplier": 0.85,
    "progression_max_size_multiplier": 1.35,
    "use_ensemble_confidence": True,
    "ensemble_probability_adjustment_enabled": True,
    "ensemble_trade_size_adjustment_enabled": True,
    "ensemble_disagreement_neutral_shrink_cap": 0.25,
    "ensemble_std_cap_c": 2.0,
    "ensemble_range_cap_c": 4.0,
    "ensemble_enable_gate": True,
    "ensemble_min_same_side_ratio": 0.67,
    "ensemble_max_std_c_for_trade": 2.5,
    "ensemble_max_range_c_for_trade": 5.0,
    "ensemble_enable_strike_disagreement_veto": True,
    "ensemble_min_size_multiplier": 0.75,
    "ensemble_max_size_multiplier": 1.15,
    "stake_fraction": 0.005,
    "stake_cap_usd": 50,
    "drawdown_position_scaling": True,
    "max_drawdown_fraction": 0.2,
    "min_drawdown_scale": 0.25,
    "station_daily_risk_fraction": 0.02,
    "portfolio_daily_risk_fraction": 0.05,
    "stoploss_daily_pnl_fraction": 0.01,
    "stoploss_consecutive_days": 3,
    "trade_stoploss": {
        "enabled": True,
        "loss_fraction": 0.25,
        "break_even_on_recovery": True,
    },
    "max_open_positions_per_station": 4,
    "max_open_positions_total": 20,
    "trade_cooldown_minutes": 30,
    "slippage_buffer_yes_fallback": 0.01,
    "min_order_size": 1,
    "price_tick": 0.001,
    "use_limit_orders": True,
    "paper_execution_realism": True,
    "paper_execution_random_seed": 0,
    "paper_fill_probability_base": 0.9,
    "paper_partial_fill_probability": 0.2,
    "paper_max_slippage_ticks": 3,
    "run_interval_minutes": 10,
    "decision_cutoff_policy": "latest_cycle_before_local_midnight",
    "lookahead_days": 4,
    "trade_window": {"start_local": "00:00", "end_local": "12:00"},
    "log_level": "INFO",
    "log_to_stdout": True,
    "log_file": "",
    "log_rotate_max_mb": 128,
    "log_rotate_backups": 20,
    "log_skip_decisions_at_info": False,
    "log_cycle_decision_summary": True,
    "log_cycle_summary_top_skip_reasons": 6,
    "write_jsonl_log": True,
    "write_csv_trades": True,
    "daily_report_time_local": "23:59",
    "telegram_templates": {"enabled": True},
    "telegram_notifications": {
        "enabled": True,
        "credentials_file": ".secrets/telegram_bot.json",
        "trades_topic_link": "https://t.me/c/3811684844/467/469",
        "daily_topic_link": "https://t.me/c/3811684844/468/471",
        "timeout_seconds": 20.0,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live trading pilot scaffold.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Only evaluate policy and log WOULD-trade decisions.")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    parser.add_argument(
        "--exit-nonzero-on-cycle-failure",
        action="store_true",
        help="Return exit code 1 when a cycle raises an exception (useful for external supervisors).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "healthcheck"],
        help="run (default) or healthcheck",
    )
    return parser.parse_args(argv)


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    cfg = deep_merge(DEFAULT_CONFIG, loaded)
    return cfg


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Manifest must be a JSON object: {path}")
    return data


def _manifest_status_is_success(manifest: Mapping[str, Any]) -> bool:
    status = str(manifest.get("status", "")).strip().lower()
    return (not status) or status == "success"


def _extract_cycle_token(manifest_path: Path, manifest: Mapping[str, Any]) -> str | None:
    cycle = str(manifest.get("cycle", "")).strip()
    if CYCLE_TOKEN_RE.fullmatch(cycle):
        return cycle

    parts = manifest_path.resolve().parts
    for idx, part in enumerate(parts):
        if part != "cycles":
            continue
        if idx + 1 >= len(parts):
            continue
        candidate = parts[idx + 1]
        if CYCLE_TOKEN_RE.fullmatch(candidate):
            return candidate
    return None


def _manifest_generated_at_rank(manifest: Mapping[str, Any]) -> int:
    raw = manifest.get("generated_at_utc")
    if raw is None:
        return -1
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return -1
    return int(ts.value)


def _resolve_manifest_probability_file(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    root_hint: Path | None = None,
) -> Path:
    rel = manifest.get("probabilities_file")
    if rel is None or not str(rel).strip():
        raise SystemExit(f"Manifest missing probabilities_file: {manifest_path}")

    rel_path = Path(str(rel).strip())
    if rel_path.is_absolute():
        if rel_path.exists():
            return rel_path.resolve()
        raise SystemExit(f"Manifest probabilities file not found: {rel_path}")

    candidate_bases: list[Path] = []
    if root_hint is not None:
        candidate_bases.append(root_hint)
    candidate_bases.extend(list(manifest_path.parents))

    seen: set[str] = set()
    checked: list[Path] = []
    for base in candidate_bases:
        candidate = (base / rel_path).resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.exists():
            return candidate

    checked_txt = ", ".join(str(p) for p in checked)
    raise SystemExit(f"Manifest probabilities file not found: {rel_path} (checked: {checked_txt})")


def _annotated_manifest(manifest: Mapping[str, Any], *, manifest_path: Path, source: str) -> dict[str, Any]:
    out = dict(manifest)
    out["_resolved_manifest_path"] = str(manifest_path.resolve())
    out["_resolved_source"] = source
    return out


def _select_latest_cycle_manifest(root: Path) -> tuple[Path, dict[str, Any]] | None:
    patterns = ("cycles/*/*/manifest.json", "cycles/*/manifest.json")
    manifest_paths: list[Path] = []
    seen_paths: set[str] = set()
    for pattern in patterns:
        for manifest_path in sorted(root.glob(pattern)):
            key = str(manifest_path.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            manifest_paths.append(manifest_path)

    candidates: list[tuple[str, int, str, Path, dict[str, Any], Path]] = []
    for manifest_path in manifest_paths:
        try:
            manifest = _load_json_dict(manifest_path)
        except (SystemExit, Exception):
            continue
        if not _manifest_status_is_success(manifest):
            continue
        cycle_token = _extract_cycle_token(manifest_path, manifest)
        if cycle_token is None:
            continue
        try:
            prob_file = _resolve_manifest_probability_file(manifest_path, manifest, root_hint=root)
        except (SystemExit, Exception):
            continue

        candidates.append(
            (
                cycle_token,
                _manifest_generated_at_rank(manifest),
                str(manifest_path.resolve()),
                prob_file,
                dict(manifest),
                manifest_path,
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    _, _, _, prob_file, manifest, manifest_path = candidates[0]
    return prob_file, _annotated_manifest(manifest, manifest_path=manifest_path, source="cycle_scan")


def resolve_probability_data_path(path: Path) -> tuple[Path, dict[str, Any] | None]:
    if path.is_file():
        if path.suffix.lower() == ".json" and "manifest" in path.name.lower():
            manifest = _load_json_dict(path)
            if not _manifest_status_is_success(manifest):
                status = str(manifest.get("status", "")).strip().lower()
                raise SystemExit(f"Manifest status is not success: {path} (status={status})")
            resolved = _resolve_manifest_probability_file(path, manifest)
            return resolved, _annotated_manifest(manifest, manifest_path=path, source="explicit_manifest")
        return path, None

    if path.is_dir():
        manifest_path = path / "latest_manifest.json"
        latest_manifest_error: str | None = None
        if manifest_path.exists():
            try:
                manifest = _load_json_dict(manifest_path)
                if not _manifest_status_is_success(manifest):
                    status = str(manifest.get("status", "")).strip().lower()
                    raise SystemExit(f"Manifest status is not success: {manifest_path} (status={status})")
                resolved = _resolve_manifest_probability_file(manifest_path, manifest, root_hint=path)
                return resolved, _annotated_manifest(manifest, manifest_path=manifest_path, source="latest_manifest")
            except (SystemExit, Exception) as exc:
                latest_manifest_error = str(exc)

        selected = _select_latest_cycle_manifest(path)
        if selected is not None:
            return selected

        for name in ("market_level_probabilities.parquet", "market_level_probabilities.csv"):
            direct_file = path / name
            if direct_file.exists():
                return direct_file.resolve(), None

        if latest_manifest_error:
            raise SystemExit(f"No valid cycle manifest found under {path}. latest_manifest.json error: {latest_manifest_error}")
        raise SystemExit(f"No valid cycle manifest found under {path}.")

    raise SystemExit(f"Probabilities path not found: {path}")


def setup_logger(
    log_path: Path,
    level: str,
    *,
    rotate_max_mb: int = 128,
    rotate_backups: int = 20,
    to_stdout: bool = True,
) -> logging.Logger:
    logger = logging.getLogger("live_pilot")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    logger.propagate = False
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    max_bytes = max(1, int(rotate_max_mb)) * 1024 * 1024
    backup_count = max(1, int(rotate_backups))
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def _derive_event_key(slug: str) -> str:
    return STRIKE_SUFFIX_RE.sub("", str(slug))


def _derive_station_from_source_path(source_path: str) -> str | None:
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


def _parse_c_token(token: str) -> int:
    if token.startswith("neg-"):
        return -int(token.split("-", 1)[1])
    return int(token)


def _f_to_c(value_f: float) -> float:
    return (float(value_f) - 32.0) * (5.0 / 9.0)


def _round_market_integer_c(value_c: float) -> int:
    return int(math.floor(float(value_c) + 0.5))


def _parse_open_market_slug(slug: str, station_lookup: dict[str, str]) -> dict[str, Any] | None:
    text = str(slug or "").strip().lower()
    m = SLUG_PREFIX_RE.search(text)
    if m is None:
        return None

    station_key = normalize_station_key(m.group(1))
    station = station_lookup.get(station_key)
    if station is None:
        return None

    strike_k: int | None = None
    m_below_c = SLUG_BELOW_C_RE.search(text)
    m_above_c = SLUG_ABOVE_C_RE.search(text)
    m_exact_c = SLUG_EXACT_C_RE.search(text)
    m_range_f = SLUG_RANGE_F_RE.search(text)
    m_below_f = SLUG_BELOW_F_RE.search(text)
    m_above_f = SLUG_ABOVE_F_RE.search(text)
    m_exact_f = SLUG_EXACT_F_RE.search(text)

    if m_below_c:
        strike_k = _parse_c_token(m_below_c.group(1))
    elif m_above_c:
        strike_k = _parse_c_token(m_above_c.group(1))
    elif m_exact_c:
        strike_k = _parse_c_token(m_exact_c.group(1))
    elif m_range_f:
        low_f = int(m_range_f.group(1))
        high_f = int(m_range_f.group(2))
        if low_f > high_f:
            low_f, high_f = high_f, low_f
        strike_k = _round_market_integer_c(_f_to_c((low_f + high_f) / 2.0))
    elif m_below_f:
        strike_k = _round_market_integer_c(_f_to_c(float(m_below_f.group(1))))
    elif m_above_f:
        strike_k = _round_market_integer_c(_f_to_c(float(m_above_f.group(1))))
    elif m_exact_f:
        strike_k = _round_market_integer_c(_f_to_c(float(m_exact_f.group(1))))

    if strike_k is None:
        return None

    event_key = _derive_event_key(text)
    return {
        "station": station,
        "station_key": station_key,
        "event_key": event_key,
        "strike_k": int(strike_k),
    }


def _to_day_iso(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _normalize_slug(value: Any) -> str:
    return _normalize_text(value).lower()


def _normalize_event_key(value: Any, *, slug_fallback: Any = None) -> str:
    event_key = _normalize_text(value)
    if not event_key and slug_fallback is not None:
        slug_text = _normalize_text(slug_fallback)
        if slug_text:
            event_key = _derive_event_key(slug_text)
    return event_key.strip().lower()


def _market_id_sort_key(value: Any) -> int:
    text = _normalize_text(value)
    if not text:
        return -1
    try:
        return int(text)
    except Exception:
        return -1


def _build_order_key(payload: Mapping[str, Any]) -> str | None:
    station_key = normalize_station_key(str(payload.get("station") or ""))
    market_day_local = _to_day_iso(payload.get("market_day_local"))
    strike_k = _safe_int(payload.get("strike_k"))
    try:
        chosen_no_ask = float(payload.get("chosen_no_ask"))
    except Exception:
        chosen_no_ask = None

    side = str(payload.get("order_side") or "buy").strip().lower()
    outcome = str(payload.get("order_outcome") or "NO").strip().upper()

    if not station_key or market_day_local is None or strike_k is None or chosen_no_ask is None:
        return None
    return "|".join(
        [
            station_key,
            market_day_local,
            str(strike_k),
            f"{chosen_no_ask:.6f}",
            side,
            outcome,
        ]
    )


def _build_execution_client(cfg: Mapping[str, Any]) -> Any:
    mode = str(cfg.get("mode", "paper")).lower()
    if mode == "paper":
        return DummyExecutionClient(
            price_tick=float(cfg.get("price_tick", 0.001)),
            conservative_fill=True,
            realism_enabled=bool(cfg.get("paper_execution_realism", True)),
            deterministic_seed=int(cfg.get("paper_execution_random_seed", 0)),
            fill_probability_base=float(cfg.get("paper_fill_probability_base", 0.9)),
            partial_fill_probability=float(cfg.get("paper_partial_fill_probability", 0.2)),
            max_slippage_ticks=int(cfg.get("paper_max_slippage_ticks", 3)),
        )
    if mode == "live":
        return RealExecutionClient()
    raise SystemExit(f"Invalid mode: {mode}")


def _trade_stoploss_settings(cfg: Mapping[str, Any]) -> tuple[bool, float, bool]:
    raw = cfg.get("trade_stoploss", {})
    if not isinstance(raw, Mapping):
        raw = {}
    enabled = bool(raw.get("enabled", True))
    try:
        loss_fraction = abs(float(raw.get("loss_fraction", 0.25)))
    except Exception:
        loss_fraction = 0.25
    loss_fraction = min(0.99, max(0.0, loss_fraction))
    break_even_on_recovery = bool(raw.get("break_even_on_recovery", True))
    return enabled, loss_fraction, break_even_on_recovery


def _parse_snapshot_ts(value: Any) -> datetime | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _compute_stoploss_mark_price(
    *,
    snapshot: dict[str, Any] | None,
    now_utc: datetime,
    max_snapshot_age_minutes: float,
) -> tuple[float | None, str | None, datetime | None, float | None, str | None]:
    if not snapshot:
        return None, None, None, None, "no_snapshot"

    no_bid = _as_float(snapshot.get("best_no_bid"))
    no_ts = _parse_snapshot_ts(snapshot.get("no_snapshot_ts_utc"))
    yes_ask = _as_float(snapshot.get("best_yes_ask"))
    yes_ts = _parse_snapshot_ts(snapshot.get("yes_snapshot_ts_utc"))

    mark_price: float | None = None
    source: str | None = None
    ts: datetime | None = None
    if no_bid is not None:
        mark_price = min(1.0, max(0.0, float(no_bid)))
        source = "NO_bid"
        ts = no_ts or yes_ts
    elif yes_ask is not None:
        mark_price = min(1.0, max(0.0, 1.0 - float(yes_ask)))
        source = "YES_ask_fallback"
        ts = yes_ts or no_ts

    if mark_price is None or ts is None:
        return None, source, ts, None, "no_market_price"

    age_minutes = max(0.0, (now_utc - ts).total_seconds() / 60.0)
    if age_minutes > float(max_snapshot_age_minutes):
        return mark_price, source, ts, age_minutes, "snapshot_too_old"

    return mark_price, source, ts, age_minutes, None


def resolve_open_market_universe(
    *,
    universe: pd.DataFrame,
    conn: psycopg.Connection,
    station_tz: dict[str, str],
    cfg: dict[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = universe.copy()
    if out.empty:
        return out, out

    station_lookup: dict[str, str] = {}
    for station in out["station"].dropna().astype(str):
        key = normalize_station_key(station)
        if key and key not in station_lookup:
            station_lookup[key] = station

    open_markets = dbmod.fetch_open_weather_markets(conn, statuses=["active"])
    if open_markets.empty:
        unmapped = out.copy()
        unmapped["unmapped_reason"] = "open_market_universe_empty"
        logger.warning("Open market universe is empty; dropping %d candidate rows", len(unmapped))
        return out.iloc[0:0].copy(), unmapped

    default_tz = str(cfg.get("timezones", {}).get("default", "UTC"))
    market_rows: list[dict[str, Any]] = []
    for row in open_markets.itertuples(index=False):
        slug = str(row.slug or "").strip()
        parsed = _parse_open_market_slug(slug, station_lookup=station_lookup)
        if parsed is None:
            continue

        end_date_utc = pd.to_datetime(row.end_date_utc, utc=True, errors="coerce")
        if pd.isna(end_date_utc):
            continue
        station = str(parsed["station"])
        tz = station_tz.get(station, default_tz)
        market_day_local = end_date_utc.tz_convert(tz).tz_localize(None).date().isoformat()

        market_rows.append(
            {
                "market_id": str(row.market_id),
                "slug": slug,
                "asset_id": None if pd.isna(getattr(row, "asset_id", pd.NA)) else str(getattr(row, "asset_id")),
                "station_key": str(parsed["station_key"]),
                "event_key": str(parsed["event_key"]).strip().lower(),
                "strike_k": int(parsed["strike_k"]),
                "market_day_local": market_day_local,
                "resolution_time": getattr(row, "resolution_time", pd.NaT),
            }
        )

    if not market_rows:
        unmapped = out.copy()
        unmapped["unmapped_reason"] = "open_market_parsing_empty"
        logger.warning("Open market universe parsing yielded zero rows; dropping %d candidate rows", len(unmapped))
        return out.iloc[0:0].copy(), unmapped

    market_idx = pd.DataFrame.from_records(market_rows)
    market_idx["market_id"] = market_idx["market_id"].map(_normalize_text)
    market_idx["slug"] = market_idx["slug"].map(_normalize_text)
    market_idx["slug_norm"] = market_idx["slug"].map(_normalize_slug)
    market_idx["event_key"] = market_idx["event_key"].map(lambda v: _normalize_event_key(v))
    market_idx["station_key"] = market_idx["station_key"].map(_normalize_text)
    market_idx["market_day_local"] = market_idx["market_day_local"].map(_to_day_iso)
    market_idx["strike_k"] = market_idx["strike_k"].map(_safe_int)
    market_idx["resolution_time"] = pd.to_datetime(market_idx.get("resolution_time"), utc=True, errors="coerce")
    market_idx["_market_id_sort"] = market_idx["market_id"].map(_market_id_sort_key)
    market_idx = market_idx.dropna(subset=["market_day_local", "strike_k"]).copy()
    market_idx = market_idx.sort_values(
        ["station_key", "market_day_local", "event_key", "strike_k", "resolution_time", "_market_id_sort"],
        ascending=[True, True, True, True, False, False],
        kind="mergesort",
    )

    market_key_cols = ["station_key", "market_day_local", "event_key", "strike_k"]
    duplicate_keys = int(market_idx.duplicated(subset=market_key_cols, keep="first").sum())
    if duplicate_keys > 0:
        logger.warning(
            "Open market universe contains %d duplicate station/day/event/strike keys (deduped to canonical rows)",
            duplicate_keys,
        )
        market_idx = market_idx.drop_duplicates(subset=market_key_cols, keep="first").copy()

    by_market_id: dict[str, dict[str, Any]] = {
        str(r.market_id): r._asdict() for r in market_idx.itertuples(index=False)
    }
    by_slug: dict[str, dict[str, Any]] = {
        str(r.slug_norm): r._asdict() for r in market_idx.itertuples(index=False)
    }
    by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for rec in market_idx.to_dict(orient="records"):
        key = (
            str(rec["station_key"]),
            str(rec["market_day_local"]),
            str(rec["event_key"]),
            int(rec["strike_k"]),
        )
        by_key[key] = rec

    out = out.copy()
    out["__station_key"] = out.get("station", "").map(lambda v: normalize_station_key(_normalize_text(v)))
    out["__day_iso"] = out.get("market_day_local").map(_to_day_iso)
    out["__strike_k"] = out.get("strike_k").map(_safe_int)
    out["__raw_market_id"] = out.get("market_id").map(_normalize_text)
    out["__raw_slug"] = out.get("slug").map(_normalize_text)
    out["__raw_slug_norm"] = out["__raw_slug"].map(_normalize_slug)
    out["__event_key_norm"] = out.apply(
        lambda r: _normalize_event_key(r.get("event_key"), slug_fallback=r.get("__raw_slug")),
        axis=1,
    )
    out["__has_valid_canonical_key"] = (
        out["__station_key"].astype(bool)
        & out["__day_iso"].notna()
        & out["__event_key_norm"].astype(bool)
        & out["__strike_k"].notna()
    )
    out["__market_id_open"] = out["__raw_market_id"].map(lambda v: bool(v) and v in by_market_id)
    out["__slug_open"] = out["__raw_slug_norm"].map(lambda v: bool(v) and v in by_slug)
    out = out.sort_values(
        ["__has_valid_canonical_key", "__market_id_open", "__slug_open"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    candidate_key_cols = ["__station_key", "__day_iso", "__event_key_norm", "__strike_k"]
    candidate_dupe_mask = out["__has_valid_canonical_key"] & out.duplicated(subset=candidate_key_cols, keep="first")
    candidate_dupe_count = int(candidate_dupe_mask.sum())
    if candidate_dupe_count > 0:
        logger.warning(
            "Candidate universe contains %d duplicate station/day/event/strike keys (deduped before mapping)",
            candidate_dupe_count,
        )
        out = out.loc[~candidate_dupe_mask].copy()

    temp_cols = [
        "__station_key",
        "__day_iso",
        "__strike_k",
        "__raw_market_id",
        "__raw_slug",
        "__raw_slug_norm",
        "__event_key_norm",
        "__has_valid_canonical_key",
        "__market_id_open",
        "__slug_open",
    ]

    mapped_records: list[dict[str, Any]] = []
    unmapped_records: list[dict[str, Any]] = []
    for rec in out.to_dict(orient="records"):
        raw_market_id = _normalize_text(rec.get("__raw_market_id"))
        raw_slug = _normalize_text(rec.get("__raw_slug"))
        raw_slug_norm = _normalize_slug(rec.get("__raw_slug_norm"))
        station_key = _normalize_text(rec.get("__station_key"))
        day_iso = _normalize_text(rec.get("__day_iso"))
        event_key = _normalize_event_key(rec.get("__event_key_norm"))
        strike_k = _safe_int(rec.get("__strike_k"))

        mapped_market: dict[str, Any] | None = None
        fallback_reason: str | None = None

        market_id_not_open = bool(raw_market_id) and raw_market_id not in by_market_id
        slug_not_open = bool(raw_slug_norm) and raw_slug_norm not in by_slug

        if raw_market_id:
            mapped_market = by_market_id.get(raw_market_id)

        if mapped_market is None and raw_slug_norm:
            mapped_market = by_slug.get(raw_slug_norm)

        if mapped_market is None:
            if not station_key:
                fallback_reason = "missing_station"
            elif not day_iso:
                fallback_reason = "invalid_market_day_local"
            elif strike_k is None:
                fallback_reason = "invalid_strike_k"
            elif not event_key:
                fallback_reason = "missing_event_key"
            else:
                mapped_market = by_key.get((station_key, day_iso, event_key, strike_k))
                if mapped_market is None:
                    if market_id_not_open:
                        fallback_reason = "market_id_not_open"
                    elif slug_not_open:
                        fallback_reason = "slug_not_open"
                    else:
                        fallback_reason = "no_station_day_strike_match"

        for col in temp_cols:
            rec.pop(col, None)
        if mapped_market is None:
            rec["unmapped_reason"] = fallback_reason or "unmapped"
            unmapped_records.append(rec)
            continue

        rec["market_id"] = mapped_market.get("market_id")
        rec["slug"] = mapped_market.get("slug") or rec.get("slug")
        rec["asset_id"] = mapped_market.get("asset_id")
        mapped_records.append(rec)

    mapped = pd.DataFrame.from_records(mapped_records)
    if mapped.empty:
        mapped = out.iloc[0:0].copy()
    unmapped = pd.DataFrame.from_records(unmapped_records)
    mapped = mapped.drop(columns=temp_cols, errors="ignore")
    unmapped = unmapped.drop(columns=temp_cols, errors="ignore")

    if not unmapped.empty:
        counts = unmapped["unmapped_reason"].astype(str).value_counts().to_dict()
        logger.warning("Unmapped probability rows dropped: count=%d reason_counts=%s", len(unmapped), counts)
        for row in unmapped.head(20).itertuples(index=False):
            logger.warning(
                "Unmapped row reason=%s station=%s day=%s strike_k=%s event_key=%s slug=%s market_id=%s",
                getattr(row, "unmapped_reason", ""),
                getattr(row, "station", ""),
                getattr(row, "market_day_local", ""),
                getattr(row, "strike_k", ""),
                getattr(row, "event_key", ""),
                getattr(row, "slug", ""),
                getattr(row, "market_id", ""),
            )
    else:
        logger.info("All universe rows mapped to open markets via master_db")
    return mapped, unmapped


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
        raise SystemExit(f"No market probability files found under: {path}")

    parts: list[pd.DataFrame] = []
    for f in files:
        if f.suffix.lower() == ".parquet":
            part = pd.read_parquet(f)
        else:
            part = pd.read_csv(f)
        part["__source_path"] = str(f)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _coerce_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
    def _parse_one(v: Any) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        try:
            if pd.isna(v):
                return default
        except Exception:
            pass
        txt = str(v).strip().lower()
        if txt in {"1", "true", "t", "yes", "y"}:
            return True
        if txt in {"0", "false", "f", "no", "n"}:
            return False
        return default

    return values.map(_parse_one)


def standardize_probabilities(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    if "slug" not in out.columns:
        raise SystemExit("Probability input must include slug column.")
    out["slug"] = out["slug"].astype("string")

    station_col = None
    for c in ["station_name", "station", "city_name", "city"]:
        if c in out.columns:
            station_col = c
            break
    if station_col is not None:
        out["station"] = out[station_col].astype("string").str.strip()
    else:
        src_station = out["__source_path"].astype("string").map(_derive_station_from_source_path)
        out["station"] = src_station.astype("string")

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

    p_model_series: pd.Series | None = None
    for c in ("p_model", "p_model_adjusted", "p_model_residual"):
        if c in out.columns:
            p_model_series = pd.to_numeric(out[c], errors="coerce")
            break
    if p_model_series is None:
        raise SystemExit("Probability input must include p_model, p_model_adjusted, or p_model_residual.")
    out["p_model"] = p_model_series

    if "p_model_raw" in out.columns:
        out["p_model_raw"] = pd.to_numeric(out["p_model_raw"], errors="coerce")
    else:
        out["p_model_raw"] = out["p_model"]
    if "p_model_adjusted" in out.columns:
        out["p_model_adjusted"] = pd.to_numeric(out["p_model_adjusted"], errors="coerce")
    else:
        out["p_model_adjusted"] = out["p_model"]

    out["p_model_raw"] = out["p_model_raw"].where(out["p_model_raw"].notna(), out["p_model"])
    out["p_model_adjusted"] = out["p_model_adjusted"].where(out["p_model_adjusted"].notna(), out["p_model"])

    if "mode_k" in out.columns:
        out["mode_k"] = pd.to_numeric(out["mode_k"], errors="coerce").astype("Int64")
    else:
        out["mode_k"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    out["market_id"] = out["market_id"].astype("string") if "market_id" in out.columns else pd.Series([pd.NA] * len(out), dtype="string")
    if "event_key" in out.columns:
        out["event_key"] = out["event_key"].astype("string").fillna(out["slug"].astype("string").map(_derive_event_key))
    else:
        out["event_key"] = out["slug"].astype("string").map(_derive_event_key)
    out["market_day_local"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    execution_raw = None
    for c in ["execution_time_utc", "decision_cycle_time_utc"]:
        if c in out.columns:
            execution_raw = out[c]
            break
    if execution_raw is None:
        out["execution_time_utc"] = pd.NaT
    else:
        out["execution_time_utc"] = pd.to_datetime(execution_raw, utc=True, errors="coerce")

    numeric_cols = [
        "pred_model_1",
        "pred_model_2",
        "pred_model_3",
        "ensemble_pred_median",
        "ensemble_pred_mean",
        "ensemble_pred_min",
        "ensemble_pred_max",
        "ensemble_pred_range",
        "ensemble_pred_std",
        "ensemble_pred_iqr",
        "ensemble_bullish_count",
        "ensemble_bearish_count",
        "ensemble_agreement_score",
        "ensemble_disagreement_score",
        "ensemble_sign_agreement_ratio",
        "ensemble_models_yes_count",
        "ensemble_models_no_count",
        "ensemble_same_side_ratio",
        "ensemble_confidence_multiplier",
        "ensemble_uncertainty_penalty",
        "ensemble_size_multiplier",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in out.columns:
        if col.startswith("pred_"):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ("ensemble_cross_strike_disagreement", "ensemble_strike_disagreement_flag"):
        if col in out.columns:
            out[col] = _coerce_bool_series(out[col], default=False)

    if "ensemble_fallback_marker" in out.columns:
        out["ensemble_fallback_marker"] = out["ensemble_fallback_marker"].astype("string").fillna("")
    else:
        out["ensemble_fallback_marker"] = pd.Series([""] * len(out), dtype="string")

    default_numeric = {
        "ensemble_agreement_score": 0.5,
        "ensemble_disagreement_score": 0.5,
        "ensemble_sign_agreement_ratio": 0.5,
        "ensemble_models_yes_count": 0.0,
        "ensemble_models_no_count": 0.0,
        "ensemble_same_side_ratio": 1.0,
        "ensemble_confidence_multiplier": 1.0,
        "ensemble_uncertainty_penalty": 0.0,
        "ensemble_size_multiplier": 1.0,
        "ensemble_bullish_count": 0.0,
        "ensemble_bearish_count": 0.0,
    }
    for col, default in default_numeric.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    if "ensemble_cross_strike_disagreement" not in out.columns:
        out["ensemble_cross_strike_disagreement"] = False
    if "ensemble_strike_disagreement_flag" not in out.columns:
        out["ensemble_strike_disagreement_flag"] = False

    out = out.dropna(subset=["station", "date", "slug", "strike_k", "p_model"]).copy()
    out["station"] = out["station"].astype(str).str.strip()
    out = out.loc[out["station"] != ""].copy()

    out["strike_k"] = out["strike_k"].astype(int)

    missing_mode = out["mode_k"].isna()
    if missing_mode.any():
        mode_df = (
            out.sort_values(["station", "date", "event_key", "p_model", "strike_k"], ascending=[True, True, True, False, True], kind="mergesort")
            .drop_duplicates(subset=["station", "date", "event_key"], keep="first")
            [["station", "date", "event_key", "strike_k"]]
            .rename(columns={"strike_k": "mode_k_derived"})
        )
        out = out.merge(mode_df, on=["station", "date", "event_key"], how="left")
        out.loc[missing_mode, "mode_k"] = out.loc[missing_mode, "mode_k_derived"]
        out = out.drop(columns=["mode_k_derived"], errors="ignore")

    out["mode_k"] = pd.to_numeric(out["mode_k"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["mode_k"]).copy()
    out["mode_k"] = out["mode_k"].astype(int)

    out["ensemble_models_yes_count"] = pd.to_numeric(out["ensemble_models_yes_count"], errors="coerce").fillna(0).astype(int)
    out["ensemble_models_no_count"] = pd.to_numeric(out["ensemble_models_no_count"], errors="coerce").fillna(0).astype(int)
    out["ensemble_bullish_count"] = pd.to_numeric(out["ensemble_bullish_count"], errors="coerce").fillna(0).astype(int)
    out["ensemble_bearish_count"] = pd.to_numeric(out["ensemble_bearish_count"], errors="coerce").fillna(0).astype(int)

    out = out.sort_values(["station", "date", "event_key", "strike_k", "slug"], kind="mergesort")
    out = out.drop_duplicates(subset=["station", "date", "event_key", "strike_k", "slug"], keep="last")
    return out


def build_station_timezone_map(cfg: dict[str, Any], stations: list[str]) -> dict[str, str]:
    csv_tz = load_station_timezones()
    out: dict[str, str] = {}
    for station in stations:
        out[station] = station_timezone(
            station,
            config_timezones=cfg.get("timezones", {}),
            fallback_timezones=csv_tz,
        )
    return out


def select_live_universe(
    *,
    prob: pd.DataFrame,
    cfg: dict[str, Any],
    station_tz: dict[str, str],
    now_utc: datetime,
) -> pd.DataFrame:
    allowlist = [str(s).strip() for s in cfg.get("stations_allowlist", []) if str(s).strip()]
    if not allowlist:
        raise SystemExit("stations_allowlist must be non-empty.")

    out = prob.loc[prob["station"].isin(allowlist)].copy()
    if out.empty:
        return out

    lookahead = int(cfg.get("lookahead_days", 4))
    keep_idx: list[int] = []
    for idx, row in out.iterrows():
        station = str(row["station"])
        tz = station_tz.get(station, str(cfg.get("timezones", {}).get("default", "UTC")))
        market_day = pd.to_datetime(row["market_day_local"], errors="coerce")
        if pd.isna(market_day):
            continue
        market_day_local = market_day.date()
        local_today = today_local(tz, now_utc=now_utc)
        if market_day_local < local_today or market_day_local > local_today + timedelta(days=lookahead):
            continue

        execution = pd.to_datetime(row.get("execution_time_utc"), utc=True, errors="coerce")
        execution_dt = None if pd.isna(execution) else execution.to_pydatetime()
        if execution_dt is not None:
            if not passes_decision_cutoff(
                execution_time_utc=execution_dt,
                market_day_local=market_day_local,
                timezone_name=tz,
                policy=str(cfg.get("decision_cutoff_policy", "latest_cycle_before_local_midnight")),
            ):
                continue
        keep_idx.append(idx)

    if not keep_idx:
        return out.iloc[0:0].copy()

    out = out.loc[keep_idx].copy()
    if "highest_temperature" in cfg.get("market_types", ["highest_temperature"]):
        out = out.loc[out["slug"].astype(str).str.startswith("highest-temperature-in-")].copy()
    return out


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _clean_for_json(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, (datetime, date)):
            out[k] = v.isoformat()
        elif isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        elif hasattr(v, "item") and callable(getattr(v, "item")):
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            try:
                if pd.isna(v):
                    out[k] = None
                    continue
            except Exception:
                pass
            out[k] = v
    return out


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        return None
    return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _fmt_num(value: Any, *, digits: int = 4) -> str:
    val = _as_float(value)
    if val is None:
        return "na"
    return f"{val:.{digits}f}"


def _decision_reason_code(clean: dict[str, Any]) -> str:
    decision = str(clean.get("decision", "")).strip().upper()
    if decision == "SKIP":
        reason = str(clean.get("skipped_reason") or "").strip()
        return reason if reason else "skip_unknown"
    if decision == "TRADE":
        return "trade_approved"
    if decision == "SELL":
        return str(clean.get("sell_reason") or "position_sold")
    if decision == "RESOLVE":
        return str(clean.get("resolution") or "position_resolved")
    return decision.lower() or "unknown"


def _build_decision_explanation(clean: dict[str, Any]) -> str:
    decision = str(clean.get("decision", "")).strip().upper()
    reason = str(clean.get("skipped_reason") or "").strip()

    edge = _as_float(clean.get("edge_after_price_check"))
    if edge is None:
        edge = _as_float(clean.get("edge"))
    edge_th = _as_float(clean.get("edge_threshold"))
    p_model = _as_float(clean.get("p_model"))
    p_model_max = _as_float(clean.get("p_model_max"))
    price = _as_float(clean.get("chosen_no_ask"))
    max_no_price = _as_float(clean.get("max_no_price"))
    strike_k = _as_int(clean.get("strike_k"))
    mode_k = _as_int(clean.get("mode_k"))
    mode_distance_min = _as_int(clean.get("mode_distance_min"))
    mode_distance = None if strike_k is None or mode_k is None else abs(strike_k - mode_k)

    progression_gate_reason = str(clean.get("progression_gate_reason") or "")
    progression_mult = _as_float(clean.get("progression_confidence_multiplier"))
    ensemble_gate_reason = str(clean.get("ensemble_gate_reason") or "")
    ensemble_mult = _as_float(clean.get("ensemble_size_multiplier"))
    size = _as_float(clean.get("size"))
    stake = _as_float(clean.get("stake_usd"))

    if decision == "TRADE":
        parts = [
            "Passed policy filters",
            f"edge={_fmt_num(edge)} threshold={_fmt_num(edge_th)}",
            f"p_model={_fmt_num(p_model)} max={_fmt_num(p_model_max)}",
            f"price={_fmt_num(price)} max={_fmt_num(max_no_price)}",
            f"mode_distance={mode_distance if mode_distance is not None else 'na'} min={mode_distance_min if mode_distance_min is not None else 'na'}",
            f"progression={progression_gate_reason or 'pass'} mult={_fmt_num(progression_mult)}",
            f"ensemble={ensemble_gate_reason or 'pass'} mult={_fmt_num(ensemble_mult)}",
            f"size={_fmt_num(size, digits=2)} stake={_fmt_num(stake, digits=2)}",
        ]
        return "; ".join(parts)

    if decision == "RESOLVE":
        pnl = _as_float(clean.get("pnl_realized"))
        resolution = str(clean.get("resolution") or "resolved")
        return f"Position resolved: resolution={resolution} pnl={_fmt_num(pnl, digits=2)}"

    if decision == "SELL":
        pnl = _as_float(clean.get("pnl_realized"))
        buy_price = _as_float(clean.get("buy_price"))
        if buy_price is None:
            buy_price = _as_float(clean.get("entry_price"))
        sell_price = _as_float(clean.get("sell_price"))
        if sell_price is None:
            sell_price = _as_float(clean.get("exit_price"))
        sell_reason = str(clean.get("sell_reason") or clean.get("resolution") or "position_sold")
        return (
            "Position sold: "
            f"reason={sell_reason} buy={_fmt_num(buy_price)} sell={_fmt_num(sell_price)} pnl={_fmt_num(pnl, digits=2)}"
        )

    if reason == "edge_too_low":
        return f"Skipped: edge below threshold (edge={_fmt_num(edge)} threshold={_fmt_num(edge_th)})."
    if reason == "p_model_too_high":
        return f"Skipped: model probability too high (p_model={_fmt_num(p_model)} max={_fmt_num(p_model_max)})."
    if reason == "mode_distance_fail":
        return (
            "Skipped: strike too close to mode "
            f"(mode_distance={mode_distance if mode_distance is not None else 'na'} min={mode_distance_min if mode_distance_min is not None else 'na'})."
        )
    if reason == "price_too_high":
        return f"Skipped: NO ask too high (price={_fmt_num(price)} max={_fmt_num(max_no_price)})."
    if reason == "health_gate_blocked":
        return f"Skipped: runtime health gate blocked trading ({clean.get('health_gate_reason') or 'unspecified'})."
    if reason in {"ensemble_high_std", "ensemble_high_range", "ensemble_low_same_side_ratio", "ensemble_strike_disagreement"}:
        return (
            "Skipped: ensemble gate failed "
            f"(reason={reason}, std={_fmt_num(clean.get('ensemble_pred_std'))}, range={_fmt_num(clean.get('ensemble_pred_range'))}, "
            f"same_side={_fmt_num(clean.get('ensemble_same_side_ratio'))})."
        )
    if reason.startswith("progression_"):
        return (
            "Skipped: progression gate failed "
            f"(reason={reason}, score={_fmt_num(clean.get('progression_confidence_score'))}, "
            f"mult={_fmt_num(clean.get('progression_confidence_multiplier'))})."
        )
    if reason:
        return f"Skipped: {reason}."
    return "No trade decision without explicit reason."


def _log_action(
    *,
    logger: logging.Logger,
    jsonl_path: Path | None,
    conn: psycopg.Connection | None,
    run_id: str,
    payload: dict[str, Any],
    emit_info_log: bool = True,
) -> None:
    clean = _clean_for_json(payload)
    clean["decision_reason_code"] = _decision_reason_code(clean)
    clean["decision_explanation"] = _build_decision_explanation(clean)
    log_method = logger.info if emit_info_log else logger.debug
    log_method(
        "[%s] %s %s reason_code=%s edge=%s price=%s explain=%s",
        clean.get("station"),
        clean.get("decision"),
        clean.get("slug") or clean.get("market_id"),
        clean.get("decision_reason_code"),
        _fmt_num(clean.get("edge_after_price_check") if clean.get("edge_after_price_check") is not None else clean.get("edge")),
        _fmt_num(clean.get("chosen_no_ask")),
        clean.get("decision_explanation"),
    )

    if jsonl_path is not None:
        _append_jsonl(jsonl_path, clean)

    if conn is not None:
        try:
            dbmod.insert_live_action(
                conn,
                run_id=run_id,
                station=clean.get("station"),
                market_id=clean.get("market_id"),
                decision=str(clean.get("decision", "")),
                payload=clean,
            )
        except Exception:
            logger.exception("Failed to insert live action row into DB")


def _should_emit_action_info_log(payload: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    decision = str(payload.get("decision") or "").upper()
    if decision in {"TRADE", "RESOLVE"}:
        return True
    if decision == "SKIP":
        return bool(cfg.get("log_skip_decisions_at_info", False))
    return True


def _update_stoploss_and_kills(
    *,
    cfg: dict[str, Any],
    state_store: PilotStateStore,
    stations: list[str],
    day_local: date,
    logger: logging.Logger,
) -> None:
    day_key = day_local.isoformat()
    nav_ref = float(cfg.get("nav_usd", state_store.nav_usd))
    threshold = -abs(float(cfg.get("stoploss_daily_pnl_fraction", 0.01))) * nav_ref

    total_pnl = state_store.daily_realized_pnl(day_local=day_key)
    for station in stations:
        station_pnl = state_store.station_daily_realized_pnl(day_local=day_key, station=station)
        hit = station_pnl < threshold
        state_store.mark_station_stoploss(day_local=day_key, station=station, triggered=hit)
        if hit:
            state_store.set_station_paused(station, True)

    stoploss = state_store.state.setdefault("stoploss", {"consecutive_days_hit": 0, "history": []})
    history = stoploss.setdefault("history", [])
    already_recorded_today = bool(history and str(history[-1].get("day_local")) == day_key)
    if not already_recorded_today:
        streak = state_store.update_stoploss_streak(day_local=day_key, stoploss_hit=(total_pnl < threshold))
        if streak >= int(cfg.get("stoploss_consecutive_days", 3)):
            state_store.set_global_kill(True)
            logger.warning("Global kill activated due to stoploss streak=%d", streak)

    portfolio_risk = state_store.portfolio_conservative_risk_used(day_local=day_key)
    portfolio_limit = float(state_store.nav_usd) * float(cfg.get("portfolio_daily_risk_fraction", 0.05))
    if portfolio_risk >= portfolio_limit:
        state_store.set_global_kill(True)
        logger.warning("Global kill activated due to portfolio daily risk limit (conservative view)")


def _settle_resolved_positions(
    *,
    conn: psycopg.Connection,
    state_store: PilotStateStore,
    run_id: str,
    logger: logging.Logger,
    jsonl_path: Path | None,
    notifier: TelegramNotifier | None,
) -> None:
    open_positions = [p for p in state_store.open_positions() if str(p.get("status", "open")) == "open"]
    if not open_positions:
        return

    market_ids = sorted({str(p.get("market_id")) for p in open_positions if p.get("market_id")})
    if not market_ids:
        return

    resolved = dbmod.fetch_resolved_outcomes(conn, market_ids=market_ids)
    if resolved.empty:
        return

    resolved_map = {str(r.market_id): r for r in resolved.itertuples(index=False)}

    for pos in open_positions:
        market_id = str(pos.get("market_id"))
        row = resolved_map.get(market_id)
        if row is None:
            continue
        if row.no_wins is None:
            continue

        entry_price = float(pos.get("entry_price"))
        size = float(pos.get("size"))
        if bool(row.no_wins):
            pnl = (1.0 - entry_price) * size
            resolution = "no_wins"
        else:
            pnl = -entry_price * size
            resolution = "yes_wins"

        closed = state_store.close_position(
            str(pos.get("position_id")),
            close_ts_utc=utc_now(),
            pnl=pnl,
            resolution=resolution,
        )
        if closed is None:
            continue

        station = str(pos.get("station"))
        day_key = today_local("UTC").isoformat()
        state_store.add_realized_pnl(day_local=day_key, station=station, pnl=pnl)
        state_store.set_nav_usd(state_store.nav_usd + pnl)

        payload = {
            "ts_utc": utc_now().isoformat(),
            "station": station,
            "market_day_local": pos.get("market_day_local"),
            "market_id": market_id,
            "slug": pos.get("slug"),
            "asset_id": pos.get("asset_id"),
            "strike_k": pos.get("strike_k"),
            "mode_k": pos.get("mode_k"),
            "p_model": pos.get("p_model"),
            "entry_price": entry_price,
            "buy_price": entry_price,
            "size": size,
            "decision": "RESOLVE",
            "skipped_reason": "",
            "order_id": pos.get("position_id"),
            "order_status": "closed",
            "pnl_realized": pnl,
            "resolution": resolution,
        }
        _log_action(logger=logger, jsonl_path=jsonl_path, conn=conn, run_id=run_id, payload=payload)
        if notifier is not None:
            notifier.notify_trade(payload, logger=logger)


def _apply_trade_stoplosses(
    *,
    cfg: Mapping[str, Any],
    state_store: PilotStateStore,
    conn: psycopg.Connection,
    snapshot_info: dbmod.SnapshotTableInfo,
    exec_client: Any,
    dry_run: bool,
    run_id: str,
    logger: logging.Logger,
    jsonl_path: Path | None,
    notifier: TelegramNotifier | None,
    now_utc: datetime,
) -> None:
    enabled, default_loss_fraction, default_break_even = _trade_stoploss_settings(cfg)
    if not enabled:
        return

    open_positions = [p for p in state_store.open_positions() if str(p.get("status", "open")) == "open"]
    if not open_positions:
        return

    market_ids = sorted({str(p.get("market_id")) for p in open_positions if p.get("market_id")})
    if not market_ids:
        return

    snapshot_df = dbmod.fetch_latest_snapshots(
        conn,
        snapshot_table=snapshot_info,
        market_ids=market_ids,
        lookback_per_outcome=int(cfg.get("snapshot_quote_lookback_per_outcome", 12)),
    )
    snapshot_map = {
        str(row.market_id): {
            "yes_snapshot_ts_utc": row.yes_snapshot_ts_utc,
            "no_snapshot_ts_utc": row.no_snapshot_ts_utc,
            "best_yes_bid": row.best_yes_bid,
            "best_yes_ask": row.best_yes_ask,
            "best_no_bid": row.best_no_bid,
            "best_no_ask": row.best_no_ask,
            "yes_bid_size": row.yes_bid_size,
            "yes_ask_size": row.yes_ask_size,
            "no_bid_size": row.no_bid_size,
            "no_ask_size": row.no_ask_size,
        }
        for row in snapshot_df.itertuples(index=False)
    }

    max_snapshot_age_minutes = float(cfg.get("max_snapshot_age_minutes", 30))
    base_tz = str(cfg.get("timezones", {}).get("default", "UTC"))
    day_key = today_local(base_tz, now_utc=now_utc).isoformat()

    for pos in open_positions:
        market_id = str(pos.get("market_id") or "").strip()
        if not market_id:
            continue

        entry_price = _as_float(pos.get("entry_price"))
        size = _as_float(pos.get("size"))
        if entry_price is None or size is None or size <= POSITION_EPS:
            continue

        stoploss_enabled = bool(pos.get("stop_loss_enabled", enabled))
        if not stoploss_enabled:
            continue

        loss_fraction = _as_float(pos.get("stop_loss_loss_fraction"))
        if loss_fraction is None:
            loss_fraction = default_loss_fraction
        loss_fraction = min(0.99, max(0.0, float(loss_fraction)))

        break_even_on_recovery = bool(pos.get("stop_loss_break_even_on_recovery", default_break_even))
        break_even_armed = bool(pos.get("stop_loss_break_even_armed", False))

        trigger_price = _as_float(pos.get("stop_loss_trigger_price"))
        if trigger_price is None:
            trigger_price = max(0.0, entry_price * (1.0 - loss_fraction))

        snapshot = snapshot_map.get(market_id)
        mark_price, mark_source, mark_ts, mark_age_minutes, mark_reason = _compute_stoploss_mark_price(
            snapshot=snapshot,
            now_utc=now_utc,
            max_snapshot_age_minutes=max_snapshot_age_minutes,
        )
        if mark_price is None or mark_reason is not None:
            logger.debug(
                "Stop-loss mark unavailable position_id=%s market_id=%s reason=%s",
                pos.get("position_id"),
                market_id,
                mark_reason or "unknown",
            )
            continue

        if break_even_on_recovery and (not break_even_armed) and mark_price > (entry_price + POSITION_EPS):
            break_even_armed = True
            trigger_price = entry_price
            logger.info(
                "Stop-loss break-even armed position_id=%s market_id=%s entry=%.4f mark=%.4f",
                pos.get("position_id"),
                market_id,
                entry_price,
                mark_price,
            )

        if break_even_armed:
            trigger_price = max(trigger_price, entry_price)

        pos["stop_loss_enabled"] = stoploss_enabled
        pos["stop_loss_loss_fraction"] = loss_fraction
        pos["stop_loss_break_even_on_recovery"] = break_even_on_recovery
        pos["stop_loss_break_even_armed"] = break_even_armed
        pos["stop_loss_trigger_price"] = trigger_price
        pos["stop_loss_last_mark_price"] = mark_price
        pos["stop_loss_last_mark_source"] = mark_source
        pos["stop_loss_last_mark_ts_utc"] = mark_ts.isoformat() if mark_ts is not None else None
        pos["stop_loss_last_mark_age_minutes"] = mark_age_minutes

        if mark_price > (trigger_price + POSITION_EPS):
            continue

        if dry_run:
            logger.info(
                "Stop-loss trigger (dry-run) position_id=%s market_id=%s mark=%.4f trigger=%.4f",
                pos.get("position_id"),
                market_id,
                mark_price,
                trigger_price,
            )
            continue

        try:
            order_id = exec_client.place_order(
                market_id=market_id,
                side="sell",
                outcome="NO",
                price=mark_price,
                size=size,
                metadata={
                    "reason": "stop_loss",
                    "trigger_price": trigger_price,
                    "mark_source": mark_source,
                    "mark_age_minutes": mark_age_minutes,
                },
            )
        except Exception:
            logger.exception(
                "Stop-loss order placement failed position_id=%s market_id=%s",
                pos.get("position_id"),
                market_id,
            )
            continue

        order_status = "submitted"
        filled_size = float(size)
        filled_price = float(mark_price)
        if hasattr(exec_client, "execution_result"):
            try:
                res = exec_client.execution_result(order_id)
            except Exception:
                res = None
            if res is not None:
                order_status = str(res.order_status)
                if res.filled_size is not None:
                    filled_size = max(0.0, float(res.filled_size))
                if res.filled_price is not None:
                    filled_price = float(res.filled_price)

        filled_size = min(size, filled_size)
        if filled_size <= POSITION_EPS:
            logger.warning(
                "Stop-loss order not filled position_id=%s market_id=%s status=%s",
                pos.get("position_id"),
                market_id,
                order_status,
            )
            continue

        pnl = (filled_price - entry_price) * filled_size
        state_store.add_realized_pnl(day_local=day_key, station=str(pos.get("station")), pnl=pnl)
        state_store.set_nav_usd(state_store.nav_usd + pnl)

        close_ts = utc_now()
        remaining_size = max(0.0, size - filled_size)
        if remaining_size <= POSITION_EPS:
            state_store.close_position(
                str(pos.get("position_id")),
                close_ts_utc=close_ts,
                pnl=pnl,
                resolution="stop_loss",
            )
        else:
            pos["size"] = remaining_size
            pos["stake_usd"] = max(0.0, entry_price * remaining_size)

        payload = {
            "ts_utc": close_ts.isoformat(),
            "station": pos.get("station"),
            "market_day_local": pos.get("market_day_local"),
            "market_id": market_id,
            "slug": pos.get("slug"),
            "asset_id": pos.get("asset_id"),
            "strike_k": pos.get("strike_k"),
            "mode_k": pos.get("mode_k"),
            "p_model": pos.get("p_model"),
            "entry_price": entry_price,
            "buy_price": entry_price,
            "sell_price": filled_price,
            "exit_price": filled_price,
            "chosen_no_ask": filled_price,
            "size": filled_size,
            "lot": filled_size,
            "decision": "SELL",
            "side": "SELL",
            "sell_reason": "stop_loss",
            "skipped_reason": "",
            "order_side": "sell",
            "order_outcome": "NO",
            "order_id": order_id,
            "order_status": order_status,
            "pnl_realized": pnl,
            "total_loss": abs(min(0.0, pnl)),
            "stop_loss_trigger_price": trigger_price,
            "stop_loss_break_even_armed": break_even_armed,
            "mark_price": mark_price,
            "mark_source": mark_source,
            "mark_age_minutes": mark_age_minutes,
            "position_id": pos.get("position_id"),
        }
        _log_action(logger=logger, jsonl_path=jsonl_path, conn=conn, run_id=run_id, payload=payload, emit_info_log=True)
        if notifier is not None:
            notifier.notify_trade(payload, logger=logger)


def _run_open_position_maintenance(
    *,
    cfg: dict[str, Any],
    state_store: PilotStateStore,
    stations: list[str],
    conn: psycopg.Connection,
    snapshot_info: dbmod.SnapshotTableInfo,
    exec_client: Any,
    dry_run: bool,
    run_id: str,
    logger: logging.Logger,
    jsonl_path: Path | None,
    notifier: TelegramNotifier | None,
    now_utc: datetime,
) -> None:
    _settle_resolved_positions(
        conn=conn,
        state_store=state_store,
        run_id=run_id,
        logger=logger,
        jsonl_path=jsonl_path,
        notifier=notifier,
    )
    _apply_trade_stoplosses(
        cfg=cfg,
        state_store=state_store,
        conn=conn,
        snapshot_info=snapshot_info,
        exec_client=exec_client,
        dry_run=dry_run,
        run_id=run_id,
        logger=logger,
        jsonl_path=jsonl_path,
        notifier=notifier,
        now_utc=now_utc,
    )
    base_tz = str(cfg.get("timezones", {}).get("default", "UTC"))
    local_day = today_local(base_tz, now_utc=now_utc)
    _update_stoploss_and_kills(
        cfg=cfg,
        state_store=state_store,
        stations=stations,
        day_local=local_day,
        logger=logger,
    )


def _should_emit_daily_report(*, cfg: dict[str, Any], state_store: PilotStateStore, now_local: datetime) -> bool:
    report_hhmm = str(cfg.get("daily_report_time_local", "23:59"))
    target_h, target_m = [int(x) for x in report_hhmm.split(":", 1)]
    target_now = now_local.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    if now_local < target_now:
        return False

    last_report = state_store.last_report_date_local()
    if last_report == now_local.date().isoformat():
        return False
    return True


def _latest_probability_date(prob: pd.DataFrame) -> date | None:
    if "date" not in prob.columns:
        return None
    latest_ts = pd.to_datetime(prob["date"], errors="coerce").max()
    if pd.isna(latest_ts):
        return None
    return latest_ts.date()


def _evaluate_runtime_health_gate(
    *,
    cfg: dict[str, Any],
    prob: pd.DataFrame,
    prob_path: Path,
    manifest: dict[str, Any] | None,
    conn: psycopg.Connection,
    snapshot_info: dbmod.SnapshotTableInfo,
    now_utc: datetime,
) -> list[str]:
    reasons: list[str] = []

    max_prob_age_days = max(0, int(cfg.get("max_probability_age_days", 2)))
    latest_prob_date = _latest_probability_date(prob)
    min_allowed_date = now_utc.date() - timedelta(days=max_prob_age_days)
    prob_fresh = latest_prob_date is not None and latest_prob_date >= min_allowed_date
    if bool(cfg.get("pause_on_stale_probabilities", True)) and not prob_fresh:
        cycle_txt = "" if manifest is None else str(manifest.get("cycle") or "").strip()
        reasons.append(
            "stale_probabilities "
            f"path={prob_path} cycle={cycle_txt or 'unknown'} "
            f"latest_date={latest_prob_date.isoformat() if latest_prob_date is not None else 'none'} "
            f"max_age_days={max_prob_age_days}"
        )

    try:
        max_ts = dbmod.fetch_snapshots_freshness(conn, snapshot_table=snapshot_info)
        if max_ts is None:
            if bool(cfg.get("pause_on_stale_snapshots", True)):
                reasons.append("stale_snapshots no snapshot rows in snapshot table")
        else:
            age_m = max(0.0, (now_utc - max_ts).total_seconds() / 60.0)
            limit = float(cfg.get("max_snapshot_age_minutes", 30))
            if bool(cfg.get("pause_on_stale_snapshots", True)) and age_m > limit:
                reasons.append(
                    "stale_snapshots "
                    f"table={snapshot_info.table_name} age_minutes={age_m:.1f} limit={limit:.1f}"
                )
    except Exception as exc:
        if bool(cfg.get("pause_on_stale_snapshots", True)):
            reasons.append(f"stale_snapshots table_check_error={exc.__class__.__name__}: {exc}")

    return reasons


def _should_emit_health_gate_alert(
    *,
    state_store: PilotStateStore,
    reason_text: str,
    now_utc: datetime,
    cooldown_minutes: float,
) -> bool:
    runtime_alerts = state_store.state.setdefault("runtime_alerts", {})
    last = runtime_alerts.get("health_gate_alert")
    if not isinstance(last, dict):
        last = {}
        runtime_alerts["health_gate_alert"] = last

    last_reason = str(last.get("reason", ""))
    last_sent_raw = last.get("sent_at_utc")
    last_sent = pd.to_datetime(last_sent_raw, utc=True, errors="coerce")
    cooldown_seconds = max(0.0, float(cooldown_minutes) * 60.0)

    if (
        bool(reason_text)
        and reason_text == last_reason
        and (not pd.isna(last_sent))
        and (now_utc - last_sent.to_pydatetime()).total_seconds() < cooldown_seconds
    ):
        return False

    last["reason"] = str(reason_text)
    last["sent_at_utc"] = now_utc.isoformat()
    return True


def _build_health_gate_blocked_policy(
    candidates: pd.DataFrame,
    *,
    reason_text: str,
    station_risk_limit: float,
    portfolio_risk_limit: float,
) -> pd.DataFrame:
    out = candidates.copy()
    def _col_series(name: str, default: Any) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series([default] * len(out))

    if "p_model_raw" not in out.columns:
        out["p_model_raw"] = pd.to_numeric(out["p_model"], errors="coerce")
    if "p_model_adjusted" not in out.columns:
        out["p_model_adjusted"] = pd.to_numeric(out["p_model"], errors="coerce")
    out["NO_true"] = 1.0 - pd.to_numeric(out["p_model"], errors="coerce")
    out["edge"] = pd.NA
    out["decision"] = "SKIP"
    out["skipped_reason"] = "health_gate_blocked"
    out["health_gate_reason"] = str(reason_text)
    out["size"] = 0.0
    out["stake_usd"] = 0.0
    out["base_size_before_progression"] = 0.0
    out["final_size_after_progression"] = 0.0
    out["base_stake_usd_before_progression"] = 0.0
    out["final_stake_usd_after_progression"] = 0.0
    out["final_size_after_ensemble"] = 0.0
    out["final_stake_usd_after_ensemble"] = 0.0
    out["progression_gate_pass"] = False
    out["progression_gate_reason"] = "health_gate_blocked"
    out["progression_confidence_score"] = 0.5
    out["progression_confidence_multiplier"] = 1.0
    out["ensemble_agreement_score"] = pd.to_numeric(_col_series("ensemble_agreement_score", 0.5), errors="coerce").fillna(0.5)
    out["ensemble_disagreement_score"] = pd.to_numeric(_col_series("ensemble_disagreement_score", 0.5), errors="coerce").fillna(0.5)
    out["ensemble_sign_agreement_ratio"] = pd.to_numeric(_col_series("ensemble_sign_agreement_ratio", 0.5), errors="coerce").fillna(0.5)
    out["ensemble_cross_strike_disagreement"] = _coerce_bool_series(
        _col_series("ensemble_cross_strike_disagreement", False),
        default=False,
    )
    out["ensemble_models_yes_count"] = pd.to_numeric(_col_series("ensemble_models_yes_count", 0), errors="coerce").fillna(0).astype(int)
    out["ensemble_models_no_count"] = pd.to_numeric(_col_series("ensemble_models_no_count", 0), errors="coerce").fillna(0).astype(int)
    out["ensemble_same_side_ratio"] = pd.to_numeric(_col_series("ensemble_same_side_ratio", 1.0), errors="coerce").fillna(1.0)
    out["ensemble_strike_disagreement_flag"] = _coerce_bool_series(
        _col_series("ensemble_strike_disagreement_flag", False),
        default=False,
    )
    out["ensemble_confidence_multiplier"] = pd.to_numeric(_col_series("ensemble_confidence_multiplier", 1.0), errors="coerce").fillna(1.0)
    out["ensemble_uncertainty_penalty"] = pd.to_numeric(_col_series("ensemble_uncertainty_penalty", 0.0), errors="coerce").fillna(0.0)
    out["ensemble_size_multiplier"] = pd.to_numeric(_col_series("ensemble_size_multiplier", 1.0), errors="coerce").fillna(1.0)
    out["ensemble_gate_pass"] = False
    out["ensemble_gate_reason"] = "health_gate_blocked"
    if "ensemble_fallback_marker" not in out.columns:
        out["ensemble_fallback_marker"] = ""
    out["risk_used_station_today"] = 0.0
    out["risk_used_station_daily"] = 0.0
    out["risk_used_station_open"] = 0.0
    out["risk_limit_station_today"] = float(station_risk_limit)
    out["risk_used_portfolio_today"] = 0.0
    out["risk_used_portfolio_daily"] = 0.0
    out["risk_used_portfolio_open"] = 0.0
    out["risk_limit_portfolio_today"] = float(portfolio_risk_limit)
    return out


def run_healthcheck(cfg: dict[str, Any]) -> int:
    checks: list[tuple[str, bool, str]] = []

    allowlist = [str(s).strip() for s in cfg.get("stations_allowlist", []) if str(s).strip()]
    checks.append(("station_allowlist_non_empty", bool(allowlist), f"count={len(allowlist)}"))

    prob_root_path = resolve_path(str(cfg.get("probabilities_path")))
    try:
        prob_path, manifest = resolve_probability_data_path(prob_root_path)
        raw = read_probability_files(prob_path)
        prob = standardize_probabilities(raw)
        latest_date = _latest_probability_date(prob)
        latest_date_txt = "none" if latest_date is None else latest_date.isoformat()
        max_prob_age_days = max(0, int(cfg.get("max_probability_age_days", 2)))
        fresh = latest_date is not None and latest_date >= utc_now().date() - timedelta(days=max_prob_age_days)
        cycle_txt = ""
        if manifest is not None and manifest.get("cycle") is not None:
            cycle_txt = f" cycle={manifest.get('cycle')}"
        manifest_txt = ""
        if manifest is not None and manifest.get("_resolved_manifest_path") is not None:
            manifest_txt = f" manifest={manifest.get('_resolved_manifest_path')}"
        checks.append(
            (
                "probabilities_recent",
                bool(fresh),
                f"path={prob_path}{cycle_txt}{manifest_txt} latest_date={latest_date_txt} max_age_days={max_prob_age_days}",
            )
        )
    except Exception as exc:
        checks.append(("probabilities_recent", False, f"path={prob_root_path} error={exc}"))

    try:
        with dbmod.connect_db(str(cfg.get("db_dsn"))) as conn:
            snap_info = dbmod.detect_snapshot_table(conn)
            max_ts = dbmod.fetch_snapshots_freshness(conn, snapshot_table=snap_info)
            checks.append(("db_connectivity", True, f"snapshot_table={snap_info.table_name}"))
            if max_ts is None:
                checks.append(("snapshots_freshness", False, "no snapshot rows"))
            else:
                age_m = max(0.0, (utc_now() - max_ts).total_seconds() / 60.0)
                limit = float(cfg.get("max_snapshot_age_minutes", 30))
                checks.append(("snapshots_freshness", age_m <= limit, f"age_minutes={age_m:.1f} limit={limit:.1f}"))
    except Exception as exc:
        checks.append(("db_connectivity", False, str(exc)))
        checks.append(("snapshots_freshness", False, "db unavailable"))

    failed = False
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed = True
    return 1 if failed else 0


def run_cycle(
    *,
    cfg: dict[str, Any],
    run_id: str,
    logger: logging.Logger,
    state_store: PilotStateStore,
    output_dir: Path,
    dry_run: bool,
    conn: psycopg.Connection,
    snapshot_info: dbmod.SnapshotTableInfo,
    notifier: TelegramNotifier | None,
) -> None:
    now = utc_now()
    jsonl_path = output_dir / "logs" / f"trades_{to_yyyymmdd(now.date())}.jsonl" if bool(cfg.get("write_jsonl_log", True)) else None
    pending_progression_dir = output_dir / "state" / "forecast_progression_pending"

    processed_batches = apply_pending_progression_updates(
        state_store.state,
        pending_progression_dir,
        logger=logger,
    )
    if processed_batches:
        state_store.persist()
        removed = 0
        for batch_path in processed_batches:
            try:
                batch_path.unlink(missing_ok=True)
                removed += 1
            except Exception as exc:
                logger.warning(
                    "Failed to remove processed forecast progression batch path=%s error=%s: %s",
                    batch_path,
                    exc.__class__.__name__,
                    exc,
                )
        logger.info(
            "Applied %d queued forecast progression batch file(s), removed=%d",
            len(processed_batches),
            removed,
        )

    prob_root_path = resolve_path(str(cfg.get("probabilities_path")))
    prob_path, manifest = resolve_probability_data_path(prob_root_path)
    raw_prob = read_probability_files(prob_path)
    prob = standardize_probabilities(raw_prob)
    if manifest is not None:
        logger.info(
            "Using probabilities manifest source=%s cycle=%s manifest=%s file=%s",
            manifest.get("_resolved_source"),
            manifest.get("cycle"),
            manifest.get("_resolved_manifest_path"),
            prob_path,
        )

    gate_reasons = _evaluate_runtime_health_gate(
        cfg=cfg,
        prob=prob,
        prob_path=prob_path,
        manifest=manifest,
        conn=conn,
        snapshot_info=snapshot_info,
        now_utc=now,
    )
    health_gate_blocked = bool(gate_reasons)
    health_gate_reason_text = "; ".join(gate_reasons)
    if health_gate_blocked:
        logger.error("Trading paused by health gate: %s", health_gate_reason_text)
        if notifier is not None and _should_emit_health_gate_alert(
            state_store=state_store,
            reason_text=health_gate_reason_text,
            now_utc=now,
            cooldown_minutes=float(cfg.get("health_gate_alert_cooldown_minutes", 30)),
        ):
            notifier.notify_alert(
                f"Live pilot paused: {health_gate_reason_text}",
                logger=logger,
                channel="trades",
            )
        elif notifier is not None:
            logger.info("Suppressed duplicate health gate alert within cooldown window")

    stations = [str(s).strip() for s in cfg.get("stations_allowlist", []) if str(s).strip()]
    station_tz = build_station_timezone_map(cfg, stations)
    exec_client = _build_execution_client(cfg)

    universe = select_live_universe(prob=prob, cfg=cfg, station_tz=station_tz, now_utc=now)
    if universe.empty:
        logger.info("No probability rows in live universe after station/date/cutoff filters")
        _run_open_position_maintenance(
            cfg=cfg,
            state_store=state_store,
            stations=stations,
            conn=conn,
            snapshot_info=snapshot_info,
            exec_client=exec_client,
            dry_run=dry_run,
            run_id=run_id,
            logger=logger,
            jsonl_path=jsonl_path,
            notifier=notifier,
            now_utc=now,
        )
        return

    universe, _unmapped = resolve_open_market_universe(
        universe=universe,
        conn=conn,
        station_tz=station_tz,
        cfg=cfg,
        logger=logger,
    )
    if universe.empty:
        logger.info("No mapped rows in open-market universe after resolver")
        _run_open_position_maintenance(
            cfg=cfg,
            state_store=state_store,
            stations=stations,
            conn=conn,
            snapshot_info=snapshot_info,
            exec_client=exec_client,
            dry_run=dry_run,
            run_id=run_id,
            logger=logger,
            jsonl_path=jsonl_path,
            notifier=notifier,
            now_utc=now,
        )
        return

    market_ids = sorted(universe["market_id"].dropna().astype(str).unique().tolist())
    snapshot_df = dbmod.fetch_latest_snapshots(
        conn,
        snapshot_table=snapshot_info,
        market_ids=market_ids,
        lookback_per_outcome=int(cfg.get("snapshot_quote_lookback_per_outcome", 12)),
    )
    snapshot_map = {
        str(row.market_id): {
            "yes_snapshot_ts_utc": row.yes_snapshot_ts_utc,
            "no_snapshot_ts_utc": row.no_snapshot_ts_utc,
            "best_yes_bid": row.best_yes_bid,
            "best_yes_ask": row.best_yes_ask,
            "best_no_bid": row.best_no_bid,
            "best_no_ask": row.best_no_ask,
            "yes_bid_size": row.yes_bid_size,
            "yes_ask_size": row.yes_ask_size,
            "no_bid_size": row.no_bid_size,
            "no_ask_size": row.no_ask_size,
        }
        for row in snapshot_df.itertuples(index=False)
    }

    records: list[dict[str, Any]] = []
    for row in universe.itertuples(index=False):
        row_dict = row._asdict()
        market_day_ts = pd.to_datetime(row.market_day_local, errors="coerce")
        if pd.isna(market_day_ts):
            continue
        snapshot = None if pd.isna(row.market_id) else snapshot_map.get(str(row.market_id))
        pricing = compute_pricing_decision(
            snapshot=snapshot,
            now_utc=now,
            max_snapshot_age_minutes=float(cfg.get("max_snapshot_age_minutes", 30)),
            slippage_buffer_yes_fallback=float(cfg.get("slippage_buffer_yes_fallback", 0.01)),
            max_spread=float(cfg.get("max_spread", 0.05)),
        )
        p_model_raw_val = row_dict.get("p_model_raw")
        if p_model_raw_val is None or pd.isna(p_model_raw_val):
            p_model_raw_val = row.p_model
        p_model_adjusted_val = row_dict.get("p_model_adjusted")
        if p_model_adjusted_val is None or pd.isna(p_model_adjusted_val):
            p_model_adjusted_val = row.p_model

        record = {
            "ts_utc": now.isoformat(),
            "station": str(row.station),
            "market_day_local": market_day_ts.date().isoformat(),
            "market_id": None if pd.isna(row.market_id) else str(row.market_id),
            "slug": str(row.slug),
            "asset_id": None if pd.isna(getattr(row, "asset_id", pd.NA)) else str(getattr(row, "asset_id")),
            "event_key": str(row.event_key),
            "strike_k": int(row.strike_k),
            "mode_k": int(row.mode_k),
            "p_model": float(row.p_model),
            "p_model_raw": float(p_model_raw_val),
            "p_model_adjusted": float(p_model_adjusted_val),
            "execution_time_utc": pd.to_datetime(row.execution_time_utc, utc=True, errors="coerce").isoformat() if not pd.isna(pd.to_datetime(row.execution_time_utc, utc=True, errors="coerce")) else None,
            "snapshot_ts_utc": pricing.snapshot_ts_utc.isoformat() if pricing.snapshot_ts_utc else None,
            "snapshot_age_minutes": pricing.snapshot_age_minutes,
            "best_yes_bid": snapshot.get("best_yes_bid") if snapshot else None,
            "best_yes_ask": snapshot.get("best_yes_ask") if snapshot else None,
            "best_no_bid": snapshot.get("best_no_bid") if snapshot else None,
            "best_no_ask": snapshot.get("best_no_ask") if snapshot else None,
            "chosen_no_ask": pricing.chosen_no_ask,
            "price_source": pricing.price_source,
            "spread": pricing.spread,
            "snapshot_skip_reason": pricing.skipped_reason,
            "edge_threshold": float(cfg.get("edge_threshold", 0.02)),
            "p_model_max": float(cfg.get("p_model_max", 0.12)),
            "max_no_price": float(cfg.get("max_no_price", 0.92)),
            "mode_distance_min": int(cfg.get("mode_distance_min", 2)),
            "NAV": float(state_store.nav_usd),
            "order_id": None,
            "order_status": None,
            "error": None,
        }
        optional_numeric = [
            "pred_model_1",
            "pred_model_2",
            "pred_model_3",
            "ensemble_pred_median",
            "ensemble_pred_mean",
            "ensemble_pred_min",
            "ensemble_pred_max",
            "ensemble_pred_range",
            "ensemble_pred_std",
            "ensemble_pred_iqr",
            "ensemble_bullish_count",
            "ensemble_bearish_count",
            "ensemble_agreement_score",
            "ensemble_disagreement_score",
            "ensemble_sign_agreement_ratio",
            "ensemble_models_yes_count",
            "ensemble_models_no_count",
            "ensemble_same_side_ratio",
            "ensemble_confidence_multiplier",
            "ensemble_uncertainty_penalty",
            "ensemble_size_multiplier",
        ]
        for key in optional_numeric:
            value = row_dict.get(key)
            if value is None or pd.isna(value):
                continue
            try:
                record[key] = float(value)
            except Exception:
                pass

        optional_bool = [
            "ensemble_cross_strike_disagreement",
            "ensemble_strike_disagreement_flag",
            "ensemble_gate_pass",
        ]
        for key in optional_bool:
            value = row_dict.get(key)
            if value is None or pd.isna(value):
                continue
            record[key] = bool(value)

        if row_dict.get("ensemble_gate_reason") is not None and not pd.isna(row_dict.get("ensemble_gate_reason")):
            record["ensemble_gate_reason"] = str(row_dict.get("ensemble_gate_reason"))
        if row_dict.get("ensemble_fallback_marker") is not None and not pd.isna(row_dict.get("ensemble_fallback_marker")):
            record["ensemble_fallback_marker"] = str(row_dict.get("ensemble_fallback_marker"))

        for key, value in row_dict.items():
            if not key.startswith("pred_"):
                continue
            if key in record:
                continue
            if value is None or pd.isna(value):
                continue
            try:
                record[key] = float(value)
            except Exception:
                continue
        records.append(record)

    candidates = pd.DataFrame.from_records(records)
    if candidates.empty:
        logger.info("No candidates generated")
        _run_open_position_maintenance(
            cfg=cfg,
            state_store=state_store,
            stations=stations,
            conn=conn,
            snapshot_info=snapshot_info,
            exec_client=exec_client,
            dry_run=dry_run,
            run_id=run_id,
            logger=logger,
            jsonl_path=jsonl_path,
            notifier=notifier,
            now_utc=now,
        )
        return

    candidates = attach_progression_features(state_store.state, candidates)

    if health_gate_blocked:
        nav_usd = float(state_store.nav_usd)
        policy_out = _build_health_gate_blocked_policy(
            candidates,
            reason_text=health_gate_reason_text,
            station_risk_limit=nav_usd * float(cfg.get("station_daily_risk_fraction", 0.02)),
            portfolio_risk_limit=nav_usd * float(cfg.get("portfolio_daily_risk_fraction", 0.05)),
        )
    else:
        policy_ctx = PolicyContext(
            nav_usd=float(state_store.nav_usd),
            nav_peak_usd=float(state_store.nav_peak_usd),
            mode_distance_min=int(cfg.get("mode_distance_min", 2)),
            p_model_max=float(cfg.get("p_model_max", 0.12)),
            edge_threshold=float(cfg.get("edge_threshold", 0.02)),
            max_no_price=float(cfg.get("max_no_price", 0.92)),
            top_n_per_event_day=int(cfg.get("top_n_per_event_day", 2)),
            stake_fraction=float(cfg.get("stake_fraction", 0.005)),
            stake_cap_usd=float(cfg.get("stake_cap_usd", 50)),
            min_order_size=float(cfg.get("min_order_size", 1)),
            station_daily_risk_fraction=float(cfg.get("station_daily_risk_fraction", 0.02)),
            portfolio_daily_risk_fraction=float(cfg.get("portfolio_daily_risk_fraction", 0.05)),
            max_open_positions_per_station=int(cfg.get("max_open_positions_per_station", 4)),
            max_open_positions_total=int(cfg.get("max_open_positions_total", 20)),
            trade_cooldown_minutes=float(cfg.get("trade_cooldown_minutes", 30)),
            drawdown_position_scaling=bool(cfg.get("drawdown_position_scaling", True)),
            max_drawdown_fraction=float(cfg.get("max_drawdown_fraction", 0.2)),
            min_drawdown_scale=float(cfg.get("min_drawdown_scale", 0.25)),
            trade_window_start_local=str(cfg.get("trade_window", {}).get("start_local", "00:00")),
            trade_window_end_local=str(cfg.get("trade_window", {}).get("end_local", "12:00")),
            use_progression_confidence=bool(cfg.get("use_progression_confidence", True)),
            progression_enable_gate=bool(cfg.get("progression_enable_gate", True)),
            progression_min_cycles_seen=int(cfg.get("progression_min_cycles_seen", 3)),
            progression_min_consecutive_candidate_cycles=int(
                cfg.get("progression_min_consecutive_candidate_cycles", 2)
            ),
            progression_enable_negative_veto=bool(cfg.get("progression_enable_negative_veto", True)),
            progression_negative_edge_trend_threshold=float(
                cfg.get("progression_negative_edge_trend_threshold", -0.01)
            ),
            progression_min_mode_consistency_ratio=float(cfg.get("progression_min_mode_consistency_ratio", 0.40)),
            progression_negative_p_model_trend_threshold=float(
                cfg.get("progression_negative_p_model_trend_threshold", 0.01)
            ),
            progression_weight_consecutive=float(cfg.get("progression_weight_consecutive", 0.30)),
            progression_weight_candidate_ratio=float(cfg.get("progression_weight_candidate_ratio", 0.20)),
            progression_weight_edge_trend=float(cfg.get("progression_weight_edge_trend", 0.20)),
            progression_weight_mode_consistency=float(cfg.get("progression_weight_mode_consistency", 0.15)),
            progression_weight_low_p_model=float(cfg.get("progression_weight_low_p_model", 0.10)),
            progression_weight_low_edge_volatility=float(cfg.get("progression_weight_low_edge_volatility", 0.05)),
            progression_edge_trend_cap=float(cfg.get("progression_edge_trend_cap", 0.05)),
            progression_enable_size_multiplier=bool(cfg.get("progression_enable_size_multiplier", True)),
            progression_min_size_multiplier=float(cfg.get("progression_min_size_multiplier", 0.85)),
            progression_max_size_multiplier=float(cfg.get("progression_max_size_multiplier", 1.35)),
            use_ensemble_confidence=bool(cfg.get("use_ensemble_confidence", True)),
            ensemble_probability_adjustment_enabled=bool(
                cfg.get("ensemble_probability_adjustment_enabled", True)
            ),
            ensemble_trade_size_adjustment_enabled=bool(
                cfg.get("ensemble_trade_size_adjustment_enabled", True)
            ),
            ensemble_disagreement_neutral_shrink_cap=float(
                cfg.get("ensemble_disagreement_neutral_shrink_cap", 0.25)
            ),
            ensemble_std_cap_c=float(cfg.get("ensemble_std_cap_c", 2.0)),
            ensemble_range_cap_c=float(cfg.get("ensemble_range_cap_c", 4.0)),
            ensemble_enable_gate=bool(cfg.get("ensemble_enable_gate", True)),
            ensemble_min_same_side_ratio=float(cfg.get("ensemble_min_same_side_ratio", 0.67)),
            ensemble_max_std_c_for_trade=float(cfg.get("ensemble_max_std_c_for_trade", 2.5)),
            ensemble_max_range_c_for_trade=float(cfg.get("ensemble_max_range_c_for_trade", 5.0)),
            ensemble_enable_strike_disagreement_veto=bool(
                cfg.get("ensemble_enable_strike_disagreement_veto", True)
            ),
            ensemble_min_size_multiplier=float(cfg.get("ensemble_min_size_multiplier", 0.75)),
            ensemble_max_size_multiplier=float(cfg.get("ensemble_max_size_multiplier", 1.15)),
        )

        policy_out = apply_policy(
            candidates=candidates,
            state_store=state_store,
            ctx=policy_ctx,
            now_utc=now,
            station_timezones=station_tz,
        )

    csv_trades_rows: list[dict[str, Any]] = []
    trade_stoploss_enabled, trade_stoploss_loss_fraction, trade_stoploss_break_even = _trade_stoploss_settings(cfg)
    trade_cooldown_minutes = max(0.0, float(cfg.get("trade_cooldown_minutes", 30)))
    state_store.prune_recent_order_keys(retention_hours=ORDER_KEY_RETENTION_HOURS, now_utc=now)
    state_store.prune_trade_cooldowns(cooldown_minutes=trade_cooldown_minutes, now_utc=now)
    cycle_reserved_order_keys: set[str] = state_store.recent_order_key_set(
        retention_hours=ORDER_KEY_RETENTION_HOURS,
        now_utc=now,
    )
    cycle_reserved_position_identity_keys: set[str] = state_store.open_position_identity_keys()
    cycle_reserved_cooldown_identity_keys: set[str] = state_store.active_trade_cooldown_keys(
        cooldown_minutes=trade_cooldown_minutes,
        now_utc=now,
    )
    decision_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()

    for row in policy_out.itertuples(index=False):
        payload = row._asdict()
        payload.setdefault("decision", "SKIP")
        payload.setdefault("skipped_reason", "")
        payload["ts_utc"] = now.isoformat()
        payload["duplicate_guard_triggered"] = False
        payload["idempotency_guard_triggered"] = False
        payload["cooldown_guard_triggered"] = False
        payload["chosen_no_ask_original"] = payload.get("chosen_no_ask")
        payload["chosen_no_ask_refreshed"] = None
        payload["edge_before_price_check"] = payload.get("edge")
        payload["edge_after_price_check"] = payload.get("edge")
        payload["price_check_skip_reason"] = None
        payload["price_drift"] = None

        day_key = str(payload.get("market_day_local"))
        station = str(payload.get("station"))
        payload["station_risk_before_trade"] = state_store.station_conservative_risk_used(day_local=day_key, station=station)
        payload["portfolio_risk_before_trade"] = state_store.portfolio_conservative_risk_used(day_local=day_key)
        payload["station_risk_after_trade"] = payload["station_risk_before_trade"]
        payload["portfolio_risk_after_trade"] = payload["portfolio_risk_before_trade"]

        if payload["decision"] == "TRADE":
            position_identity_key = state_store.position_identity_key(
                station=payload.get("station"),
                market_day_local=payload.get("market_day_local"),
                strike_k=payload.get("strike_k"),
            )
            payload["position_identity_key"] = position_identity_key
            payload["order_side"] = "buy"
            payload["order_outcome"] = "NO"
            payload["order_key"] = None
            if dry_run:
                payload["order_status"] = "dry_run_would_trade"
                payload["order_id"] = None
            else:
                market_id = str(payload.get("market_id") or "").strip()
                try:
                    original_price = float(payload.get("chosen_no_ask"))
                except (TypeError, ValueError):
                    original_price = None
                try:
                    no_true = 1.0 - float(payload.get("p_model"))
                except (TypeError, ValueError):
                    no_true = None

                if position_identity_key and (
                    position_identity_key in cycle_reserved_position_identity_keys
                    or state_store.has_open_position_identity(
                        station=payload.get("station"),
                        market_day_local=payload.get("market_day_local"),
                        strike_k=payload.get("strike_k"),
                    )
                ):
                    cycle_reserved_position_identity_keys.add(position_identity_key)
                    payload["decision"] = "SKIP"
                    payload["skipped_reason"] = "already_open_position"
                    payload["order_status"] = "skipped_already_open_position"
                    payload["error"] = f"duplicate_position_identity: {position_identity_key}"
                    payload["duplicate_guard_triggered"] = True
                elif position_identity_key and (
                    position_identity_key in cycle_reserved_cooldown_identity_keys
                    or state_store.is_trade_cooldown_active(
                        position_identity_key,
                        cooldown_minutes=trade_cooldown_minutes,
                        now_utc=now,
                    )
                ):
                    cycle_reserved_cooldown_identity_keys.add(position_identity_key)
                    payload["decision"] = "SKIP"
                    payload["skipped_reason"] = "trade_cooldown"
                    payload["order_status"] = "skipped_trade_cooldown"
                    payload["error"] = f"trade_cooldown_active: {position_identity_key}"
                    payload["cooldown_guard_triggered"] = True
                elif not market_id or original_price is None:
                    payload["decision"] = "SKIP"
                    payload["skipped_reason"] = "price_check_failed"
                    payload["order_status"] = "skipped_price_check_failed"
                    payload["error"] = "pre_order_price_check_failed: missing_market_or_price"
                else:
                    try:
                        refreshed_snapshot_df = dbmod.fetch_latest_snapshots(
                            conn,
                            snapshot_table=snapshot_info,
                            market_ids=[market_id],
                            lookback_per_outcome=int(cfg.get("snapshot_quote_lookback_per_outcome", 12)),
                        )
                        refreshed_snapshot = None
                        if not refreshed_snapshot_df.empty:
                            refreshed_row = refreshed_snapshot_df.iloc[0]
                            refreshed_snapshot = {
                                "yes_snapshot_ts_utc": refreshed_row.get("yes_snapshot_ts_utc"),
                                "no_snapshot_ts_utc": refreshed_row.get("no_snapshot_ts_utc"),
                                "best_yes_bid": refreshed_row.get("best_yes_bid"),
                                "best_yes_ask": refreshed_row.get("best_yes_ask"),
                                "best_no_bid": refreshed_row.get("best_no_bid"),
                                "best_no_ask": refreshed_row.get("best_no_ask"),
                                "yes_bid_size": refreshed_row.get("yes_bid_size"),
                                "yes_ask_size": refreshed_row.get("yes_ask_size"),
                                "no_bid_size": refreshed_row.get("no_bid_size"),
                                "no_ask_size": refreshed_row.get("no_ask_size"),
                            }

                        refreshed_pricing = compute_pricing_decision(
                            snapshot=refreshed_snapshot,
                            now_utc=utc_now(),
                            max_snapshot_age_minutes=float(cfg.get("max_snapshot_age_minutes", 30)),
                            slippage_buffer_yes_fallback=float(cfg.get("slippage_buffer_yes_fallback", 0.01)),
                            max_spread=float(cfg.get("max_spread", 0.05)),
                        )
                    except Exception as exc:
                        payload["decision"] = "SKIP"
                        payload["skipped_reason"] = "price_check_failed"
                        payload["order_status"] = "skipped_price_check_failed"
                        payload["error"] = f"pre_order_price_check_failed: {exc.__class__.__name__}: {exc}"
                    else:
                        payload["chosen_no_ask_refreshed"] = refreshed_pricing.chosen_no_ask
                        payload["price_check_skip_reason"] = refreshed_pricing.skipped_reason

                        if refreshed_pricing.skipped_reason is not None or refreshed_pricing.chosen_no_ask is None:
                            payload["decision"] = "SKIP"
                            payload["skipped_reason"] = "price_check_failed"
                            payload["order_status"] = "skipped_price_check_failed"
                            payload["error"] = (
                                "pre_order_price_check_failed: "
                                f"{refreshed_pricing.skipped_reason or 'no_snapshot'}"
                            )
                        else:
                            refreshed_price = float(refreshed_pricing.chosen_no_ask)
                            price_drift = refreshed_price - original_price
                            payload["price_drift"] = float(price_drift)
                            if no_true is not None:
                                payload["edge_after_price_check"] = float(no_true - refreshed_price)
                            if price_drift > float(cfg.get("price_drift_tolerance", 0.01)):
                                payload["decision"] = "SKIP"
                                payload["skipped_reason"] = "price_drift"
                                payload["order_status"] = "skipped_price_drift"
                                payload["error"] = (
                                    "pre_order_price_drift: "
                                    f"original={original_price:.6f} refreshed={refreshed_price:.6f} drift={price_drift:.6f}"
                                )
                            else:
                                payload["chosen_no_ask"] = refreshed_price
                                payload["price_source"] = refreshed_pricing.price_source
                                payload["snapshot_ts_utc"] = (
                                    refreshed_pricing.snapshot_ts_utc.isoformat()
                                    if refreshed_pricing.snapshot_ts_utc is not None
                                    else None
                                )
                                payload["snapshot_age_minutes"] = refreshed_pricing.snapshot_age_minutes
                                payload["spread"] = refreshed_pricing.spread
                                payload["snapshot_skip_reason"] = refreshed_pricing.skipped_reason
                                payload["stake_usd"] = float(refreshed_price) * float(payload["size"])

                if payload["decision"] == "TRADE":
                    order_key = _build_order_key(payload)
                    payload["order_key"] = order_key

                    if order_key is None:
                        payload["decision"] = "SKIP"
                        payload["skipped_reason"] = "order_key_failed"
                        payload["order_status"] = "skipped_order_key_failed"
                        payload["error"] = "order_key_failed: missing_required_fields"
                    elif order_key in cycle_reserved_order_keys or state_store.has_recent_order_key(
                        order_key,
                        retention_hours=ORDER_KEY_RETENTION_HOURS,
                        now_utc=now,
                    ):
                        cycle_reserved_order_keys.add(order_key)
                        payload["decision"] = "SKIP"
                        payload["skipped_reason"] = "duplicate_order_key"
                        payload["order_status"] = "skipped_duplicate_order_key"
                        payload["error"] = f"duplicate_order_key: {order_key}"
                        payload["idempotency_guard_triggered"] = True
                    else:
                        try:
                            state_store.record_recent_order_key(
                                order_key,
                                retention_hours=ORDER_KEY_RETENTION_HOURS,
                                now_utc=utc_now(),
                            )
                            cycle_reserved_order_keys.add(order_key)
                            # Persist reservation before placement to prevent restart duplicates.
                            state_store.persist()
                        except Exception as exc:
                            payload["decision"] = "SKIP"
                            payload["skipped_reason"] = "order_key_failed"
                            payload["order_status"] = "skipped_order_key_failed"
                            payload["error"] = f"order_key_record_failed: {exc.__class__.__name__}: {exc}"

                if payload["decision"] == "TRADE":
                    if position_identity_key:
                        cycle_reserved_position_identity_keys.add(position_identity_key)
                    try:
                        order_id = exec_client.place_order(
                            market_id=str(payload["market_id"]),
                            side="buy",
                            outcome="NO",
                            price=float(payload["chosen_no_ask"]),
                            size=float(payload["size"]),
                            metadata={
                                "spread": payload.get("spread"),
                                "snapshot_age_minutes": payload.get("snapshot_age_minutes"),
                                "price_source": payload.get("price_source"),
                            },
                        )
                        payload["order_id"] = order_id
                        payload["order_status"] = "submitted"
                        filled_size = float(payload.get("size") or 0.0)

                        if hasattr(exec_client, "execution_result"):
                            res = exec_client.execution_result(order_id)
                            if res is not None:
                                payload["order_status"] = res.order_status
                                if res.filled_size is not None:
                                    filled_size = max(0.0, float(res.filled_size))
                                if res.filled_price is not None:
                                    payload["chosen_no_ask"] = float(res.filled_price)
                        payload["filled_size"] = filled_size

                        if filled_size <= 0:
                            payload["decision"] = "SKIP"
                            payload["skipped_reason"] = "not_filled"
                            payload["error"] = "order_not_filled"
                        else:
                            payload["size"] = filled_size
                            payload["stake_usd"] = float(payload["chosen_no_ask"]) * float(filled_size)
                            state_store.add_trade(day_local=day_key, station=station, risk_used=float(payload.get("stake_usd") or 0.0))
                            entry_price = float(payload["chosen_no_ask"])
                            stop_loss_trigger_price = max(0.0, entry_price * (1.0 - trade_stoploss_loss_fraction))

                            pos_id = state_store.add_open_position(
                                {
                                    "station": station,
                                    "market_day_local": day_key,
                                    "market_id": payload.get("market_id"),
                                    "slug": payload.get("slug"),
                                    "asset_id": payload.get("asset_id"),
                                    "strike_k": payload.get("strike_k"),
                                    "mode_k": payload.get("mode_k"),
                                    "p_model": payload.get("p_model"),
                                    "entry_price": payload.get("chosen_no_ask"),
                                    "size": payload.get("size"),
                                    "stake_usd": payload.get("stake_usd"),
                                    "edge_at_entry": payload.get("edge_after_price_check"),
                                    "price_source": payload.get("price_source"),
                                    "order_key": payload.get("order_key"),
                                    "stop_loss_enabled": trade_stoploss_enabled,
                                    "stop_loss_loss_fraction": trade_stoploss_loss_fraction,
                                    "stop_loss_break_even_on_recovery": trade_stoploss_break_even,
                                    "stop_loss_break_even_armed": False,
                                    "stop_loss_trigger_price": stop_loss_trigger_price,
                                }
                            )
                            payload["position_id"] = pos_id
                            if position_identity_key:
                                state_store.record_trade_cooldown(
                                    position_identity_key,
                                    cooldown_minutes=trade_cooldown_minutes,
                                    now_utc=utc_now(),
                                )
                                cycle_reserved_cooldown_identity_keys.add(position_identity_key)

                            payload["station_risk_after_trade"] = state_store.station_conservative_risk_used(
                                day_local=day_key,
                                station=station,
                            )
                            payload["portfolio_risk_after_trade"] = state_store.portfolio_conservative_risk_used(
                                day_local=day_key
                            )
                    except Exception as exc:
                        payload["decision"] = "SKIP"
                        payload["skipped_reason"] = "kill_switch_active" if state_store.is_global_kill() else "risk_limit_hit"
                        payload["order_status"] = "error"
                        payload["error"] = f"{exc.__class__.__name__}: {exc}"
                        logger.exception("Order placement failed")
        else:
            state_store.add_skip(day_local=str(payload["market_day_local"]), station=str(payload["station"]))

        decision_label = str(payload.get("decision") or "UNKNOWN").upper()
        decision_counts[decision_label] += 1
        if decision_label == "SKIP":
            reason = str(payload.get("skipped_reason") or "").strip() or "unspecified"
            skip_reason_counts[reason] += 1

        _log_action(
            logger=logger,
            jsonl_path=jsonl_path,
            conn=conn,
            run_id=run_id,
            payload=payload,
            emit_info_log=_should_emit_action_info_log(payload, cfg),
        )
        if payload.get("decision") == "TRADE" and notifier is not None:
            notifier.notify_trade(payload, logger=logger)
        csv_trades_rows.append(payload)

    if bool(cfg.get("log_cycle_decision_summary", True)) and decision_counts:
        top_n = max(0, int(cfg.get("log_cycle_summary_top_skip_reasons", 6)))
        if top_n > 0 and skip_reason_counts:
            top_skip = ", ".join(
                f"{reason}:{count}" for reason, count in skip_reason_counts.most_common(top_n)
            )
        else:
            top_skip = "none"
        logger.info(
            "Cycle decision summary total=%d trades=%d skips=%d top_skip_reasons=%s",
            sum(decision_counts.values()),
            int(decision_counts.get("TRADE", 0)),
            int(decision_counts.get("SKIP", 0)),
            top_skip,
        )

    _run_open_position_maintenance(
        cfg=cfg,
        state_store=state_store,
        stations=stations,
        conn=conn,
        snapshot_info=snapshot_info,
        exec_client=exec_client,
        dry_run=dry_run,
        run_id=run_id,
        logger=logger,
        jsonl_path=jsonl_path,
        notifier=notifier,
        now_utc=now,
    )

    if bool(cfg.get("write_csv_trades", True)) and csv_trades_rows:
        csv_path = output_dir / "logs" / f"trades_{to_yyyymmdd(now.date())}.csv"
        pd.DataFrame(csv_trades_rows).to_csv(csv_path, index=False)

    base_tz = str(cfg.get("timezones", {}).get("default", "UTC"))
    local_now = now.astimezone(ZoneInfo(base_tz))
    if _should_emit_daily_report(cfg=cfg, state_store=state_store, now_local=local_now):
        report = generate_daily_report(
            output_dir=output_dir,
            logs_dir=output_dir / "logs",
            state_store=state_store,
            day_local=local_now.date(),
            stations=stations,
            nav_seed=float(cfg.get("nav_usd", 10000)),
            now_utc=now,
        )
        state_store.set_last_report_date_local(local_now.date())
        try:
            dbmod.insert_daily_report(
                conn,
                report_day_local=local_now.date().isoformat(),
                payload=report["summary"],
            )
        except Exception:
            logger.exception("Failed to write daily report to DB")

        logger.info("Daily report written: %s", report["json_path"])
        if notifier is not None:
            notifier.notify_daily_report(telegram_text_path=Path(report["telegram_path"]), logger=logger)

    state_store.persist()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    output_dir = resolve_path(str(cfg.get("output_dir", "live_trading")))
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "daily").mkdir(parents=True, exist_ok=True)
    (output_dir / "state").mkdir(parents=True, exist_ok=True)

    if args.command == "healthcheck":
        return run_healthcheck(cfg)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    today_utc = utc_now().date()
    explicit_log_path = str(cfg.get("log_file", "") or "").strip()
    if explicit_log_path:
        log_path = resolve_path(explicit_log_path)
    else:
        log_path = output_dir / "logs" / f"live_pilot_{to_yyyymmdd(today_utc)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        log_path,
        str(cfg.get("log_level", "INFO")),
        rotate_max_mb=int(cfg.get("log_rotate_max_mb", 128)),
        rotate_backups=int(cfg.get("log_rotate_backups", 20)),
        to_stdout=bool(cfg.get("log_to_stdout", True)),
    )

    logger.info("Starting live pilot run_id=%s mode=%s dry_run=%s", run_id, cfg.get("mode"), args.dry_run)

    try:
        state_store = PilotStateStore(output_dir / "state", nav_usd=float(cfg.get("nav_usd", 10000)))
    except Exception as exc:
        logger.error("State store setup failure: %s", exc)
        return 1
    notifier = TelegramNotifier.from_config(
        cfg=cfg,
        repo_root=REPO_ROOT,
        send_enabled=(not bool(args.dry_run)),
        logger=logger,
    )

    conn: psycopg.Connection | None = None
    snapshot_info: dbmod.SnapshotTableInfo | None = None

    try:
        while True:
            cycle_started = utc_now()

            if conn is None or bool(getattr(conn, "closed", False)):
                try:
                    conn = dbmod.connect_db(str(cfg.get("db_dsn")))
                    dbmod.ensure_live_pilot_tables(conn)
                    snapshot_info = dbmod.detect_snapshot_table(conn)
                    logger.info("DB connection ready snapshot_table=%s", snapshot_info.table_name)
                except Exception as exc:
                    logger.error("DB setup failure: %s", exc)
                    logger.error(traceback.format_exc())
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    conn = None
                    snapshot_info = None
                    if bool(args.exit_nonzero_on_cycle_failure):
                        logger.error("Exiting with non-zero due to --exit-nonzero-on-cycle-failure")
                        return 1
                    if args.once:
                        return 1

                    wait_s = max(1.0, float(cfg.get("run_interval_minutes", 10)) * 60.0)
                    logger.info("Sleeping %.1f seconds before DB reconnect attempt", wait_s)
                    time.sleep(wait_s)
                    continue

            try:
                if snapshot_info is None:
                    raise RuntimeError("snapshot_info unavailable")
                run_cycle(
                    cfg=cfg,
                    run_id=run_id,
                    logger=logger,
                    state_store=state_store,
                    output_dir=output_dir,
                    dry_run=bool(args.dry_run),
                    conn=conn,
                    snapshot_info=snapshot_info,
                    notifier=notifier,
                )
            except psycopg.Error as exc:
                logger.error("Cycle DB failure: %s", exc)
                logger.error(traceback.format_exc())
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                conn = None
                snapshot_info = None
                if bool(args.exit_nonzero_on_cycle_failure):
                    logger.error("Exiting with non-zero due to --exit-nonzero-on-cycle-failure")
                    return 1
            except Exception as exc:
                logger.error("Cycle failure: %s", exc)
                logger.error(traceback.format_exc())
                if bool(args.exit_nonzero_on_cycle_failure):
                    logger.error("Exiting with non-zero due to --exit-nonzero-on-cycle-failure")
                    return 1

            if args.once:
                break

            elapsed = (utc_now() - cycle_started).total_seconds()
            wait_s = max(1.0, float(cfg.get("run_interval_minutes", 10)) * 60.0 - elapsed)
            logger.info("Cycle complete. Sleeping %.1f seconds", wait_s)
            time.sleep(wait_s)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        try:
            state_store.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
