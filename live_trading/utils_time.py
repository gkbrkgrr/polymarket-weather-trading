from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCATIONS_CSV = REPO_ROOT / "locations.csv"


def normalize_station_key(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def load_station_timezones(locations_csv: Path | None = None) -> dict[str, str]:
    path = locations_csv or DEFAULT_LOCATIONS_CSV
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "name" not in df.columns or "timezone" not in df.columns:
        return {}

    tz_by_station: dict[str, str] = {}
    for row in df.itertuples(index=False):
        station = str(getattr(row, "name", "")).strip()
        tz = str(getattr(row, "timezone", "")).strip()
        if station and tz:
            tz_by_station[station] = tz
    return tz_by_station


def station_timezone(
    station: str,
    *,
    config_timezones: dict[str, Any] | None,
    fallback_timezones: dict[str, str] | None = None,
) -> str:
    fallback = fallback_timezones or {}
    cfg = config_timezones or {}
    station_overrides = cfg.get("stations") if isinstance(cfg, dict) else None

    if isinstance(station_overrides, dict):
        if station in station_overrides and str(station_overrides[station]).strip():
            return str(station_overrides[station]).strip()
        key = normalize_station_key(station)
        for cand_station, cand_tz in station_overrides.items():
            if normalize_station_key(str(cand_station)) == key and str(cand_tz).strip():
                return str(cand_tz).strip()

    if station in fallback and str(fallback[station]).strip():
        return str(fallback[station]).strip()

    default_tz = cfg.get("default") if isinstance(cfg, dict) else None
    if default_tz and str(default_tz).strip():
        return str(default_tz).strip()

    return "UTC"


def utc_now() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def to_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def parse_local_day(value: Any) -> date | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def market_day_local_for_timestamp(timestamp_utc: datetime, timezone_name: str) -> date:
    local_dt = timestamp_utc.astimezone(ZoneInfo(timezone_name))
    return local_dt.date()


def decision_cutoff_utc_for_market_day(
    market_day_local: date,
    *,
    timezone_name: str,
    policy: str,
) -> datetime:
    if policy != "latest_cycle_before_local_midnight":
        raise ValueError(f"Unsupported decision cutoff policy: {policy}")
    day_start_local = datetime.combine(market_day_local, time.min, tzinfo=ZoneInfo(timezone_name))
    return day_start_local.astimezone(ZoneInfo("UTC"))


def passes_decision_cutoff(
    *,
    execution_time_utc: datetime | None,
    market_day_local: date,
    timezone_name: str,
    policy: str,
) -> bool:
    if execution_time_utc is None:
        return False
    cutoff_utc = decision_cutoff_utc_for_market_day(
        market_day_local,
        timezone_name=timezone_name,
        policy=policy,
    )
    return execution_time_utc < cutoff_utc


def parse_hhmm(value: str) -> time:
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM string: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid HH:MM string: {value!r}")
    return time(hour=hour, minute=minute)


def is_within_trade_window(*, now_local: datetime, start_local_hhmm: str, end_local_hhmm: str) -> bool:
    start_t = parse_hhmm(start_local_hhmm)
    end_t = parse_hhmm(end_local_hhmm)
    current_t = now_local.timetz().replace(tzinfo=None)

    if start_t <= end_t:
        return start_t <= current_t <= end_t
    return current_t >= start_t or current_t <= end_t


def today_local(timezone_name: str, *, now_utc: datetime | None = None) -> date:
    base = now_utc or utc_now()
    return base.astimezone(ZoneInfo(timezone_name)).date()


def to_yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")
