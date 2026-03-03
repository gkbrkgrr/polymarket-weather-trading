#!/usr/bin/env python3
"""
Build per-cycle, per-city GFS training datasets from archived GRIB2 files.

Expected input layout:
- GFS archive cycles under:
  data/raster_data/gfs_archive/YYYYMMDDHH/*.grib2
- Station metadata:
  stations.csv (preferred) or locations.csv (fallback for this repository)
- Observations:
  master_db.station_observations

Timestamp derivation for file naming:
- For each cycle directory, first/last timestamps in output filename are taken
  directly from GRIB message metadata valid times (UTC), not inferred from
  expected lead schedules.

Cumulative/integrated variables (tp, ssrd/dswrf):
- stepRange is parsed as [start-end] hours relative to issue time.
- Interval end time is issue_time_utc + end_step_hours.
- Intervals are assigned to local day by interval-end local date.
- Daily totals are computed by summing interval amounts in that local day.
- No assumptions are made about fixed 3-hour spacing.

Run:
  python scripts/build_gfs_trainset.py --start 2021010100 --end 2021123100
  python scripts/build_gfs_trainset.py --start 2021010100 --end 2021123100 --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependencies. Install with: pip install numpy pandas pyarrow"
    ) from exc

try:
    from eccodes import (
        codes_get,
        codes_get_array,
        codes_get_values,
        codes_grib_new_from_file,
        codes_release,
    )
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'eccodes'. Install with: pip install eccodes cfgrib"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from master_db import get_daily_tmax_by_station, resolve_master_postgres_dsn

DEFAULT_ARCHIVE_ROOT = (
    REPO_ROOT / "data" / "raster_data" / "gfs_archive"
)
DEFAULT_TRAIN_ROOT = REPO_ROOT / "data" / "train_data"
DEFAULT_STATIONS_CSV = REPO_ROOT / "locations.csv"
DEFAULT_OBS_ROOT = REPO_ROOT / "data" / "observations"
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "trainset_build.log"

MODEL_NAME = "gfs"

CITY_TIMEZONE_MAP: dict[str, str] = {
    "London": "Europe/London",
    "NYC": "America/New_York",
    "Atlanta": "America/New_York",
    "Seattle": "America/Los_Angeles",
    "Dallas": "America/Chicago",
    "Toronto": "America/Toronto",
    "Seoul": "Asia/Seoul",
    "BuenosAires": "America/Argentina/Buenos_Aires",
    "Ankara": "Europe/Istanbul",
}

OUTPUT_COLUMNS = [
    "station_lat",
    "station_lon",
    "station_elev_m",
    "city_name",
    "local_timezone",
    "model",
    "issue_time_utc",
    "target_date_local",
    "tmax_time_utc",
    "lead_time_hours",
    "issue_hour_utc_sin",
    "issue_hour_utc_cos",
    "tmax_obs_c",
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
    "day_of_year",
    "day_of_year_sin",
    "day_of_year_cos",
    "month",
]

REQUIRED_FEATURE_VARS = ("t2m", "td2m", "u10", "v10", "tcc", "tp", "ssrd")


@dataclass(frozen=True)
class Station:
    city_name: str
    station_lat: float
    station_lon: float
    station_elev_m: float
    local_timezone: str


def setup_logging(
    log_path: Path,
    *,
    logger_name: str = "build_gfs_trainset",
    to_stdout: bool = True,
) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def parse_cycle_name(cycle_name: str) -> datetime:
    return datetime.strptime(cycle_name, "%Y%m%d%H").replace(tzinfo=timezone.utc)


def format_cycle(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H")


def parse_yyyymmdd_hhmm(date_int: int, time_int: int) -> datetime:
    hhmm = int(time_int)
    hh = hhmm // 100
    mm = hhmm % 100
    return datetime.strptime(f"{int(date_int):08d}{hh:02d}{mm:02d}", "%Y%m%d%H%M").replace(
        tzinfo=timezone.utc
    )


def parse_lat_lon_compact(value: str) -> tuple[float, float]:
    match = re.fullmatch(
        r"\s*([0-9.]+)\s*([NS])\s*([0-9.]+)\s*([EW])\s*",
        str(value).strip(),
    )
    if not match:
        raise ValueError(f"Invalid lat/lon compact format: {value!r}")
    lat = float(match.group(1)) * (1.0 if match.group(2) == "N" else -1.0)
    lon = float(match.group(3)) * (1.0 if match.group(4) == "E" else -1.0)
    return lat, lon


def normalize_city_name(city_name: str) -> str:
    return str(city_name).strip()


def load_stations(stations_csv: Path, logger: logging.Logger) -> list[Station]:
    source = stations_csv
    if not source.exists():
        fallback = stations_csv.parent / "locations.csv"
        if fallback.exists():
            logger.warning(
                "stations.csv not found at %s; falling back to %s",
                source,
                fallback,
            )
            source = fallback
        else:
            raise FileNotFoundError(
                f"Stations file not found: {source}. Provide stations.csv with city_name/lat/lon/elev."
            )

    df = pd.read_csv(source)
    cols = {c.lower(): c for c in df.columns}

    stations: list[Station] = []
    if all(k in cols for k in ("city_name", "lat", "lon", "elev")):
        city_col = cols["city_name"]
        lat_col = cols["lat"]
        lon_col = cols["lon"]
        elev_col = cols["elev"]
        tz_col = cols.get("timezone")
        for _, row in df.iterrows():
            city = normalize_city_name(row[city_col])
            if not city:
                continue
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            elev_m = float(row[elev_col])
            tz_file = str(row[tz_col]).strip() if tz_col and pd.notna(row[tz_col]) else ""
            tz_map = CITY_TIMEZONE_MAP.get(city, "")
            tz = tz_file or tz_map
            if not tz:
                raise ValueError(f"No timezone found for city '{city}'. Add timezone column or CITY_TIMEZONE_MAP.")
            if tz_file and tz_map and tz_file != tz_map:
                logger.warning(
                    "Timezone mismatch for %s: stations.csv=%s, map=%s. Using stations.csv value.",
                    city,
                    tz_file,
                    tz_map,
                )
            stations.append(
                Station(
                    city_name=city,
                    station_lat=lat,
                    station_lon=lon,
                    station_elev_m=elev_m,
                    local_timezone=tz,
                )
            )
    elif all(k in cols for k in ("name", "lat_lon", "elevation")):
        city_col = cols["name"]
        latlon_col = cols["lat_lon"]
        elev_col = cols["elevation"]
        tz_col = cols.get("timezone")
        for _, row in df.iterrows():
            city = normalize_city_name(row[city_col])
            if not city:
                continue
            lat, lon = parse_lat_lon_compact(str(row[latlon_col]))
            elev_m = float(row[elev_col])
            tz_file = str(row[tz_col]).strip() if tz_col and pd.notna(row[tz_col]) else ""
            tz_map = CITY_TIMEZONE_MAP.get(city, "")
            tz = tz_file or tz_map
            if not tz:
                raise ValueError(f"No timezone found for city '{city}'. Add timezone column or CITY_TIMEZONE_MAP.")
            if tz_file and tz_map and tz_file != tz_map:
                logger.warning(
                    "Timezone mismatch for %s: csv=%s, map=%s. Using csv value.",
                    city,
                    tz_file,
                    tz_map,
                )
            stations.append(
                Station(
                    city_name=city,
                    station_lat=lat,
                    station_lon=lon,
                    station_elev_m=elev_m,
                    local_timezone=tz,
                )
            )
    else:
        raise ValueError(
            f"Unsupported station schema in {source}. "
            "Expected either columns city_name,lat,lon,elev[,timezone] "
            "or name,lat_lon,elevation[,timezone]."
        )

    if not stations:
        raise ValueError(f"No valid stations found in {source}")

    logger.info("Loaded %d stations from %s", len(stations), source)
    return stations


def detect_observation_time_column(df: pd.DataFrame) -> str:
    preferred = [
        "observed_at_local",
        "observed_at_utc",
        "observed_at",
        "timestamp",
        "valid_time_utc",
        "datetime",
        "time",
        "date",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for key in preferred:
        if key in cols_lower:
            return cols_lower[key]
    for col in df.columns:
        c = col.lower()
        if "observed" in c or "time" in c or "date" in c:
            return col
    raise ValueError("Could not detect observation timestamp column.")


def detect_temperature_celsius(
    df: pd.DataFrame,
    city_name: str,
    logger: logging.Logger,
) -> pd.Series:
    cols_lower = {c.lower(): c for c in df.columns}
    if "temperature_c" in cols_lower:
        return pd.to_numeric(df[cols_lower["temperature_c"]], errors="coerce").astype(float)
    if "temp_c" in cols_lower:
        return pd.to_numeric(df[cols_lower["temp_c"]], errors="coerce").astype(float)
    if "temperature_f" in cols_lower:
        f = pd.to_numeric(df[cols_lower["temperature_f"]], errors="coerce").astype(float)
        return (f - 32.0) * (5.0 / 9.0)
    if "temp_f" in cols_lower:
        f = pd.to_numeric(df[cols_lower["temp_f"]], errors="coerce").astype(float)
        return (f - 32.0) * (5.0 / 9.0)
    for col in df.columns:
        c = col.lower()
        if "temperature" in c or c == "temp":
            logger.warning(
                "Obs temp unit ambiguous for city=%s column=%s; assuming Celsius.",
                city_name,
                col,
            )
            return pd.to_numeric(df[col], errors="coerce").astype(float)
    raise ValueError(f"Could not detect temperature column for city={city_name}")


def parse_observation_times_to_utc(
    raw_series: pd.Series,
    *,
    city_name: str,
    local_timezone: str,
    column_name: str,
    logger: logging.Logger,
) -> pd.Series:
    if isinstance(raw_series.dtype, pd.DatetimeTZDtype):
        return pd.to_datetime(raw_series, utc=True, errors="coerce")

    if pd.api.types.is_datetime64_dtype(raw_series):
        if "local" in column_name.lower():
            logger.info(
                "Naive datetime obs treated as local time for city=%s column=%s",
                city_name,
                column_name,
            )
            return raw_series.dt.tz_localize(local_timezone).dt.tz_convert("UTC")
        logger.warning(
            "Naive datetime obs treated as UTC for city=%s column=%s",
            city_name,
            column_name,
        )
        return raw_series.dt.tz_localize("UTC")

    s = raw_series.astype("string")
    has_explicit_offset = s.str.contains(r"(?:Z$|[+\-]\d{2}:?\d{2}$)", regex=True, na=False).any()
    if has_explicit_offset:
        return pd.to_datetime(s, utc=True, errors="coerce")

    parsed_naive = pd.to_datetime(s, errors="coerce")
    if "local" in column_name.lower():
        logger.info(
            "Naive string obs treated as local time for city=%s column=%s",
            city_name,
            column_name,
        )
        return parsed_naive.dt.tz_localize(local_timezone).dt.tz_convert("UTC")

    logger.warning(
        "Naive string obs treated as UTC for city=%s column=%s",
        city_name,
        column_name,
    )
    return parsed_naive.dt.tz_localize("UTC")


def load_obs(
    city_name: str,
    obs_root: Path,
    local_timezone: str,
    logger: logging.Logger,
    obs_dsn: str | None = None,
) -> pd.DataFrame:
    del obs_root, local_timezone
    dsn = resolve_master_postgres_dsn(explicit_dsn=obs_dsn)
    out = get_daily_tmax_by_station(stations=[city_name], master_dsn=dsn)
    if out.empty:
        logger.warning("Observation rows missing for city=%s in station_observations", city_name)
        return pd.DataFrame({"city_name": [], "target_date_local": [], "tmax_obs_c": []})

    out = out.rename(columns={"city_name": "city_name", "target_date_local": "target_date_local"})
    out["city_name"] = city_name
    out["target_date_local"] = pd.to_datetime(out["target_date_local"], errors="coerce").dt.date
    out["tmax_obs_c"] = pd.to_numeric(out["tmax_obs_c"], errors="coerce")
    out = out.dropna(subset=["target_date_local", "tmax_obs_c"]).copy()
    return out


def parse_step_range(step_range: str | int | float | None) -> tuple[float | None, float | None]:
    if step_range is None:
        return None, None
    s = str(step_range).strip()
    if not s:
        return None, None
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:\s*-\s*(-?\d+(?:\.\d+)?))?", s)
    if not match:
        return None, None
    a = float(match.group(1))
    b = float(match.group(2)) if match.group(2) is not None else None
    if b is None:
        return 0.0, a
    return a, b


def round_if_close(value: float | None, tol: float = 1e-6) -> int | None:
    if value is None or math.isnan(value):
        return None
    iv = int(round(value))
    if abs(value - iv) <= tol:
        return iv
    return None


def _safe_get(gid: int, key: str):
    try:
        return codes_get(gid, key)
    except Exception:
        return None


def inspect_cycle_gribs(
    cycle_dir: Path,
    logger: logging.Logger,
) -> tuple[datetime, list[dict[str, object]], datetime, datetime]:
    files = sorted(cycle_dir.glob("*.grib2"))
    if not files:
        raise FileNotFoundError(f"No .grib2 files found in {cycle_dir}")

    metadata: list[dict[str, object]] = []
    issue_candidates: set[datetime] = set()
    valid_times: list[datetime] = []

    for path in files:
        with open(path, "rb") as f:
            msg_idx = 0
            while True:
                gid = codes_grib_new_from_file(f)
                if gid is None:
                    break
                msg_idx += 1
                data_date = _safe_get(gid, "dataDate")
                data_time = _safe_get(gid, "dataTime")
                valid_date = _safe_get(gid, "validityDate")
                valid_time = _safe_get(gid, "validityTime")
                step_range = _safe_get(gid, "stepRange")
                start_step, end_step = parse_step_range(step_range)
                issue_time_utc = parse_yyyymmdd_hhmm(int(data_date), int(data_time))
                valid_time_utc = parse_yyyymmdd_hhmm(int(valid_date), int(valid_time))
                issue_candidates.add(issue_time_utc)
                valid_times.append(valid_time_utc)

                metadata.append(
                    {
                        "file_path": path,
                        "message_index": msg_idx,
                        "shortName": _safe_get(gid, "shortName"),
                        "name": _safe_get(gid, "name"),
                        "paramId": _safe_get(gid, "paramId"),
                        "typeOfLevel": _safe_get(gid, "typeOfLevel"),
                        "level": _safe_get(gid, "level"),
                        "stepType": _safe_get(gid, "stepType"),
                        "units": _safe_get(gid, "units"),
                        "stepRange": str(step_range) if step_range is not None else "",
                        "start_step_hours": start_step,
                        "end_step_hours": end_step,
                        "issue_time_utc": issue_time_utc,
                        "valid_time_utc": valid_time_utc,
                    }
                )
                codes_release(gid)

    if not metadata:
        raise ValueError(f"No GRIB messages found in cycle {cycle_dir}")
    if len(issue_candidates) > 1:
        logger.warning(
            "Multiple issue times detected in %s: %s. Using earliest.",
            cycle_dir,
            ", ".join(sorted(format_cycle(x) for x in issue_candidates)),
        )
    issue_time_utc = min(issue_candidates)
    first_valid = min(valid_times)
    last_valid = max(valid_times)
    return issue_time_utc, metadata, first_valid, last_valid


def _ensure_ascending(
    coord: np.ndarray,
    values: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    if coord.size < 2:
        return coord, values
    if float(coord[0]) <= float(coord[-1]):
        return coord, values
    return coord[::-1].copy(), np.flip(values, axis=axis)


def _normalize_lon_for_grid(lon: float, lons: np.ndarray) -> float:
    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))
    out = float(lon)
    if lon_min >= 0.0 and lon_max > 180.0 and lon < 0.0:
        out = lon % 360.0
    return out


def bilinear_interpolate(
    lats: np.ndarray,
    lons: np.ndarray,
    values_2d: np.ndarray,
    *,
    lat: float,
    lon: float,
) -> float:
    lats, values_2d = _ensure_ascending(lats, values_2d, axis=0)
    lons, values_2d = _ensure_ascending(lons, values_2d, axis=1)

    lon = _normalize_lon_for_grid(lon, lons)
    lat = float(np.clip(lat, float(lats[0]), float(lats[-1])))
    lon = float(np.clip(lon, float(lons[0]), float(lons[-1])))

    i1 = int(np.searchsorted(lats, lat, side="right"))
    j1 = int(np.searchsorted(lons, lon, side="right"))
    i0 = max(i1 - 1, 0)
    j0 = max(j1 - 1, 0)
    i1 = min(i1, lats.size - 1)
    j1 = min(j1, lons.size - 1)

    lat0 = float(lats[i0])
    lat1 = float(lats[i1])
    lon0 = float(lons[j0])
    lon1 = float(lons[j1])

    dlat = lat1 - lat0
    dlon = lon1 - lon0
    wlat = 0.0 if dlat == 0.0 else (lat - lat0) / dlat
    wlon = 0.0 if dlon == 0.0 else (lon - lon0) / dlon
    wlat = float(np.clip(wlat, 0.0, 1.0))
    wlon = float(np.clip(wlon, 0.0, 1.0))

    q00 = float(values_2d[i0, j0])
    q01 = float(values_2d[i0, j1])
    q10 = float(values_2d[i1, j0])
    q11 = float(values_2d[i1, j1])

    corners = np.array([q00, q01, q10, q11], dtype=float)
    if np.any(np.isnan(corners)):
        pts = np.array(
            [(lat0, lon0), (lat0, lon1), (lat1, lon0), (lat1, lon1)],
            dtype=float,
        )
        d2 = (pts[:, 0] - lat) ** 2 + (pts[:, 1] - lon) ** 2
        for idx in np.argsort(d2):
            if not math.isnan(corners[idx]):
                return float(corners[idx])
        return float("nan")

    return float(
        q00 * (1 - wlat) * (1 - wlon)
        + q01 * (1 - wlat) * wlon
        + q10 * wlat * (1 - wlon)
        + q11 * wlat * wlon
    )


def classify_message(short_name: str | None, level_type: str | None, level: object) -> str | None:
    s = str(short_name or "").strip()
    lt = str(level_type or "").strip()
    lev = None
    try:
        lev = int(level)
    except Exception:
        lev = None

    if s == "2t" and lt == "heightAboveGround" and lev == 2:
        return "t2m"
    if s == "2d" and lt == "heightAboveGround" and lev == 2:
        return "td2m"
    if s == "2r" and lt == "heightAboveGround" and lev == 2:
        return "rh2m"
    if s == "10u" and lt == "heightAboveGround" and lev == 10:
        return "u10"
    if s == "10v" and lt == "heightAboveGround" and lev == 10:
        return "v10"
    if s == "tcc":
        return "tcc"
    if s == "tp":
        return "tp"
    if s in {"sdswrf", "dswrf", "ssrd"}:
        return "ssrd"
    return None


def read_grid_from_gid(gid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nj = int(codes_get(gid, "Nj"))
    ni = int(codes_get(gid, "Ni"))
    lats = np.asarray(codes_get_array(gid, "distinctLatitudes"), dtype=float)
    lons = np.asarray(codes_get_array(gid, "distinctLongitudes"), dtype=float)
    values = np.asarray(codes_get_values(gid), dtype=float)
    if values.size != nj * ni:
        raise ValueError(f"Unexpected grid size values={values.size} Nj*Ni={nj * ni}")
    values2d = values.reshape(nj, ni)
    return lats, lons, values2d


def extract_point_timeseries(
    cycle_dir: Path,
    stations: list[Station],
    issue_time_utc: datetime,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[tuple[object, object, object, object]]], dict[str, set[str]]]:
    files = sorted(cycle_dir.glob("*.grib2"))
    if not files:
        raise FileNotFoundError(f"No GRIB2 files in cycle dir: {cycle_dir}")

    instantaneous_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    used_meta: dict[str, set[tuple[object, object, object, object]]] = {}
    step_ranges_seen: dict[str, set[str]] = {"tp": set(), "ssrd": set()}

    for path in files:
        with open(path, "rb") as f:
            while True:
                gid = codes_grib_new_from_file(f)
                if gid is None:
                    break
                try:
                    short_name = _safe_get(gid, "shortName")
                    var = classify_message(
                        short_name,
                        _safe_get(gid, "typeOfLevel"),
                        _safe_get(gid, "level"),
                    )
                    if var is None:
                        continue

                    step_range_raw = _safe_get(gid, "stepRange")
                    start_step, end_step = parse_step_range(step_range_raw)
                    start_step_i = round_if_close(start_step)
                    end_step_i = round_if_close(end_step)

                    data_date = int(codes_get(gid, "dataDate"))
                    data_time = int(codes_get(gid, "dataTime"))
                    valid_date = int(codes_get(gid, "validityDate"))
                    valid_time = int(codes_get(gid, "validityTime"))
                    msg_issue = parse_yyyymmdd_hhmm(data_date, data_time)
                    msg_valid = parse_yyyymmdd_hhmm(valid_date, valid_time)
                    if msg_issue != issue_time_utc:
                        logger.warning(
                            "Message issue time differs from cycle issue in %s: msg=%s cycle=%s",
                            path.name,
                            msg_issue.isoformat(),
                            issue_time_utc.isoformat(),
                        )

                    units = _safe_get(gid, "units")
                    step_type = _safe_get(gid, "stepType")
                    param_id = _safe_get(gid, "paramId")
                    used_meta.setdefault(var, set()).add((short_name, param_id, step_type, units))

                    lats, lons, grid = read_grid_from_gid(gid)

                    if var in {"tp", "ssrd"}:
                        sr = str(step_range_raw) if step_range_raw is not None else ""
                        step_ranges_seen[var].add(sr)
                        if start_step_i is None or end_step_i is None:
                            logger.warning(
                                "Skipping %s message with non-integer stepRange=%r in %s",
                                var,
                                step_range_raw,
                                path.name,
                            )
                            continue
                        interval_hours = end_step_i - start_step_i
                        if interval_hours <= 0:
                            logger.warning(
                                "Skipping %s message with invalid interval stepRange=%s in %s",
                                var,
                                step_range_raw,
                                path.name,
                            )
                            continue
                        end_valid_utc = issue_time_utc + timedelta(hours=end_step_i)
                        for st in stations:
                            value = bilinear_interpolate(
                                lats,
                                lons,
                                grid,
                                lat=st.station_lat,
                                lon=st.station_lon,
                            )
                            interval_rows.append(
                                {
                                    "city_name": st.city_name,
                                    "var": var,
                                    "value": value,
                                    "units": str(units) if units is not None else "",
                                    "step_range": sr,
                                    "start_step_hours": int(start_step_i),
                                    "end_step_hours": int(end_step_i),
                                    "interval_hours": int(interval_hours),
                                    "interval_end_valid_time_utc": end_valid_utc,
                                    "issue_time_utc": issue_time_utc,
                                }
                            )
                    else:
                        valid_time_utc = msg_valid
                        for st in stations:
                            value = bilinear_interpolate(
                                lats,
                                lons,
                                grid,
                                lat=st.station_lat,
                                lon=st.station_lon,
                            )
                            instantaneous_rows.append(
                                {
                                    "city_name": st.city_name,
                                    "valid_time_utc": valid_time_utc,
                                    "var": var,
                                    "value": value,
                                }
                            )
                finally:
                    codes_release(gid)

    instant_df = pd.DataFrame.from_records(instantaneous_rows)
    if instant_df.empty:
        instant_wide = pd.DataFrame(columns=["city_name", "valid_time_utc"])
    else:
        instant_df["valid_time_utc"] = pd.to_datetime(instant_df["valid_time_utc"], utc=True)
        instant_wide = (
            instant_df.pivot_table(
                index=["city_name", "valid_time_utc"],
                columns="var",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )
        instant_wide.columns.name = None

    interval_df = pd.DataFrame.from_records(interval_rows)
    if not interval_df.empty:
        interval_df["interval_end_valid_time_utc"] = pd.to_datetime(
            interval_df["interval_end_valid_time_utc"], utc=True
        )
        interval_df["issue_time_utc"] = pd.to_datetime(interval_df["issue_time_utc"], utc=True)
    return instant_wide, interval_df, used_meta, step_ranges_seen


def relative_humidity_from_t_td_c(t_c: float, td_c: float) -> float:
    if math.isnan(t_c) or math.isnan(td_c):
        return float("nan")
    # Magnus approximation over water (temperatures in Celsius).
    a = 17.625
    b = 243.04
    gamma_t = (a * t_c) / (b + t_c)
    gamma_td = (a * td_c) / (b + td_c)
    rh = 100.0 * math.exp(gamma_td - gamma_t)
    return float(np.clip(rh, 0.0, 100.0))


def convert_tcc_to_percent(series: pd.Series, units_seen: Iterable[str]) -> pd.Series:
    units_l = " ".join(str(u).lower() for u in units_seen if u is not None)
    out = pd.to_numeric(series, errors="coerce").astype(float)
    if "%" in units_l or "percent" in units_l:
        return out
    if out.dropna().empty:
        return out
    if float(out.max(skipna=True)) <= 1.5:
        return out * 100.0
    return out


def convert_rh_to_percent(series: pd.Series, units_seen: Iterable[str]) -> pd.Series:
    units_l = " ".join(str(u).lower() for u in units_seen if u is not None)
    out = pd.to_numeric(series, errors="coerce").astype(float)
    if "%" in units_l or "percent" in units_l:
        return out.clip(lower=0.0, upper=100.0)
    if out.dropna().empty:
        return out
    if float(out.max(skipna=True)) <= 1.5:
        return (out * 100.0).clip(lower=0.0, upper=100.0)
    return out.clip(lower=0.0, upper=100.0)


def convert_tp_to_mm(values: pd.Series, units: pd.Series) -> tuple[pd.Series, str]:
    unit_set = {str(u).strip().lower() for u in units.dropna().unique().tolist()}
    vals = pd.to_numeric(values, errors="coerce").astype(float)
    if not unit_set:
        return vals, "unknown->assumed-mm"
    if any("kg" in u and "m" in u for u in unit_set):
        return vals, "kg/m^2->mm(1:1)"
    if any(re.fullmatch(r"m|meter|metre|m\*\*-?1|m\^?1", u) for u in unit_set):
        return vals * 1000.0, "m->mm(*1000)"
    if any("mm" in u for u in unit_set):
        return vals, "mm(no-conversion)"
    return vals, f"unknown-units({sorted(unit_set)})-assumed-mm"


def convert_ssrd_to_jm2(values: pd.Series, units: pd.Series, interval_hours: pd.Series) -> tuple[pd.Series, str]:
    unit_set = {str(u).strip().lower() for u in units.dropna().unique().tolist()}
    vals = pd.to_numeric(values, errors="coerce").astype(float)
    ih = pd.to_numeric(interval_hours, errors="coerce").astype(float)
    if not unit_set:
        return vals, "unknown->assumed-J/m^2"
    if any("j" in u for u in unit_set):
        return vals, "J/m^2(no-conversion)"
    if any("w" in u for u in unit_set):
        return vals * ih * 3600.0, "W/m^2->J/m^2(*interval_hours*3600)"
    return vals, f"unknown-units({sorted(unit_set)})-assumed-J/m^2"


def select_non_overlapping_intervals(
    df_var: pd.DataFrame,
    *,
    city_name: str,
    var: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if df_var.empty:
        return df_var
    work = df_var.copy()
    work = work.sort_values(["end_step_hours", "interval_hours", "start_step_hours"], kind="mergesort")

    selected: list[pd.Series] = []
    for _, row in work.iterrows():
        if not selected:
            selected.append(row)
            continue
        prev = selected[-1]
        prev_start = int(prev["start_step_hours"])
        prev_end = int(prev["end_step_hours"])
        cur_start = int(row["start_step_hours"])
        cur_end = int(row["end_step_hours"])

        if cur_start == prev_start and cur_end > prev_end:
            selected[-1] = row
            continue
        if cur_start >= prev_end:
            selected.append(row)
            continue
        # Overlap with different start; skip the later candidate and keep deterministic output.
        if cur_end > prev_end and cur_start < prev_end:
            logger.warning(
                "Overlapping %s intervals for city=%s: kept [%d-%d], skipped [%d-%d]",
                var,
                city_name,
                prev_start,
                prev_end,
                cur_start,
                cur_end,
            )

    out = pd.DataFrame(selected).reset_index(drop=True)
    out = out.sort_values(["end_step_hours", "start_step_hours"], kind="mergesort")

    prev_end: int | None = None
    for _, row in out.iterrows():
        start_h = int(row["start_step_hours"])
        end_h = int(row["end_step_hours"])
        if prev_end is not None and start_h > prev_end:
            logger.warning(
                "Gap detected for %s city=%s: previous end=%d current start=%d",
                var,
                city_name,
                prev_end,
                start_h,
            )
        prev_end = end_h
    return out


def compute_daily_interval_totals(
    interval_df: pd.DataFrame,
    stations: list[Station],
    logger: logging.Logger,
) -> pd.DataFrame:
    if interval_df.empty:
        return pd.DataFrame(columns=["city_name", "target_date_local", "tp_day_total", "ssrd_day_total"])

    tz_map = {st.city_name: st.local_timezone for st in stations}
    out_rows: list[dict[str, object]] = []

    for city_name, city_df in interval_df.groupby("city_name", sort=False):
        local_tz = tz_map[city_name]
        city_rows: dict[tuple[str, date], dict[str, float]] = {}
        for var in ("tp", "ssrd"):
            var_df_raw = city_df[city_df["var"] == var].copy()
            if var_df_raw.empty:
                continue
            var_df = select_non_overlapping_intervals(
                var_df_raw,
                city_name=city_name,
                var=var,
                logger=logger,
            )
            if var == "tp":
                converted, conversion_note = convert_tp_to_mm(var_df["value"], var_df["units"])
                var_df["converted"] = converted
            else:
                converted, conversion_note = convert_ssrd_to_jm2(
                    var_df["value"],
                    var_df["units"],
                    var_df["interval_hours"],
                )
                var_df["converted"] = converted

            neg_mask = pd.to_numeric(var_df["converted"], errors="coerce") < 0.0
            neg_count = int(neg_mask.sum())
            if neg_count > 0:
                logger.warning(
                    "Negative increments encountered and corrected city=%s var=%s count=%d",
                    city_name,
                    var,
                    neg_count,
                )
                var_df.loc[neg_mask, "converted"] = 0.0

            logger.info(
                "Interval conversion city=%s var=%s units=%s conversion=%s selected=%d raw=%d negative_corrected=%d",
                city_name,
                var,
                sorted({str(u) for u in var_df_raw['units'].dropna().unique().tolist()}),
                conversion_note,
                len(var_df),
                len(var_df_raw),
                neg_count,
            )

            local_end = var_df["interval_end_valid_time_utc"].dt.tz_convert(local_tz)
            local_day = local_end.dt.date
            for d, amount in zip(local_day.tolist(), var_df["converted"].tolist()):
                key = (city_name, d)
                if key not in city_rows:
                    city_rows[key] = {"tp_day_total": float("nan"), "ssrd_day_total": float("nan")}
                col = "tp_day_total" if var == "tp" else "ssrd_day_total"
                current = city_rows[key][col]
                city_rows[key][col] = float(amount) if math.isnan(current) else float(current + amount)

        for (cn, d), vals in city_rows.items():
            out_rows.append(
                {
                    "city_name": cn,
                    "target_date_local": d,
                    "tp_day_total": vals["tp_day_total"],
                    "ssrd_day_total": vals["ssrd_day_total"],
                }
            )

    if not out_rows:
        return pd.DataFrame(columns=["city_name", "target_date_local", "tp_day_total", "ssrd_day_total"])
    return pd.DataFrame.from_records(out_rows)


def compute_daily_features(
    stations: list[Station],
    issue_time_utc: datetime,
    instant_wide: pd.DataFrame,
    interval_df: pd.DataFrame,
    obs_by_city: dict[str, pd.DataFrame],
    used_meta: dict[str, set[tuple[object, object, object, object]]],
    logger: logging.Logger,
) -> pd.DataFrame:
    tz_map = {st.city_name: st.local_timezone for st in stations}
    station_map = {st.city_name: st for st in stations}

    interval_daily = compute_daily_interval_totals(interval_df, stations, logger)
    if interval_daily.empty:
        interval_daily_idx: dict[tuple[str, date], tuple[float, float]] = {}
    else:
        interval_daily_idx = {
            (str(r.city_name), r.target_date_local): (
                float(r.tp_day_total) if pd.notna(r.tp_day_total) else float("nan"),
                float(r.ssrd_day_total) if pd.notna(r.ssrd_day_total) else float("nan"),
            )
            for r in interval_daily.itertuples(index=False)
        }

    if instant_wide.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows: list[dict[str, object]] = []
    tcc_units = [x[3] for x in used_meta.get("tcc", set())]
    rh_units = [x[3] for x in used_meta.get("rh2m", set())]
    issue_hour_utc = (
        float(issue_time_utc.astimezone(timezone.utc).hour)
        + float(issue_time_utc.astimezone(timezone.utc).minute) / 60.0
    )
    issue_hour_utc_sin = math.sin(2.0 * math.pi * issue_hour_utc / 24.0)
    issue_hour_utc_cos = math.cos(2.0 * math.pi * issue_hour_utc / 24.0)

    for city_name, city_df in instant_wide.groupby("city_name", sort=False):
        st = station_map.get(city_name)
        if st is None:
            continue
        tz_name = tz_map[city_name]
        city_ts = city_df.copy().sort_values("valid_time_utc", kind="mergesort")
        city_ts["valid_time_utc"] = pd.to_datetime(city_ts["valid_time_utc"], utc=True)

        if "t2m" in city_ts:
            city_ts["t2m_c"] = pd.to_numeric(city_ts["t2m"], errors="coerce").astype(float) - 273.15
        else:
            city_ts["t2m_c"] = np.nan
        if "td2m" in city_ts:
            city_ts["td2m_c"] = pd.to_numeric(city_ts["td2m"], errors="coerce").astype(float) - 273.15
        else:
            city_ts["td2m_c"] = np.nan

        if "u10" in city_ts and "v10" in city_ts:
            u = pd.to_numeric(city_ts["u10"], errors="coerce").astype(float)
            v = pd.to_numeric(city_ts["v10"], errors="coerce").astype(float)
            city_ts["ws10_mps"] = np.sqrt(u * u + v * v)
        else:
            city_ts["ws10_mps"] = np.nan

        if "tcc" in city_ts:
            city_ts["tcc_pct"] = convert_tcc_to_percent(city_ts["tcc"], tcc_units)
        else:
            city_ts["tcc_pct"] = np.nan

        if "rh2m" in city_ts:
            city_ts["rh2m_pct"] = convert_rh_to_percent(city_ts["rh2m"], rh_units)
        else:
            city_ts["rh2m_pct"] = np.nan

        local_time = city_ts["valid_time_utc"].dt.tz_convert(tz_name)
        city_ts["target_date_local"] = local_time.dt.date

        obs_df = obs_by_city.get(city_name)
        obs_map: dict[date, float] = {}
        if obs_df is not None and not obs_df.empty:
            obs_map = {
                r.target_date_local: float(r.tmax_obs_c)
                for r in obs_df.itertuples(index=False)
            }

        for target_day, day_df in city_ts.groupby("target_date_local", sort=True):
            t2m_vals = pd.to_numeric(day_df["t2m_c"], errors="coerce").astype(float)
            if t2m_vals.dropna().empty:
                tmax_raw_c = float("nan")
                tmax_time_utc = pd.NaT
                lead_time_hours = float("nan")
                td2m_at_tmax_c = float("nan")
                rh2m_at_tmax_c = float("nan")
                tcc_at_tmax_pct = float("nan")
                ws10_at_tmax_mps = float("nan")
            else:
                tmax_idx = t2m_vals.idxmax(skipna=True)
                tmax_raw_c = float(day_df.loc[tmax_idx, "t2m_c"])
                tmax_time_utc = pd.Timestamp(day_df.loc[tmax_idx, "valid_time_utc"]).tz_convert("UTC")
                lead_time_hours = (
                    tmax_time_utc.to_pydatetime() - issue_time_utc
                ).total_seconds() / 3600.0
                td2m_at_tmax_c = float(day_df.loc[tmax_idx, "td2m_c"])
                rh_raw = float(day_df.loc[tmax_idx, "rh2m_pct"])
                if math.isnan(rh_raw):
                    rh2m_at_tmax_c = relative_humidity_from_t_td_c(
                        float(day_df.loc[tmax_idx, "t2m_c"]),
                        float(day_df.loc[tmax_idx, "td2m_c"]),
                    )
                else:
                    rh2m_at_tmax_c = float(np.clip(rh_raw, 0.0, 100.0))
                tcc_at_tmax_pct = float(pd.to_numeric(day_df.loc[tmax_idx, "tcc_pct"], errors="coerce"))
                ws10_at_tmax_mps = float(pd.to_numeric(day_df.loc[tmax_idx, "ws10_mps"], errors="coerce"))

            tmin = float(t2m_vals.min(skipna=True)) if not t2m_vals.dropna().empty else float("nan")
            tmax = float(t2m_vals.max(skipna=True)) if not t2m_vals.dropna().empty else float("nan")
            t2m_diurnal_range = (tmax - tmin) if (not math.isnan(tmax) and not math.isnan(tmin)) else float("nan")

            d = target_day
            doy = int(datetime(d.year, d.month, d.day).timetuple().tm_yday)
            month = int(d.month)
            sin_doy = math.sin(2.0 * math.pi * doy / 365.25)
            cos_doy = math.cos(2.0 * math.pi * doy / 365.25)

            tp_day_total, ssrd_day_total = interval_daily_idx.get((city_name, d), (float("nan"), float("nan")))

            row = {
                "station_lat": float(st.station_lat),
                "station_lon": float(st.station_lon),
                "station_elev_m": float(st.station_elev_m),
                "city_name": city_name,
                "local_timezone": st.local_timezone,
                "model": MODEL_NAME,
                "issue_time_utc": pd.Timestamp(issue_time_utc),
                "target_date_local": d,
                "tmax_time_utc": tmax_time_utc,
                "lead_time_hours": float(lead_time_hours),
                "issue_hour_utc_sin": float(issue_hour_utc_sin),
                "issue_hour_utc_cos": float(issue_hour_utc_cos),
                "tmax_obs_c": obs_map.get(d, float("nan")),
                "tmax_raw_c": float(tmax_raw_c),
                "tcc_mean_pct": float(pd.to_numeric(day_df["tcc_pct"], errors="coerce").mean(skipna=True)),
                "tcc_max_pct": float(pd.to_numeric(day_df["tcc_pct"], errors="coerce").max(skipna=True)),
                "tcc_at_tmax_pct": float(tcc_at_tmax_pct),
                "ws10_mean_mps": float(pd.to_numeric(day_df["ws10_mps"], errors="coerce").mean(skipna=True)),
                "ws10_max_mps": float(pd.to_numeric(day_df["ws10_mps"], errors="coerce").max(skipna=True)),
                "ws10_at_tmax_mps": float(ws10_at_tmax_mps),
                "td2m_mean_c": float(pd.to_numeric(day_df["td2m_c"], errors="coerce").mean(skipna=True)),
                "td2m_at_tmax_c": float(td2m_at_tmax_c),
                "rh2m_at_tmax_c": float(rh2m_at_tmax_c),
                "t2m_diurnal_range": float(t2m_diurnal_range),
                "ssrd_day_total": float(ssrd_day_total),
                "tp_day_total": float(tp_day_total),
                "day_of_year": float(doy),
                "day_of_year_sin": float(sin_doy),
                "day_of_year_cos": float(cos_doy),
                "month": float(month),
            }
            rows.append(row)

    out_df = pd.DataFrame.from_records(rows)
    if out_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out_df = out_df[OUTPUT_COLUMNS]
    return out_df


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[OUTPUT_COLUMNS]

    string_cols = ["city_name", "local_timezone", "model"]
    for col in string_cols:
        out[col] = out[col].astype("string")

    out["issue_time_utc"] = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    out["tmax_time_utc"] = pd.to_datetime(out["tmax_time_utc"], utc=True, errors="coerce")
    out["target_date_local"] = pd.to_datetime(out["target_date_local"], errors="coerce").dt.date

    numeric_cols = [c for c in OUTPUT_COLUMNS if c not in string_cols + ["issue_time_utc", "tmax_time_utc", "target_date_local"]]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        out[col] = out[col].round(3)
    return out


def write_parquet(
    df_city: pd.DataFrame,
    train_root: Path,
    city_name: str,
    first_valid: datetime,
    last_valid: datetime,
) -> Path:
    first_str = format_cycle(first_valid)
    last_str = format_cycle(last_valid)
    out_dir = train_root / MODEL_NAME / city_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{MODEL_NAME}_train_{first_str}_{last_str}.parquet"

    data = enforce_schema(df_city)
    tmp_path = out_path.with_suffix(".parquet.part")
    if tmp_path.exists():
        tmp_path.unlink()
    data.to_parquet(tmp_path, index=False, engine="pyarrow")
    tmp_path.replace(out_path)
    return out_path


def log_variable_choices(
    used_meta: dict[str, set[tuple[object, object, object, object]]],
    step_ranges_seen: dict[str, set[str]],
    logger: logging.Logger,
) -> None:
    for var in ("t2m", "td2m", "rh2m", "u10", "v10", "tcc", "tp", "ssrd"):
        entries = used_meta.get(var, set())
        if not entries:
            logger.warning("Missing GRIB variable for %s", var)
            continue
        formatted = sorted(
            f"shortName={sn},paramId={pid},stepType={st},units={u}"
            for sn, pid, st, u in entries
        )
        logger.info("Variable mapping %s: %s", var, " | ".join(formatted))

    for var in ("tp", "ssrd"):
        ranges = sorted(step_ranges_seen.get(var, set()))
        if ranges:
            logger.info("stepRange values seen for %s: %s", var, ", ".join(ranges))


def list_cycles_in_range(archive_root: Path, start: str, end: str) -> list[Path]:
    valid = re.compile(r"^\d{10}$")
    cycles = [
        p
        for p in archive_root.iterdir()
        if p.is_dir() and valid.fullmatch(p.name) and start <= p.name <= end
    ]
    return sorted(cycles, key=lambda p: p.name)


def split_cycles_for_workers(cycles: list[Path], workers: int) -> list[list[Path]]:
    if workers <= 1 or len(cycles) <= 1:
        return [cycles]
    workers = min(workers, len(cycles))
    chunk_size = int(math.ceil(len(cycles) / workers))
    chunks: list[list[Path]] = []
    for i in range(0, len(cycles), chunk_size):
        chunk = cycles[i : i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


def worker_log_path(base_log_path: Path, worker_id: int) -> Path:
    suffix = base_log_path.suffix or ".log"
    stem = base_log_path.stem
    return base_log_path.with_name(f"{stem}.worker{worker_id}{suffix}")


def process_cycle_chunk(
    *,
    worker_id: int,
    cycle_dirs: list[str],
    stations_csv: str,
    obs_root: str,
    obs_dsn: str | None,
    train_root: str,
    base_log_file: str,
) -> dict[str, object]:
    cycle_paths = [Path(x) for x in cycle_dirs]
    log_path = worker_log_path(Path(base_log_file), worker_id)
    logger = setup_logging(
        log_path,
        logger_name=f"build_gfs_trainset.worker{worker_id}",
        to_stdout=False,
    )

    stations = load_stations(Path(stations_csv), logger)
    obs_by_city: dict[str, pd.DataFrame] = {}
    for st in stations:
        obs_by_city[st.city_name] = load_obs(
            city_name=st.city_name,
            obs_root=Path(obs_root),
            local_timezone=st.local_timezone,
            logger=logger,
            obs_dsn=obs_dsn,
        )

    failures = 0
    for cycle_dir in cycle_paths:
        try:
            build_for_cycle(
                cycle_dir,
                stations=stations,
                obs_by_city=obs_by_city,
                train_root=Path(train_root),
                logger=logger,
            )
        except Exception as exc:
            failures += 1
            logger.exception("Failed cycle %s: %s", cycle_dir, exc)
            continue

    return {
        "worker_id": worker_id,
        "cycles": len(cycle_paths),
        "failures": failures,
        "log_file": str(log_path),
        "first_cycle": cycle_paths[0].name if cycle_paths else "",
        "last_cycle": cycle_paths[-1].name if cycle_paths else "",
    }


def build_for_cycle(
    cycle_dir: Path,
    *,
    stations: list[Station],
    obs_by_city: dict[str, pd.DataFrame],
    train_root: Path,
    logger: logging.Logger,
) -> None:
    logger.info("Processing cycle: %s", cycle_dir)
    issue_time_utc, metadata, first_valid, last_valid = inspect_cycle_gribs(cycle_dir, logger)
    logger.info("Detected issue_time_utc: %s", issue_time_utc.isoformat())
    logger.info(
        "Detected valid range for filename: first=%s last=%s",
        first_valid.isoformat(),
        last_valid.isoformat(),
    )
    if first_valid > issue_time_utc:
        logger.warning(
            "Cycle appears to miss f000 or earlier messages: first_valid=%s issue=%s",
            first_valid.isoformat(),
            issue_time_utc.isoformat(),
        )
    if not metadata:
        raise ValueError(f"No metadata/messages found in {cycle_dir}")

    instant_wide, interval_df, used_meta, step_ranges_seen = extract_point_timeseries(
        cycle_dir=cycle_dir,
        stations=stations,
        issue_time_utc=issue_time_utc,
        logger=logger,
    )
    log_variable_choices(used_meta, step_ranges_seen, logger)

    missing_required = [v for v in REQUIRED_FEATURE_VARS if v not in used_meta]
    if missing_required:
        logger.warning("Cycle %s missing required variables: %s", cycle_dir.name, ", ".join(missing_required))

    features_df = compute_daily_features(
        stations=stations,
        issue_time_utc=issue_time_utc,
        instant_wide=instant_wide,
        interval_df=interval_df,
        obs_by_city=obs_by_city,
        used_meta=used_meta,
        logger=logger,
    )

    if features_df.empty:
        logger.warning("No daily rows computed for cycle %s; writing empty city files", cycle_dir.name)

    for st in stations:
        city_name = st.city_name
        city_df = features_df[features_df["city_name"] == city_name] if not features_df.empty else pd.DataFrame(columns=OUTPUT_COLUMNS)
        out_path = write_parquet(
            df_city=city_df,
            train_root=train_root,
            city_name=city_name,
            first_valid=first_valid,
            last_valid=last_valid,
        )
        logger.info("Wrote parquet: %s", out_path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-cycle GFS training parquet files.")
    p.add_argument("--start", required=True, help="Start cycle inclusive, YYYYMMDDHH")
    p.add_argument("--end", required=True, help="End cycle inclusive, YYYYMMDDHH")
    p.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    p.add_argument("--stations-csv", default=str(DEFAULT_STATIONS_CSV))
    p.add_argument(
        "--obs-root",
        default=str(DEFAULT_OBS_ROOT),
        help="Deprecated (observations now read from DB); kept for compatibility.",
    )
    p.add_argument(
        "--obs-dsn",
        default=None,
        help="Optional DSN for master_db observation reads (defaults to MASTER_POSTGRES_DSN/config-derived value).",
    )
    p.add_argument("--train-root", default=str(DEFAULT_TRAIN_ROOT))
    p.add_argument("--log-file", default=str(DEFAULT_LOG_PATH))
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers; cycles are split into non-overlapping chunks.",
    )
    return p.parse_args(argv)


def validate_cycle_arg(value: str, name: str) -> None:
    if not re.fullmatch(r"\d{10}", value):
        raise ValueError(f"{name} must be YYYYMMDDHH, got: {value}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    validate_cycle_arg(args.start, "--start")
    validate_cycle_arg(args.end, "--end")
    if args.start > args.end:
        raise SystemExit("--start must be <= --end")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")

    logger = setup_logging(Path(args.log_file).expanduser().resolve())
    archive_root = Path(args.archive_root).expanduser().resolve()
    stations_csv = Path(args.stations_csv).expanduser().resolve()
    obs_root = Path(args.obs_root).expanduser().resolve()
    obs_dsn = resolve_master_postgres_dsn(explicit_dsn=args.obs_dsn)
    train_root = Path(args.train_root).expanduser().resolve()

    if not archive_root.exists():
        raise SystemExit(f"Archive root does not exist: {archive_root}")

    cycles = list_cycles_in_range(archive_root, args.start, args.end)
    logger.info(
        "Found %d cycle directories in [%s, %s] under %s",
        len(cycles),
        args.start,
        args.end,
        archive_root,
    )
    if not cycles:
        return 0

    workers = min(int(args.workers), len(cycles))
    failures = 0

    if workers <= 1:
        stations = load_stations(stations_csv, logger)
        obs_by_city: dict[str, pd.DataFrame] = {}
        for st in stations:
            obs_by_city[st.city_name] = load_obs(
                city_name=st.city_name,
                obs_root=obs_root,
                local_timezone=st.local_timezone,
                logger=logger,
                obs_dsn=obs_dsn,
            )
        for cycle_dir in cycles:
            try:
                build_for_cycle(
                    cycle_dir,
                    stations=stations,
                    obs_by_city=obs_by_city,
                    train_root=train_root,
                    logger=logger,
                )
            except Exception as exc:
                failures += 1
                logger.exception("Failed cycle %s: %s", cycle_dir, exc)
                continue
    else:
        chunks = split_cycles_for_workers(cycles, workers)
        logger.info(
            "Parallel mode enabled: workers=%d chunks=%d (non-overlapping cycle chunks)",
            workers,
            len(chunks),
        )
        for idx, chunk in enumerate(chunks, start=1):
            logger.info(
                "Worker %d assigned %d cycles: %s -> %s",
                idx,
                len(chunk),
                chunk[0].name,
                chunk[-1].name,
            )

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_cycle_chunk,
                    worker_id=idx,
                    cycle_dirs=[str(p) for p in chunk],
                    stations_csv=str(stations_csv),
                    obs_root=str(obs_root),
                    obs_dsn=obs_dsn,
                    train_root=str(train_root),
                    base_log_file=str(Path(args.log_file).expanduser().resolve()),
                )
                for idx, chunk in enumerate(chunks, start=1)
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    failures += int(result["failures"])
                    logger.info(
                        "Worker %s done: cycles=%s failures=%s range=%s->%s log=%s",
                        result["worker_id"],
                        result["cycles"],
                        result["failures"],
                        result["first_cycle"],
                        result["last_cycle"],
                        result["log_file"],
                    )
                except Exception as exc:
                    failures += 1
                    logger.exception("Worker process failed: %s", exc)

    logger.info("Completed. cycles=%d failures=%d", len(cycles), failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
