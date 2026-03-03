import argparse
import csv
import dataclasses
import datetime as dt
import http.client
import json
import math
import os
import re
import socket
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from master_db import (  # noqa: E402
    ensure_master_db_ready,
    get_latest_observed_date_local,
    upsert_station_observations,
)


DEFAULT_LOCATIONS_CSV = os.path.join(_REPO_ROOT, "locations.csv")
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "data", "observations")
DEFAULT_UNITS = "e"  # English/Imperial units from weather.com APIs (temp=F, precip=in)
DEFAULT_CHUNK_DAYS = 30
DEFAULT_BOOTSTRAP_DAYS = 30
DEFAULT_OUTPUT_FORMAT = "db"  # db|jsonl|parquet (parquet is deprecated alias for db)
DEFAULT_RETRIES = 3
DEFAULT_RETRY_BACKOFF_S = 1.0


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_yyyy_mm_dd(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _date_to_yyyymmdd(value: dt.date) -> str:
    return value.strftime("%Y%m%d")


def _sleep_seconds(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _is_retryable_http_error(error: urllib.error.HTTPError) -> bool:
    code = getattr(error, "code", None)
    if code in {408, 429, 500, 502, 503, 504}:
        return True
    return False


def _retry_sleep_s(attempt_index: int, base_backoff_s: float) -> float:
    if base_backoff_s <= 0:
        return 0.0
    # exponential backoff: 1x, 2x, 4x...
    return base_backoff_s * (2**attempt_index)


def http_get_text(
    url: str,
    *,
    timeout_s: int = 30,
    headers: Optional[dict[str, str]] = None,
    retries: int = DEFAULT_RETRIES,
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (polymarket-weather-trading/observations)",
            **(headers or {}),
        },
        method="GET",
    )
    last_error: Optional[Exception] = None
    for attempt in range(max(1, retries + 1)):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
            return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            last_error = exc
            if not _is_retryable_http_error(exc) or attempt >= retries:
                raise
            _log(f"HTTP {exc.code} fetching {url} (attempt {attempt+1}/{retries+1}); retrying...")
        except (TimeoutError, socket.timeout, http.client.IncompleteRead, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            _log(f"Error fetching {url} ({type(exc).__name__}) (attempt {attempt+1}/{retries+1}); retrying...")
        _sleep_seconds(_retry_sleep_s(attempt, retry_backoff_s))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


def http_get_json(
    url: str,
    *,
    timeout_s: int = 30,
    headers: Optional[dict[str, str]] = None,
    retries: int = DEFAULT_RETRIES,
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "polymarket-weather-trading/observations",
            **(headers or {}),
        },
        method="GET",
    )
    last_error: Optional[Exception] = None
    for attempt in range(max(1, retries + 1)):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
            return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if not _is_retryable_http_error(exc) or attempt >= retries:
                raise
            _log(f"HTTP {exc.code} fetching {url} (attempt {attempt+1}/{retries+1}); retrying...")
        except (TimeoutError, socket.timeout, http.client.IncompleteRead, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            _log(f"Error fetching {url} ({type(exc).__name__}) (attempt {attempt+1}/{retries+1}); retrying...")
        _sleep_seconds(_retry_sleep_s(attempt, retry_backoff_s))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


def extract_weather_com_api_key(wunderground_html: str) -> str:
    key_match = re.search(r'"SUN_API_KEY"\s*:\s*"([^"]+)"', wunderground_html)
    if key_match:
        return key_match.group(1)

    query_match = re.search(r"\bapiKey=([0-9a-f]{32})\b", wunderground_html)
    if query_match:
        return query_match.group(1)

    raise RuntimeError("Could not find weather.com apiKey in Wunderground HTML")


def extract_iana_timezone(wunderground_html: str) -> str:
    tz_match = re.search(r'ianaTimeZone"\s*:\s*"([^"]+)"', wunderground_html)
    if tz_match:
        return tz_match.group(1)
    raise RuntimeError("Could not find ianaTimeZone in Wunderground HTML")


@dataclasses.dataclass(frozen=True)
class Station:
    name: str
    url: str
    lat_lon: str
    country_code: str
    station_code: str
    timezone: Optional[str] = None

    @property
    def weather_com_location_id(self) -> str:
        return f"{self.station_code}:9:{self.country_code}"


def parse_station_from_wunderground_url(
    *,
    name: str,
    url: str,
    lat_lon: str,
    timezone: Optional[str] = None,
) -> Station:
    parsed = urllib.parse.urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    try:
        daily_index = parts.index("daily")
    except ValueError as exc:
        raise ValueError(f"Unexpected Wunderground URL (missing /daily/): {url}") from exc

    if daily_index + 1 >= len(parts) or len(parts) < 2:
        raise ValueError(f"Unexpected Wunderground URL structure: {url}")

    country = parts[daily_index + 1].upper()
    station_code = parts[-1].upper()
    if not re.fullmatch(r"[A-Z0-9]{3,6}", station_code):
        raise ValueError(f"Unexpected station code in URL ({station_code!r}): {url}")

    timezone = timezone.strip() if isinstance(timezone, str) and timezone.strip() else None
    return Station(
        name=name,
        url=url,
        lat_lon=lat_lon,
        country_code=country,
        station_code=station_code,
        timezone=timezone,
    )


def load_stations(locations_csv_path: str) -> list[Station]:
    stations: list[Station] = []
    with open(locations_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row_number, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            url = (row.get("url") or "").strip()
            lat_lon = (row.get("lat_lon") or "").strip()
            timezone = (row.get("timezone") or "").strip()
            if not name or not url:
                continue
            try:
                stations.append(
                    parse_station_from_wunderground_url(name=name, url=url, lat_lon=lat_lon, timezone=timezone)
                )
            except Exception as exc:
                _log(
                    f"Skipping invalid station row {row_number} in {locations_csv_path}: "
                    f"name={name!r} url={url!r} ({type(exc).__name__}: {exc})"
                )
                continue
    if not stations:
        raise RuntimeError(f"No stations found in {locations_csv_path}")
    return stations


def iter_date_chunks(start: dt.date, end: dt.date, *, chunk_days: int) -> Iterable[tuple[dt.date, dt.date]]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be > 0")
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + dt.timedelta(days=chunk_days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + dt.timedelta(days=1)


def fetch_historical_observations(
    *,
    api_key: str,
    station: Station,
    start: dt.date,
    end: dt.date,
    units: str = DEFAULT_UNITS,
    timeout_s: int = 30,
    retries: int = DEFAULT_RETRIES,
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
) -> list[dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "units": units,
        "startDate": _date_to_yyyymmdd(start),
        "endDate": _date_to_yyyymmdd(end),
    }
    location_id = station.weather_com_location_id
    url = (
        "https://api.weather.com/v1/location/"
        + urllib.parse.quote(location_id, safe="")
        + "/observations/historical.json?"
        + urllib.parse.urlencode(params)
    )
    payload = http_get_json(url, timeout_s=timeout_s, retries=retries, retry_backoff_s=retry_backoff_s)
    observations = payload.get("observations", [])
    if not isinstance(observations, list):
        return []
    return [o for o in observations if isinstance(o, dict)]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _epoch_s_to_utc_dt(epoch_s: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(epoch_s, tz=dt.timezone.utc)


def _parse_observed_at_local(value: str) -> Optional[dt.datetime]:
    value = value.strip()
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _extract_observed_at_local(record: dict[str, Any]) -> Optional[str]:
    value = record.get("observed_at_local") or record.get("obs_time_local") or record.get("valid_time_local")
    if isinstance(value, str) and value.strip():
        return value.strip()
    valid_time_gmt = record.get("valid_time_gmt")
    if isinstance(valid_time_gmt, int):
        return _epoch_s_to_utc_dt(valid_time_gmt).isoformat()
    return None


def _normalize_record(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    observed_at_local = _extract_observed_at_local(record)
    if observed_at_local is None:
        return None
    normalized = dict(record)
    normalized["observed_at_local"] = observed_at_local
    normalized.pop("valid_time_gmt", None)
    normalized.pop("observed_at_utc", None)
    normalized.pop("wunderground_url", None)
    normalized.pop("weather_com_location_id", None)
    normalized.pop("obs_time_local", None)
    normalized.pop("valid_time_local", None)
    return normalized


def _round_half_away_from_zero(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def observations_to_records(
    *,
    station: Station,
    observations: list[dict[str, Any]],
    tzinfo: dt.tzinfo,
    scraped_at_utc: dt.datetime,
    include_precip: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for obs in observations:
        valid_time_gmt = obs.get("valid_time_gmt")
        if not isinstance(valid_time_gmt, int):
            continue
        local_dt = dt.datetime.fromtimestamp(valid_time_gmt, tz=dt.timezone.utc).astimezone(tzinfo)

        temp_f = _to_float(obs.get("temp"))
        if temp_f is None:
            continue
        temp_c = (temp_f - 32.0) * (5.0 / 9.0)
        temp_f_int = _round_half_away_from_zero(temp_f)
        temp_c_int = _round_half_away_from_zero(temp_c)

        record: dict[str, Any] = {
            "station": station.name,
            "observed_at_local": local_dt.isoformat(),
            "temperature_f": temp_f_int,
            "temperature_c": temp_c_int,
            "scraped_at_utc": scraped_at_utc.isoformat(),
        }
        record["precipitation_hourly_in"] = _to_float(obs.get("precip_hrly")) if include_precip else None
        record["precipitation_total_in"] = _to_float(obs.get("precip_total")) if include_precip else None
        records.append(record)
    return records


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Parquet support requires a working pyarrow install. "
            "If pyarrow fails to import in this environment, run with --format jsonl, "
            "or fix your environment (commonly: downgrade to numpy<2, or reinstall a NumPy-2-compatible pyarrow)."
        ) from exc


def _read_existing_jsonl_records(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _write_jsonl_records(records: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def load_existing_parquet_records(path: str) -> Optional[list[dict[str, Any]]]:
    _require_pyarrow()
    import pyarrow.parquet as pq

    if not os.path.exists(path):
        return None
    table = pq.read_table(path)
    return table.to_pylist()


def write_parquet_records(records: list[dict[str, Any]], path: str) -> None:
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd")


def upsert_station_records(*, new_records: list[dict[str, Any]], master_dsn: str) -> dict[str, int]:
    return upsert_station_observations(records=new_records, master_dsn=master_dsn)


def upsert_station_records_jsonl(*, output_path: str, new_records: list[dict[str, Any]]) -> int:
    if not new_records:
        return 0

    existing_records = _read_existing_jsonl_records(output_path)
    by_time: dict[str, dict[str, Any]] = {}
    for record in existing_records:
        normalized = _normalize_record(record)
        if normalized is None:
            continue
        by_time[normalized["observed_at_local"]] = normalized

    inserted = 0
    for record in new_records:
        normalized = _normalize_record(record)
        if normalized is None:
            continue
        key = normalized["observed_at_local"]
        if key not in by_time:
            inserted += 1
        by_time[key] = normalized

    combined = [by_time[k] for k in sorted(by_time.keys())]
    _write_jsonl_records(combined, output_path)
    return inserted


def _infer_station_output_path(output_dir: str, station: Station) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", station.name).strip("_")
    return os.path.join(output_dir, f"{safe_name}")


def _latest_observed_date_local_parquet(output_path: str) -> Optional[dt.date]:
    existing = load_existing_parquet_records(output_path)
    if not existing:
        return None
    latest: Optional[dt.date] = None
    for record in existing:
        observed_at_local = _extract_observed_at_local(record)
        if observed_at_local is None:
            continue
        parsed = _parse_observed_at_local(observed_at_local)
        if parsed is None:
            continue
        observed_date = parsed.date()
        latest = observed_date if latest is None else max(latest, observed_date)
    return latest


def _latest_observed_date_local_jsonl(output_path: str) -> Optional[dt.date]:
    latest: Optional[dt.date] = None
    if not os.path.exists(output_path):
        return None
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            observed_at_local = _extract_observed_at_local(record)
            if observed_at_local is None:
                continue
            parsed = _parse_observed_at_local(observed_at_local)
            if parsed is None:
                continue
            observed_date = parsed.date()
            latest = observed_date if latest is None else max(latest, observed_date)
    return latest


def _normalize_output_format(value: str) -> str:
    value = (value or "").strip().lower()
    if value == "parquet":
        _log("--format parquet is deprecated; using database mode instead.")
        return "db"
    if value in {"db", "jsonl"}:
        return value
    raise ValueError("--format must be one of: db, jsonl (parquet is accepted as alias for db)")


def scrape_once(
    *,
    stations: list[Station],
    output_dir: str,
    start_date: Optional[dt.date],
    end_date: dt.date,
    chunk_days: int,
    throttle_s: float,
    timeout_s: int,
    dry_run: bool,
    output_format: str,
    master_dsn: Optional[str],
    retries: int,
    retry_backoff_s: float,
    fail_fast: bool,
) -> None:
    output_format = _normalize_output_format(output_format)
    resolved_master_dsn: Optional[str] = None
    if output_format == "db" and not dry_run:
        resolved_master_dsn = ensure_master_db_ready(master_dsn=master_dsn)

    shared_api_key: Optional[str] = None
    shared_api_key_station: Optional[str] = None
    for station in stations:
        try:
            include_precip = station.name.strip().casefold() == "nyc"
            base_path = _infer_station_output_path(output_dir, station)
            output_path = base_path + ".jsonl"
            output_target = "master_db.station_observations" if output_format == "db" else output_path

            if start_date is None:
                if dry_run:
                    latest = None
                else:
                    latest = (
                        get_latest_observed_date_local(station.name, master_dsn=resolved_master_dsn)
                        if output_format == "db"
                        else _latest_observed_date_local_jsonl(output_path)
                    )
                if latest is None:
                    station_start = end_date - dt.timedelta(days=DEFAULT_BOOTSTRAP_DAYS)
                else:
                    station_start = latest - dt.timedelta(days=1)
            else:
                station_start = start_date

            station_start = min(station_start, end_date)

            html: Optional[str] = None
            tzinfo: dt.tzinfo | None = None
            wu_timezone: Optional[str] = None
            if station.timezone:
                try:
                    tzinfo = ZoneInfo(station.timezone)
                except Exception as exc:
                    _log(
                        f"{station.name}: invalid locations.csv timezone {station.timezone!r} "
                        f"({type(exc).__name__}: {exc}); falling back to Wunderground"
                    )
                    tzinfo = None
            need_station_page = (shared_api_key is None) or (tzinfo is None)
            if need_station_page:
                try:
                    html = http_get_text(
                        station.url,
                        timeout_s=timeout_s,
                        retries=retries,
                        retry_backoff_s=retry_backoff_s,
                    )
                except Exception as exc:
                    if shared_api_key is None:
                        raise RuntimeError(
                            f"Could not fetch station page for API key discovery: {station.url}"
                        ) from exc
                    _log(
                        f"{station.name}: failed fetching station page "
                        f"({type(exc).__name__}: {exc}); using cached API key from "
                        f"{shared_api_key_station or 'previous station'}"
                    )

            station_api_key: Optional[str] = None
            if html is not None:
                try:
                    station_api_key = extract_weather_com_api_key(html)
                except Exception as exc:
                    if shared_api_key is None:
                        raise RuntimeError(
                            f"Could not extract weather.com api key from {station.url}"
                        ) from exc
                    _log(
                        f"{station.name}: failed to extract weather.com api key "
                        f"({type(exc).__name__}: {exc}); using cached key from "
                        f"{shared_api_key_station or 'previous station'}"
                    )

            if station_api_key:
                api_key = station_api_key
                if shared_api_key is None:
                    shared_api_key = station_api_key
                    shared_api_key_station = station.name
            elif shared_api_key is not None:
                api_key = shared_api_key
            else:
                raise RuntimeError("No weather.com api key available")

            try:
                if html is not None:
                    wu_timezone = extract_iana_timezone(html)
            except Exception as exc:
                wu_timezone = None
                _log(
                    f"{station.name}: failed to extract Wunderground timezone "
                    f"({type(exc).__name__}: {exc})"
                )
            if tzinfo is None:
                if wu_timezone:
                    try:
                        tzinfo = ZoneInfo(wu_timezone)
                    except Exception as exc:
                        _log(
                            f"{station.name}: invalid Wunderground timezone {wu_timezone!r} "
                            f"({type(exc).__name__}: {exc}); falling back to UTC"
                        )
                        tzinfo = dt.timezone.utc
                else:
                    _log(f"{station.name}: no timezone available; falling back to UTC")
                    tzinfo = dt.timezone.utc
            if station.timezone and wu_timezone and station.timezone != wu_timezone:
                _log(
                    f"{station.name}: timezone mismatch locations.csv={station.timezone!r} "
                    f"Wunderground={wu_timezone!r}; using locations.csv"
                )

            scraped_at = _utcnow()
            total_inserted = 0
            total_updated = 0
            total_unchanged = 0
            buffered: list[dict[str, Any]] = []
            for chunk_start, chunk_end in iter_date_chunks(station_start, end_date, chunk_days=chunk_days):
                try:
                    observations = fetch_historical_observations(
                        api_key=api_key,
                        station=station,
                        start=chunk_start,
                        end=chunk_end,
                        timeout_s=timeout_s,
                        retries=retries,
                        retry_backoff_s=retry_backoff_s,
                    )
                except Exception as exc:
                    _log(
                        f"{station.name}: failed fetching {chunk_start}..{chunk_end} "
                        f"({type(exc).__name__}: {exc}); skipping chunk"
                    )
                    if fail_fast:
                        raise
                    continue

                records = observations_to_records(
                    station=station,
                    observations=observations,
                    tzinfo=tzinfo,
                    scraped_at_utc=scraped_at,
                    include_precip=include_precip,
                )
                if dry_run:
                    buffered.extend(records)
                else:
                    if output_format == "db":
                        stats = upsert_station_records(new_records=records, master_dsn=resolved_master_dsn or "")
                        total_inserted += stats["inserted"]
                        total_updated += stats["updated"]
                        total_unchanged += stats["unchanged"]
                    else:
                        inserted = upsert_station_records_jsonl(output_path=output_path, new_records=records)
                        total_inserted += inserted
                _sleep_seconds(throttle_s)

            if dry_run:
                unique = {r.get("observed_at_local") for r in buffered if isinstance(r.get("observed_at_local"), str)}
                sample = buffered[-1] if buffered else None
                print(f"{station.name}: scraped {len(unique)} unique timestamps (dry-run). sample={sample}")
            else:
                if output_format == "db":
                    print(
                        f"{station.name}: inserted={total_inserted} updated={total_updated} "
                        f"unchanged={total_unchanged} into {output_target}"
                    )
                else:
                    print(f"{station.name}: added {total_inserted} new rows to {output_target}")
        except Exception as exc:
            _log(f"{station.name}: failed ({type(exc).__name__}: {exc}); skipping station")
            if fail_fast:
                raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Wunderground-backed weather.com observations into master_db.station_observations "
            "(or JSONL if requested)."
        )
    )
    parser.add_argument("--locations", default=DEFAULT_LOCATIONS_CSV, help="Path to locations.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for jsonl files")
    parser.add_argument(
        "--start-date",
        type=_parse_yyyy_mm_dd,
        default=None,
        help="Backfill start date (YYYY-MM-DD). If omitted, uses last stored timestamp or ~30 days bootstrap.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_yyyy_mm_dd,
        default=None,
        help="Backfill end date (YYYY-MM-DD), default: today",
    )
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS, help="Days per API request chunk")
    parser.add_argument("--timeout-s", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--throttle-s", type=float, default=0.5, help="Sleep between requests, seconds")
    parser.add_argument("--dry-run", action="store_true", help="Scrape and print counts, but do not write output")
    parser.add_argument(
        "--format",
        default=DEFAULT_OUTPUT_FORMAT,
        help="Output format: db (default), jsonl, or deprecated parquet alias (mapped to db).",
    )
    parser.add_argument(
        "--master-dsn",
        default=None,
        help="Optional DSN for master_db (defaults to MASTER_POSTGRES_DSN or config-derived value).",
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="HTTP retry attempts on timeouts/5xx")
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_S,
        help="Base seconds for exponential backoff between retries",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on first station/chunk failure (default: continue)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only scrape a station name (repeatable). Matches case-insensitively on locations.csv 'name'.",
    )
    parser.add_argument(
        "--loop-minutes",
        type=float,
        default=0.0,
        help="If set (>0), run forever and rescrape every N minutes (e.g. 30).",
    )
    parser.add_argument(
        "--live-after-backfill",
        action="store_true",
        help=(
            "If set with --start-date, run the backfill once, then keep looping without --start-date "
            "(incremental live updates). Requires --loop-minutes > 0."
        ),
    )
    args = parser.parse_args()

    stations = load_stations(args.locations)
    if args.only:
        only = {s.casefold() for s in args.only}
        stations = [s for s in stations if s.name.casefold() in only]
    if not stations:
        raise RuntimeError("No stations selected")

    start_date_for_loop = args.start_date
    while True:
        end_date = args.end_date or dt.date.today()
        scrape_once(
            stations=stations,
            output_dir=args.output_dir,
            start_date=start_date_for_loop,
            end_date=end_date,
            chunk_days=args.chunk_days,
            throttle_s=args.throttle_s,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            output_format=args.format,
            master_dsn=args.master_dsn,
            retries=args.retries,
            retry_backoff_s=args.retry_backoff_s,
            fail_fast=args.fail_fast,
        )
        if args.loop_minutes and args.loop_minutes > 0:
            if args.live_after_backfill and start_date_for_loop is not None:
                start_date_for_loop = None
            _sleep_seconds(args.loop_minutes * 60.0)
        else:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
