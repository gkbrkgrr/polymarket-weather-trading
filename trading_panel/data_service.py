from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import psycopg

from master_db import get_historical_daily_tmax_bounds as fetch_historical_daily_tmax_bounds

from .config import ModelSpec

REQUIRED_PRED_COLUMNS = ["issue_time_utc", "target_date_local", "Forecast"]
CYCLE_TOKEN_PATTERN = re.compile(r"(\d{10})$")
TITLE_BETWEEN_PATTERN = re.compile(r"be between\s+(-?\d+)\s*-\s*(-?\d+)\s*°([CF])", re.IGNORECASE)
TITLE_EXACT_PATTERN = re.compile(r"be\s+(-?\d+)\s*°([CF])\s+on", re.IGNORECASE)
TITLE_HIGHER_PATTERN = re.compile(r"be\s+(-?\d+)\s*°([CF])\s+or higher", re.IGNORECASE)
TITLE_BELOW_PATTERN = re.compile(r"be\s+(-?\d+)\s*°([CF])\s+or below", re.IGNORECASE)
TITLE_DATE_PATTERN = re.compile(r"\bon\s+([A-Za-z]+)\s+(\d{1,2})\??", re.IGNORECASE)


@dataclass(frozen=True)
class FileEntry:
    token: str
    issue_time_utc: dt.datetime
    path: Path


@dataclass(frozen=True)
class StationMeta:
    timezone: str
    is_usa: bool
    resolution_source_key: str


@dataclass(frozen=True)
class ObservationSnapshot:
    observed_at_local: dt.datetime
    rounded_value: int


class PanelDataService:
    def __init__(
        self,
        *,
        model_specs: tuple[ModelSpec, ...],
        locations_csv: Path,
        master_dsn: str,
        cache_ttl_seconds: int,
        default_history_days: int,
        parquet_read_workers: int,
    ) -> None:
        self.model_specs = model_specs
        self.locations_csv = locations_csv
        self.master_dsn = master_dsn
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))
        self.parquet_read_workers = max(1, int(parquet_read_workers))

        self.station_meta = self._load_station_meta(locations_csv)
        self.file_index = self._build_file_index(model_specs)
        self.stations = self._discover_common_stations()
        self.history_days = self._infer_history_days(default_history_days)
        self.historical_bounds = self._load_historical_bounds()
        self.resolved_yes_markets = self._load_resolved_yes_markets()

        self._cache_lock = threading.Lock()
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def get_panel_payload(self, day: dt.date, *, force_refresh: bool = False) -> dict[str, Any]:
        cache_key = day.isoformat()
        now = time.monotonic()
        if not force_refresh:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                loaded_at, payload = cached
                if now - loaded_at <= self.cache_ttl_seconds:
                    return payload

        payload = self._build_payload(day)
        with self._cache_lock:
            self._cache[cache_key] = (time.monotonic(), payload)
        return payload

    def warm_cache_for_day(self, day: dt.date) -> None:
        self.get_panel_payload(day, force_refresh=True)

    def _build_payload(self, day: dt.date) -> dict[str, Any]:
        payload_started = time.perf_counter()
        db_observations_ms = 0.0

        now_utc = dt.datetime.now(dt.timezone.utc)
        db_started = time.perf_counter()
        last_max_obs = self._fetch_last_max_observations(day)
        db_observations_ms += (time.perf_counter() - db_started) * 1000.0
        month_day = day.strftime("%m-%d")

        station_meta_lookup: dict[str, StationMeta] = {}
        for station in self.stations:
            station_meta_lookup[station] = self.station_meta.get(
                station,
                StationMeta(timezone="UTC", is_usa=False, resolution_source_key=""),
            )

        parquet_started = time.perf_counter()
        station_model_frames, parquet_task_sum_ms = self._load_station_model_days_parallel(
            day=day,
            station_meta_lookup=station_meta_lookup,
        )
        parquet_reads_ms = (time.perf_counter() - parquet_started) * 1000.0

        stations_payload: list[dict[str, Any]] = []
        for station in self.stations:
            meta = station_meta_lookup[station]
            unit = "F" if meta.is_usa else "C"
            last_max_snapshot = last_max_obs.get(station)
            current_local_time = self._format_current_local_time(day, meta, now_utc)
            last_observation = self._format_last_max_observation(last_max_snapshot)
            reference_lines = self._build_reference_lines(
                station=station,
                month_day=month_day,
                is_usa=meta.is_usa,
                last_max_snapshot=last_max_snapshot,
            )

            traces: list[dict[str, Any]] = []
            for spec in self.model_specs:
                model_df = station_model_frames.get(
                    (station, spec.key),
                    pd.DataFrame(columns=["issue_time_utc", "forecast_rounded"]),
                )
                traces.append(
                    {
                        "model_key": spec.key,
                        "label": spec.label,
                        "color": spec.color,
                        "x": [
                            ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                            for ts in model_df["issue_time_utc"]
                        ],
                        "y": model_df["forecast_rounded"].tolist(),
                    }
                )

            resolved_market = self.resolved_yes_markets.get(
                (station, day),
                {"left": "Max. Temp.: N/A", "right": ""},
            )

            stations_payload.append(
                {
                    "station": station,
                    "unit": unit,
                    "current_local_time": current_local_time,
                    "last_observation": last_observation,
                    "resolved_yes_market_left": resolved_market["left"],
                    "resolved_yes_market_right": resolved_market["right"],
                    "reference_lines": reference_lines,
                    "traces": traces,
                }
            )

        payload_build_ms = (time.perf_counter() - payload_started) * 1000.0
        return {
            "date": day.isoformat(),
            "generated_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "history_days": self.history_days,
            "station_count": len(stations_payload),
            "timings_ms": {
                "db_observations": round(db_observations_ms, 2),
                "parquet_reads": round(parquet_reads_ms, 2),
                "parquet_reads_task_sum": round(parquet_task_sum_ms, 2),
                "payload_build": round(payload_build_ms, 2),
            },
            "stations": stations_payload,
        }

    def _load_station_model_days_parallel(
        self,
        *,
        day: dt.date,
        station_meta_lookup: dict[str, StationMeta],
    ) -> tuple[dict[tuple[str, str], pd.DataFrame], float]:
        tasks: list[tuple[str, ModelSpec, bool]] = []
        for station in self.stations:
            meta = station_meta_lookup[station]
            for spec in self.model_specs:
                tasks.append((station, spec, meta.is_usa))

        out: dict[tuple[str, str], pd.DataFrame] = {}
        parquet_task_sum_ms = 0.0
        if not tasks:
            return out, parquet_task_sum_ms

        def _worker(station: str, model_spec: ModelSpec, is_usa: bool) -> tuple[str, str, pd.DataFrame, float]:
            started = time.perf_counter()
            df = self._load_station_model_day(
                station=station,
                day=day,
                model_spec=model_spec,
                is_usa=is_usa,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return station, model_spec.key, df, elapsed_ms

        max_workers = min(self.parquet_read_workers, len(tasks))
        if max_workers <= 1:
            for station, spec, is_usa in tasks:
                st, model_key, frame, elapsed_ms = _worker(station, spec, is_usa)
                out[(st, model_key)] = frame
                parquet_task_sum_ms += elapsed_ms
            return out, parquet_task_sum_ms

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_worker, station, spec, is_usa) for station, spec, is_usa in tasks]
            for future in as_completed(futures):
                station, model_key, frame, elapsed_ms = future.result()
                out[(station, model_key)] = frame
                parquet_task_sum_ms += elapsed_ms

        return out, parquet_task_sum_ms

    def _load_station_model_day(
        self,
        *,
        station: str,
        day: dt.date,
        model_spec: ModelSpec,
        is_usa: bool,
    ) -> pd.DataFrame:
        files = self._candidate_files(station=station, day=day, model_key=model_spec.key)
        if not files:
            return pd.DataFrame(columns=["issue_time_utc", "forecast_rounded"])

        parts: list[pd.DataFrame] = []
        for file_path in files:
            try:
                part = pd.read_parquet(
                    file_path,
                    columns=REQUIRED_PRED_COLUMNS,
                    filters=[("target_date_local", "==", day)],
                )
            except (TypeError, ValueError, NotImplementedError):
                part = pd.read_parquet(file_path, columns=REQUIRED_PRED_COLUMNS)
            if part.empty:
                continue

            part["target_date_local"] = pd.to_datetime(part["target_date_local"], errors="coerce").dt.date
            part = part[part["target_date_local"] == day]
            if part.empty:
                continue

            parts.append(part[["issue_time_utc", "Forecast"]])

        if not parts:
            return pd.DataFrame(columns=["issue_time_utc", "forecast_rounded"])

        combined = pd.concat(parts, ignore_index=True)
        combined["issue_time_utc"] = pd.to_datetime(combined["issue_time_utc"], utc=True, errors="coerce")
        combined["Forecast"] = pd.to_numeric(combined["Forecast"], errors="coerce")
        combined = combined.dropna(subset=["issue_time_utc", "Forecast"])
        if combined.empty:
            return pd.DataFrame(columns=["issue_time_utc", "forecast_rounded"])

        if is_usa:
            combined["Forecast"] = (combined["Forecast"] * 9.0 / 5.0) + 32.0

        combined["forecast_rounded"] = combined["Forecast"].round().astype(int)
        combined = combined.sort_values("issue_time_utc", kind="mergesort")
        combined = combined.drop_duplicates(subset=["issue_time_utc"], keep="last")
        combined = combined[["issue_time_utc", "forecast_rounded"]].reset_index(drop=True)
        return combined

    def _candidate_files(self, *, station: str, day: dt.date, model_key: str) -> list[Path]:
        entries = self.file_index[model_key].get(station, [])
        if not entries:
            return []

        start_day = day - dt.timedelta(days=self.history_days)
        selected = [
            entry.path
            for entry in entries
            if start_day <= entry.issue_time_utc.date() <= day
        ]
        return selected

    def _fetch_last_max_observations(self, day: dt.date) -> dict[str, ObservationSnapshot]:
        day_start = dt.datetime.combine(day, dt.time.min)
        day_end = day_start + dt.timedelta(days=1)
        query = (
            "SELECT "
            "station, observed_at_local, temperature_f, temperature_c "
            "FROM station_observations "
            "WHERE station = ANY(%s) "
            "AND observed_at_local >= %s "
            "AND observed_at_local < %s "
            "ORDER BY station, observed_at_local"
        )

        best_by_station: dict[str, tuple[float, dt.datetime]] = {}
        with psycopg.connect(self.master_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (self.stations, day_start, day_end))
                rows = cur.fetchall()

        for station, observed_at_local, temperature_f, temperature_c in rows:
            meta = self.station_meta.get(
                station,
                StationMeta(timezone="UTC", is_usa=False, resolution_source_key=""),
            )
            value = self._resolve_observed_value(
                is_usa=meta.is_usa,
                temperature_f=temperature_f,
                temperature_c=temperature_c,
            )
            if value is None:
                continue
            numeric_value = float(value)
            best = best_by_station.get(station)
            if best is None or numeric_value > best[0] or (
                numeric_value == best[0] and observed_at_local > best[1]
            ):
                best_by_station[station] = (numeric_value, observed_at_local)

        out: dict[str, ObservationSnapshot] = {}
        for station, (max_value, observed_at_local) in best_by_station.items():
            out[station] = ObservationSnapshot(
                observed_at_local=observed_at_local,
                rounded_value=int(round(max_value)),
            )
        return out

    @staticmethod
    def _resolve_observed_value(*, is_usa: bool, temperature_f: Any, temperature_c: Any) -> float | None:
        if is_usa:
            if temperature_f is not None:
                return float(temperature_f)
            if temperature_c is not None:
                return (float(temperature_c) * 9.0 / 5.0) + 32.0
            return None

        if temperature_c is not None:
            return float(temperature_c)
        if temperature_f is not None:
            return (float(temperature_f) - 32.0) * 5.0 / 9.0
        return None

    def _format_current_local_time(self, day: dt.date, meta: StationMeta, now_utc: dt.datetime) -> str:
        zone = self._get_zone(meta.timezone)
        station_now = now_utc.astimezone(zone)
        if day < station_now.date():
            clock = "23:59:59"
            offset_ref = dt.datetime.combine(day, dt.time(23, 59, 59), tzinfo=zone)
        else:
            clock = station_now.strftime("%H:%M:%S")
            offset_ref = station_now
        offset = offset_ref.utcoffset() or dt.timedelta(0)
        return f"Local Time: {clock} ({self._format_utc_offset(offset)})"

    @staticmethod
    def _format_last_max_observation(
        latest: ObservationSnapshot | None,
    ) -> str:
        if latest is None:
            return "Last Max. Observation: N/A"
        return (
            f"Last Max. Observation: {latest.rounded_value} "
            f"at {latest.observed_at_local.strftime('%H:%M:%S')}"
        )

    @staticmethod
    def _format_utc_offset(offset: dt.timedelta) -> str:
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        abs_minutes = abs(total_minutes)
        hours = abs_minutes // 60
        minutes = abs_minutes % 60
        if minutes == 0:
            return f"UTC{sign}{hours}"
        return f"UTC{sign}{hours}:{minutes:02d}"

    def _build_reference_lines(
        self,
        *,
        station: str,
        month_day: str,
        is_usa: bool,
        last_max_snapshot: ObservationSnapshot | None,
    ) -> list[dict[str, Any]]:
        lines: list[dict[str, Any]] = []

        bounds = self.historical_bounds.get((station, month_day))
        if bounds is not None:
            min_key = "hist_min_f" if is_usa else "hist_min_c"
            max_key = "hist_max_f" if is_usa else "hist_max_c"
            min_value = bounds.get(min_key)
            max_value = bounds.get(max_key)
            if min_value is not None:
                lines.append(
                    {
                        "label": "Historical Min",
                        "value": min_value,
                        "color": "#9ca3af",
                        "dash": "dash",
                        "width": 1.5,
                    }
                )
            if max_value is not None:
                lines.append(
                    {
                        "label": "Historical Max",
                        "value": max_value,
                        "color": "#9ca3af",
                        "dash": "dash",
                        "width": 1.5,
                    }
                )

        if last_max_snapshot is not None:
            lines.append(
                {
                    "label": "Last Max. Observation",
                    "value": float(last_max_snapshot.rounded_value),
                    "color": "#ffffff",
                    "dash": "solid",
                    "width": 2.2,
                }
            )

        return lines

    @staticmethod
    def _get_zone(timezone_name: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            return ZoneInfo("UTC")

    @staticmethod
    def _load_station_meta(locations_csv: Path) -> dict[str, StationMeta]:
        if not locations_csv.exists():
            raise FileNotFoundError(f"Locations CSV not found: {locations_csv}")

        df = pd.read_csv(locations_csv)
        required = {"name", "timezone", "url"}
        missing = required - set(df.columns)
        if missing:
            missing_msg = ", ".join(sorted(missing))
            raise ValueError(f"Locations CSV missing required columns: {missing_msg}")

        out: dict[str, StationMeta] = {}
        for row in df.itertuples(index=False):
            station = str(row.name).strip()
            if not station:
                continue
            timezone_name = str(row.timezone).strip() or "UTC"
            is_usa = "/daily/us/" in str(row.url).strip().lower()
            source_key = PanelDataService._normalize_source_url(str(row.url))
            out[station] = StationMeta(
                timezone=timezone_name,
                is_usa=is_usa,
                resolution_source_key=source_key,
            )
        return out

    @staticmethod
    def _extract_cycle_token(path: Path) -> str | None:
        match = CYCLE_TOKEN_PATTERN.search(path.stem)
        if match is None:
            return None
        return match.group(1)

    def _build_file_index(self, model_specs: tuple[ModelSpec, ...]) -> dict[str, dict[str, list[FileEntry]]]:
        index: dict[str, dict[str, list[FileEntry]]] = {}

        for spec in model_specs:
            if not spec.root.exists():
                raise FileNotFoundError(f"Model root does not exist: {spec.root}")

            station_map: dict[str, list[FileEntry]] = {}
            for station_dir in sorted(spec.root.iterdir()):
                if not station_dir.is_dir():
                    continue

                entries: list[FileEntry] = []
                for file_path in sorted(station_dir.glob("*.parquet")):
                    token = self._extract_cycle_token(file_path)
                    if token is None:
                        continue
                    issue_time_utc = dt.datetime.strptime(token, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
                    entries.append(FileEntry(token=token, issue_time_utc=issue_time_utc, path=file_path))

                if entries:
                    station_map[station_dir.name] = entries

            if not station_map:
                raise RuntimeError(f"No station parquet files found in model root: {spec.root}")

            index[spec.key] = station_map

        return index

    def _discover_common_stations(self) -> list[str]:
        station_sets = [set(stations.keys()) for stations in self.file_index.values()]
        if not station_sets:
            raise RuntimeError("No model station indexes discovered.")

        common = sorted(set.intersection(*station_sets))
        if not common:
            raise RuntimeError("No common stations found across configured model roots.")

        return common

    def _infer_history_days(self, default_history_days: int) -> int:
        inferred = 0
        for spec in self.model_specs:
            station_entries = self.file_index[spec.key].get(self.stations[0], [])
            if not station_entries:
                continue
            sample_file = station_entries[-1].path
            sample = pd.read_parquet(sample_file, columns=["issue_time_utc", "target_date_local"])
            sample["issue_time_utc"] = pd.to_datetime(sample["issue_time_utc"], utc=True, errors="coerce")
            sample["target_date_local"] = pd.to_datetime(sample["target_date_local"], errors="coerce")
            sample = sample.dropna(subset=["issue_time_utc", "target_date_local"])
            if sample.empty:
                continue
            lead_days = (sample["target_date_local"].dt.date - sample["issue_time_utc"].dt.date).map(lambda delta: delta.days)
            inferred = max(inferred, int(lead_days.max()))

        return max(int(default_history_days), inferred + 1)

    def _load_historical_bounds(self) -> dict[tuple[str, str], dict[str, float | None]]:
        df = fetch_historical_daily_tmax_bounds(
            stations=self.stations,
            start_date=dt.date(2000, 1, 1),
            master_dsn=self.master_dsn,
        )
        if df.empty:
            return {}

        out: dict[tuple[str, str], dict[str, float | None]] = {}
        for row in df.itertuples(index=False):
            out[(row.city_name, row.month_day)] = {
                "hist_min_c": self._as_float_or_none(row.hist_min_c),
                "hist_max_c": self._as_float_or_none(row.hist_max_c),
                "hist_min_f": self._as_float_or_none(row.hist_min_f),
                "hist_max_f": self._as_float_or_none(row.hist_max_f),
            }
        return out

    @staticmethod
    def _as_float_or_none(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _load_resolved_yes_markets(self) -> dict[tuple[str, dt.date], dict[str, str]]:
        source_to_station = {
            meta.resolution_source_key: station
            for station, meta in self.station_meta.items()
            if meta.resolution_source_key
        }
        if not source_to_station:
            return {}

        query = (
            "SELECT title, raw, resolution_time "
            "FROM markets "
            "WHERE status = 'resolved' "
            "AND lower(title) LIKE '%highest temperature%'"
        )

        selected: dict[tuple[str, dt.date], tuple[pd.Timestamp, dict[str, str]]] = {}
        with psycopg.connect(self.master_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()

        for title, raw, resolution_time in rows:
            raw_dict = self._coerce_raw_json(raw)
            if not raw_dict:
                continue

            source_key = self._normalize_source_url(raw_dict.get("resolutionSource"))
            station = source_to_station.get(source_key)
            if station is None:
                continue

            if not self._is_yes_resolved(raw_dict):
                continue

            target_day = self._extract_market_day(title=title, raw=raw_dict, resolution_time=resolution_time)
            if target_day is None:
                continue

            summary = self._format_resolved_market_summary(title)
            if summary is None:
                continue

            resolution_ts = pd.to_datetime(resolution_time, utc=True, errors="coerce")
            if pd.isna(resolution_ts):
                resolution_ts = pd.Timestamp("1970-01-01T00:00:00Z")
                market_view = {"left": summary, "right": ""}
            else:
                station_zone = self._get_zone(self.station_meta[station].timezone)
                resolution_local = resolution_ts.tz_convert(station_zone).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                market_view = {
                    "left": summary,
                    "right": f"RESOLVED at {resolution_local}",
                }

            key = (station, target_day)
            existing = selected.get(key)
            if existing is None or resolution_ts > existing[0]:
                selected[key] = (resolution_ts, market_view)

        return {k: v[1] for k, v in selected.items()}

    @staticmethod
    def _coerce_raw_json(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _normalize_source_url(value: Any) -> str:
        if value is None:
            return ""
        raw = str(value).strip()
        if not raw:
            return ""
        parsed = urlsplit(raw)
        host = parsed.netloc.strip().lower()
        path = unquote(parsed.path.strip()).rstrip("/").lower()
        return f"{host}{path}"

    @staticmethod
    def _normalize_json_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return parsed
        return []

    @classmethod
    def _is_yes_resolved(cls, raw: dict[str, Any]) -> bool:
        resolved_outcome = raw.get("resolvedOutcome") or raw.get("winning_outcome") or raw.get("winner")
        if isinstance(resolved_outcome, str) and resolved_outcome.strip().lower() == "yes":
            return True

        outcomes = cls._normalize_json_list(raw.get("outcomes"))
        prices = cls._normalize_json_list(raw.get("outcomePrices"))
        if not prices:
            return False

        yes_index = None
        for idx, label in enumerate(outcomes):
            if str(label).strip().lower() == "yes":
                yes_index = idx
                break
        if yes_index is None:
            yes_index = 0
        if yes_index >= len(prices):
            return False

        try:
            yes_price = float(prices[yes_index])
        except (TypeError, ValueError):
            return False
        return yes_price >= 0.999

    @classmethod
    def _extract_market_day(
        cls,
        *,
        title: str,
        raw: dict[str, Any],
        resolution_time: Any,
    ) -> dt.date | None:
        date_match = TITLE_DATE_PATTERN.search(str(title))
        if date_match is None:
            return None

        month_str = date_match.group(1)
        day_int = int(date_match.group(2))

        month_num = None
        for fmt in ("%B", "%b"):
            try:
                month_num = dt.datetime.strptime(month_str.title(), fmt).month
                break
            except ValueError:
                continue
        if month_num is None:
            return None

        year = cls._infer_market_year(raw=raw, resolution_time=resolution_time)
        if year is None:
            return None
        try:
            return dt.date(year, month_num, day_int)
        except ValueError:
            return None

    @staticmethod
    def _infer_market_year(*, raw: dict[str, Any], resolution_time: Any) -> int | None:
        for key in ("endDateIso", "endDate", "closedTime", "resolutionTime", "updatedAt"):
            value = raw.get(key)
            if value is None:
                continue
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            return int(ts.year)

        ts = pd.to_datetime(resolution_time, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return int(ts.year)

    @staticmethod
    def _format_resolved_market_summary(title: str) -> str | None:
        text = str(title)
        between_match = TITLE_BETWEEN_PATTERN.search(text)
        if between_match is not None:
            low = between_match.group(1)
            high = between_match.group(2)
            unit = between_match.group(3).upper()
            return f"Max. Temp. between {low} - {high} \N{DEGREE SIGN}{unit}"

        exact_match = TITLE_EXACT_PATTERN.search(text)
        if exact_match is not None:
            temp = exact_match.group(1)
            unit = exact_match.group(2).upper()
            return f"Max. Temp. {temp} \N{DEGREE SIGN}{unit}"

        higher_match = TITLE_HIGHER_PATTERN.search(text)
        if higher_match is not None:
            temp = higher_match.group(1)
            unit = higher_match.group(2).upper()
            return f"Max. Temp. {temp} \N{DEGREE SIGN}{unit} or higher"

        below_match = TITLE_BELOW_PATTERN.search(text)
        if below_match is not None:
            temp = below_match.group(1)
            unit = below_match.group(2).upper()
            return f"Max. Temp. {temp} \N{DEGREE SIGN}{unit} or below"

        return None
