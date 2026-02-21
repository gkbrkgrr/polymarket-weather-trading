from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "station_name": ("station_name", "station", "Station", "station_id", "stationId"),
    "city": ("city", "City", "city_name", "CityName"),
    "issue_time_utc": (
        "issue_time_utc",
        "InitTimeUTC",
        "init_time",
        "issue_time",
        "cycle",
    ),
    "target_date": (
        "valid_date_local",
        "target_date_local",
        "target_date",
        "valid_date",
    ),
    "valid_time_local": ("valid_time_local", "valid_local_time", "valid_time"),
    "valid_time_utc": (
        "valid_time_utc",
        "target_time_utc",
        "tmax_time_utc",
        "valid_datetime_utc",
    ),
    "lead_hours": (
        "lead_hours",
        "LeadHour",
        "lead_time_hours",
        "lead",
        "lead_hour",
        "lead_time",
    ),
    "cycle": ("cycle", "Cycle"),
    "local_timezone": ("local_timezone", "timezone", "tz", "time_zone"),
    "tmax_obs": (
        "tmax_obs",
        "tmax_obs_c",
        "obs",
        "observation",
        "tmax_actual",
        "tmax_observed",
        "actual",
        "y_true",
    ),
}

PREDICTION_CANDIDATES = (
    "tmax_pred",
    "Forecast",
    "forecast",
    "prediction",
    "pred",
    "yhat",
    "y_pred",
)

LAT_ALIASES = ("station_lat", "lat", "latitude")
LON_ALIASES = ("station_lon", "lon", "longitude")


@dataclass(slots=True)
class NormalizationResult:
    normalized: pd.DataFrame
    dropped_rows: int
    warnings: list[str]


def _build_col_lookup(columns: pd.Index) -> dict[str, str]:
    return {str(c).lower(): str(c) for c in columns}


def _first_present(
    lookup: dict[str, str],
    candidates: tuple[str, ...],
) -> str | None:
    for cand in candidates:
        found = lookup.get(cand.lower())
        if found is not None:
            return found
    return None


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _normalize_cycle(cycle_raw: pd.Series, issue_time_utc: pd.Series) -> pd.Series:
    out = pd.to_numeric(cycle_raw, errors="coerce")
    missing = out.isna()
    if missing.any():
        out.loc[missing] = issue_time_utc.loc[missing].dt.hour
    return out.astype(int)


def _detect_prediction_column(df: pd.DataFrame) -> str | None:
    lookup = _build_col_lookup(df.columns)
    for cand in PREDICTION_CANDIDATES:
        col = lookup.get(cand.lower())
        if col is None:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            return col

    best_col: str | None = None
    best_score = -1e9
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        non_null = vals.notna().sum()
        if non_null == 0:
            continue
        name = str(col).lower()
        score = float(non_null)
        if any(token in name for token in ("pred", "forecast", "yhat")):
            score += 10000
        if "tmax" in name:
            score += 2000
        if any(
            token in name
            for token in (
                "obs",
                "actual",
                "raw",
                "lead",
                "time",
                "date",
                "lat",
                "lon",
                "elev",
                "day_of_year",
                "month",
                "ws",
                "rh",
                "td",
                "tp",
                "ssrd",
            )
        ):
            score -= 5000
        if score > best_score:
            best_score = score
            best_col = str(col)

    if best_col is None or best_score < 1:
        return None
    return best_col


def _derive_station_name(
    df: pd.DataFrame,
    lookup: dict[str, str],
    city: pd.Series,
) -> tuple[pd.Series, list[str]]:
    warnings: list[str] = []
    station_col = _first_present(lookup, ALIAS_MAP["station_name"])
    if station_col is not None:
        station = df[station_col].astype("string").str.strip()
        return station.fillna(city.astype("string")), warnings

    lat_col = _first_present(lookup, LAT_ALIASES)
    lon_col = _first_present(lookup, LON_ALIASES)
    if lat_col is not None and lon_col is not None:
        lat = pd.to_numeric(df[lat_col], errors="coerce").round(4)
        lon = pd.to_numeric(df[lon_col], errors="coerce").round(4)
        station = (
            city.astype("string")
            + "__"
            + lat.astype("string")
            + "_"
            + lon.astype("string")
        )
        warnings.append(
            "station_name missing: derived from city + station_lat/station_lon"
        )
        return station, warnings

    warnings.append("station_name missing: falling back to city")
    return city.astype("string"), warnings


def _derive_target_date(
    *,
    df: pd.DataFrame,
    lookup: dict[str, str],
    issue_time_utc: pd.Series,
    lead_hours: pd.Series,
) -> tuple[pd.Series, list[str]]:
    warnings: list[str] = []

    target_col = _first_present(lookup, ALIAS_MAP["target_date"])
    if target_col is not None:
        target = pd.to_datetime(df[target_col], errors="coerce").dt.normalize()
        return target, warnings

    valid_local_col = _first_present(lookup, ALIAS_MAP["valid_time_local"])
    if valid_local_col is not None:
        target = pd.to_datetime(df[valid_local_col], errors="coerce").dt.normalize()
        if target.isna().any():
            s = df[valid_local_col].astype("string")
            target = target.fillna(pd.to_datetime(s.str.slice(0, 10), errors="coerce"))
            target = pd.to_datetime(target, errors="coerce").dt.normalize()
        warnings.append("target_date derived from valid_time_local")
        return target, warnings

    valid_utc_col = _first_present(lookup, ALIAS_MAP["valid_time_utc"])
    if valid_utc_col is not None:
        valid_utc = pd.to_datetime(df[valid_utc_col], utc=True, errors="coerce")
        tz_col = _first_present(lookup, ALIAS_MAP["local_timezone"])
        if tz_col is None:
            warnings.append(
                "No local timezone column; target_date derived from valid_time_utc in UTC"
            )
            return valid_utc.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize(), warnings

        target = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        tz_values = df[tz_col].astype("string")
        for tz_name in sorted({x for x in tz_values.dropna().unique() if str(x).strip()}):
            mask = tz_values == tz_name
            try:
                target.loc[mask] = (
                    valid_utc.loc[mask].dt.tz_convert(str(tz_name)).dt.tz_localize(None).dt.normalize()
                )
            except Exception:
                warnings.append(
                    f"Invalid timezone {tz_name!r}; used UTC target_date for those rows"
                )
                target.loc[mask] = (
                    valid_utc.loc[mask].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
                )
        missing_tz = tz_values.isna() | (tz_values.str.strip() == "")
        if missing_tz.any():
            warnings.append(
                "Some rows have no timezone; used UTC target_date for those rows"
            )
            target.loc[missing_tz] = (
                valid_utc.loc[missing_tz].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
            )
        return target, warnings

    warnings.append("target_date missing: derived from issue_time_utc + lead_hours in UTC")
    target = (issue_time_utc + pd.to_timedelta(lead_hours, unit="h")).dt.tz_convert("UTC")
    target = target.dt.tz_localize(None).dt.normalize()
    return target, warnings


def normalize_prediction_frame(
    *,
    df: pd.DataFrame,
    model_name: str,
    city_hint: str,
    source_path: Path,
    issue_time_hint: pd.Timestamp,
) -> NormalizationResult:
    warnings: list[str] = []
    lookup = _build_col_lookup(df.columns)

    city_col = _first_present(lookup, ALIAS_MAP["city"])
    if city_col is None:
        city = pd.Series(city_hint, index=df.index, dtype="string")
        warnings.append("city column missing: filled from folder name")
    else:
        city = df[city_col].astype("string").fillna(city_hint)

    issue_col = _first_present(lookup, ALIAS_MAP["issue_time_utc"])
    if issue_col is None:
        issue_time_utc = pd.Series(issue_time_hint, index=df.index)
        warnings.append("issue_time_utc missing: filled from filename cycle token")
    else:
        issue_time_utc = _to_datetime_utc(df[issue_col])
        if issue_time_utc.isna().all():
            issue_time_utc = pd.Series(issue_time_hint, index=df.index)
            warnings.append(
                "issue_time_utc present but unparsable: filled from filename cycle token"
            )
        else:
            issue_time_utc = issue_time_utc.fillna(issue_time_hint)

    lead_col = _first_present(lookup, ALIAS_MAP["lead_hours"])
    if lead_col is None:
        raise ValueError(
            f"Missing lead-hours column in {source_path}. Expected aliases: {ALIAS_MAP['lead_hours']}"
        )
    lead_hours = pd.to_numeric(df[lead_col], errors="coerce").astype(float)

    cycle_col = _first_present(lookup, ALIAS_MAP["cycle"])
    if cycle_col is None:
        cycle = issue_time_utc.dt.hour.astype(int)
    else:
        cycle = _normalize_cycle(df[cycle_col], issue_time_utc)

    target_date, target_warnings = _derive_target_date(
        df=df,
        lookup=lookup,
        issue_time_utc=issue_time_utc,
        lead_hours=lead_hours,
    )
    warnings.extend(target_warnings)

    pred_col = _detect_prediction_column(df)
    if pred_col is None:
        raise ValueError(
            f"Could not detect prediction column in {source_path}. Add an alias for tmax_pred/prediction/yhat/forecast."
        )
    tmax_pred = pd.to_numeric(df[pred_col], errors="coerce").astype(float)

    obs_col = _first_present(lookup, ALIAS_MAP["tmax_obs"])
    if obs_col is None:
        tmax_obs = pd.Series(np.nan, index=df.index, dtype=float)
    else:
        tmax_obs = pd.to_numeric(df[obs_col], errors="coerce").astype(float)

    station_name, station_warnings = _derive_station_name(df, lookup, city)
    warnings.extend(station_warnings)

    out = pd.DataFrame(
        {
            "model_name": model_name,
            "city": city.astype("string"),
            "station_name": station_name.astype("string"),
            "issue_time_utc": issue_time_utc,
            "cycle": cycle.astype(int),
            "lead_hours": lead_hours.astype(float),
            "target_date": target_date,
            "tmax_pred": tmax_pred,
            "tmax_obs": tmax_obs,
            "source_file": str(source_path),
            "_row_idx": df.index.astype(int),
        }
    )

    out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce").dt.normalize()
    required = [
        "city",
        "station_name",
        "issue_time_utc",
        "lead_hours",
        "target_date",
        "tmax_pred",
    ]

    rows_before = len(out)
    out = out.dropna(subset=required).copy()
    out["city"] = out["city"].astype("string").str.strip()
    out["station_name"] = out["station_name"].astype("string").str.strip()
    out = out[(out["city"] != "") & (out["station_name"] != "")].copy()

    dropped = rows_before - len(out)
    if dropped:
        warnings.append(
            f"Dropped {dropped} rows due to missing required normalized fields"
        )

    if out.empty:
        raise ValueError(
            f"No valid rows remain in {source_path} after schema normalization; please verify core columns."
        )

    return NormalizationResult(normalized=out, dropped_rows=dropped, warnings=warnings)
