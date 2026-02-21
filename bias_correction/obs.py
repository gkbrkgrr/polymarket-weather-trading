from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OBS_CITY_ALIASES = ("city", "city_name")
OBS_STATION_ALIASES = ("station_name", "station", "station_id")
OBS_TARGET_DATE_ALIASES = (
    "target_date_local",
    "target_date",
    "valid_date_local",
    "date",
)
OBS_LOCAL_TIME_ALIASES = (
    "observed_at_local",
    "valid_time_local",
    "local_time",
)
OBS_UTC_TIME_ALIASES = (
    "observed_at_utc",
    "valid_time_utc",
    "timestamp_utc",
    "datetime_utc",
)
OBS_TEMP_C_ALIASES = ("tmax_obs", "tmax_obs_c", "temperature_c", "temp_c")
OBS_TEMP_F_ALIASES = ("temperature_f", "temp_f")


def _lookup(columns: pd.Index) -> dict[str, str]:
    return {str(c).lower(): str(c) for c in columns}


def _first(lookup: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        col = lookup.get(alias.lower())
        if col is not None:
            return col
    return None


def _derive_obs_target_date(df: pd.DataFrame, lookup: dict[str, str]) -> pd.Series:
    date_col = _first(lookup, OBS_TARGET_DATE_ALIASES)
    if date_col is not None:
        return pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    local_col = _first(lookup, OBS_LOCAL_TIME_ALIASES)
    if local_col is not None:
        text = df[local_col].astype("string")
        return pd.to_datetime(text.str.slice(0, 10), errors="coerce").dt.normalize()

    utc_col = _first(lookup, OBS_UTC_TIME_ALIASES)
    if utc_col is not None:
        return (
            pd.to_datetime(df[utc_col], utc=True, errors="coerce")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .dt.normalize()
        )

    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def _derive_obs_temperature(df: pd.DataFrame, lookup: dict[str, str]) -> pd.Series:
    temp_c_col = _first(lookup, OBS_TEMP_C_ALIASES)
    if temp_c_col is not None:
        return pd.to_numeric(df[temp_c_col], errors="coerce").astype(float)

    temp_f_col = _first(lookup, OBS_TEMP_F_ALIASES)
    if temp_f_col is not None:
        f = pd.to_numeric(df[temp_f_col], errors="coerce").astype(float)
        return (f - 32.0) * (5.0 / 9.0)

    return pd.Series(np.nan, index=df.index, dtype=float)


def _normalize_obs_frame(
    *,
    df: pd.DataFrame,
    city_hint: str,
    source_path: Path,
) -> pd.DataFrame:
    lookup = _lookup(df.columns)

    city_col = _first(lookup, OBS_CITY_ALIASES)
    if city_col is None:
        city = pd.Series(city_hint, index=df.index, dtype="string")
    else:
        city = df[city_col].astype("string").fillna(city_hint)

    station_col = _first(lookup, OBS_STATION_ALIASES)
    if station_col is None:
        station = city.astype("string")
    else:
        station = df[station_col].astype("string").fillna(city.astype("string"))

    target_date = _derive_obs_target_date(df, lookup)
    temperature_c = _derive_obs_temperature(df, lookup)

    out = pd.DataFrame(
        {
            "city": city.astype("string").str.strip(),
            "station_name": station.astype("string").str.strip(),
            "target_date": target_date,
            "tmax_obs_external": temperature_c,
            "source": str(source_path),
        }
    )
    out = out.dropna(subset=["city", "target_date", "tmax_obs_external"]).copy()
    if out.empty:
        return out

    out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["target_date"])
    if out.empty:
        return out

    grouped = (
        out.groupby(["city", "station_name", "target_date"], as_index=False)["tmax_obs_external"]
        .max()
        .sort_values(["city", "station_name", "target_date"], kind="mergesort")
    )
    return grouped


def load_observation_daily(
    *,
    obs_source_path: Path | None,
    logger,
) -> pd.DataFrame | None:
    if obs_source_path is None:
        return None
    if not obs_source_path.exists():
        raise FileNotFoundError(f"obs_source_path not found: {obs_source_path}")

    frames: list[pd.DataFrame] = []
    if obs_source_path.is_dir():
        files = sorted(obs_source_path.glob("*.parquet"))
        logger.info(
            "Loading external observations from directory: %s (%d files)",
            obs_source_path,
            len(files),
        )
        for path in files:
            df_obs = pd.read_parquet(path)
            city_hint = path.stem
            normalized = _normalize_obs_frame(df=df_obs, city_hint=city_hint, source_path=path)
            if not normalized.empty:
                frames.append(normalized)
    else:
        logger.info("Loading external observations from file: %s", obs_source_path)
        df_obs = pd.read_parquet(obs_source_path)
        normalized = _normalize_obs_frame(
            df=df_obs,
            city_hint="",
            source_path=obs_source_path,
        )
        if not normalized.empty:
            frames.append(normalized)

    if not frames:
        logger.warning("No external observations were parsed from %s", obs_source_path)
        return None

    all_obs = pd.concat(frames, ignore_index=True)
    all_obs = (
        all_obs.groupby(["city", "station_name", "target_date"], as_index=False)["tmax_obs_external"]
        .max()
        .sort_values(["city", "station_name", "target_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    logger.info("Loaded %d external daily observation rows", len(all_obs))
    return all_obs


def attach_external_observations(
    *,
    normalized_df: pd.DataFrame,
    obs_daily: pd.DataFrame | None,
    logger,
) -> pd.DataFrame:
    if obs_daily is None or obs_daily.empty or normalized_df.empty:
        return normalized_df

    out = normalized_df.copy()

    obs_has_city_date_dupes = obs_daily.duplicated(subset=["city", "target_date"]).any()
    if obs_has_city_date_dupes:
        join_keys = ["city", "station_name", "target_date"]
        obs_join = obs_daily[["city", "station_name", "target_date", "tmax_obs_external"]].copy()
    else:
        join_keys = ["city", "target_date"]
        obs_join = (
            obs_daily[["city", "target_date", "tmax_obs_external"]]
            .groupby(["city", "target_date"], as_index=False)["tmax_obs_external"]
            .max()
        )

    merged = out.merge(obs_join, on=join_keys, how="left")
    before_missing = merged["tmax_obs"].isna().sum()
    merged["tmax_obs"] = merged["tmax_obs"].fillna(merged["tmax_obs_external"])
    after_missing = merged["tmax_obs"].isna().sum()
    filled = before_missing - after_missing

    if filled > 0:
        logger.info(
            "Filled %d missing observation values from external source using keys=%s",
            filled,
            ",".join(join_keys),
        )

    return merged.drop(columns=["tmax_obs_external"], errors="ignore")
