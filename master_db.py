from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import yaml


MASTER_DB_NAME = "master_db"
DEFAULT_MASTER_DSN = "postgresql://archive_user:password@127.0.0.1:5432/master_db"

OBS_COLUMNS = (
    "station",
    "observed_at_local",
    "temperature_f",
    "temperature_c",
    "precipitation_hourly_in",
    "precipitation_total_in",
    "scraped_at_utc",
)

_MET_COLUMNS = (
    "temperature_f",
    "temperature_c",
    "precipitation_hourly_in",
    "precipitation_total_in",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_schema_path() -> Path:
    return _repo_root() / "migrations" / "master_db_schema.sql"


def _load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (_repo_root() / "config.yaml")
    if not cfg_path.exists():
        return {}
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _with_dbname(base_dsn: str, dbname: str) -> str:
    parts = conninfo_to_dict(base_dsn)
    if not parts:
        return base_dsn
    parts["dbname"] = dbname
    return make_conninfo(**parts)


def resolve_master_postgres_dsn(
    *,
    explicit_dsn: str | None = None,
    config_path: Path | None = None,
) -> str:
    if explicit_dsn:
        return explicit_dsn

    env_master = os.getenv("MASTER_POSTGRES_DSN")
    if env_master:
        return env_master

    env_base = os.getenv("POSTGRES_DSN")
    if env_base:
        return _with_dbname(env_base, MASTER_DB_NAME)

    cfg = _load_config(config_path)
    cfg_master = cfg.get("master_postgres_dsn")
    if isinstance(cfg_master, str) and cfg_master.strip():
        return cfg_master.strip()

    cfg_base = cfg.get("postgres_dsn")
    if isinstance(cfg_base, str) and cfg_base.strip():
        return _with_dbname(cfg_base.strip(), MASTER_DB_NAME)

    return DEFAULT_MASTER_DSN


def _resolve_admin_postgres_dsn(master_dsn: str) -> str:
    return _with_dbname(master_dsn, "postgres")


def _dbname_from_dsn(dsn: str) -> str:
    parts = conninfo_to_dict(dsn)
    dbname = parts.get("dbname")
    if dbname:
        return dbname
    return MASTER_DB_NAME


def ensure_master_db_exists(*, master_dsn: str | None = None, config_path: Path | None = None) -> str:
    resolved_master_dsn = resolve_master_postgres_dsn(explicit_dsn=master_dsn, config_path=config_path)
    admin_dsn = _resolve_admin_postgres_dsn(resolved_master_dsn)
    dbname = _dbname_from_dsn(resolved_master_dsn)

    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))

    return resolved_master_dsn


def ensure_master_schema(
    *,
    master_dsn: str | None = None,
    schema_sql_path: Path | None = None,
    config_path: Path | None = None,
) -> str:
    resolved_master_dsn = resolve_master_postgres_dsn(explicit_dsn=master_dsn, config_path=config_path)
    ddl_path = schema_sql_path or _default_schema_path()
    ddl = ddl_path.read_text(encoding="utf-8")

    with psycopg.connect(resolved_master_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

    return resolved_master_dsn


def ensure_master_db_ready(
    *,
    master_dsn: str | None = None,
    schema_sql_path: Path | None = None,
    config_path: Path | None = None,
) -> str:
    resolved_master_dsn = ensure_master_db_exists(master_dsn=master_dsn, config_path=config_path)
    ensure_master_schema(
        master_dsn=resolved_master_dsn,
        schema_sql_path=schema_sql_path,
        config_path=config_path,
    )
    return resolved_master_dsn


def _parse_local_wall_time(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = dt.datetime.fromisoformat(value)
    else:
        raise ValueError(f"Unsupported observed_at_local value: {value!r}")

    if parsed.tzinfo is not None:
        return parsed.replace(tzinfo=None)
    return parsed


def _parse_scraped_at_utc(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = dt.datetime.fromisoformat(value)
    else:
        raise ValueError(f"Unsupported scraped_at_utc value: {value!r}")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def normalize_station_observation_records(records: Sequence[dict[str, Any]]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for record in records:
        station = str(record.get("station") or "").strip()
        observed_raw = record.get("observed_at_local")
        scraped_raw = record.get("scraped_at_utc")
        if not station or observed_raw is None or scraped_raw is None:
            continue

        rows.append(
            (
                station,
                _parse_local_wall_time(observed_raw),
                _to_int_or_none(record.get("temperature_f")),
                _to_int_or_none(record.get("temperature_c")),
                _to_float_or_none(record.get("precipitation_hourly_in")),
                _to_float_or_none(record.get("precipitation_total_in")),
                _parse_scraped_at_utc(scraped_raw),
            )
        )
    return rows


def _fetch_df(dsn: str, query: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _coerce_local_bound(value: Any, *, end_exclusive: bool) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, dt.date):
        parsed = dt.datetime.combine(value, dt.time.min)
        if end_exclusive:
            parsed += dt.timedelta(days=1)
    elif isinstance(value, str):
        parsed = dt.datetime.fromisoformat(value)
        if len(value.strip()) == 10 and end_exclusive:
            parsed += dt.timedelta(days=1)
    else:
        raise ValueError(f"Unsupported date/datetime bound: {value!r}")

    if parsed.tzinfo is not None:
        return parsed.replace(tzinfo=None)
    return parsed


def get_station_obs(
    station: str,
    start: dt.date | dt.datetime | str | None,
    end: dt.date | dt.datetime | str | None,
    *,
    master_dsn: str | None = None,
) -> pd.DataFrame:
    dsn = resolve_master_postgres_dsn(explicit_dsn=master_dsn)
    predicates = ["station = %s"]
    params: list[Any] = [station]

    if start is not None:
        predicates.append("observed_at_local >= %s")
        params.append(_coerce_local_bound(start, end_exclusive=False))

    if end is not None:
        predicates.append("observed_at_local < %s")
        params.append(_coerce_local_bound(end, end_exclusive=True))

    query = (
        "SELECT station, observed_at_local, temperature_f, temperature_c, "
        "precipitation_hourly_in, precipitation_total_in, scraped_at_utc "
        "FROM station_observations "
        f"WHERE {' AND '.join(predicates)} "
        "ORDER BY observed_at_local"
    )
    return _fetch_df(dsn, query, params)


def get_obs_for_day(
    station: str,
    day: dt.date | dt.datetime | str,
    *,
    master_dsn: str | None = None,
) -> pd.DataFrame:
    if isinstance(day, dt.datetime):
        base_day = day.date()
    elif isinstance(day, dt.date):
        base_day = day
    elif isinstance(day, str):
        parsed = dt.datetime.fromisoformat(day)
        base_day = parsed.date()
    else:
        raise ValueError(f"Unsupported day value: {day!r}")

    return get_station_obs(
        station,
        start=base_day,
        end=base_day,
        master_dsn=master_dsn,
    )


def get_latest_observed_date_local(
    station: str,
    *,
    master_dsn: str | None = None,
) -> dt.date | None:
    dsn = resolve_master_postgres_dsn(explicit_dsn=master_dsn)
    query = "SELECT MAX(observed_at_local::date) AS max_day FROM station_observations WHERE station = %s"
    out = _fetch_df(dsn, query, (station,))
    if out.empty or out.iloc[0]["max_day"] is None:
        return None
    value = out.iloc[0]["max_day"]
    return value if isinstance(value, dt.date) else pd.Timestamp(value).date()


def get_daily_tmax_by_station(
    *,
    stations: Sequence[str] | None = None,
    master_dsn: str | None = None,
) -> pd.DataFrame:
    dsn = resolve_master_postgres_dsn(explicit_dsn=master_dsn)

    where = ""
    params: list[Any] = []
    if stations:
        where = "WHERE station = ANY(%s)"
        params.append(list(stations))

    query = (
        "SELECT "
        "station AS city_name, "
        "observed_at_local::date AS target_date_local, "
        "MAX(COALESCE(temperature_c::double precision, ((temperature_f - 32.0) * (5.0 / 9.0)))) AS tmax_obs_c "
        "FROM station_observations "
        f"{where} "
        "GROUP BY station, observed_at_local::date "
        "ORDER BY station, observed_at_local::date"
    )
    return _fetch_df(dsn, query, params)


def get_historical_daily_tmax_bounds(
    *,
    stations: Sequence[str],
    start_date: dt.date,
    master_dsn: str | None = None,
) -> pd.DataFrame:
    if not stations:
        return pd.DataFrame(
            columns=["city_name", "month_day", "hist_min_c", "hist_max_c", "hist_min_f", "hist_max_f"]
        )

    dsn = resolve_master_postgres_dsn(explicit_dsn=master_dsn)
    query = (
        "WITH daily AS ("
        "  SELECT "
        "    station AS city_name, "
        "    observed_at_local::date AS local_date, "
        "    MAX(COALESCE(temperature_c::double precision, ((temperature_f - 32.0) * (5.0 / 9.0)))) AS daily_tmax_c, "
        "    MAX(COALESCE(temperature_f::double precision, ((temperature_c * 9.0 / 5.0) + 32.0))) AS daily_tmax_f "
        "  FROM station_observations "
        "  WHERE station = ANY(%s) AND observed_at_local::date >= %s "
        "  GROUP BY station, observed_at_local::date"
        ") "
        "SELECT "
        "  city_name, "
        "  to_char(local_date, 'MM-DD') AS month_day, "
        "  MIN(daily_tmax_c) AS hist_min_c, "
        "  MAX(daily_tmax_c) AS hist_max_c, "
        "  MIN(daily_tmax_f) AS hist_min_f, "
        "  MAX(daily_tmax_f) AS hist_max_f "
        "FROM daily "
        "GROUP BY city_name, to_char(local_date, 'MM-DD') "
        "ORDER BY city_name, month_day"
    )
    return _fetch_df(dsn, query, [list(stations), start_date])


def upsert_station_observations(
    *,
    records: Sequence[dict[str, Any]],
    master_dsn: str | None = None,
    ensure_schema: bool = False,
) -> dict[str, int]:
    rows = normalize_station_observation_records(records)
    if not rows:
        return {"inserted": 0, "updated": 0, "unchanged": 0}

    dsn = resolve_master_postgres_dsn(explicit_dsn=master_dsn)
    if ensure_schema:
        ensure_master_db_ready(master_dsn=dsn)

    incoming_dedup_cte = (
        "WITH incoming AS ("
        "  SELECT DISTINCT ON (station, observed_at_local) "
        "    station, observed_at_local, temperature_f, temperature_c, "
        "    precipitation_hourly_in, precipitation_total_in, scraped_at_utc "
        "  FROM _incoming_station_observations "
        "  ORDER BY station, observed_at_local, scraped_at_utc DESC"
        ") "
    )

    diff_predicate = " OR ".join(f"e.{col} IS DISTINCT FROM i.{col}" for col in _MET_COLUMNS)
    update_predicate = " OR ".join(f"station_observations.{col} IS DISTINCT FROM EXCLUDED.{col}" for col in _MET_COLUMNS)

    stats_query = (
        incoming_dedup_cte
        + "SELECT "
        "  COUNT(*) FILTER (WHERE e.station IS NULL) AS inserted, "
        f"  COUNT(*) FILTER (WHERE e.station IS NOT NULL AND ({diff_predicate})) AS updated, "
        f"  COUNT(*) FILTER (WHERE e.station IS NOT NULL AND NOT ({diff_predicate})) AS unchanged "
        "FROM incoming i "
        "LEFT JOIN station_observations e "
        "  ON e.station = i.station AND e.observed_at_local = i.observed_at_local"
    )

    upsert_sql = (
        incoming_dedup_cte
        + "INSERT INTO station_observations ("
        "  station, observed_at_local, temperature_f, temperature_c, "
        "  precipitation_hourly_in, precipitation_total_in, scraped_at_utc"
        ") "
        "SELECT "
        "  station, observed_at_local, temperature_f, temperature_c, "
        "  precipitation_hourly_in, precipitation_total_in, scraped_at_utc "
        "FROM incoming "
        "ON CONFLICT (station, observed_at_local) DO UPDATE SET "
        "  temperature_f = EXCLUDED.temperature_f, "
        "  temperature_c = EXCLUDED.temperature_c, "
        "  precipitation_hourly_in = EXCLUDED.precipitation_hourly_in, "
        "  precipitation_total_in = EXCLUDED.precipitation_total_in, "
        "  scraped_at_utc = EXCLUDED.scraped_at_utc "
        f"WHERE {update_predicate}"
    )

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE _incoming_station_observations ("
                "station TEXT NOT NULL, "
                "observed_at_local TIMESTAMP WITHOUT TIME ZONE NOT NULL, "
                "temperature_f INTEGER NULL, "
                "temperature_c INTEGER NULL, "
                "precipitation_hourly_in DOUBLE PRECISION NULL, "
                "precipitation_total_in DOUBLE PRECISION NULL, "
                "scraped_at_utc TIMESTAMPTZ NOT NULL"
                ") ON COMMIT DROP"
            )

            with cur.copy(
                "COPY _incoming_station_observations "
                "(station, observed_at_local, temperature_f, temperature_c, "
                "precipitation_hourly_in, precipitation_total_in, scraped_at_utc) "
                "FROM STDIN"
            ) as copy:
                for row in rows:
                    copy.write_row(row)

            cur.execute(stats_query)
            inserted, updated, unchanged = cur.fetchone()
            cur.execute(upsert_sql)
        conn.commit()

    return {
        "inserted": int(inserted or 0),
        "updated": int(updated or 0),
        "unchanged": int(unchanged or 0),
    }
