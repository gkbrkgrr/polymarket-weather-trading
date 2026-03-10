from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass
class SnapshotTableInfo:
    table_name: str


def connect_db(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn)


def fetch_dataframe(conn: psycopg.Connection, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)
        rows = cur.fetchall()
        if not cur.description:
            return pd.DataFrame()
        columns = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=columns)


def ensure_live_pilot_tables(conn: psycopg.Connection) -> None:
    ddl_actions = """
        CREATE TABLE IF NOT EXISTS live_pilot_actions (
            action_id BIGSERIAL PRIMARY KEY,
            ts_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
            run_id TEXT NOT NULL,
            station TEXT NULL,
            market_id TEXT NULL,
            decision TEXT NOT NULL,
            payload JSONB NOT NULL
        );
    """
    ddl_reports = """
        CREATE TABLE IF NOT EXISTS live_pilot_reports (
            report_id BIGSERIAL PRIMARY KEY,
            ts_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
            report_day_local DATE NOT NULL,
            payload JSONB NOT NULL
        );
    """
    ddl_idx = """
        CREATE INDEX IF NOT EXISTS live_pilot_actions_ts_idx ON live_pilot_actions(ts_utc DESC);
        CREATE INDEX IF NOT EXISTS live_pilot_reports_ts_idx ON live_pilot_reports(ts_utc DESC);
    """
    with conn.cursor() as cur:
        cur.execute(ddl_actions)
        cur.execute(ddl_reports)
        cur.execute(ddl_idx)
    conn.commit()


def detect_snapshot_table(conn: psycopg.Connection) -> SnapshotTableInfo:
    query = """
        SELECT
            CASE
                WHEN to_regclass('public.snapshots') IS NOT NULL THEN 'snapshots'
                WHEN to_regclass('public.book_snapshots') IS NOT NULL THEN 'book_snapshots'
                ELSE NULL
            END AS table_name
    """
    out = fetch_dataframe(conn, query)
    if out.empty or out.iloc[0]["table_name"] is None:
        raise RuntimeError("Could not find snapshots table (expected snapshots or book_snapshots).")
    return SnapshotTableInfo(table_name=str(out.iloc[0]["table_name"]))


def _parse_json_maybe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _extract_asset_id_from_raw(raw: Any, preferred_outcome_index: int = 1) -> str | None:
    obj = _parse_json_maybe(raw)
    if not isinstance(obj, dict):
        return None

    clob_ids = _parse_json_maybe(obj.get("clobTokenIds"))
    if isinstance(clob_ids, list) and len(clob_ids) > preferred_outcome_index:
        val = clob_ids[preferred_outcome_index]
        if val is not None and str(val).strip():
            return str(val)

    token_lists = []
    for key in ("tokens", "outcomes", "outcomeTokens", "outcome_tokens"):
        cand = obj.get(key)
        if isinstance(cand, list):
            token_lists.append(cand)

    for tokens in token_lists:
        for tok in tokens:
            if not isinstance(tok, dict):
                continue
            idx = tok.get("outcomeIndex")
            if idx is None:
                idx = tok.get("outcome_index")
            if idx is None:
                idx = tok.get("index")
            label = str(tok.get("outcome") or tok.get("label") or tok.get("outcome_label") or "").strip().lower()
            if idx is not None:
                try:
                    idx_ok = int(idx) == int(preferred_outcome_index)
                except Exception:
                    idx_ok = False
            else:
                idx_ok = label == "no"
            if not idx_ok:
                continue
            for k in ("asset_id", "token_id", "tokenId", "id", "outcome_id", "outcomeId"):
                value = tok.get(k)
                if value is not None and str(value).strip():
                    return str(value)

    return None


def fetch_market_metadata(
    conn: psycopg.Connection,
    *,
    slugs: list[str] | None = None,
    market_ids: list[str] | None = None,
) -> pd.DataFrame:
    slugs = [str(s) for s in (slugs or []) if str(s).strip()]
    market_ids = [str(m) for m in (market_ids or []) if str(m).strip()]
    if not slugs and not market_ids:
        return pd.DataFrame(columns=["market_id", "slug", "asset_id", "status", "resolution_time"])

    query = """
        SELECT
            m.market_id::text AS market_id,
            m.slug,
            m.status,
            m.resolution_time,
            m.raw,
            o.outcome_id::text AS no_outcome_id
        FROM markets m
        LEFT JOIN outcomes o
          ON o.market_id = m.market_id
         AND o.outcome_index = 1
        WHERE (
            (%(slugs)s::text[] IS NOT NULL AND m.slug = ANY(%(slugs)s))
            OR
            (%(market_ids)s::text[] IS NOT NULL AND m.market_id = ANY(%(market_ids)s))
        )
    """
    out = fetch_dataframe(
        conn,
        query,
        params={
            "slugs": slugs or None,
            "market_ids": market_ids or None,
        },
    )
    if out.empty:
        return out

    out["asset_id"] = out["no_outcome_id"].astype("string")
    missing = out["asset_id"].isna() | (out["asset_id"].astype("string").str.strip() == "")
    if missing.any():
        out.loc[missing, "asset_id"] = out.loc[missing, "raw"].map(_extract_asset_id_from_raw)

    out["market_id"] = out["market_id"].astype("string")
    out["slug"] = out["slug"].astype("string")
    out["asset_id"] = out["asset_id"].astype("string")
    out = out.drop(columns=["raw", "no_outcome_id"], errors="ignore")
    out = out.drop_duplicates(subset=["market_id"], keep="last")
    return out


def fetch_open_weather_markets(
    conn: psycopg.Connection,
    *,
    statuses: list[str] | None = None,
) -> pd.DataFrame:
    status_list = [str(s).strip() for s in (statuses or ["active"]) if str(s).strip()]
    if not status_list:
        status_list = ["active"]

    query = """
        SELECT
            m.market_id::text AS market_id,
            m.slug,
            m.status,
            m.resolution_time,
            m.raw ->> 'endDate' AS end_date_utc,
            m.raw,
            o.outcome_id::text AS no_outcome_id
        FROM markets m
        LEFT JOIN outcomes o
          ON o.market_id = m.market_id
         AND o.outcome_index = 1
        WHERE m.status = ANY(%(statuses)s::text[])
          AND m.slug ILIKE 'highest-temperature-in-%%'
    """
    out = fetch_dataframe(conn, query, params={"statuses": status_list})
    if out.empty:
        return out

    out["asset_id"] = out["no_outcome_id"].astype("string")
    missing = out["asset_id"].isna() | (out["asset_id"].astype("string").str.strip() == "")
    if missing.any():
        out.loc[missing, "asset_id"] = out.loc[missing, "raw"].map(_extract_asset_id_from_raw)

    out["market_id"] = out["market_id"].astype("string")
    out["slug"] = out["slug"].astype("string")
    out["status"] = out["status"].astype("string")
    out["asset_id"] = out["asset_id"].astype("string")
    out["resolution_time"] = pd.to_datetime(out["resolution_time"], utc=True, errors="coerce")
    out["end_date_utc"] = pd.to_datetime(out["end_date_utc"], utc=True, errors="coerce")
    out = out.drop(columns=["raw", "no_outcome_id"], errors="ignore")
    out = out.drop_duplicates(subset=["market_id"], keep="last")
    return out


def fetch_latest_snapshots(
    conn: psycopg.Connection,
    *,
    snapshot_table: SnapshotTableInfo,
    market_ids: list[str],
    lookback_per_outcome: int = 12,
) -> pd.DataFrame:
    market_ids = [str(m) for m in market_ids if str(m).strip()]
    if not market_ids:
        return pd.DataFrame(
            columns=[
                "market_id",
                "yes_snapshot_ts_utc",
                "no_snapshot_ts_utc",
                "best_yes_bid",
                "best_yes_ask",
                "best_no_bid",
                "best_no_ask",
                "yes_bid_size",
                "yes_ask_size",
                "no_bid_size",
                "no_ask_size",
            ]
        )

    try:
        lookback = max(1, int(lookback_per_outcome))
    except Exception:
        lookback = 12

    query = sql.SQL(
        """
        WITH ranked AS (
            SELECT
                s.market_id::text AS market_id,
                s.ts,
                s.outcome_index,
                s.best_bid::double precision AS best_bid,
                s.best_ask::double precision AS best_ask,
                s.bid_size::double precision AS bid_size,
                s.ask_size::double precision AS ask_size,
                ROW_NUMBER() OVER (
                    PARTITION BY s.market_id, s.outcome_index
                    ORDER BY s.ts DESC
                ) AS rn
            FROM {table_name} s
            WHERE s.market_id = ANY(%(market_ids)s)
              AND s.outcome_index IN (0, 1)
        )
        SELECT
            market_id,
            ts,
            outcome_index,
            best_bid,
            best_ask,
            bid_size,
            ask_size
        FROM ranked
        WHERE rn <= %(lookback)s
        ORDER BY market_id, outcome_index, ts DESC
        """
    ).format(table_name=sql.Identifier(snapshot_table.table_name))

    with conn.cursor() as cur:
        cur.execute(query, {"market_ids": market_ids, "lookback": lookback})
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    raw = pd.DataFrame(rows, columns=cols)
    if raw.empty:
        return raw

    raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="coerce")
    raw["outcome_index"] = pd.to_numeric(raw["outcome_index"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["market_id", "outcome_index", "ts"]).copy()
    raw["outcome_index"] = raw["outcome_index"].astype(int)

    records: list[dict[str, Any]] = []
    for market_id, grp in raw.groupby("market_id", sort=False):
        row: dict[str, Any] = {
            "market_id": str(market_id),
            "yes_snapshot_ts_utc": pd.NaT,
            "no_snapshot_ts_utc": pd.NaT,
            "best_yes_bid": None,
            "best_yes_ask": None,
            "best_no_bid": None,
            "best_no_ask": None,
            "yes_bid_size": None,
            "yes_ask_size": None,
            "no_bid_size": None,
            "no_ask_size": None,
        }

        yes_grp = grp.loc[grp["outcome_index"] == 0].sort_values("ts", ascending=False, kind="mergesort")
        if not yes_grp.empty:
            yes_pick = yes_grp.loc[yes_grp["best_bid"].notna()].head(1)
            if yes_pick.empty:
                yes_pick = yes_grp.loc[yes_grp["best_ask"].notna()].head(1)
            if yes_pick.empty:
                yes_pick = yes_grp.head(1)
            q = yes_pick.iloc[0]
            row["yes_snapshot_ts_utc"] = q["ts"]
            row["best_yes_bid"] = q["best_bid"]
            row["best_yes_ask"] = q["best_ask"]
            row["yes_bid_size"] = q["bid_size"]
            row["yes_ask_size"] = q["ask_size"]

        no_grp = grp.loc[grp["outcome_index"] == 1].sort_values("ts", ascending=False, kind="mergesort")
        if not no_grp.empty:
            no_pick = no_grp.loc[no_grp["best_ask"].notna()].head(1)
            if no_pick.empty:
                no_pick = no_grp.loc[no_grp["best_bid"].notna()].head(1)
            if no_pick.empty:
                no_pick = no_grp.head(1)
            q = no_pick.iloc[0]
            row["no_snapshot_ts_utc"] = q["ts"]
            row["best_no_bid"] = q["best_bid"]
            row["best_no_ask"] = q["best_ask"]
            row["no_bid_size"] = q["bid_size"]
            row["no_ask_size"] = q["ask_size"]

        records.append(row)

    out = pd.DataFrame.from_records(records)
    out["yes_snapshot_ts_utc"] = pd.to_datetime(out["yes_snapshot_ts_utc"], utc=True, errors="coerce")
    out["no_snapshot_ts_utc"] = pd.to_datetime(out["no_snapshot_ts_utc"], utc=True, errors="coerce")
    return out


def fetch_recent_trades(
    conn: psycopg.Connection,
    *,
    market_ids: list[str],
    lookback_hours: float = 24.0,
) -> pd.DataFrame:
    market_ids = [str(m) for m in market_ids if str(m).strip()]
    if not market_ids:
        return pd.DataFrame(columns=["market_id", "ts", "outcome_index", "side", "price", "size"])

    cutoff = datetime.utcnow() - timedelta(hours=float(lookback_hours))
    query = """
        SELECT
            market_id::text AS market_id,
            ts,
            outcome_index,
            side,
            price::double precision AS price,
            size::double precision AS size
        FROM trades
        WHERE market_id = ANY(%(market_ids)s)
          AND ts >= %(cutoff)s
        ORDER BY ts DESC
    """
    out = fetch_dataframe(conn, query, params={"market_ids": market_ids, "cutoff": cutoff})
    if out.empty:
        return out
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    return out


def _parse_yes_prob_from_outcome_prices(value: Any) -> float | None:
    parsed = _parse_json_maybe(value)
    if not isinstance(parsed, list) or not parsed:
        return None
    try:
        return float(parsed[0])
    except (TypeError, ValueError):
        return None


def fetch_resolved_outcomes(conn: psycopg.Connection, *, market_ids: list[str]) -> pd.DataFrame:
    market_ids = [str(m) for m in market_ids if str(m).strip()]
    if not market_ids:
        return pd.DataFrame(columns=["market_id", "resolved_at_utc", "yes_prob_resolved", "no_wins"])

    query = """
        SELECT
            market_id::text AS market_id,
            status,
            resolution_time,
            raw ->> 'outcomePrices' AS outcome_prices
        FROM markets
        WHERE market_id = ANY(%(market_ids)s)
          AND (status = 'resolved' OR resolution_time IS NOT NULL)
    """
    out = fetch_dataframe(conn, query, params={"market_ids": market_ids})
    if out.empty:
        return out

    out["resolved_at_utc"] = pd.to_datetime(out["resolution_time"], utc=True, errors="coerce")
    out["yes_prob_resolved"] = out["outcome_prices"].map(_parse_yes_prob_from_outcome_prices)
    out["no_wins"] = out["yes_prob_resolved"].map(lambda x: None if x is None else bool(float(x) < 0.5))
    out = out[["market_id", "resolved_at_utc", "yes_prob_resolved", "no_wins"]]
    out = out.drop_duplicates(subset=["market_id"], keep="last")
    return out


def fetch_snapshots_freshness(conn: psycopg.Connection, *, snapshot_table: SnapshotTableInfo) -> datetime | None:
    query = sql.SQL("SELECT MAX(ts) AS max_ts FROM {table_name}").format(
        table_name=sql.Identifier(snapshot_table.table_name)
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        row = cur.fetchone()
    if not row or row.get("max_ts") is None:
        return None
    ts = pd.to_datetime(row["max_ts"], utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def insert_live_action(
    conn: psycopg.Connection,
    *,
    run_id: str,
    station: str | None,
    market_id: str | None,
    decision: str,
    payload: dict[str, Any],
) -> None:
    query = """
        INSERT INTO live_pilot_actions (run_id, station, market_id, decision, payload)
        VALUES (%(run_id)s, %(station)s, %(market_id)s, %(decision)s, %(payload)s)
    """
    with conn.cursor() as cur:
        cur.execute(
            query,
            {
                "run_id": run_id,
                "station": station,
                "market_id": market_id,
                "decision": decision,
                "payload": Jsonb(payload),
            },
        )
    conn.commit()


def insert_daily_report(conn: psycopg.Connection, *, report_day_local: str, payload: dict[str, Any]) -> None:
    query = """
        INSERT INTO live_pilot_reports (report_day_local, payload)
        VALUES (%(report_day_local)s::date, %(payload)s)
    """
    with conn.cursor() as cur:
        cur.execute(
            query,
            {
                "report_day_local": report_day_local,
                "payload": Jsonb(payload),
            },
        )
    conn.commit()
