from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from polymarket_archive.models import GammaMarket, GammaOutcome, Trade
from polymarket_archive.utils import CursorState


class Database:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5) -> None:
        self._dsn = dsn
        self._pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False)

    async def open(self) -> None:
        await self._pool.open()

    async def close(self) -> None:
        await self._pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncConnection:
        async with self._pool.connection() as conn:
            yield conn

    async def init_db(self, schema_path: Path) -> None:
        ddl = schema_path.read_text(encoding="utf-8")
        async with self.connection() as conn:
            await conn.execute(ddl)

    async def ensure_schema(self, schema_path: Path) -> None:
        await self.init_db(schema_path)

    async def upsert_markets(self, markets: Sequence[GammaMarket]) -> None:
        if not markets:
            return
        rows = [
            (
                market.market_id,
                market.slug,
                market.title,
                market.status,
                market.event_start_time,
                market.resolution_time,
                Jsonb(market.raw),
            )
            for market in markets
        ]
        sql = (
            "INSERT INTO markets (market_id, slug, title, status, event_start_time, resolution_time, raw) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (market_id) DO UPDATE SET "
            "slug = EXCLUDED.slug, "
            "title = EXCLUDED.title, "
            "status = EXCLUDED.status, "
            "event_start_time = EXCLUDED.event_start_time, "
            "resolution_time = EXCLUDED.resolution_time, "
            "raw = EXCLUDED.raw, "
            "updated_at = now()"
        )
        async with self.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(sql, rows)

    async def upsert_outcomes(self, outcomes: Sequence[GammaOutcome], market_id: str) -> None:
        if not outcomes:
            return
        rows = [
            (
                market_id,
                outcome.outcome_id,
                outcome.outcome_label,
                outcome.outcome_index,
                Jsonb(outcome.raw),
            )
            for outcome in outcomes
        ]
        sql = (
            "INSERT INTO outcomes (market_id, outcome_id, outcome_label, outcome_index, raw) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (market_id, outcome_index) DO UPDATE SET "
            "outcome_id = EXCLUDED.outcome_id, "
            "outcome_label = EXCLUDED.outcome_label, "
            "raw = EXCLUDED.raw"
        )
        async with self.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(sql, rows)

    async def insert_trades(self, trades: Sequence[Trade], conn: AsyncConnection | None = None) -> None:
        if not trades:
            return
        rows = [
            (
                trade.trade_id,
                trade.market_id,
                trade.ts,
                trade.outcome_id,
                trade.outcome_index,
                trade.side,
                trade.price,
                trade.size,
                trade.tx_hash,
                Jsonb(trade.raw),
            )
            for trade in trades
        ]
        sql = (
            "INSERT INTO trades (trade_id, market_id, ts, outcome_id, outcome_index, side, price, size, tx_hash, raw) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (trade_id) DO NOTHING"
        )
        if conn is None:
            async with self.connection() as local_conn:
                async with local_conn.cursor() as cur:
                    await cur.executemany(sql, rows)
        else:
            async with conn.cursor() as cur:
                await cur.executemany(sql, rows)

    async def get_cursor(self, market_id: str) -> CursorState | None:
        async with self.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT last_ts, last_tiebreak FROM cursors WHERE market_id = %s",
                    (market_id,),
                )
                row = await cur.fetchone()
        if not row:
            return None
        return CursorState(last_ts=row[0], last_tiebreak=row[1])

    async def upsert_cursor(
        self, market_id: str, cursor: CursorState, conn: AsyncConnection | None = None
    ) -> None:
        sql = (
            "INSERT INTO cursors (market_id, last_ts, last_tiebreak) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (market_id) DO UPDATE SET "
            "last_ts = EXCLUDED.last_ts, "
            "last_tiebreak = EXCLUDED.last_tiebreak, "
            "updated_at = now()"
        )
        args = (market_id, cursor.last_ts, cursor.last_tiebreak)
        if conn is None:
            async with self.connection() as local_conn:
                await local_conn.execute(sql, args)
        else:
            await conn.execute(sql, args)

    async def insert_book_snapshot(
        self,
        market_id: str,
        ts,
        outcome_index: int | None,
        best_bid,
        best_ask,
        bid_size,
        ask_size,
        raw,
        conn: AsyncConnection | None = None,
    ) -> None:
        sql = (
            "INSERT INTO book_snapshots (market_id, ts, outcome_index, best_bid, best_ask, bid_size, ask_size, raw) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (market_id, ts, outcome_index) DO NOTHING"
        )
        args = (market_id, ts, outcome_index, best_bid, best_ask, bid_size, ask_size, Jsonb(raw))
        if conn is None:
            async with self.connection() as local_conn:
                async with local_conn.cursor() as cur:
                    await cur.execute(sql, args)
        else:
            async with conn.cursor() as cur:
                await cur.execute(sql, args)

    async def list_market_ids(self) -> list[str]:
        async with self.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT market_id FROM markets")
                rows = await cur.fetchall()
        return [row[0] for row in rows]

    async def list_market_condition_ids(self) -> list[tuple[str, str | None]]:
        async with self.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT market_id, raw->>'conditionId' FROM markets"
                )
                rows = await cur.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def get_condition_id(self, market_id: str) -> str | None:
        async with self.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT raw->>'conditionId' FROM markets WHERE market_id = %s",
                    (market_id,),
                )
                row = await cur.fetchone()
        if not row:
            return None
        return row[0]

    async def list_markets(self) -> list[GammaMarket]:
        async with self.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT market_id, slug, title, status, event_start_time, resolution_time, raw FROM markets"
                )
                rows = await cur.fetchall()
        markets: list[GammaMarket] = []
        for row in rows:
            markets.append(
                GammaMarket(
                    market_id=row[0],
                    slug=row[1],
                    title=row[2],
                    status=row[3],
                    event_start_time=row[4],
                    resolution_time=row[5],
                    raw=row[6],
                    outcomes=[],
                )
            )
        return markets

    async def compact_resolved_book_snapshots(
        self,
        *,
        as_of: datetime | None = None,
        grace_minutes: int = 60,
        bucket_seconds_recent: int = 20,
        bucket_seconds_mid: int = 30,
        bucket_seconds_old: int = 60,
    ) -> dict[str, int]:
        if grace_minutes < 0:
            raise ValueError("grace_minutes must be >= 0")
        for value in (bucket_seconds_recent, bucket_seconds_mid, bucket_seconds_old):
            if int(value) < 1:
                raise ValueError("bucket seconds must be >= 1")
        now_utc = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)
        tiers = _resolved_compaction_tiers(
            now_utc=now_utc,
            grace_minutes=int(grace_minutes),
            bucket_seconds_recent=int(bucket_seconds_recent),
            bucket_seconds_mid=int(bucket_seconds_mid),
            bucket_seconds_old=int(bucket_seconds_old),
        )
        totals: dict[str, int] = {
            "tiers": 0,
            "buckets_compacted": 0,
            "rows_updated": 0,
            "rows_deleted": 0,
        }
        async with self.connection() as conn:
            await conn.execute(
                """
                CREATE TEMP TABLE IF NOT EXISTS _book_snapshot_compact_stage (
                    market_id TEXT NOT NULL,
                    outcome_index INT NULL,
                    bucket_ts TIMESTAMPTZ NOT NULL,
                    keep_ts TIMESTAMPTZ NOT NULL,
                    n INT NOT NULL,
                    first_ts TIMESTAMPTZ NOT NULL,
                    last_ts TIMESTAMPTZ NOT NULL,
                    bid_min NUMERIC(10,6) NULL,
                    bid_max NUMERIC(10,6) NULL,
                    ask_min NUMERIC(10,6) NULL,
                    ask_max NUMERIC(10,6) NULL,
                    mid_min NUMERIC(10,6) NULL,
                    mid_max NUMERIC(10,6) NULL
                ) ON COMMIT DROP
                """
            )
            for tier in tiers:
                stats = await self._compact_resolved_book_snapshots_for_tier(
                    conn,
                    bucket_seconds=int(tier["bucket_seconds"]),
                    resolved_after=tier["resolved_after"],
                    resolved_before=tier["resolved_before"],
                )
                totals["tiers"] += 1
                totals["buckets_compacted"] += int(stats["buckets_compacted"])
                totals["rows_updated"] += int(stats["rows_updated"])
                totals["rows_deleted"] += int(stats["rows_deleted"])
        return totals

    async def _compact_resolved_book_snapshots_for_tier(
        self,
        conn: AsyncConnection,
        *,
        bucket_seconds: int,
        resolved_after: datetime | None,
        resolved_before: datetime | None,
    ) -> dict[str, int]:
        conditions = [
            "lower(coalesce(m.status, '')) NOT IN ('active', 'open')",
            "coalesce(m.resolution_time, m.updated_at) IS NOT NULL",
        ]
        params: dict[str, object] = {"bucket_seconds": int(bucket_seconds)}
        if resolved_after is not None:
            conditions.append("coalesce(m.resolution_time, m.updated_at) > %(resolved_after)s")
            params["resolved_after"] = resolved_after
        if resolved_before is not None:
            conditions.append("coalesce(m.resolution_time, m.updated_at) <= %(resolved_before)s")
            params["resolved_before"] = resolved_before
        where_sql = " AND ".join(conditions)
        bucket_expr = (
            "to_timestamp(floor(extract(epoch from bs.ts) / %(bucket_seconds)s) * %(bucket_seconds)s)"
        )

        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE TABLE _book_snapshot_compact_stage")
            await cur.execute(
                f"""
                INSERT INTO _book_snapshot_compact_stage (
                    market_id,
                    outcome_index,
                    bucket_ts,
                    keep_ts,
                    n,
                    first_ts,
                    last_ts,
                    bid_min,
                    bid_max,
                    ask_min,
                    ask_max,
                    mid_min,
                    mid_max
                )
                SELECT
                    bs.market_id,
                    bs.outcome_index,
                    {bucket_expr} AS bucket_ts,
                    max(bs.ts) AS keep_ts,
                    count(*)::INT AS n,
                    min(bs.ts) AS first_ts,
                    max(bs.ts) AS last_ts,
                    min(bs.best_bid) AS bid_min,
                    max(bs.best_bid) AS bid_max,
                    min(bs.best_ask) AS ask_min,
                    max(bs.best_ask) AS ask_max,
                    min(
                        CASE
                            WHEN bs.best_bid IS NOT NULL AND bs.best_ask IS NOT NULL THEN ((bs.best_bid + bs.best_ask) / 2.0)
                            ELSE NULL
                        END
                    ) AS mid_min,
                    max(
                        CASE
                            WHEN bs.best_bid IS NOT NULL AND bs.best_ask IS NOT NULL THEN ((bs.best_bid + bs.best_ask) / 2.0)
                            ELSE NULL
                        END
                    ) AS mid_max
                FROM book_snapshots bs
                JOIN markets m ON m.market_id = bs.market_id
                WHERE {where_sql}
                GROUP BY bs.market_id, bs.outcome_index, {bucket_expr}
                HAVING count(*) > 1
                """,
                params,
            )
            buckets_compacted = int(cur.rowcount or 0)
            if buckets_compacted == 0:
                return {"buckets_compacted": 0, "rows_updated": 0, "rows_deleted": 0}

            await cur.execute(
                """
                UPDATE book_snapshots bs
                SET raw = jsonb_build_object(
                    'v', 1,
                    'kind', 'book_snapshot_compact',
                    'bucket_seconds', %(bucket_seconds)s,
                    'n', s.n,
                    'first_ts', s.first_ts,
                    'last_ts', s.last_ts,
                    'close', jsonb_build_object(
                        'best_bid', bs.best_bid,
                        'best_ask', bs.best_ask,
                        'bid_size', bs.bid_size,
                        'ask_size', bs.ask_size
                    ),
                    'range', jsonb_build_object(
                        'bid_min', s.bid_min,
                        'bid_max', s.bid_max,
                        'ask_min', s.ask_min,
                        'ask_max', s.ask_max
                    ),
                    'mid', jsonb_build_object(
                        'min', s.mid_min,
                        'max', s.mid_max
                    )
                )
                FROM _book_snapshot_compact_stage s
                WHERE bs.market_id = s.market_id
                  AND bs.outcome_index IS NOT DISTINCT FROM s.outcome_index
                  AND bs.ts = s.keep_ts
                """,
                params,
            )
            rows_updated = int(cur.rowcount or 0)

            await cur.execute(
                """
                DELETE FROM book_snapshots bs
                USING _book_snapshot_compact_stage s
                WHERE bs.market_id = s.market_id
                  AND bs.outcome_index IS NOT DISTINCT FROM s.outcome_index
                  AND to_timestamp(
                        floor(extract(epoch from bs.ts) / %(bucket_seconds)s) * %(bucket_seconds)s
                      ) = s.bucket_ts
                  AND bs.ts <> s.keep_ts
                """,
                params,
            )
            rows_deleted = int(cur.rowcount or 0)
        return {
            "buckets_compacted": buckets_compacted,
            "rows_updated": rows_updated,
            "rows_deleted": rows_deleted,
        }


def _resolved_compaction_tiers(
    *,
    now_utc: datetime,
    grace_minutes: int,
    bucket_seconds_recent: int,
    bucket_seconds_mid: int,
    bucket_seconds_old: int,
) -> list[dict[str, object]]:
    now = now_utc.astimezone(timezone.utc)
    grace_cutoff = now - timedelta(minutes=max(0, int(grace_minutes)))
    day_cutoff = now - timedelta(hours=24)
    week_cutoff = now - timedelta(days=7)
    return [
        {
            "name": "resolved_recent",
            "bucket_seconds": int(bucket_seconds_recent),
            "resolved_after": day_cutoff,
            "resolved_before": grace_cutoff,
        },
        {
            "name": "resolved_mid",
            "bucket_seconds": int(bucket_seconds_mid),
            "resolved_after": week_cutoff,
            "resolved_before": day_cutoff,
        },
        {
            "name": "resolved_old",
            "bucket_seconds": int(bucket_seconds_old),
            "resolved_after": None,
            "resolved_before": week_cutoff,
        },
    ]
