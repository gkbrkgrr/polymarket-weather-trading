from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable, Sequence

from psycopg import AsyncConnection
from psycopg.errors import UndefinedTable
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
        try:
            async with self.connection() as conn:
                await conn.execute("SELECT 1 FROM markets LIMIT 1")
        except UndefinedTable:
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
