from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from polymarket_archive.data_client import DataClient, parse_trade
from polymarket_archive.db import Database
from polymarket_archive.utils import CursorState


async def ingest_trades_for_market(
    db: Database,
    client: DataClient,
    market_id: str,
    condition_id: str | None,
    start_ts: datetime,
    end_ts: datetime | None,
    page_size: int,
) -> CursorState:
    cursor = await db.get_cursor(market_id)
    if cursor is None:
        cursor = CursorState(start_ts, "")

    next_cursor: str | None = None
    current_cursor = cursor
    effective_start = current_cursor.last_ts

    market_param = condition_id or await db.get_condition_id(market_id) or market_id

    while True:
        page, next_cursor, _ = await client.list_trades(
            market_param=market_param,
            market_id=market_id,
            cursor=next_cursor,
            limit=page_size,
            start_ts=effective_start,
            end_ts=end_ts,
        )
        trades = _parse_trades(client, page, market_id)
        filtered = _filter_trades(trades, current_cursor, start_ts, end_ts)
        if filtered:
            filtered.sort(key=lambda t: (t.ts, t.trade_id))
            new_cursor = CursorState(filtered[-1].ts, filtered[-1].trade_id)
            async with db.connection() as conn:
                async with conn.transaction():
                    await db.insert_trades(filtered, conn=conn)
                    await db.upsert_cursor(market_id, new_cursor, conn=conn)
            current_cursor = new_cursor
        if not page or not next_cursor:
            break

    return current_cursor


def _filter_trades(
    trades: Iterable, cursor: CursorState, start_ts: datetime, end_ts: datetime | None
) -> list:
    results = []
    for trade in trades:
        if trade.ts < start_ts:
            continue
        if end_ts is not None and trade.ts > end_ts:
            continue
        if trade.ts < cursor.last_ts:
            continue
        if trade.ts == cursor.last_ts and trade.trade_id <= cursor.last_tiebreak:
            continue
        results.append(trade)
    return results


def _parse_trades(client: DataClient, payloads: Iterable[dict], market_id: str) -> list:
    parsed = []
    for payload in payloads:
        trade = parse_trade(payload, fallback_market_id=market_id)
        if trade is None:
            _log_trade_error(client, payload, market_id)
            continue
        parsed.append(trade)
    return parsed


def _log_trade_error(client: DataClient, payload: dict, market_id: str | None) -> None:
    ts = datetime.now(timezone.utc)
    client.raw_sink.write_record(
        "error",
        ts,
        {"url": "", "params": {}, "headers_redacted": {}, "cursor": None},
        {"error": "trade_parse_failed", "payload": payload},
        client.run_id,
        market_id=market_id or payload.get("market") or payload.get("market_id"),
    )
