from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import ValidationError

from polymarket_archive.http import build_request_info, fetch_json, RequestLimiter
from polymarket_archive.models import Trade
from polymarket_archive.raw_sink import RawSink
from polymarket_archive.utils import coerce_decimal, parse_datetime, surrogate_trade_id


class DataClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient,
        limiter: RequestLimiter,
        max_retries: int,
        raw_sink: RawSink,
        run_id: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.limiter = limiter
        self.max_retries = max_retries
        self.raw_sink = raw_sink
        self.run_id = run_id

    async def list_trades(
        self,
        market_param: str,
        market_id: str | None,
        cursor: str | None,
        limit: int,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, datetime]:
        url = f"{self.base_url}/trades"
        params: dict[str, Any] = {"limit": limit, "market": market_param}
        if cursor:
            params["cursor"] = cursor
        if start_ts:
            params["start"] = start_ts.astimezone(timezone.utc).isoformat()
        if end_ts:
            params["end"] = end_ts.astimezone(timezone.utc).isoformat()
        ts = datetime.now(timezone.utc)
        payload = await fetch_json(
            self.client, "GET", url, params, limiter=self.limiter, max_retries=self.max_retries
        )
        request_info = build_request_info(url, params, cursor=cursor)
        raw_market_id = market_id or market_param
        self.raw_sink.write_record(
            "data_trades", ts, request_info, payload, self.run_id, market_id=raw_market_id
        )
        trades = _extract_trades(payload)
        next_cursor = _extract_next_cursor(payload)
        return trades, next_cursor, ts


def _extract_trades(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("trades"), list):
            return payload["trades"]
    if isinstance(payload, list):
        return payload
    return []


def _extract_next_cursor(payload: Any) -> str | None:
    if isinstance(payload, dict):
        return (
            payload.get("next_cursor")
            or payload.get("nextCursor")
            or payload.get("cursor")
            or payload.get("next")
        )
    return None


def parse_trade(payload: dict[str, Any], fallback_market_id: str | None = None) -> Trade | None:
    market_id = payload.get("market") or payload.get("market_id") or payload.get("marketId")
    if not market_id and fallback_market_id:
        market_id = fallback_market_id
    if not market_id:
        return None
    ts = parse_datetime(
        payload.get("timestamp")
        or payload.get("created_at")
        or payload.get("createdAt")
        or payload.get("ts")
    )
    if ts is None:
        return None
    price = coerce_decimal(payload.get("price"))
    size = coerce_decimal(payload.get("size") or payload.get("amount"))
    if price is None or size is None:
        return None
    trade_id = payload.get("id") or payload.get("trade_id") or payload.get("tradeId")
    outcome_id = payload.get("outcome_id") or payload.get("outcome") or payload.get("outcomeId")
    outcome_index = payload.get("outcome_index") or payload.get("outcomeIndex")
    side = payload.get("side")
    tx_hash = payload.get("tx_hash") or payload.get("transactionHash")
    if trade_id is None:
        trade_id = surrogate_trade_id(
            str(market_id), ts, price, size, str(outcome_id) if outcome_id else None, side, tx_hash, outcome_index
        )
    try:
        return Trade(
            trade_id=str(trade_id),
            market_id=str(market_id),
            ts=ts,
            outcome_id=str(outcome_id) if outcome_id is not None else None,
            outcome_index=int(outcome_index) if outcome_index is not None else None,
            side=side,
            price=price,
            size=size,
            tx_hash=tx_hash,
            raw=payload,
        )
    except (ValidationError, ValueError):
        return None
