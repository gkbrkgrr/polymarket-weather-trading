from __future__ import annotations

from datetime import datetime, timezone
import asyncio
from typing import Awaitable, Callable, Iterable

import httpx

from polymarket_archive.gamma_client import GammaClient, filter_markets, parse_market
from polymarket_archive.models import GammaMarket


async def discover_markets(
    client: GammaClient,
    title_filter: str,
    page_size: int,
    on_batch: Callable[[list[GammaMarket]], Awaitable[None]] | None = None,
) -> list[GammaMarket]:
    markets: list[GammaMarket] = []
    offset = 0
    search: str | None = title_filter
    while True:
        try:
            page, _ = await client.list_markets(search, page_size, offset)
        except httpx.HTTPStatusError:
            if search is not None:
                search = None
                offset = 0
                markets = []
                continue
            raise
        if not page:
            break
        parsed = _parse_markets(client, page)
        filtered = filter_markets(parsed, title_filter)
        if filtered and on_batch is not None:
            result = on_batch(filtered)
            if asyncio.iscoroutine(result):
                await result
        markets.extend(filtered)
        next_offset = offset + len(page)
        if next_offset <= offset:
            break
        offset = next_offset
    return markets


def _parse_markets(client: GammaClient, payloads: Iterable[dict]) -> list[GammaMarket]:
    parsed: list[GammaMarket] = []
    for payload in payloads:
        market = parse_market(payload)
        if market is None:
            _log_validation_error(client, payload, "market_parse_failed")
            continue
        parsed.append(market)
    return parsed


def _log_validation_error(client: GammaClient, payload: dict, message: str) -> None:
    ts = datetime.now(timezone.utc)
    client.raw_sink.write_record(
        "error",
        ts,
        {"url": "", "params": {}, "headers_redacted": {}, "cursor": None},
        {"error": message, "payload": payload},
        client.run_id,
    )
