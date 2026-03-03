from __future__ import annotations

from datetime import datetime, timezone
import asyncio
from typing import Awaitable, Callable, Iterable

from polymarket_archive.gamma_client import GammaClient, filter_markets, parse_market
from polymarket_archive.models import GammaMarket


async def discover_markets(
    client: GammaClient,
    title_filters: Iterable[str],
    market_filters: Iterable[str],
    market_tag_ids: Iterable[int] | None,
    target_market_ids: Iterable[str],
    page_size: int,
    start_date_min: datetime | None = None,
    on_batch: Callable[[list[GammaMarket]], Awaitable[None]] | None = None,
) -> list[GammaMarket]:
    markets: dict[str, GammaMarket] = {}
    tag_ids = [int(tag) for tag in (market_tag_ids or []) if str(tag).strip()]
    if not tag_ids:
        tag_ids = [None]

    for tag_id in tag_ids:
        offset = 0
        previous_page_ids: tuple[str, ...] | None = None
        seen_page_signatures: set[tuple[str, ...]] = set()
        while True:
            page, _ = await client.list_markets(
                search=None,
                limit=page_size,
                offset=offset,
                start_date_min=start_date_min,
                tag_id=tag_id,
            )
            if not page:
                break

            raw_page_ids: tuple[str, ...] = tuple(
                str(payload.get("id") or payload.get("market_id") or payload.get("marketId") or "")
                for payload in page
                if isinstance(payload, dict)
            )
            if raw_page_ids:
                # Some Gamma endpoints may ignore offset and repeat the same page forever.
                if previous_page_ids is not None and raw_page_ids == previous_page_ids:
                    break
                if raw_page_ids in seen_page_signatures:
                    break
                seen_page_signatures.add(raw_page_ids)
                previous_page_ids = raw_page_ids

            parsed = _parse_markets(client, page)
            filtered = filter_markets(parsed, title_filters, market_filters, target_market_ids)
            if filtered and on_batch is not None:
                result = on_batch(filtered)
                if asyncio.iscoroutine(result):
                    await result

            for market in filtered:
                markets[market.market_id] = market

            next_offset = offset + len(page)
            if next_offset <= offset:
                break
            offset = next_offset
    return list(markets.values())


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
