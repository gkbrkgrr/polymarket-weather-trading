from __future__ import annotations

import asyncio
import logging
import orjson
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

import httpx

from polymarket_archive.config import Settings
from polymarket_archive.clob_client import ClobClient, ClobRestClient
from polymarket_archive.data_client import DataClient
from polymarket_archive.db import Database
from polymarket_archive.discovery import discover_markets
from polymarket_archive.gamma_client import GammaClient
from polymarket_archive.http import RequestLimiter
from polymarket_archive.ingest_markets import ingest_markets
from polymarket_archive.ingest_trades import ingest_trades_for_market
from polymarket_archive.raw_sink import RawSink


def _title_filters(settings: Settings) -> list[str]:
    terms: list[str] = []
    for term in settings.market_title_contains:
        value = str(term).strip()
        if value:
            terms.append(value)
    if settings.title_filter and str(settings.title_filter).strip():
        terms.append(str(settings.title_filter).strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        lower = term.lower()
        if lower in seen:
            continue
        seen.add(lower)
        deduped.append(term)
    return deduped


async def run_backfill(
    settings: Settings,
    db: Database,
    start_ts: datetime,
    end_ts: datetime | None,
    run_id: str | None = None,
) -> None:
    run_id = run_id or str(uuid4())
    async with httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_seconds)) as client:
        limiter = RequestLimiter(settings.rate_limit_per_second)
        raw_sink = RawSink(settings.raw_dir)
        gamma = GammaClient(
            settings.gamma_base_url, client, limiter, settings.max_retries, raw_sink, run_id
        )
        data = DataClient(
            settings.data_base_url, client, limiter, settings.max_retries, raw_sink, run_id
        )
        async def _on_batch(batch) -> None:
            await ingest_markets(db, batch)

        title_filters = _title_filters(settings)
        markets = await discover_markets(
            gamma,
            title_filters,
            settings.market_filters,
            settings.market_tag_ids,
            settings.target_market_ids,
            settings.markets_page_size,
            start_date_min=start_ts,
            on_batch=_on_batch,
        )
        await _ingest_trades_concurrently(
            db, data, markets, start_ts, end_ts, settings.concurrency, settings.trades_page_size
        )


async def run_live_once(settings: Settings, db: Database, run_id: str | None = None) -> None:
    run_id = run_id or str(uuid4())
    now = datetime.now(timezone.utc)
    async with httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_seconds)) as client:
        limiter = RequestLimiter(settings.rate_limit_per_second)
        raw_sink = RawSink(settings.raw_dir)
        gamma = GammaClient(
            settings.gamma_base_url, client, limiter, settings.max_retries, raw_sink, run_id
        )
        data = DataClient(
            settings.data_base_url, client, limiter, settings.max_retries, raw_sink, run_id
        )
        async def _on_batch(batch) -> None:
            await ingest_markets(db, batch)

        title_filters = _title_filters(settings)
        await discover_markets(
            gamma,
            title_filters,
            settings.market_filters,
            settings.market_tag_ids,
            settings.target_market_ids,
            settings.markets_page_size,
            start_date_min=settings.backfill_start,
            on_batch=_on_batch,
        )
        market_ids = await db.list_market_condition_ids()
        await _ingest_trades_for_market_ids(
            db,
            data,
            market_ids,
            settings.backfill_start,
            now,
            settings.concurrency,
            settings.trades_page_size,
        )


async def run_live_loop(settings: Settings, db: Database) -> None:
    run_id = str(uuid4())
    last_discovery: datetime | None = None
    logger = logging.getLogger(__name__)
    async with httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_seconds)) as client:
        limiter = RequestLimiter(settings.rate_limit_per_second)
        raw_sink = RawSink(settings.raw_dir)
        gamma = GammaClient(
            settings.gamma_base_url, client, limiter, settings.max_retries, raw_sink, run_id
        )
        data = DataClient(
            settings.data_base_url, client, limiter, settings.max_retries, raw_sink, run_id
        )
        clob_task: asyncio.Task | None = None
        clob_rest_task: asyncio.Task | None = None
        clob_client: ClobClient | None = None
        rest_client: ClobRestClient | None = None
        token_map: dict[str, tuple[str, int | None]] = {}
        if settings.feature_clob:
            token_map = await _build_clob_token_map(db)
            clob_client = ClobClient(
                settings.clob_ws_url,
                db,
                raw_sink,
                run_id,
                settings.book_snapshot_interval_seconds,
            )
            rest_client = ClobRestClient(
                settings.clob_base_url,
                client,
                RequestLimiter(settings.rate_limit_per_second),
                settings.max_retries,
                db,
                raw_sink,
                run_id,
                settings.book_snapshot_interval_seconds,
                settings.concurrency,
            )

            def _start_rest() -> None:
                nonlocal clob_rest_task
                if rest_client is not None and clob_rest_task is None:
                    clob_rest_task = asyncio.create_task(rest_client.run(token_map))
                    clob_rest_task.add_done_callback(
                        lambda task: _log_task_result(task, logger, "clob_rest")
                    )

            def _start_ws() -> None:
                nonlocal clob_task
                if clob_client is None or clob_task is not None:
                    return
                if not token_map:
                    return
                clob_task = asyncio.create_task(clob_client.run(token_map))

                def _on_ws_done(task: asyncio.Task) -> None:
                    if task.cancelled():
                        return
                    try:
                        task.result()
                    except Exception as exc:
                        logger.warning("clob task failed: %s", exc)
                        _start_rest()

                clob_task.add_done_callback(_on_ws_done)

            if settings.clob_ws_url:
                _start_ws()
            else:
                _start_rest()

        while True:
            now = datetime.now(timezone.utc)
            if last_discovery is None or (now - last_discovery).total_seconds() >= settings.discovery_interval_seconds:
                async def _on_batch(batch) -> None:
                    await ingest_markets(db, batch)

                title_filters = _title_filters(settings)
                await discover_markets(
                    gamma,
                    title_filters,
                    settings.market_filters,
                    settings.market_tag_ids,
                    settings.target_market_ids,
                    settings.markets_page_size,
                    start_date_min=settings.backfill_start,
                    on_batch=_on_batch,
                )
                last_discovery = now
                if settings.feature_clob:
                    await _refresh_clob_token_map(db, token_map, logger)
                    if settings.clob_ws_url:
                        _start_ws()
                    else:
                        _start_rest()

            market_ids = await db.list_market_condition_ids()
            await _ingest_trades_for_market_ids(
                db,
                data,
                market_ids,
                settings.backfill_start,
                now,
                settings.concurrency,
                settings.trades_page_size,
            )
            await asyncio.sleep(settings.poll_interval_seconds)
        if clob_task:
            clob_task.cancel()


def _log_task_result(task: asyncio.Task, logger: logging.Logger, name: str) -> None:
    if task.cancelled():
        return
    try:
        task.result()
    except Exception as exc:
        logger.warning("%s task failed: %s", name, exc)


async def _build_clob_token_map(db: Database) -> dict[str, tuple[str, int | None]]:
    token_map: dict[str, tuple[str, int | None]] = {}
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT market_id, status, raw->>'active' AS raw_active, raw->>'clobTokenIds' "
                "FROM markets"
            )
            rows = await cur.fetchall()
    for market_id, status, raw_active, token_blob in rows:
        status_value = (status or "").lower()
        raw_active_flag = (raw_active or "").lower() == "true"
        if status_value not in {"active", "open"} and not raw_active_flag:
            continue
        tokens = _parse_token_blob(token_blob)
        for idx, token_id in enumerate(tokens):
            token_map[str(token_id)] = (market_id, idx)
    return token_map


def _parse_token_blob(value: str | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            decoded = orjson.loads(value)
        except orjson.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
        return [value]
    return []


async def _refresh_clob_token_map(
    db: Database, token_map: dict[str, tuple[str, int | None]], logger: logging.Logger
) -> None:
    latest = await _build_clob_token_map(db)
    added = 0
    for token_id, mapping in latest.items():
        if token_id not in token_map:
            token_map[token_id] = mapping
            added += 1
    if added:
        logger.info("Added %d new CLOB token ids", added)


async def _ingest_trades_concurrently(
    db: Database,
    data: DataClient,
    markets: Iterable,
    start_ts: datetime,
    end_ts: datetime | None,
    concurrency: int,
    page_size: int,
) -> None:
    logger = logging.getLogger(__name__)
    sem = asyncio.Semaphore(concurrency)

    async def _run(market):
        async with sem:
            condition_id = None
            if isinstance(market.raw, dict):
                condition_id = market.raw.get("conditionId")
            try:
                await ingest_trades_for_market(
                    db, data, market.market_id, condition_id, start_ts, end_ts, page_size
                )
            except Exception as exc:
                logger.warning("trade ingest failed market_id=%s err=%s", market.market_id, exc)

    await asyncio.gather(*[_run(market) for market in markets])


async def _ingest_trades_for_market_ids(
    db: Database,
    data: DataClient,
    market_ids: Iterable[tuple[str, str | None]],
    start_ts: datetime,
    end_ts: datetime | None,
    concurrency: int,
    page_size: int,
) -> None:
    logger = logging.getLogger(__name__)
    sem = asyncio.Semaphore(concurrency)

    async def _run(market_pair: tuple[str, str | None]) -> None:
        async with sem:
            market_id, condition_id = market_pair
            try:
                await ingest_trades_for_market(
                    db, data, market_id, condition_id, start_ts, end_ts, page_size
                )
            except Exception as exc:
                logger.warning("trade ingest failed market_id=%s err=%s", market_id, exc)

    await asyncio.gather(*[_run(market_pair) for market_pair in market_ids])
