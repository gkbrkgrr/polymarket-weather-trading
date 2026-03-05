from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx

from polymarket_archive.db import Database
from polymarket_archive.raw_sink import RawSink
from polymarket_archive.utils import coerce_decimal
from polymarket_archive.http import RequestLimiter, build_request_info, fetch_json


class ClobClient:
    def __init__(
        self,
        ws_url: str,
        db: Database,
        raw_sink: RawSink,
        run_id: str,
        snapshot_interval_seconds: int,
    ) -> None:
        self.ws_url = ws_url.rstrip("/")
        self.db = db
        self.raw_sink = raw_sink
        self.run_id = run_id
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self._last_snapshot: dict[tuple[str, int | None], datetime] = {}
        self._last_quote: dict[
            tuple[str, int | None],
            tuple[Decimal | None, Decimal | None, Decimal | None, Decimal | None],
        ] = {}
        self._subscribed_assets: set[str] = set()
        self._logger = logging.getLogger(__name__)

    async def run(self, token_map: dict[str, tuple[str, int | None]]) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("websockets is required for FEATURE_CLOB") from exc

        self._logger.info("Starting CLOB websocket to %s", self.ws_url)
        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))
            try:
                await self._subscribe(ws, token_map)
                while True:
                    try:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        await self._subscribe(ws, token_map)
                        continue
                    await self._handle_message(raw_msg, token_map)
                    await self._subscribe(ws, token_map)
            finally:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(10)
            await ws.send("PING")

    async def _subscribe(self, ws, token_map: dict[str, tuple[str, int | None]]) -> None:
        new_asset_ids = [token_id for token_id in token_map.keys() if token_id not in self._subscribed_assets]
        if not new_asset_ids:
            return

        self._logger.info("Subscribing to %d CLOB books", len(new_asset_ids))
        chunks = _chunked(new_asset_ids, size=500)
        for idx, chunk in enumerate(chunks):
            if idx == 0 and not self._subscribed_assets:
                payload = {
                    "assets_ids": chunk,
                    "type": "market",
                    "custom_feature_enabled": True,
                }
            else:
                payload = {
                    "assets_ids": chunk,
                    "operation": "subscribe",
                    "custom_feature_enabled": True,
                }
            await ws.send(json.dumps(payload))
        self._subscribed_assets.update(new_asset_ids)

    async def _handle_message(
        self, raw_msg: Any, token_map: dict[str, tuple[str, int | None]]
    ) -> None:
        if isinstance(raw_msg, bytes):
            try:
                raw_msg = raw_msg.decode("utf-8")
            except UnicodeDecodeError:
                return
        try:
            message = json.loads(raw_msg)
        except json.JSONDecodeError:
            return
        if not isinstance(message, dict):
            return

        ts = _extract_ts(message)
        token_id = _extract_token_id(message)
        market_id = None
        outcome_index = None
        if token_id and token_id in token_map:
            market_id, outcome_index = token_map[token_id]
        raw_market_id = market_id or token_id
        self.raw_sink.write_record(
            "clob_book",
            ts,
            {"url": self.ws_url, "params": {}, "headers_redacted": {}, "cursor": None},
            message,
            self.run_id,
            market_id=raw_market_id,
        )

        snapshot = _extract_snapshot(message, outcome_index)
        if snapshot is None or market_id is None:
            return

        outcome_index, best_bid, best_ask, bid_size, ask_size = snapshot
        key = (market_id, outcome_index)
        current_quote = (best_bid, best_ask, bid_size, ask_size)
        if not _quote_changed(current_quote, self._last_quote.get(key)):
            return
        if not _should_write(ts, self._last_snapshot.get(key), self.snapshot_interval_seconds):
            return

        await self.db.insert_book_snapshot(
            market_id=market_id,
            ts=ts,
            outcome_index=outcome_index,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            raw=message,
        )
        self._last_snapshot[key] = ts
        self._last_quote[key] = current_quote


def _extract_ts(message: dict[str, Any]) -> datetime:
    data = message.get("data") if isinstance(message.get("data"), dict) else message
    ts_value = data.get("timestamp") or data.get("ts") or data.get("time")
    if isinstance(ts_value, (int, float)):
        if ts_value > 10_000_000_000:
            ts_value = ts_value / 1000
        return datetime.fromtimestamp(ts_value, tz=timezone.utc)
    if isinstance(ts_value, str):
        try:
            if ts_value.isdigit() and len(ts_value) > 10:
                return datetime.fromtimestamp(int(ts_value) / 1000, tz=timezone.utc)
            if ts_value.endswith("Z"):
                ts_value = ts_value[:-1] + "+00:00"
            return datetime.fromisoformat(ts_value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _extract_token_id(message: dict[str, Any]) -> str | None:
    data = message.get("data") if isinstance(message.get("data"), dict) else message
    return (
        data.get("tokenId")
        or data.get("token_id")
        or data.get("asset_id")
        or data.get("assetId")
        or data.get("asset")
        or data.get("market")
    )


def _extract_snapshot(
    message: dict[str, Any],
    default_outcome_index: int | None,
) -> tuple[int | None, Decimal | None, Decimal | None, Decimal | None, Decimal | None] | None:
    data = message.get("data") if isinstance(message.get("data"), dict) else message
    bids = data.get("bids") or data.get("bid") or []
    asks = data.get("asks") or data.get("ask") or []
    best_bid = None
    best_ask = None
    bid_size = None
    ask_size = None
    if bids or asks:
        best_bid, bid_size = _best_level(bids, prefer_max=True)
        best_ask, ask_size = _best_level(asks, prefer_max=False)
    else:
        # best_bid_ask style payloads may carry only top-of-book fields.
        best_bid = coerce_decimal(data.get("best_bid") or data.get("bestBid") or data.get("bid"))
        best_ask = coerce_decimal(data.get("best_ask") or data.get("bestAsk") or data.get("ask"))
        bid_size = coerce_decimal(
            data.get("best_bid_size") or data.get("bestBidSize") or data.get("bid_size")
        )
        ask_size = coerce_decimal(
            data.get("best_ask_size") or data.get("bestAskSize") or data.get("ask_size")
        )
        if best_bid is None and best_ask is None:
            return None
    outcome_index = data.get("outcomeIndex") or data.get("outcome_index") or default_outcome_index
    if outcome_index is not None:
        try:
            outcome_index = int(outcome_index)
        except (TypeError, ValueError):
            outcome_index = None
    return outcome_index, best_bid, best_ask, bid_size, ask_size


def _best_level(levels: Any, prefer_max: bool) -> tuple[Decimal | None, Decimal | None]:
    parsed: list[tuple[Decimal, Decimal]] = []
    if isinstance(levels, list):
        for level in levels:
            price, size = _parse_level(level)
            if price is None or size is None:
                continue
            parsed.append((price, size))
    if not parsed:
        return None, None
    if prefer_max:
        return max(parsed, key=lambda item: item[0])
    return min(parsed, key=lambda item: item[0])


def _parse_level(level: Any) -> tuple[Decimal | None, Decimal | None]:
    if isinstance(level, dict):
        price = coerce_decimal(level.get("price") or level.get("p"))
        size = coerce_decimal(level.get("size") or level.get("s"))
        return price, size
    if isinstance(level, (list, tuple)) and len(level) >= 2:
        price = coerce_decimal(level[0])
        size = coerce_decimal(level[1])
        return price, size
    return None, None


def _should_write(ts: datetime, last_ts: datetime | None, interval_seconds: int) -> bool:
    if last_ts is None:
        return True
    return (ts - last_ts).total_seconds() >= interval_seconds


def _chunked(values: list[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [values[i : i + size] for i in range(0, len(values), size)]


def _quote_changed(
    current: tuple[Decimal | None, Decimal | None, Decimal | None, Decimal | None],
    previous: tuple[Decimal | None, Decimal | None, Decimal | None, Decimal | None] | None,
) -> bool:
    if previous is None:
        return True
    return current != previous


class ClobRestClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient,
        limiter: RequestLimiter,
        max_retries: int,
        db: Database,
        raw_sink: RawSink,
        run_id: str,
        snapshot_interval_seconds: int,
        concurrency: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.limiter = limiter
        self.max_retries = max_retries
        self.db = db
        self.raw_sink = raw_sink
        self.run_id = run_id
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.concurrency = max(1, concurrency)
        self._last_snapshot: dict[tuple[str, int | None], datetime] = {}
        self._last_quote: dict[
            tuple[str, int | None],
            tuple[Decimal | None, Decimal | None, Decimal | None, Decimal | None],
        ] = {}
        self._skip_tokens: dict[str, datetime] = {}
        self._logger = logging.getLogger(__name__)

    async def run(self, token_map: dict[str, tuple[str, int | None]]) -> None:
        self._logger.info("Starting CLOB REST polling from %s", self.base_url)
        while True:
            await self._poll_once(token_map)
            await asyncio.sleep(self.snapshot_interval_seconds)

    async def _poll_once(self, token_map: dict[str, tuple[str, int | None]]) -> None:
        sem = asyncio.Semaphore(self.concurrency)
        now = datetime.now(timezone.utc)

        async def _fetch(token_id: str, market_id: str, outcome_index: int | None) -> None:
            last_skip = self._skip_tokens.get(token_id)
            if last_skip and (now - last_skip).total_seconds() < 21600:
                return
            async with sem:
                url = f"{self.base_url}/book"
                params = {"token_id": token_id}
                try:
                    payload = await fetch_json(
                        self.client, "GET", url, params, self.limiter, self.max_retries
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        self._skip_tokens[token_id] = now
                        return
                    raise
                ts = _extract_ts(payload if isinstance(payload, dict) else {})
                request_info = build_request_info(url, params, cursor=None)
                self.raw_sink.write_record(
                    "clob_book", ts, request_info, payload, self.run_id, market_id=market_id
                )
                if not isinstance(payload, dict):
                    return
                snapshot = _extract_snapshot(payload, outcome_index)
                if snapshot is None:
                    return
                snap_outcome, best_bid, best_ask, bid_size, ask_size = snapshot
                key = (market_id, snap_outcome)
                current_quote = (best_bid, best_ask, bid_size, ask_size)
                if not _quote_changed(current_quote, self._last_quote.get(key)):
                    return
                if not _should_write(
                    ts, self._last_snapshot.get(key), self.snapshot_interval_seconds
                ):
                    return
                await self.db.insert_book_snapshot(
                    market_id=market_id,
                    ts=ts,
                    outcome_index=snap_outcome,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_size=bid_size,
                    ask_size=ask_size,
                    raw=payload,
                )
                self._last_snapshot[key] = ts
                self._last_quote[key] = current_quote

        tasks = [
            _fetch(token_id, market_id, outcome_index)
            for token_id, (market_id, outcome_index) in token_map.items()
        ]
        if tasks:
            await asyncio.gather(*tasks)
