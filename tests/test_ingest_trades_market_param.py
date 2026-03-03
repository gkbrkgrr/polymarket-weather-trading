from __future__ import annotations

from datetime import datetime, timezone
import unittest

from polymarket_archive.ingest_trades import ingest_trades_for_market


class _DummyDb:
    async def get_cursor(self, _market_id: str):
        return None


class _DummyDataClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def list_trades(
        self,
        market_param: str,
        market_id: str | None,
        cursor: str | None,
        limit: int,
        start_ts=None,
        end_ts=None,
    ):
        self.calls.append(
            {
                "market_param": market_param,
                "market_id": market_id,
                "cursor": cursor,
                "limit": limit,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        )
        return [], None, datetime.now(timezone.utc)


class IngestTradesMarketParamTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_condition_id_when_available(self) -> None:
        db = _DummyDb()
        client = _DummyDataClient()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        await ingest_trades_for_market(
            db=db,
            client=client,
            market_id="1080315",
            condition_id="0xabc",
            start_ts=start,
            end_ts=start,
            page_size=200,
        )

        self.assertEqual(client.calls[0]["market_param"], "0xabc")

    async def test_falls_back_to_market_id_when_condition_id_missing(self) -> None:
        db = _DummyDb()
        client = _DummyDataClient()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        await ingest_trades_for_market(
            db=db,
            client=client,
            market_id="1080315",
            condition_id=None,
            start_ts=start,
            end_ts=start,
            page_size=200,
        )

        self.assertEqual(client.calls[0]["market_param"], "1080315")
