from __future__ import annotations

from datetime import datetime, timezone
import unittest

from polymarket_archive.discovery import discover_markets


class _FakeGammaClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def list_markets(self, search, limit, offset, start_date_min=None, tag_id=None):
        self.calls.append(
            {
                "search": search,
                "limit": limit,
                "offset": offset,
                "start_date_min": start_date_min,
                "tag_id": tag_id,
            }
        )
        pages = {
            0: [
                {"id": "1", "question": "Will BTC be above 1M?"},
                {"id": "2", "question": "Will ETH be above 50k?"},
            ],
            2: [
                {
                    "id": "3",
                    "question": "Will the highest temperature in London be 5C on Jan 3?",
                }
            ],
            3: [],
        }
        return pages.get(offset, []), datetime.now(timezone.utc)


class DiscoveryPagingTest(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_scans_multiple_pages_until_empty(self) -> None:
        client = _FakeGammaClient()
        start_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

        found = await discover_markets(
            client=client,
            title_filters=["highest temperature in"],
            market_filters=[],
            market_tag_ids=[84],
            target_market_ids=[],
            page_size=500,
            start_date_min=start_ts,
            on_batch=None,
        )

        self.assertEqual([m.market_id for m in found], ["3"])
        self.assertEqual([c["offset"] for c in client.calls], [0, 2, 3])
        self.assertTrue(all(c["search"] is None for c in client.calls))
        self.assertTrue(all(c["start_date_min"] == start_ts for c in client.calls))
        self.assertTrue(all(c["tag_id"] == 84 for c in client.calls))
