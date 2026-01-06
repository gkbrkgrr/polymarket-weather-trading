import os
import tempfile
import unittest

import polymarket_archive as pa


class TargetMatchingTests(unittest.TestCase):
    def test_matches_daily_contract(self) -> None:
        t = pa.Target(name="Seoul", match_any=["Highest temperature in Seoul"], match_all=[])
        self.assertTrue(t.matches("Highest temperature in Seoul January 6"))
        self.assertTrue(t.matches("Highest Temperature In Seoul January 7"))
        self.assertFalse(t.matches("Highest temperature in Toronto January 6"))

    def test_match_all_and_any(self) -> None:
        t = pa.Target(
            name="NYC precip",
            match_any=["Precipitation in NYC"],
            match_all=["monthly total precipitation in inches"],
        )
        self.assertTrue(t.matches("Precipitation in NYC (monthly total precipitation in inches) for Jan"))
        self.assertFalse(t.matches("Precipitation in NYC total precipitation for Jan"))


class GammaListMarketsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_http_get_json = pa.http_get_json

    def tearDown(self) -> None:
        pa.http_get_json = self._orig_http_get_json

    def test_accepts_list_shape(self) -> None:
        calls: list[str] = []

        def fake_http(url: str, timeout_s: int = 30):
            calls.append(url)
            return [{"id": "1", "question": "A"}, {"id": "2", "question": "B"}]

        pa.http_get_json = fake_http  # type: ignore[assignment]
        markets = list(pa.gamma_list_markets("https://example.com", limit=200, max_pages=10))
        self.assertEqual([m["id"] for m in markets], ["1", "2"])
        self.assertEqual(len(calls), 1)

    def test_accepts_dict_shape(self) -> None:
        calls: list[str] = []

        def fake_http(url: str, timeout_s: int = 30):
            calls.append(url)
            return {"value": [{"id": "1", "question": "A"}], "Count": 1}

        pa.http_get_json = fake_http  # type: ignore[assignment]
        markets = list(pa.gamma_list_markets("https://example.com", limit=200, max_pages=10))
        self.assertEqual([m["id"] for m in markets], ["1"])
        self.assertEqual(len(calls), 1)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_gamma_list_markets = pa.gamma_list_markets
        self._orig_gamma_get_market = pa.gamma_get_market

    def tearDown(self) -> None:
        pa.gamma_list_markets = self._orig_gamma_list_markets  # type: ignore[assignment]
        pa.gamma_get_market = self._orig_gamma_get_market  # type: ignore[assignment]

    def test_discover_then_snapshot_writes_rows(self) -> None:
        market = {
            "id": "999",
            "question": "Highest temperature in Seoul January 6",
            "slug": "highest-temperature-in-seoul-jan-6",
            "conditionId": "0xabc",
            "category": "weather",
            "endDate": "2026-01-06T23:59:59Z",
            "createdAt": "2026-01-06T00:00:00Z",
            "active": True,
            "closed": False,
            "archived": False,
            "volumeNum": 123.0,
            "liquidityNum": 456.0,
            "lastTradePrice": 0.42,
            "bestBid": 0.41,
            "bestAsk": 0.43,
            "spread": 0.02,
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.42","0.58"]',
            "clobTokenIds": '["1","2"]',
        }

        def fake_list(*args, **kwargs):
            yield market

        def fake_get(*args, **kwargs):
            return market

        pa.gamma_list_markets = fake_list  # type: ignore[assignment]
        pa.gamma_get_market = fake_get  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.sqlite3")
            conn = pa.connect_db(db_path)
            try:
                pa.init_db(conn)
                config = {
                    "gamma_base_url": "https://example.com",
                    "db_path": db_path,
                    "discovery": {
                        "active": True,
                        "closed": False,
                        "archived": False,
                        "limit": 200,
                        "max_pages": 1,
                        "timeout_s": 1,
                    },
                    "snapshot_timeout_s": 1,
                    "targets": [
                        {"name": "Highest temperature in Seoul", "match_any": ["Highest temperature in Seoul"], "match_all": []}
                    ],
                }

                matches = pa.discover(conn, config)
                self.assertEqual(len(matches), 1)

                snap_count = pa.snapshot(conn, config)
                self.assertEqual(snap_count, 1)

                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM markets")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT COUNT(*) FROM market_snapshots")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT COUNT(*) FROM outcome_prices")
                self.assertEqual(cur.fetchone()[0], 2)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
