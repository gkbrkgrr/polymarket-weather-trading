from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import uuid
import unittest

import pandas as pd
import psycopg

from master_db import (
    ensure_master_schema,
    get_obs_for_day,
    get_station_obs,
    upsert_station_observations,
)


def _resolve_test_dsn() -> str:
    explicit = os.getenv("TEST_POSTGRES_DSN")
    if explicit:
        return explicit

    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if cfg_path.exists():
        text = cfg_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip().startswith("postgres_dsn:"):
                return line.split(":", 1)[1].strip()

    raise RuntimeError("No TEST_POSTGRES_DSN and no postgres_dsn in config.yaml")


def _record(
    *,
    station: str,
    observed_at_local: str,
    temperature_f: int,
    temperature_c: int,
    scraped_at_utc: str,
    precipitation_hourly_in: float | None = None,
    precipitation_total_in: float | None = None,
) -> dict[str, object]:
    return {
        "station": station,
        "observed_at_local": observed_at_local,
        "temperature_f": temperature_f,
        "temperature_c": temperature_c,
        "precipitation_hourly_in": precipitation_hourly_in,
        "precipitation_total_in": precipitation_total_in,
        "scraped_at_utc": scraped_at_utc,
    }


class StationObservationsUpsertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dsn = _resolve_test_dsn()
        ensure_master_schema(master_dsn=cls.dsn)

    def setUp(self) -> None:
        self.station = f"pytest_station_{uuid.uuid4().hex}"

    def tearDown(self) -> None:
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM station_observations WHERE station = %s", (self.station,))
            conn.commit()

    def test_upsert_semantics_and_queries(self) -> None:
        observed_at = "2026-03-03T12:30:00+03:00"

        first = _record(
            station=self.station,
            observed_at_local=observed_at,
            temperature_f=68,
            temperature_c=20,
            scraped_at_utc="2026-03-03T09:31:00+00:00",
        )

        stats_first = upsert_station_observations(records=[first], master_dsn=self.dsn)
        self.assertEqual(stats_first, {"inserted": 1, "updated": 0, "unchanged": 0})

        rows_after_first = get_station_obs(self.station, None, None, master_dsn=self.dsn)
        self.assertEqual(len(rows_after_first), 1)
        first_scraped = rows_after_first.iloc[0]["scraped_at_utc"]

        identical = _record(
            station=self.station,
            observed_at_local=observed_at,
            temperature_f=68,
            temperature_c=20,
            scraped_at_utc="2026-03-03T09:45:00+00:00",
        )
        stats_identical = upsert_station_observations(records=[identical], master_dsn=self.dsn)
        self.assertEqual(stats_identical, {"inserted": 0, "updated": 0, "unchanged": 1})

        rows_after_identical = get_station_obs(self.station, None, None, master_dsn=self.dsn)
        self.assertEqual(len(rows_after_identical), 1)
        self.assertEqual(rows_after_identical.iloc[0]["scraped_at_utc"], first_scraped)

        changed = _record(
            station=self.station,
            observed_at_local=observed_at,
            temperature_f=70,
            temperature_c=21,
            scraped_at_utc="2026-03-03T10:00:00+00:00",
        )
        stats_changed = upsert_station_observations(records=[changed], master_dsn=self.dsn)
        self.assertEqual(stats_changed, {"inserted": 0, "updated": 1, "unchanged": 0})

        rows_after_changed = get_station_obs(self.station, None, None, master_dsn=self.dsn)
        self.assertEqual(len(rows_after_changed), 1)
        self.assertEqual(int(rows_after_changed.iloc[0]["temperature_c"]), 21)
        self.assertEqual(int(rows_after_changed.iloc[0]["temperature_f"]), 70)
        self.assertNotEqual(rows_after_changed.iloc[0]["scraped_at_utc"], first_scraped)

        day_rows = get_obs_for_day(self.station, dt.date(2026, 3, 3), master_dsn=self.dsn)
        self.assertEqual(len(day_rows), 1)

        next_day_rows = get_obs_for_day(self.station, dt.date(2026, 3, 4), master_dsn=self.dsn)
        self.assertTrue(next_day_rows.empty)

        # Datetime columns should arrive as datetime dtypes without caller-side conversion.
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(rows_after_changed["observed_at_local"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(rows_after_changed["scraped_at_utc"]))


if __name__ == "__main__":
    unittest.main()
