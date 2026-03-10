from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_GATHERER_ROOT = ROOT / "data_gatherer"
if str(DATA_GATHERER_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_GATHERER_ROOT))

from polymarket_archive.clob_client import _build_snapshot_raw_metadata
from polymarket_archive.db import _resolved_compaction_tiers
from polymarket_archive.jobs import _refresh_clob_token_map


def test_snapshot_raw_metadata_is_compact() -> None:
    payload = {
        "type": "book",
        "data": {
            "timestamp": "2026-03-09T12:00:01Z",
            "bids": [["0.40", "100"], ["0.39", "120"]],
            "asks": [["0.42", "80"]],
        },
    }
    out = _build_snapshot_raw_metadata(
        payload=payload,
        source="clob_ws",
        token_id="tok-1",
        market_id="m-1",
        outcome_index=1,
        event_ts=datetime(2026, 3, 9, 12, 0, 2, tzinfo=timezone.utc),
    )

    assert out["v"] == 1
    assert out["kind"] == "book_snapshot_meta"
    assert out["source"] == "clob_ws"
    assert out["token_id"] == "tok-1"
    assert out["market_id"] == "m-1"
    assert out["outcome_index"] == 1
    assert out["message_ts"] == "2026-03-09T12:00:01Z"
    assert out["bid_levels"] == 2
    assert out["ask_levels"] == 1


def test_resolved_compaction_tiers_match_policy() -> None:
    now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
    tiers = _resolved_compaction_tiers(
        now_utc=now,
        grace_minutes=60,
        bucket_seconds_recent=20,
        bucket_seconds_mid=30,
        bucket_seconds_old=60,
    )

    assert [int(t["bucket_seconds"]) for t in tiers] == [20, 30, 60]
    assert tiers[0]["resolved_after"] == datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
    assert tiers[0]["resolved_before"] == datetime(2026, 3, 9, 11, 0, tzinfo=timezone.utc)
    assert tiers[1]["resolved_after"] == datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    assert tiers[1]["resolved_before"] == datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
    assert tiers[2]["resolved_after"] is None
    assert tiers[2]["resolved_before"] == datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)


def test_refresh_clob_token_map_reconciles_active_open_set(monkeypatch, caplog) -> None:
    token_map = {
        "tok-a": ("m-a", 0),
        "tok-b": ("m-b", 1),
    }

    async def _fake_build_token_map(db):
        return {
            "tok-a": ("m-a-remapped", 0),
            "tok-c": ("m-c", 1),
        }

    monkeypatch.setattr("polymarket_archive.jobs._build_clob_token_map", _fake_build_token_map)
    caplog.set_level(logging.INFO)

    asyncio.run(_refresh_clob_token_map(db=None, token_map=token_map, logger=logging.getLogger("test.token.map")))

    assert token_map == {
        "tok-a": ("m-a-remapped", 0),
        "tok-c": ("m-c", 1),
    }
    assert any("added=1" in rec.message and "removed=1" in rec.message and "remapped=1" in rec.message for rec in caplog.records)
