from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from live_trading.policy import PolicyContext, apply_policy
from live_trading.run_live_pilot import resolve_open_market_universe


class _StateStoreNoStationPauseCalls:
    def is_global_kill(self) -> bool:
        return False

    def is_station_paused(self, station: str) -> bool:
        raise AssertionError("station-level kill switch should not be consulted")

    def open_positions(self):
        return []

    def position_identity_key(self, *, station, market_day_local, strike_k):
        day = pd.to_datetime(market_day_local).date().isoformat()
        return f"{station}|{day}|{int(strike_k)}"

    def station_risk_used(self, *, day_local: str, station: str) -> float:
        return 0.0

    def station_open_risk(self, *, day_local: str, station: str) -> float:
        return 0.0

    def portfolio_risk_used(self, *, day_local: str) -> float:
        return 0.0

    def portfolio_open_risk(self, *, day_local: str) -> float:
        return 0.0

    def is_trade_cooldown_active(self, position_identity_key: str, *, cooldown_minutes: float, now_utc: datetime) -> bool:
        return False

    def open_position_count_for_station(self, station: str) -> int:
        return 0

    def open_position_count(self) -> int:
        return 0


def _base_policy_context() -> PolicyContext:
    return PolicyContext(
        nav_usd=1000.0,
        nav_peak_usd=1000.0,
        mode_distance_min=2,
        p_model_max=0.12,
        edge_threshold=0.02,
        max_no_price=0.92,
        top_n_per_event_day=1,
        stake_fraction=0.01,
        stake_cap_usd=25.0,
        min_order_size=1.0,
        station_daily_risk_fraction=1.0,
        portfolio_daily_risk_fraction=1.0,
        max_open_positions_per_station=10,
        max_open_positions_total=100,
        trade_cooldown_minutes=5.0,
        drawdown_position_scaling=False,
        max_drawdown_fraction=0.2,
        min_drawdown_scale=0.25,
        trade_window_start_local="00:00",
        trade_window_end_local="23:59",
        use_progression_confidence=False,
        progression_enable_gate=False,
        use_ensemble_confidence=False,
        ensemble_enable_gate=False,
        ensemble_trade_size_adjustment_enabled=False,
    )


def test_apply_policy_does_not_check_station_pause() -> None:
    candidates = pd.DataFrame(
        [
            {
                "station": "Dallas",
                "market_day_local": "2026-03-08",
                "event_key": "highest-temperature-in-dallas-on-march-8-2026",
                "strike_k": 11,
                "mode_k": 14,
                "p_model": 0.05,
                "chosen_no_ask": 0.70,
                "snapshot_skip_reason": "",
                "slug": "highest-temperature-in-dallas-on-march-8-2026-11c",
                "market_id": "2001",
            }
        ]
    )

    out = apply_policy(
        candidates=candidates,
        state_store=_StateStoreNoStationPauseCalls(),
        ctx=_base_policy_context(),
        now_utc=datetime(2026, 3, 8, 9, 0, tzinfo=timezone.utc),
        station_timezones={"Dallas": "UTC"},
    )

    assert len(out) == 1
    assert str(out.iloc[0]["skipped_reason"]) != "kill_switch_active"


def test_resolve_open_market_universe_dedupes_open_market_keys(monkeypatch, caplog) -> None:
    universe = pd.DataFrame(
        [
            {
                "station": "Dallas",
                "market_day_local": "2026-03-08",
                "event_key": "highest-temperature-in-dallas-on-march-8-2026",
                "strike_k": 11,
                "market_id": "9999",
                "slug": "highest-temperature-in-dallas-on-march-8-2026-11c",
            }
        ]
    )

    open_rows = pd.DataFrame(
        [
            {
                "market_id": "2001",
                "slug": "highest-temperature-in-dallas-on-march-8-2026-11c",
                "asset_id": "asset-old",
                "end_date_utc": "2026-03-08T18:00:00Z",
                "resolution_time": "2026-03-08T20:00:00Z",
            },
            {
                "market_id": "2002",
                "slug": "highest-temperature-in-dallas-on-march-8-2026-11c",
                "asset_id": "asset-new",
                "end_date_utc": "2026-03-08T18:00:00Z",
                "resolution_time": "2026-03-08T21:00:00Z",
            },
        ]
    )

    monkeypatch.setattr(
        "live_trading.run_live_pilot.dbmod.fetch_open_weather_markets",
        lambda conn, statuses: open_rows,
    )

    caplog.set_level(logging.WARNING)
    mapped, unmapped = resolve_open_market_universe(
        universe=universe,
        conn=None,
        station_tz={"Dallas": "UTC"},
        cfg={"timezones": {"default": "UTC"}},
        logger=logging.getLogger("test.live.mapping"),
    )

    assert len(mapped) == 1
    assert unmapped.empty
    assert str(mapped.iloc[0]["market_id"]) == "2002"
    assert any("duplicate station/day/event/strike keys" in rec.message for rec in caplog.records)


def test_resolve_open_market_universe_dedupes_candidate_rows_before_mapping(monkeypatch, caplog) -> None:
    universe = pd.DataFrame(
        [
            {
                "station": "Dallas",
                "market_day_local": "2026-03-08",
                "event_key": "highest-temperature-in-dallas-on-march-8-2026",
                "strike_k": 11,
                "market_id": "9999",
                "slug": "highest-temperature-in-dallas-on-march-8-2026-legacy-11c",
            },
            {
                "station": "Dallas",
                "market_day_local": "2026-03-08",
                "event_key": "highest-temperature-in-dallas-on-march-8-2026",
                "strike_k": 11,
                "market_id": "",
                "slug": "highest-temperature-in-dallas-on-march-8-2026-11c",
            },
        ]
    )

    open_rows = pd.DataFrame(
        [
            {
                "market_id": "3001",
                "slug": "highest-temperature-in-dallas-on-march-8-2026-11c",
                "asset_id": "asset-3001",
                "end_date_utc": "2026-03-08T18:00:00Z",
                "resolution_time": "2026-03-08T20:00:00Z",
            }
        ]
    )

    monkeypatch.setattr(
        "live_trading.run_live_pilot.dbmod.fetch_open_weather_markets",
        lambda conn, statuses: open_rows,
    )

    caplog.set_level(logging.WARNING)
    mapped, unmapped = resolve_open_market_universe(
        universe=universe,
        conn=None,
        station_tz={"Dallas": "UTC"},
        cfg={"timezones": {"default": "UTC"}},
        logger=logging.getLogger("test.live.mapping"),
    )

    assert len(mapped) == 1
    assert unmapped.empty
    assert str(mapped.iloc[0]["market_id"]) == "3001"
    assert any("Candidate universe contains" in rec.message for rec in caplog.records)


def test_resolve_open_market_universe_maps_by_canonical_key_when_ids_are_stale(monkeypatch) -> None:
    universe = pd.DataFrame(
        [
            {
                "station": "Dallas",
                "market_day_local": "2026-03-08",
                "event_key": "highest-temperature-in-dallas-on-march-8-2026",
                "strike_k": 11,
                "market_id": "9999",
                "slug": "highest-temperature-in-dallas-on-march-8-2026-retired-11c",
            }
        ]
    )

    open_rows = pd.DataFrame(
        [
            {
                "market_id": "4001",
                "slug": "highest-temperature-in-dallas-on-march-8-2026-11c",
                "asset_id": "asset-4001",
                "end_date_utc": "2026-03-08T18:00:00Z",
                "resolution_time": "2026-03-08T20:00:00Z",
            }
        ]
    )

    monkeypatch.setattr(
        "live_trading.run_live_pilot.dbmod.fetch_open_weather_markets",
        lambda conn, statuses: open_rows,
    )

    mapped, unmapped = resolve_open_market_universe(
        universe=universe,
        conn=None,
        station_tz={"Dallas": "UTC"},
        cfg={"timezones": {"default": "UTC"}},
        logger=logging.getLogger("test.live.mapping"),
    )

    assert len(mapped) == 1
    assert unmapped.empty
    assert str(mapped.iloc[0]["market_id"]) == "4001"
