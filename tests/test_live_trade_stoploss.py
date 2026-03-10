from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from live_trading.db import SnapshotTableInfo
from live_trading.execution import DummyExecutionClient
from live_trading.run_live_pilot import _apply_trade_stoplosses
from live_trading.state import PilotStateStore
from live_trading.telegram_notify import format_trade_message
from live_trading.utils_time import today_local


class _TestNotifier:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def notify_trade(self, payload: dict[str, object], *, logger: object) -> None:
        self.payloads.append(dict(payload))


def _snapshot_df(*, market_id: str, no_bid: float, ts: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_id": market_id,
                "yes_snapshot_ts_utc": pd.Timestamp(ts),
                "no_snapshot_ts_utc": pd.Timestamp(ts),
                "best_yes_bid": 1.0 - no_bid - 0.01,
                "best_yes_ask": 1.0 - no_bid + 0.01,
                "best_no_bid": no_bid,
                "best_no_ask": no_bid + 0.01,
                "yes_bid_size": 100.0,
                "yes_ask_size": 100.0,
                "no_bid_size": 100.0,
                "no_ask_size": 100.0,
            }
        ]
    )


def test_apply_trade_stoploss_executes_sell_at_loss_threshold(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
    state = PilotStateStore(state_dir=tmp_path / "state", nav_usd=1000.0)
    try:
        state.add_open_position(
            {
                "station": "Dallas",
                "market_day_local": "2026-03-10",
                "market_id": "m-1",
                "slug": "highest-temperature-in-dallas-on-march-10-2026-20c",
                "asset_id": "asset-1",
                "strike_k": 20,
                "mode_k": 22,
                "p_model": 0.1,
                "entry_price": 0.80,
                "size": 10.0,
                "stake_usd": 8.0,
                "edge_at_entry": 0.03,
                "price_source": "NO_ask",
                "stop_loss_enabled": True,
                "stop_loss_loss_fraction": 0.25,
                "stop_loss_break_even_on_recovery": True,
                "stop_loss_break_even_armed": False,
                "stop_loss_trigger_price": 0.60,
            }
        )

        monkeypatch.setattr(
            "live_trading.run_live_pilot.dbmod.fetch_latest_snapshots",
            lambda conn, snapshot_table, market_ids, **kwargs: _snapshot_df(market_id="m-1", no_bid=0.59, ts=now),
        )

        notifier = _TestNotifier()
        exec_client = DummyExecutionClient(
            realism_enabled=False,
            conservative_fill=False,
            price_tick=0.001,
        )

        _apply_trade_stoplosses(
            cfg={
                "trade_stoploss": {
                    "enabled": True,
                    "loss_fraction": 0.25,
                    "break_even_on_recovery": True,
                },
                "max_snapshot_age_minutes": 30,
                "timezones": {"default": "UTC"},
            },
            state_store=state,
            conn=None,
            snapshot_info=SnapshotTableInfo(table_name="book_snapshots"),
            exec_client=exec_client,
            dry_run=False,
            run_id="test-stoploss",
            logger=logging.getLogger("test.stoploss"),
            jsonl_path=None,
            notifier=notifier,
            now_utc=now,
        )

        position = state.open_positions()[0]
        assert str(position.get("status")) == "closed"
        assert len(notifier.payloads) == 1
        assert notifier.payloads[0].get("sell_reason") == "stop_loss"
        assert notifier.payloads[0].get("decision") == "SELL"
        assert abs(float(position.get("pnl_realized")) + 2.1) < 1e-9
        assert abs(state.nav_usd - 997.9) < 1e-9
        assert abs(state.daily_realized_pnl(today_local("UTC", now_utc=now).isoformat()) + 2.1) < 1e-9
    finally:
        state.close()


def test_apply_trade_stoploss_break_even_rearm_then_sell(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
    state = PilotStateStore(state_dir=tmp_path / "state", nav_usd=1000.0)
    try:
        state.add_open_position(
            {
                "station": "Dallas",
                "market_day_local": "2026-03-10",
                "market_id": "m-2",
                "slug": "highest-temperature-in-dallas-on-march-10-2026-21c",
                "asset_id": "asset-2",
                "strike_k": 21,
                "mode_k": 23,
                "p_model": 0.1,
                "entry_price": 0.50,
                "size": 8.0,
                "stake_usd": 4.0,
                "edge_at_entry": 0.03,
                "price_source": "NO_ask",
                "stop_loss_enabled": True,
                "stop_loss_loss_fraction": 0.25,
                "stop_loss_break_even_on_recovery": True,
                "stop_loss_break_even_armed": False,
                "stop_loss_trigger_price": 0.375,
            }
        )

        snapshots = [
            _snapshot_df(market_id="m-2", no_bid=0.55, ts=now),
            _snapshot_df(market_id="m-2", no_bid=0.49, ts=now),
        ]

        def _fetch_latest_snapshots(conn, snapshot_table, market_ids, **kwargs):
            return snapshots.pop(0)

        monkeypatch.setattr(
            "live_trading.run_live_pilot.dbmod.fetch_latest_snapshots",
            _fetch_latest_snapshots,
        )

        notifier = _TestNotifier()
        exec_client = DummyExecutionClient(
            realism_enabled=False,
            conservative_fill=False,
            price_tick=0.001,
        )

        base_cfg = {
            "trade_stoploss": {
                "enabled": True,
                "loss_fraction": 0.25,
                "break_even_on_recovery": True,
            },
            "max_snapshot_age_minutes": 30,
            "timezones": {"default": "UTC"},
        }

        _apply_trade_stoplosses(
            cfg=base_cfg,
            state_store=state,
            conn=None,
            snapshot_info=SnapshotTableInfo(table_name="book_snapshots"),
            exec_client=exec_client,
            dry_run=False,
            run_id="test-break-even-1",
            logger=logging.getLogger("test.stoploss"),
            jsonl_path=None,
            notifier=notifier,
            now_utc=now,
        )

        first_pass_position = state.open_positions()[0]
        assert str(first_pass_position.get("status")) == "open"
        assert bool(first_pass_position.get("stop_loss_break_even_armed")) is True
        assert float(first_pass_position.get("stop_loss_trigger_price")) == 0.5
        assert len(notifier.payloads) == 0

        _apply_trade_stoplosses(
            cfg=base_cfg,
            state_store=state,
            conn=None,
            snapshot_info=SnapshotTableInfo(table_name="book_snapshots"),
            exec_client=exec_client,
            dry_run=False,
            run_id="test-break-even-2",
            logger=logging.getLogger("test.stoploss"),
            jsonl_path=None,
            notifier=notifier,
            now_utc=now,
        )

        final_position = state.open_positions()[0]
        assert str(final_position.get("status")) == "closed"
        assert len(notifier.payloads) == 1
        assert notifier.payloads[0].get("sell_reason") == "stop_loss"
        assert abs(float(final_position.get("pnl_realized")) + 0.08) < 1e-9
    finally:
        state.close()


def test_format_trade_message_stop_loss_template() -> None:
    message = format_trade_message(
        {
            "decision": "SELL",
            "sell_reason": "stop_loss",
            "station": "Dallas",
            "market_day_local": "2026-03-10",
            "strike_k": 20,
            "buy_price": 0.80,
            "sell_price": 0.59,
            "size": 10.0,
            "pnl_realized": -2.1,
        }
    )
    assert message is not None
    assert "Stop Loss" in message
    assert "Buy Price: 0.800" in message
    assert "Sell Price: 0.590" in message
    assert "Lot: 10.00" in message
    assert "Total Loss: 2.10" in message
