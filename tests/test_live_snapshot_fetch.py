from __future__ import annotations

from datetime import datetime, timezone

from live_trading.db import SnapshotTableInfo, fetch_latest_snapshots


class _CursorDescription:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCursor:
    def __init__(self, *, rows: list[tuple[object, ...]], columns: list[str]) -> None:
        self._rows = rows
        self.description = [_CursorDescription(c) for c in columns]
        self.params: dict[str, object] | None = None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query, params) -> None:
        self.params = dict(params or {})

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._rows)


class _FakeConnection:
    def __init__(self, *, rows: list[tuple[object, ...]], columns: list[str]) -> None:
        self._cursor = _FakeCursor(rows=rows, columns=columns)

    def cursor(self) -> _FakeCursor:
        return self._cursor


def test_fetch_latest_snapshots_prefers_latest_usable_quote_rows() -> None:
    market_id = "m-1"
    ts_latest = datetime(2026, 3, 10, 18, 40, tzinfo=timezone.utc)
    ts_usable = datetime(2026, 3, 10, 18, 34, tzinfo=timezone.utc)

    columns = ["market_id", "ts", "outcome_index", "best_bid", "best_ask", "bid_size", "ask_size"]
    rows = [
        (market_id, ts_latest, 0, None, 0.004, 100.0, 100.0),
        (market_id, ts_usable, 0, 0.001, 0.004, 100.0, 100.0),
        (market_id, ts_latest, 1, 0.996, None, 100.0, 100.0),
        (market_id, ts_usable, 1, 0.996, 0.999, 100.0, 100.0),
    ]
    conn = _FakeConnection(rows=rows, columns=columns)

    out = fetch_latest_snapshots(
        conn,
        snapshot_table=SnapshotTableInfo(table_name="book_snapshots"),
        market_ids=[market_id],
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert str(row["market_id"]) == market_id
    assert row["yes_snapshot_ts_utc"].to_pydatetime() == ts_usable
    assert row["no_snapshot_ts_utc"].to_pydatetime() == ts_usable
    assert float(row["best_yes_bid"]) == 0.001
    assert float(row["best_no_ask"]) == 0.999


def test_fetch_latest_snapshots_passes_lookback_param() -> None:
    columns = ["market_id", "ts", "outcome_index", "best_bid", "best_ask", "bid_size", "ask_size"]
    conn = _FakeConnection(rows=[], columns=columns)

    fetch_latest_snapshots(
        conn,
        snapshot_table=SnapshotTableInfo(table_name="book_snapshots"),
        market_ids=["m-2"],
        lookback_per_outcome=7,
    )

    assert conn.cursor().params is not None
    assert int(conn.cursor().params.get("lookback")) == 7
