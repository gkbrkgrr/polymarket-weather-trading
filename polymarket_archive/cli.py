from __future__ import annotations

import argparse
import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from polymarket_archive.config import load_settings
from polymarket_archive.db import Database
from polymarket_archive.jobs import run_backfill, run_live_loop
from polymarket_archive.log import configure_logging
from polymarket_archive.utils import parse_datetime


def _schema_path() -> Path:
    return Path(__file__).resolve().parent / "ddl" / "schema.sql"


def _parse_time(value: str) -> datetime:
    if value == "now":
        return datetime.now(timezone.utc)
    parsed = parse_datetime(value)
    if parsed is None:
        raise ValueError(f"Invalid timestamp: {value}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(prog="polymarket_archive")
    parser.add_argument("--config", dest="config_path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db")

    backfill = subparsers.add_parser("backfill")
    backfill.add_argument("--start", default="2026-01-01T00:00:00Z")
    backfill.add_argument("--end", default="now")

    subparsers.add_parser("run-live")
    subparsers.add_parser("run")

    subparsers.add_parser("test")

    export = subparsers.add_parser("export-parquet")
    export.add_argument("--date", required=True)
    export.add_argument("--output", default="data/exports")

    args = parser.parse_args()

    if args.command == "test":
        raise SystemExit(subprocess.call(["pytest", "-q"]))

    settings = load_settings(args.config_path)
    configure_logging(settings.log_level)

    db = Database(settings.postgres_dsn)

    if args.command == "init-db":
        asyncio.run(_init_db(db))
        return

    if args.command == "backfill":
        start_ts = _parse_time(args.start)
        end_ts = _parse_time(args.end) if args.end != "now" else datetime.now(timezone.utc)
        asyncio.run(_run_backfill(db, settings, start_ts, end_ts))
        return

    if args.command == "run-live":
        asyncio.run(_run_live(db, settings))
        return

    if args.command == "run":
        start_ts = settings.backfill_start
        end_ts = datetime.now(timezone.utc)
        asyncio.run(_run_full(db, settings, start_ts, end_ts))
        return

    if args.command == "export-parquet":
        asyncio.run(_export_parquet(db, settings, args.date, args.output))
        return


def _init_db(db: Database) -> asyncio.Future:
    async def _inner() -> None:
        await db.open()
        try:
            await db.init_db(_schema_path())
        finally:
            await db.close()

    return _inner()


def _run_backfill(db: Database, settings, start_ts, end_ts) -> asyncio.Future:
    async def _inner() -> None:
        await db.open()
        try:
            await db.ensure_schema(_schema_path())
            await run_backfill(settings, db, start_ts, end_ts)
        finally:
            await db.close()

    return _inner()


def _run_live(db: Database, settings) -> asyncio.Future:
    async def _inner() -> None:
        await db.open()
        try:
            await db.ensure_schema(_schema_path())
            await run_live_loop(settings, db)
        finally:
            await db.close()

    return _inner()


def _run_full(db: Database, settings, start_ts, end_ts) -> asyncio.Future:
    async def _inner() -> None:
        await db.open()
        try:
            await db.ensure_schema(_schema_path())
            await run_backfill(settings, db, start_ts, end_ts)
            await run_live_loop(settings, db)
        finally:
            await db.close()

    return _inner()


async def _export_parquet(db: Database, settings, date_str: str, output: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required for export-parquet") from exc

    target_date = datetime.fromisoformat(date_str)
    start_ts = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    end_ts = start_ts.replace(hour=23, minute=59, second=59)

    await db.open()
    try:
        async with db.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT trade_id, market_id, ts, outcome_id, outcome_index, side, price, size, tx_hash, raw "
                    "FROM trades WHERE ts >= %s AND ts <= %s",
                    (start_ts, end_ts),
                )
                rows = await cur.fetchall()
        columns = [
            "trade_id",
            "market_id",
            "ts",
            "outcome_id",
            "outcome_index",
            "side",
            "price",
            "size",
            "tx_hash",
            "raw",
        ]
        table = pa.Table.from_pylist([dict(zip(columns, row)) for row in rows])
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path / f"trades_{date_str}.parquet")
    finally:
        await db.close()


if __name__ == "__main__":
    main()
