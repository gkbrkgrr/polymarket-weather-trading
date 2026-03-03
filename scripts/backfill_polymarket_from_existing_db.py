#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import psycopg
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from master_db import ensure_master_db_ready, resolve_master_postgres_dsn


DEFAULT_SOURCE_DSN = "postgresql://archive_user:password@127.0.0.1:5432/polymarket_archive"


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: tuple[str, ...]
    merge_sql: str


TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        name="markets",
        columns=(
            "market_id",
            "slug",
            "title",
            "status",
            "event_start_time",
            "resolution_time",
            "raw",
            "updated_at",
        ),
        merge_sql=(
            "INSERT INTO markets (market_id, slug, title, status, event_start_time, resolution_time, raw, updated_at) "
            "SELECT market_id, slug, title, status, event_start_time, resolution_time, raw, updated_at FROM {stg} "
            "ON CONFLICT (market_id) DO UPDATE SET "
            "slug = EXCLUDED.slug, "
            "title = EXCLUDED.title, "
            "status = EXCLUDED.status, "
            "event_start_time = EXCLUDED.event_start_time, "
            "resolution_time = EXCLUDED.resolution_time, "
            "raw = EXCLUDED.raw, "
            "updated_at = EXCLUDED.updated_at"
        ),
    ),
    TableSpec(
        name="outcomes",
        columns=("market_id", "outcome_id", "outcome_label", "outcome_index", "raw"),
        merge_sql=(
            "INSERT INTO outcomes (market_id, outcome_id, outcome_label, outcome_index, raw) "
            "SELECT market_id, outcome_id, outcome_label, outcome_index, raw FROM {stg} "
            "ON CONFLICT (market_id, outcome_index) DO UPDATE SET "
            "outcome_id = EXCLUDED.outcome_id, "
            "outcome_label = EXCLUDED.outcome_label, "
            "raw = EXCLUDED.raw"
        ),
    ),
    TableSpec(
        name="trades",
        columns=(
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
        ),
        merge_sql=(
            "INSERT INTO trades (trade_id, market_id, ts, outcome_id, outcome_index, side, price, size, tx_hash, raw) "
            "SELECT trade_id, market_id, ts, outcome_id, outcome_index, side, price, size, tx_hash, raw FROM {stg} "
            "ON CONFLICT (trade_id) DO NOTHING"
        ),
    ),
    TableSpec(
        name="cursors",
        columns=("market_id", "last_ts", "last_tiebreak", "updated_at"),
        merge_sql=(
            "INSERT INTO cursors (market_id, last_ts, last_tiebreak, updated_at) "
            "SELECT market_id, last_ts, last_tiebreak, updated_at FROM {stg} "
            "ON CONFLICT (market_id) DO UPDATE SET "
            "last_ts = EXCLUDED.last_ts, "
            "last_tiebreak = EXCLUDED.last_tiebreak, "
            "updated_at = EXCLUDED.updated_at"
        ),
    ),
    TableSpec(
        name="book_snapshots",
        columns=(
            "market_id",
            "ts",
            "outcome_index",
            "best_bid",
            "best_ask",
            "bid_size",
            "ask_size",
            "raw",
        ),
        merge_sql=(
            "INSERT INTO book_snapshots (market_id, ts, outcome_index, best_bid, best_ask, bid_size, ask_size, raw) "
            "SELECT market_id, ts, outcome_index, best_bid, best_ask, bid_size, ask_size, raw FROM {stg} "
            "ON CONFLICT (market_id, ts, outcome_index) DO NOTHING"
        ),
    ),
)


def _load_repo_config() -> dict:
    cfg_path = REPO_ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def resolve_source_dsn(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = sys.modules.get("os")
    # lazy import keeps startup minimal
    import os

    if os.getenv("SOURCE_POSTGRES_DSN"):
        return os.environ["SOURCE_POSTGRES_DSN"]
    cfg = _load_repo_config()
    from_cfg = cfg.get("postgres_dsn")
    if isinstance(from_cfg, str) and from_cfg.strip():
        return from_cfg.strip()
    return DEFAULT_SOURCE_DSN


def _copy_table_to_staging(src: psycopg.Connection, tgt: psycopg.Connection, spec: TableSpec, stg_name: str) -> int:
    cols = ", ".join(spec.columns)
    with tgt.cursor() as tcur:
        tcur.execute(f"DROP TABLE IF EXISTS {stg_name}")
        tcur.execute(f"CREATE TEMP TABLE {stg_name} AS SELECT {cols} FROM {spec.name} WHERE false")

    copy_out_sql = f"COPY (SELECT {cols} FROM {spec.name}) TO STDOUT WITH (FORMAT BINARY)"
    copy_in_sql = f"COPY {stg_name} ({cols}) FROM STDIN WITH (FORMAT BINARY)"

    with src.cursor() as scur:
        with scur.copy(copy_out_sql) as copy_out:
            with tgt.cursor() as tcur:
                with tcur.copy(copy_in_sql) as copy_in:
                    for chunk in copy_out:
                        copy_in.write(chunk)

    with tgt.cursor() as tcur:
        tcur.execute(f"SELECT COUNT(*) FROM {stg_name}")
        return int(tcur.fetchone()[0])


def _merge_staging(tgt: psycopg.Connection, spec: TableSpec, stg_name: str) -> int:
    sql = spec.merge_sql.format(stg=stg_name)
    with tgt.cursor() as tcur:
        tcur.execute(sql)
        return int(tcur.rowcount or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill Polymarket tables into master_db by copying from an existing Polymarket PostgreSQL DB."
        )
    )
    parser.add_argument("--source-dsn", default=None, help="Source Polymarket PostgreSQL DSN.")
    parser.add_argument("--target-dsn", default=None, help="Target master_db PostgreSQL DSN.")
    parser.add_argument(
        "--tables",
        nargs="*",
        default=[spec.name for spec in TABLE_SPECS],
        help="Subset of tables to copy (default: all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dsn = resolve_source_dsn(args.source_dsn)
    target_dsn = ensure_master_db_ready(master_dsn=resolve_master_postgres_dsn(explicit_dsn=args.target_dsn))

    requested = {name.strip() for name in args.tables if name.strip()}
    specs = [spec for spec in TABLE_SPECS if spec.name in requested]
    if not specs:
        raise SystemExit("No valid tables selected.")

    totals: dict[str, tuple[int, int]] = {}

    with psycopg.connect(source_dsn) as src, psycopg.connect(target_dsn) as tgt:
        for spec in specs:
            stg_name = f"_stg_{spec.name}"
            staged = _copy_table_to_staging(src, tgt, spec, stg_name)
            merged = _merge_staging(tgt, spec, stg_name)
            tgt.commit()
            totals[spec.name] = (staged, merged)
            print(f"{spec.name}: staged={staged} merged={merged}")

    print("SUMMARY")
    for name in [spec.name for spec in specs]:
        staged, merged = totals[name]
        skipped = staged - merged
        print(f"{name}: staged={staged} merged={merged} skipped={skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
