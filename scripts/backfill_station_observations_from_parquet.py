#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from master_db import ensure_master_db_ready, upsert_station_observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill station_observations in master_db from legacy per-station parquet files."
        )
    )
    parser.add_argument(
        "--obs-dir",
        type=Path,
        default=REPO_ROOT / "data" / "observations",
        help="Directory containing legacy per-station parquet files.",
    )
    parser.add_argument(
        "--master-dsn",
        type=str,
        default=None,
        help="Optional DSN for master_db.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Rows per batch for parquet->DB upsert.",
    )
    parser.add_argument(
        "--only-station",
        action="append",
        default=[],
        help="Restrict to station stem(s) from parquet filenames; repeatable.",
    )
    return parser.parse_args()


def _to_records(rows: list[dict[str, Any]], station_fallback: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        station = str(row.get("station") or station_fallback).strip()
        observed_at_local = row.get("observed_at_local")
        scraped_at_utc = row.get("scraped_at_utc")
        if not station or observed_at_local is None or scraped_at_utc is None:
            continue
        out.append(
            {
                "station": station,
                "observed_at_local": observed_at_local,
                "temperature_f": row.get("temperature_f"),
                "temperature_c": row.get("temperature_c"),
                "precipitation_hourly_in": row.get("precipitation_hourly_in"),
                "precipitation_total_in": row.get("precipitation_total_in"),
                "scraped_at_utc": scraped_at_utc,
            }
        )
    return out


def main() -> int:
    args = parse_args()
    obs_dir = args.obs_dir.expanduser().resolve()
    if not obs_dir.exists():
        raise SystemExit(f"Observation directory not found: {obs_dir}")

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    dsn = ensure_master_db_ready(master_dsn=args.master_dsn)
    selected = {s.strip().casefold() for s in args.only_station if s.strip()}

    parquet_files = sorted(obs_dir.glob("*.parquet"))
    if selected:
        parquet_files = [p for p in parquet_files if p.stem.casefold() in selected]

    if not parquet_files:
        raise SystemExit(f"No parquet files selected under {obs_dir}")

    total_inserted = 0
    total_updated = 0
    total_unchanged = 0
    total_rows = 0

    for path in parquet_files:
        station_hint = path.stem
        pf = pq.ParquetFile(path)
        file_inserted = 0
        file_updated = 0
        file_unchanged = 0
        file_rows = 0

        for batch in pf.iter_batches(batch_size=args.batch_size):
            rows = batch.to_pylist()
            records = _to_records(rows, station_hint)
            if not records:
                continue
            stats = upsert_station_observations(records=records, master_dsn=dsn)
            n = len(records)
            file_rows += n
            file_inserted += stats["inserted"]
            file_updated += stats["updated"]
            file_unchanged += stats["unchanged"]

        total_rows += file_rows
        total_inserted += file_inserted
        total_updated += file_updated
        total_unchanged += file_unchanged
        print(
            f"{path.name}: rows={file_rows} inserted={file_inserted} "
            f"updated={file_updated} unchanged={file_unchanged}"
        )

    print(
        f"TOTAL: files={len(parquet_files)} rows={total_rows} inserted={total_inserted} "
        f"updated={total_updated} unchanged={total_unchanged}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

