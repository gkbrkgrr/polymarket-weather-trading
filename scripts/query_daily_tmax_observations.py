#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from master_db import get_daily_tmax_by_station, resolve_master_postgres_dsn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query daily station Tmax observations from master_db and print JSON records."
    )
    parser.add_argument(
        "--station",
        action="append",
        default=[],
        help="Station/city name to filter by. May be repeated.",
    )
    parser.add_argument(
        "--obs-dsn",
        default=None,
        help="Optional master_db DSN override.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dsn = resolve_master_postgres_dsn(explicit_dsn=args.obs_dsn)
    stations = args.station if args.station else None

    out = get_daily_tmax_by_station(stations=stations, master_dsn=dsn)
    if out.empty:
        sys.stdout.write("[]")
        return 0

    out = out.loc[:, ["city_name", "target_date_local", "tmax_obs_c"]].copy()
    out["target_date_local"] = pd.to_datetime(out["target_date_local"], errors="coerce").dt.date
    out["target_date_local"] = out["target_date_local"].astype("string")
    out["tmax_obs_c"] = pd.to_numeric(out["tmax_obs_c"], errors="coerce")
    out = out.dropna(subset=["city_name", "target_date_local", "tmax_obs_c"])

    records = out.to_dict(orient="records")
    sys.stdout.write(json.dumps(records, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
