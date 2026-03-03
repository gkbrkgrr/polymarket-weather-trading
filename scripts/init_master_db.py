#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from master_db import ensure_master_db_ready, resolve_master_postgres_dsn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create master_db and apply unified schema.")
    parser.add_argument(
        "--master-dsn",
        default=None,
        help="Optional DSN for master_db (defaults to MASTER_POSTGRES_DSN/config-derived value).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "migrations" / "master_db_schema.sql",
        help="Schema SQL path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dsn = resolve_master_postgres_dsn(explicit_dsn=args.master_dsn)
    ensure_master_db_ready(master_dsn=dsn, schema_sql_path=args.schema)
    print(f"master_db initialized using DSN: {dsn}")
    print(f"Applied schema: {args.schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
