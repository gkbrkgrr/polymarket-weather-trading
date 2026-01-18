# Polymarket Highest Temperature Markets Archive

Production-grade archive for Polymarket markets whose titles include "Highest temperature" (case-insensitive). It initializes PostgreSQL schema, backfills trades from 2026-01-01T00:00:00Z, and keeps polling for new markets and trades with idempotent writes.

## Features
- Discovery of matching markets (open, closed, resolved all included)
- Backfill from 2026-01-01T00:00:00Z to now, then continuous live polling
- Async HTTP with retries, exponential backoff, jitter, and strict timeouts
- Per-market cursors for incremental trade ingestion
- Deterministic JSONL.gz raw archive for every API response
- PostgreSQL schema with conflict-safe upserts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:
```bash
pip install -e ".[parquet]"
pip install -e ".[test]"
```

## PostgreSQL

Set `POSTGRES_DSN` and initialize the schema:

```bash
export POSTGRES_DSN=postgresql://archive_user:password@127.0.0.1:5432/polymarket_archive
polymarket_archive init-db
```

## Configuration

Configuration can come from environment variables or a YAML/JSON file passed via `--config`.

Supported environment variables:
- `POSTGRES_DSN`
- `GAMMA_BASE_URL` (default `https://gamma-api.polymarket.com`)
- `DATA_BASE_URL` (default `https://data-api.polymarket.com`)
- `CLOB_WS_URL` (default `wss://ws-subscriptions-clob.polymarket.com/ws`)
- `CLOB_BASE_URL` (default `https://clob.polymarket.com`)
- `TITLE_FILTER` (default `Highest temperature`)
- `BACKFILL_START` (default `2026-01-01T00:00:00Z`)
- `POLL_INTERVAL_SECONDS` (default `30`)
- `DISCOVERY_INTERVAL_SECONDS` (default `300`)
- `CONCURRENCY` (default `10`)
- `BOOK_SNAPSHOT_INTERVAL_SECONDS` (default `5`)
- `RAW_DIR` (default `./data/raw`)
- `FEATURE_CLOB` (default `false`)

Example `config.yaml`:
```yaml
postgres_dsn: postgresql://archive_user:password@127.0.0.1:5432/polymarket_archive
gamma_base_url: https://gamma-api.polymarket.com
data_base_url: https://data-api.polymarket.com
clob_ws_url: wss://ws-subscriptions-clob.polymarket.com/ws
clob_base_url: https://clob.polymarket.com
title_filter: Highest temperature
backfill_start: 2026-01-01T00:00:00Z
poll_interval_seconds: 30
discovery_interval_seconds: 300
concurrency: 10
book_snapshot_interval_seconds: 5
raw_dir: ./data/raw
feature_clob: false
```

## Usage

Backfill and run live polling:
```bash
polymarket_archive run --config config.yaml
```

Backfill only:
```bash
polymarket_archive backfill --start 2026-01-01T00:00:00Z --end now
```

Live mode only:
```bash
polymarket_archive run-live
```

Enable CLOB top-of-book snapshots (optional, requires `websockets`):
```bash
pip install -e ".[clob]"
FEATURE_CLOB=true polymarket_archive run-live
```

If the websocket endpoint is unavailable (HTTP 404), the runner will fall back to polling the REST `GET /book` endpoint for each token id.

## Raw archive layout
- `data/raw/gamma_discovery/YYYY/MM/DD/HH/discovery_YYYYMMDD_HH.jsonl.gz`
- `data/raw/gamma_markets/YYYY/MM/DD/HH/gamma_markets_YYYYMMDD_HH.jsonl.gz`
- `data/raw/data_trades/YYYY/MM/DD/HH/market_id=<id>/trades_YYYYMMDD_HH.jsonl.gz`
- `data/raw/errors/YYYY/MM/DD/HH/errors_YYYYMMDD_HH.jsonl.gz`

Each JSONL record contains:
- `ingest_ts` (UTC ISO8601)
- `source` (`gamma_discovery`, `gamma_markets`, `data_trades`, `clob_book`, `error`)
- `request` (url, params, headers_redacted, cursor/page info)
- `payload` (raw API response)
- `run_id`
- `market_id` when applicable

## Tests

```bash
POSTGRES_DSN=postgresql://archive_user:password@127.0.0.1:5432/polymarket_archive pytest -q
```

If `POSTGRES_DSN` is not set, database-backed tests are skipped.
