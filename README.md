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
export POSTGRES_DSN=postgresql://archive_user:password@127.0.0.1:5432/master_db
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
postgres_dsn: postgresql://archive_user:password@127.0.0.1:5432/master_db
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
POSTGRES_DSN=postgresql://archive_user:password@127.0.0.1:5432/master_db pytest -q
```

If `POSTGRES_DSN` is not set, database-backed tests are skipped.

## Bias Correction Pipeline

The repo now includes a production-oriented, leakage-safe bias-correction pipeline for daily Tmax prediction archives:

- CLI entrypoint: `python bias_correction/run_bias_correction.py`
- Config example: `bias_correction/config.example.yaml`
- Artifacts/logs: `bias_correction/artifacts/<run_timestamp>/`

### Expected Input Schema

Prediction parquet files can vary by model, but each file must be normalizable to:

- `station_name` (aliases: `station`, `Station`, `station_id`; if missing, fallback derived from `station_lat/station_lon`, otherwise city)
- `city` (aliases: `City`, `city_name`)
- `issue_time_utc` (aliases: `InitTimeUTC`, `init_time`, `cycle`, `issue_time`)
- `target_date` day label (aliases: `valid_date_local`, `target_date_local`, `valid_time_local`; if only UTC valid time exists, UTC/local fallback is applied with warning)
- `lead_hours` (aliases: `LeadHour`, `lead_time_hours`)
- `cycle` (derived from `issue_time_utc` hour if missing)
- `tmax_pred` prediction (auto-detected from common names such as `tmax_pred`, `prediction`, `yhat`, `Forecast`)
- `tmax_obs` observations (`tmax_obs`, `tmax_obs_c`, etc.) or loaded from `master_db.station_observations` (override via `--obs_source_dsn`)

### Leakage-Safety Design

- Stage A (EWMA bias) is strictly online.
- For issue time `T`, correction uses only residual history from issue times `< T`.
- Rows sharing the same issue time are scored first, then history is updated.
- Stage B is trained on Stage-A-corrected residual targets with time-based validation split.

### Outputs

- Output folders are sibling model directories with suffix `_biascorrected`:
  - `data/ml_predictions/xgb_biascorrected/...`
  - `data/ml_predictions/xgb_opt_biascorrected/...`
  - future models auto-discovered as `data/ml_predictions/<new_model>_biascorrected/...`
- City subfolder and filename are preserved; only parent model directory changes.

### Run Examples

Train + backfill all discovered models:

```bash
python bias_correction/run_bias_correction.py \
  --predictions_root "/home/gkbrkgrr/Desktop/polymarket-weather-trading/data/ml_predictions" \
  --train_start "2025010100" --train_end "2025123118" \
  --backfill_start "2026010100" --backfill_end "latest" \
  --rolling_window_days 45 \
  --ewma_halflife_days 14 \
  --min_history 30 \
  --model_strategy "single_residual_with_model_feature" \
  --obs_source_dsn "postgresql://archive_user:password@127.0.0.1:5432/master_db" \
  --n_jobs 8
```

Only one model:

```bash
python bias_correction/run_bias_correction.py \
  --predictions_root "/home/gkbrkgrr/Desktop/polymarket-weather-trading/data/ml_predictions" \
  --models "xgb_opt" \
  --train_start "2025010100" --train_end "2025123118" \
  --backfill_start "2026010100" --backfill_end "latest" \
  --obs_source_dsn "postgresql://archive_user:password@127.0.0.1:5432/master_db"
```

Dry-run discovery and write plan:

```bash
python bias_correction/run_bias_correction.py \
  --predictions_root "/home/gkbrkgrr/Desktop/polymarket-weather-trading/data/ml_predictions" \
  --train_start "2025010100" --train_end "2025123118" \
  --backfill_start "2026010100" --backfill_end "latest" \
  --dry_run
```

### Adding New Model Directories

Place files under:

`data/ml_predictions/<new_model>/<City>/*.parquet`

No code changes are needed if the parquet schema can be normalized by aliases/fallback rules above.
