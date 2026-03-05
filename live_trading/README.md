# Live Pilot Builder

This directory contains a production-minded, minimal scaffold for running a NO-only live/paper trading pilot against Polymarket weather markets.

## What It Does

- Loads precomputed market probabilities (`p_model`) from pipeline outputs.
- Pulls latest orderbook snapshots from `master_db` (`snapshots` or `book_snapshots`).
- Applies Sprint-2-scale NO-only rules:
  - `|k - mode_k| >= mode_distance_min`
  - `p_model <= p_model_max`
  - edge threshold, max price cap, spread cap, snapshot freshness
  - top-N per `(station, market_day_local, event_key)`
- Enforces risk controls:
  - station + portfolio daily risk budgets
  - max open positions station/portfolio
  - station pause and global kill switch handling
- Routes orders through an abstract execution interface:
  - `DummyExecutionClient` for paper fills (default)
  - `RealExecutionClient` stub for later real exchange integration
- Logs actions to:
  - `live_trading/logs/live_pilot_YYYYMMDD.log`
  - `live_trading/logs/trades_YYYYMMDD.jsonl`
  - DB table `live_pilot_actions`
- Writes daily reports to:
  - `live_trading/reports/daily/YYYYMMDD_summary.json`
  - `live_trading/reports/daily/YYYYMMDD_summary.csv`
  - `live_trading/reports/daily/YYYYMMDD_telegram.txt`
  - DB table `live_pilot_reports`
- Sends Telegram notifications from the same process:
  - Trade/resolution events to the Trades topic
  - Daily report text payload to the Daily topic

## Safety Notes

- Default mode is `paper`.
- No API keys are required for this implementation.
- `--dry-run` performs selection/logging only and does not place any orders.
- `mode: live` currently points to a stub `RealExecutionClient` and is intentionally not wired to real credentials.

## How To Run

Use `env_poly` as requested, then run from repo root:

```bash
conda activate env_poly
python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml --dry-run --once
```

Run continuously (every `run_interval_minutes`):

```bash
python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml
```

Healthcheck:

```bash
python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml healthcheck
```

## Example Config

See [`config.live_pilot.yaml`](./config.live_pilot.yaml).

Key fields:
- `mode`: `paper` or `live` (default `paper`)
- `db_dsn`: master Postgres DSN
- `probabilities_path`: precomputed probabilities file/directory
- `stations_allowlist`, thresholds, risk limits, execution settings, scheduling, reporting
- `telegram_notifications`: topic links + credentials for in-process sends

## Files

- `run_live_pilot.py`: CLI entrypoint and orchestration loop
- `policy.py`: strategy filters + risk/kill checks
- `pricing.py`: snapshot/NO-ask pricing logic
- `execution.py`: execution abstraction + dummy client + live stub
- `state.py`: persistent pilot state (`live_trading/state/live_state.json`)
- `reporting.py`: daily report builders (JSON/CSV/telegram text)
- `db.py`: master_db query + DB logging helpers
- `utils_time.py`: station timezone and market-day/cutoff utilities
