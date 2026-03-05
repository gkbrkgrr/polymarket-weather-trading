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
- Runtime health gates pause new trades when probabilities or snapshots are stale and emit explicit alerts.

## How To Run

Use `env_poly` as requested, then run from repo root:

```bash
python scripts/live_market_probabilities.py --calibration-version residual_oof_v1
```

This writes cycle-versioned artifacts under `reports/live_probabilities/cycles/<cycle>/...` and atomically updates `reports/live_probabilities/latest_manifest.json`.
It also freezes calibration inputs under `reports/live_probabilities/calibrations/<calibration_version>/...` so reruns with the same cycle + calibration version are deterministic.
The live pilot resolves the latest successful cycle from that manifest by default, and falls back to scanning `cycles/*/*/manifest.json` and `cycles/*/manifest.json` if `latest_manifest.json` is stale/invalid.

Then run the pilot:

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

## LT-07 Automation (4x/day)

To automate end-to-end cycle processing at each nominal GFS publish window, use:

```bash
python scripts/trigger_gfs_cycle_pipeline.py
```

What it does:
- infers the latest publish-eligible cycle from UTC schedule (`00z/06z/12z/18z`)
- runs `scripts/run_gfs_cycle_pipeline.py --cycle <cycle> ...`
- verifies `reports/live_probabilities/latest_manifest.json` was updated for that cycle
- persists idempotency state in `live_trading/state/gfs_cycle_trigger_state.json` so each cycle runs once unless `--force` is used

Useful checks:

```bash
# print resolved cycle + downstream command without executing
python scripts/trigger_gfs_cycle_pipeline.py --dry-run

# deterministic test at a fixed UTC clock
python scripts/trigger_gfs_cycle_pipeline.py --now-utc 202603061540 --dry-run
```

Systemd templates are provided for production scheduling:
- `deploy/polymarket-gfs-cycle-trigger.service`
- `deploy/polymarket-gfs-cycle-trigger.timer`

Install example:

```bash
# first edit ExecStart/WorkingDirectory in the service file for your host paths
sudo cp deploy/polymarket-gfs-cycle-trigger.service /etc/systemd/system/
sudo cp deploy/polymarket-gfs-cycle-trigger.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now polymarket-gfs-cycle-trigger.timer
systemctl list-timers --all | grep polymarket-gfs-cycle-trigger
```

## Example Config

See [`config.live_pilot.yaml`](./config.live_pilot.yaml).

Key fields:
- `mode`: `paper` or `live` (default `paper`)
- `db_dsn`: master Postgres DSN
- `probabilities_path`: precomputed probabilities file/directory (default: `reports/live_probabilities`; auto-resolves newest valid cycle from manifests)
- `max_probability_age_days`, `max_snapshot_age_minutes`: freshness thresholds for runtime health gates
- `pause_on_stale_probabilities`, `pause_on_stale_snapshots`: enable/disable runtime trade pause on stale data
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
