# Live Pilot Builder

This directory contains a production-minded, minimal scaffold for running a NO-only live/paper trading pilot against Polymarket weather markets.

## What It Does

- Loads precomputed market probabilities (`p_model`) from pipeline outputs.
- Resolves live open markets from `master_db` by `market_id`/`slug` and `(station, day, event_key, strike_k)` fallback mapping.
- Drops unmapped probability rows with explicit reason logging.
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
  - Health-gate pause alerts with duplicate suppression (`health_gate_alert_cooldown_minutes`)

## Safety Notes

- Default mode is `paper`.
- No API keys are required for this implementation.
- `--dry-run` performs selection/logging only and does not place any orders.
- `mode: live` currently points to a stub `RealExecutionClient` and is intentionally not wired to real credentials.
- Runtime health gates pause new trades when probabilities or snapshots are stale and emit explicit alerts.

## Current End-to-End Pipeline

1. **Cycle scheduling and orchestration (LT-07)**  
   `scripts/trigger_gfs_cycle_pipeline.py` runs on each publish window and calls `scripts/run_gfs_cycle_pipeline.py`.
2. **Forecast + probability generation**  
   The cycle pipeline runs raw extraction, station Tmax predictors, market probability generation, progression-state update, and forecast reporting; outputs are written under `reports/live_probabilities/cycles/<cycle>/...` and promoted via `latest_manifest.json`.
3. **Live pilot runtime (LT-08, LT-09)**  
   `live_trading/run_live_pilot.py` resolves the latest valid probabilities artifact, maps to open markets from `master_db`, applies freshness/health gates, selector policy, risk controls, and execution (paper/live mode abstraction).
4. **Parity validation (LT-10)**  
   `scripts/selector_parity_harness.py` validates backtest/live selector behavior on identical inputs.
5. **Continuous paper canary (LT-11)**  
   `scripts/run_paper_trading_canary.py` supervises long-running paper operation with retries, status reporting, and notifications.
6. **Go-live and rollback operations (LT-12)**  
   `scripts/live_kill_switch.py` plus the go-live checklist and rollback playbook provide operator controls and incident procedures.

## How To Run

Use `env_poly` as requested, then run from repo root.

Step 1: refresh market probabilities (or rely on LT-07 scheduler):

```bash
conda activate env_poly
python scripts/live_market_probabilities.py --calibration-version residual_oof_v1
```

This writes cycle-versioned artifacts under `reports/live_probabilities/cycles/<cycle>/...` and atomically updates `reports/live_probabilities/latest_manifest.json`.
It also freezes calibration inputs under `reports/live_probabilities/calibrations/<calibration_version>/...` so reruns with the same cycle + calibration version are deterministic.
The live pilot resolves the latest successful cycle from that manifest by default, and falls back to scanning `cycles/*/*/manifest.json` and `cycles/*/manifest.json` if `latest_manifest.json` is stale/invalid.

Step 2: run readiness healthcheck:

```bash
python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml healthcheck
```

Step 3: smoke a single paper cycle (recommended):

```bash
python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml --dry-run --once
```

Step 4: start continuous paper trading (uses `mode: paper` in config):

```bash
python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml
```

Optional: run under an external supervisor and force non-zero on cycle failure:

```bash
python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml --once --exit-nonzero-on-cycle-failure
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

## LT-10 Selector Parity Harness

Validate backtest-vs-live selector parity on identical candidate inputs:

```bash
python scripts/selector_parity_harness.py --config live_trading/config.live_pilot.yaml
```

The harness:
- builds candidates from live probabilities + current snapshots
- runs live selector policy and backtest selector on the same rows
- reports selected-set parity (Jaccard, mismatch rate, precision/recall) and rank-order mismatches
- exits non-zero when tolerance checks fail

Report output:
- `reports/selector_parity/selector_parity_<UTC_TIMESTAMP>.json`

## LT-11 Paper-Trading Canary (10-14 Days)

Run the canary supervisor:

```bash
python scripts/run_paper_trading_canary.py --config live_trading/config.live_pilot.yaml --duration-days 10
```

For a full two-week canary:

```bash
python scripts/run_paper_trading_canary.py --config live_trading/config.live_pilot.yaml --duration-days 14
```

What it adds on top of `run_live_pilot.py`:
- executes `run_live_pilot.py --once --exit-nonzero-on-cycle-failure` in a resilient loop
- runs periodic `healthcheck` calls and emits Telegram alerts on failures
- sends canary lifecycle notifications (start, heartbeat, cycle failure, completion)
- stops with `critical_failure` after configurable consecutive failed cycles
- persists canary state/report for auditability

Canary artifacts:
- state: `live_trading/state/paper_canary_state.json`
- per-run report: `live_trading/reports/canary/lt11_<...>.json`
- latest report pointer: `live_trading/reports/canary/latest_canary_status.json`
- canary log: `live_trading/logs/paper_canary_YYYYMMDD_<id>.log`

Short smoke test (local/dev):

```bash
python scripts/run_paper_trading_canary.py \
  --config live_trading/config.live_pilot.yaml \
  --duration-days 0.01 \
  --allow-short-duration \
  --dry-run \
  --skip-telegram \
  --interval-minutes 0.05 \
  --healthcheck-every-cycles 1 \
  --max-consecutive-failures 1
```

Optional systemd service template:
- `deploy/polymarket-paper-canary.service`

## LT-12 Go-Live and Rollback Runbooks

Operator kill-switch CLI:

```bash
python scripts/live_kill_switch.py status
python scripts/live_kill_switch.py enable-global --reason "manual risk halt"
python scripts/live_kill_switch.py disable-global --reason "incident resolved"
python scripts/live_kill_switch.py pause-station Atlanta --reason "station anomaly"
python scripts/live_kill_switch.py unpause-station Atlanta --reason "station recovered"
```

Formal runbooks:
- `live_trading/LIVE_GO_LIVE_CHECKLIST.md`
- `live_trading/LIVE_ROLLBACK_PLAYBOOK.md`

## Example Config

See [`config.live_pilot.yaml`](./config.live_pilot.yaml).

Key fields:
- `mode`: `paper` or `live` (default `paper`)
- `db_dsn`: master Postgres DSN
- `probabilities_path`: precomputed probabilities file/directory (default: `reports/live_probabilities`; auto-resolves newest valid cycle from manifests)
- `max_probability_age_days`, `max_snapshot_age_minutes`: freshness thresholds for runtime health gates
- `pause_on_stale_probabilities`, `pause_on_stale_snapshots`: enable/disable runtime trade pause on stale data
- `health_gate_alert_cooldown_minutes`: suppress repeated identical health-gate Telegram alerts
- `stations_allowlist`, thresholds, risk limits, execution settings, scheduling, reporting
- `telegram_notifications`: topic links + credentials for in-process sends

## Files

- `run_live_pilot.py`: CLI entrypoint and orchestration loop
- `scripts/trigger_gfs_cycle_pipeline.py`: idempotent cycle trigger orchestrator (LT-07)
- `scripts/selector_parity_harness.py`: backtest/live selector parity validator (LT-10)
- `scripts/run_paper_trading_canary.py`: paper canary supervisor with reporting/alerts (LT-11)
- `scripts/live_kill_switch.py`: operator kill-switch CLI for incident response (LT-12)
- `policy.py`: strategy filters + risk/kill checks
- `pricing.py`: snapshot/NO-ask pricing logic
- `execution.py`: execution abstraction + dummy client + live stub
- `state.py`: persistent pilot state (`live_trading/state/live_state.json`)
- `reporting.py`: daily report builders (JSON/CSV/telegram text)
- `db.py`: master_db query + DB logging helpers
- `utils_time.py`: station timezone and market-day/cutoff utilities
- `LIVE_GO_LIVE_CHECKLIST.md`: sign-off gate for live execution (LT-12)
- `LIVE_ROLLBACK_PLAYBOOK.md`: emergency rollback procedure (LT-12)
