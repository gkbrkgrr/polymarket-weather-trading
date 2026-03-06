# LT-12 Live Execution Go-Live Checklist

This checklist must be fully completed and signed before enabling `mode: live`.

## Context

- Scope: Polymarket weather live execution path in `live_trading/run_live_pilot.py`.
- Dependency gate: LT-11 canary completed successfully for at least 10 days.
- Required artifacts:
- `live_trading/reports/canary/latest_canary_status.json`
- `live_trading/state/live_state.json`
- `live_trading/logs/`
- `live_trading/reports/daily/`

## Hard Blockers (Must Be `YES`)

| ID | Gate | Owner | Status (`YES`/`NO`) | Evidence |
| --- | --- | --- | --- | --- |
| G0 | Real execution path is implemented and tested (current `RealExecutionClient` is not stubbed). | Eng |  |  |
| G1 | LT-11 canary duration is in range 10-14 days and ended with `status=completed`. | Eng/Ops |  |  |
| G2 | Canary run has no `critical_failure` and no unresolved runtime incident. | Eng/Ops |  |  |
| G3 | `python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml healthcheck` returns `0`. | Ops |  |  |
| G4 | Telegram trade + daily topics receive expected messages in non-dry execution. | Ops |  |  |
| G5 | Kill-switch drill completed (global on/off and station pause/unpause). | Ops |  |  |

## Readiness Checklist

| ID | Check | Owner | Status (`YES`/`NO`) | Evidence |
| --- | --- | --- | --- | --- |
| R1 | `config.live_pilot.yaml` reviewed: `mode=live`, risk limits, stale gates, and station allowlist. | Strategy |  |  |
| R2 | Latest probabilities manifest points to expected cycle and status `success`. | Eng |  |  |
| R3 | Snapshot freshness is within configured `max_snapshot_age_minutes`. | Eng |  |  |
| R4 | DB writes validated for `live_pilot_actions` and `live_pilot_reports`. | Eng |  |  |
| R5 | Runtime process manager configuration reviewed (service unit, restart policy, logs). | Ops |  |  |
| R6 | Incident channel and on-call contacts confirmed for launch window. | Ops |  |  |

## Kill-Switch Drill (Required)

Use the same host and config planned for live.

1. Check current status:
```bash
python scripts/live_kill_switch.py status
```
2. Enable global kill:
```bash
python scripts/live_kill_switch.py enable-global --reason "LT-12 drill"
```
3. Verify pilot skips trades with `kill_switch_active`.
4. Disable global kill:
```bash
python scripts/live_kill_switch.py disable-global --reason "LT-12 drill complete"
```
5. Pause and unpause one station:
```bash
python scripts/live_kill_switch.py pause-station Atlanta --reason "LT-12 drill"
python scripts/live_kill_switch.py unpause-station Atlanta --reason "LT-12 drill complete"
```

## Launch Checklist (T-30 min to T+60 min)

1. Freeze config changes and archive the launch config snapshot.
2. Confirm all Hard Blockers are `YES`.
3. Start/rotate runtime process and verify startup logs.
4. Monitor first hour:
- no uncaught cycle failures
- no stale-data health-gate incidents
- expected action/report logging and Telegram messages
5. If any critical anomaly appears, execute rollback playbook immediately.

## Sign-Off

All signers acknowledge that every `YES` gate above is backed by evidence.

| Role | Name | Date (UTC) | Signature | Result (`APPROVED`/`REJECTED`) | Notes |
| --- | --- | --- | --- | --- | --- |
| Strategy Owner |  |  |  |  |  |
| Engineering Owner |  |  |  |  |  |
| Operations Owner |  |  |  |  |  |

