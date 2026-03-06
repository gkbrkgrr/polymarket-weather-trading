# LT-12 Live Rollback Playbook

Use this playbook for any live-trading incident where risk must be reduced immediately.

## Incident Triggers

- unexpected order behavior
- repeated runtime exceptions or process crash loop
- stale snapshot/probability gate failures that do not auto-recover
- abnormal PnL drawdown or risk-limit breach
- external exchange/data outage

## Immediate Response (T+0 to T+5 min)

1. Enable global kill switch:
```bash
python scripts/live_kill_switch.py enable-global --reason "INC-<id> immediate containment"
```
2. Confirm kill switch is active:
```bash
python scripts/live_kill_switch.py status
```
3. Stop the running live process in your process manager:
```bash
# example (replace with your unit/process name)
sudo systemctl stop polymarket-live-pilot.service
```
4. Post incident notice in ops channel with timestamp, operator, and trigger.

## Containment Verification (T+5 to T+15 min)

1. Verify no new live trade submissions appear after kill activation.
2. Check latest pilot logs:
```bash
tail -n 200 live_trading/logs/live_pilot_$(date -u +%Y%m%d).log
```
3. Verify action stream:
```bash
tail -n 200 live_trading/logs/trades_$(date -u +%Y%m%d).jsonl
```
4. Optional DB verification:
```sql
SELECT ts_utc, run_id, station, market_id, decision
FROM live_pilot_actions
ORDER BY ts_utc DESC
LIMIT 50;
```

## Rollback Path

1. Keep `global_kill=true` until root cause is confirmed.
2. Revert runtime mode to paper if needed:
```yaml
mode: paper
```
3. Restart pilot in safe mode (`--dry-run` recommended first):
```bash
python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml --dry-run --once
```
4. If issue is station-specific, keep global kill off but pause affected station(s):
```bash
python scripts/live_kill_switch.py pause-station <Station> --reason "INC-<id> scoped containment"
```
5. Resume paper canary monitoring until stability is proven.

## Recovery and Return to Live

1. Root cause documented with fix PR/commit.
2. Healthcheck passes:
```bash
python live_trading/run_live_pilot.py --config live_trading/config.live_pilot.yaml healthcheck
```
3. Run targeted dry-run validation on impacted stations/markets.
4. Obtain fresh go-live approval against `LIVE_GO_LIVE_CHECKLIST.md`.
5. Disable global kill only after approval:
```bash
python scripts/live_kill_switch.py disable-global --reason "INC-<id> resolved"
```

## Post-Incident Requirements

- incident timeline with UTC timestamps
- detection source and blast radius
- root cause and corrective actions
- explicit decision log for re-enable or extended paper-only period

