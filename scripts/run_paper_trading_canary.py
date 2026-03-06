#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
DEFAULT_CONFIG_PATH = REPO_ROOT / "live_trading" / "config.live_pilot.yaml"
DEFAULT_STATE_FILE = REPO_ROOT / "live_trading" / "state" / "paper_canary_state.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "live_trading" / "reports" / "canary"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_trading.telegram_notify import TelegramNotifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resilient 10-14 day paper-trading canary for live_trading/run_live_pilot.py "
            "with heartbeat reporting and Telegram notifications."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Live pilot YAML config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--duration-days",
        type=float,
        default=10.0,
        help="Canary run duration in days (recommended: 10-14).",
    )
    parser.add_argument(
        "--allow-short-duration",
        action="store_true",
        help="Allow duration under 1 day (useful for smoke tests).",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=None,
        help="Cycle interval override. Defaults to run_interval_minutes from live pilot config.",
    )
    parser.add_argument(
        "--heartbeat-minutes",
        type=float,
        default=60.0,
        help="Heartbeat cadence for canary status alerts.",
    )
    parser.add_argument(
        "--healthcheck-every-cycles",
        type=int,
        default=6,
        help="Run live pilot healthcheck every N cycles (set 0 to disable periodic healthchecks).",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=6,
        help="Stop canary with critical failure when consecutive cycle failures reach this threshold.",
    )
    parser.add_argument(
        "--failure-backoff-seconds",
        type=float,
        default=60.0,
        help="Minimum sleep after a failed cycle before retrying.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to run live_pilot subprocess calls.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root path.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help=f"Persistent canary state JSON path (default: {DEFAULT_STATE_FILE}).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Canary report directory (default: {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument("--max-history", type=int, default=2000, help="Maximum cycle records to keep in state history.")
    parser.add_argument("--dry-run", action="store_true", help="Forward --dry-run to live pilot cycle runs.")
    parser.add_argument("--skip-telegram", action="store_true", help="Disable Telegram notifications for this canary run.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Canary logger level.")
    return parser.parse_args(argv)


def utc_now() -> datetime:
    return datetime.now(UTC)


def resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def setup_logger(log_path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("paper_canary")
    logger.handlers = []
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_live_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Live pilot config must be a mapping: {path}")
    return dict(loaded)


def append_history(state: dict[str, Any], record: dict[str, Any], *, max_history: int) -> None:
    history = state.get("history")
    if not isinstance(history, list):
        history = []
    history.append(record)
    keep = max(1, int(max_history))
    state["history"] = history[-keep:]


def _tail_text(text: str, *, max_chars: int = 2000, max_lines: int = 20) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    lines = cleaned.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        clipped = clipped[-max_chars:]
    return clipped


def run_command(cmd: list[str], *, cwd: Path) -> tuple[int, str, str, float]:
    started = utc_now()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    duration = max(0.0, (utc_now() - started).total_seconds())
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or ""), duration


def format_heartbeat_message(state: dict[str, Any], *, end_at: datetime) -> str:
    remaining_hours = max(0.0, (end_at - utc_now()).total_seconds() / 3600.0)
    return (
        "LT-11 canary heartbeat "
        f"run_id={state.get('run_id')} "
        f"cycles={state.get('cycles_total', 0)} "
        f"success={state.get('cycles_success', 0)} "
        f"failed={state.get('cycles_failed', 0)} "
        f"consecutive_failures={state.get('consecutive_failures', 0)}/"
        f"{state.get('max_consecutive_failures', 0)} "
        f"remaining_hours={remaining_hours:.1f}"
    )


def notify(
    notifier: TelegramNotifier | None,
    message: str,
    *,
    logger: logging.Logger,
    channel: str,
) -> None:
    if notifier is None:
        return
    notifier.notify_alert(message, logger=logger, channel=channel)


def persist_state(
    *,
    state: dict[str, Any],
    state_file: Path,
    report_file: Path,
    latest_report_file: Path,
) -> None:
    atomic_write_json(state_file, state)
    atomic_write_json(report_file, state)
    atomic_write_json(latest_report_file, state)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if float(args.duration_days) <= 0.0:
        raise SystemExit("--duration-days must be > 0")
    if float(args.duration_days) < 1.0 and not bool(args.allow_short_duration):
        raise SystemExit("--duration-days < 1 requires --allow-short-duration")
    if int(args.max_consecutive_failures) <= 0:
        raise SystemExit("--max-consecutive-failures must be >= 1")
    if float(args.heartbeat_minutes) <= 0.0:
        raise SystemExit("--heartbeat-minutes must be > 0")
    if float(args.failure_backoff_seconds) < 0.0:
        raise SystemExit("--failure-backoff-seconds must be >= 0")
    if int(args.healthcheck_every_cycles) < 0:
        raise SystemExit("--healthcheck-every-cycles must be >= 0")
    if int(args.max_history) <= 0:
        raise SystemExit("--max-history must be >= 1")

    repo_root = resolve_path(Path(args.repo_root), REPO_ROOT)
    config_path = resolve_path(Path(args.config), repo_root)
    state_file = resolve_path(Path(args.state_file), repo_root)
    report_dir = resolve_path(Path(args.report_dir), repo_root)
    logs_dir = (repo_root / "live_trading" / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    live_cfg = load_live_config(config_path)
    interval_minutes = (
        float(args.interval_minutes)
        if args.interval_minutes is not None
        else float(live_cfg.get("run_interval_minutes", 10.0))
    )
    if interval_minutes <= 0.0:
        raise SystemExit("Cycle interval must be > 0 minutes")

    started_at = utc_now()
    end_at = started_at + timedelta(days=float(args.duration_days))
    run_id = f"lt11_{started_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    log_path = logs_dir / f"paper_canary_{started_at.strftime('%Y%m%d')}_{run_id.split('_')[-1]}.log"
    logger = setup_logger(log_path=log_path, level=str(args.log_level))

    if float(args.duration_days) < 10.0 or float(args.duration_days) > 14.0:
        logger.warning("duration_days=%s is outside the LT-11 recommended range [10, 14]", args.duration_days)

    state: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "target_end_at_utc": end_at.isoformat(),
        "completed_at_utc": None,
        "updated_at_utc": started_at.isoformat(),
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "python_bin": str(Path(args.python_bin)),
        "interval_minutes": float(interval_minutes),
        "duration_days": float(args.duration_days),
        "dry_run": bool(args.dry_run),
        "healthcheck_every_cycles": int(args.healthcheck_every_cycles),
        "max_consecutive_failures": int(args.max_consecutive_failures),
        "failure_backoff_seconds": float(args.failure_backoff_seconds),
        "heartbeat_minutes": float(args.heartbeat_minutes),
        "cycles_total": 0,
        "cycles_success": 0,
        "cycles_failed": 0,
        "healthcheck_failures": 0,
        "consecutive_failures": 0,
        "max_consecutive_failures_seen": 0,
        "last_success_at_utc": None,
        "last_failure_at_utc": None,
        "last_heartbeat_at_utc": None,
        "history": [],
    }

    report_file = report_dir / f"{run_id}.json"
    latest_report_file = report_dir / "latest_canary_status.json"
    persist_state(
        state=state,
        state_file=state_file,
        report_file=report_file,
        latest_report_file=latest_report_file,
    )

    notifier: TelegramNotifier | None = None
    if not bool(args.skip_telegram):
        notifier = TelegramNotifier.from_config(
            cfg=live_cfg,
            repo_root=repo_root,
            send_enabled=(not bool(args.dry_run)),
            logger=logger,
        )

    logger.info(
        "Starting LT-11 canary run_id=%s duration_days=%.2f interval_minutes=%.3f dry_run=%s",
        run_id,
        float(args.duration_days),
        float(interval_minutes),
        bool(args.dry_run),
    )
    notify(
        notifier,
        (
            "LT-11 canary started "
            f"run_id={run_id} duration_days={float(args.duration_days):.2f} "
            f"interval_minutes={float(interval_minutes):.2f} dry_run={bool(args.dry_run)}"
        ),
        logger=logger,
        channel="daily",
    )

    healthcheck_cmd = [
        str(Path(args.python_bin)),
        str(repo_root / "live_trading" / "run_live_pilot.py"),
        "--config",
        str(config_path),
        "healthcheck",
    ]

    cycle_cmd = [
        str(Path(args.python_bin)),
        str(repo_root / "live_trading" / "run_live_pilot.py"),
        "--config",
        str(config_path),
        "--once",
        "--exit-nonzero-on-cycle-failure",
    ]
    if bool(args.dry_run):
        cycle_cmd.append("--dry-run")

    next_heartbeat_at = started_at + timedelta(minutes=float(args.heartbeat_minutes))

    try:
        while utc_now() < end_at:
            cycle_idx = int(state["cycles_total"]) + 1
            cycle_started = utc_now()
            healthcheck_rc: int | None = None

            run_healthcheck = int(args.healthcheck_every_cycles) > 0 and (
                cycle_idx == 1 or cycle_idx % int(args.healthcheck_every_cycles) == 0
            )
            if run_healthcheck:
                hc_rc, hc_stdout, hc_stderr, hc_duration = run_command(healthcheck_cmd, cwd=repo_root)
                healthcheck_rc = int(hc_rc)
                if hc_rc != 0:
                    state["healthcheck_failures"] = int(state["healthcheck_failures"]) + 1
                    logger.error(
                        "Healthcheck failed cycle=%d rc=%d duration_s=%.1f",
                        cycle_idx,
                        hc_rc,
                        hc_duration,
                    )
                    detail = _tail_text(f"{hc_stdout}\n{hc_stderr}")
                    notify(
                        notifier,
                        (
                            f"LT-11 canary healthcheck failed run_id={run_id} cycle={cycle_idx} "
                            f"rc={hc_rc} detail={detail or 'none'}"
                        ),
                        logger=logger,
                        channel="trades",
                    )
                else:
                    logger.info("Healthcheck passed cycle=%d duration_s=%.1f", cycle_idx, hc_duration)

            rc, stdout, stderr, duration_s = run_command(cycle_cmd, cwd=repo_root)
            cycle_completed = utc_now()

            state["cycles_total"] = int(state["cycles_total"]) + 1
            record: dict[str, Any] = {
                "cycle_index": cycle_idx,
                "started_at_utc": cycle_started.isoformat(),
                "completed_at_utc": cycle_completed.isoformat(),
                "duration_seconds": float(duration_s),
                "return_code": int(rc),
                "status": "success" if rc == 0 else "failed",
                "healthcheck_return_code": healthcheck_rc,
            }

            if rc == 0:
                state["cycles_success"] = int(state["cycles_success"]) + 1
                state["consecutive_failures"] = 0
                state["last_success_at_utc"] = cycle_completed.isoformat()
                logger.info("Cycle success idx=%d duration_s=%.1f", cycle_idx, duration_s)
            else:
                state["cycles_failed"] = int(state["cycles_failed"]) + 1
                state["consecutive_failures"] = int(state["consecutive_failures"]) + 1
                state["last_failure_at_utc"] = cycle_completed.isoformat()
                error_tail = _tail_text(f"{stdout}\n{stderr}")
                if error_tail:
                    record["error_tail"] = error_tail
                logger.error(
                    "Cycle failed idx=%d rc=%d consecutive_failures=%d/%d",
                    cycle_idx,
                    rc,
                    int(state["consecutive_failures"]),
                    int(state["max_consecutive_failures"]),
                )
                notify(
                    notifier,
                    (
                        f"LT-11 canary cycle failure run_id={run_id} cycle={cycle_idx} rc={rc} "
                        f"consecutive={int(state['consecutive_failures'])}/"
                        f"{int(state['max_consecutive_failures'])}"
                    ),
                    logger=logger,
                    channel="trades",
                )

            state["max_consecutive_failures_seen"] = max(
                int(state["max_consecutive_failures_seen"]),
                int(state["consecutive_failures"]),
            )
            state["updated_at_utc"] = cycle_completed.isoformat()

            append_history(state, record, max_history=int(args.max_history))
            persist_state(
                state=state,
                state_file=state_file,
                report_file=report_file,
                latest_report_file=latest_report_file,
            )

            if int(state["consecutive_failures"]) >= int(state["max_consecutive_failures"]):
                state["status"] = "critical_failure"
                state["completed_at_utc"] = utc_now().isoformat()
                state["updated_at_utc"] = state["completed_at_utc"]
                persist_state(
                    state=state,
                    state_file=state_file,
                    report_file=report_file,
                    latest_report_file=latest_report_file,
                )
                msg = (
                    "LT-11 canary stopped with critical failure "
                    f"run_id={run_id} cycles={int(state['cycles_total'])} "
                    f"consecutive_failures={int(state['consecutive_failures'])}"
                )
                logger.error(msg)
                notify(notifier, msg, logger=logger, channel="trades")
                notify(notifier, msg, logger=logger, channel="daily")
                return 1

            now = utc_now()
            if now >= next_heartbeat_at:
                state["last_heartbeat_at_utc"] = now.isoformat()
                state["updated_at_utc"] = now.isoformat()
                persist_state(
                    state=state,
                    state_file=state_file,
                    report_file=report_file,
                    latest_report_file=latest_report_file,
                )
                heartbeat_msg = format_heartbeat_message(state, end_at=end_at)
                logger.info(heartbeat_msg)
                notify(notifier, heartbeat_msg, logger=logger, channel="daily")
                next_heartbeat_at = now + timedelta(minutes=float(args.heartbeat_minutes))

            remaining_s = max(0.0, (end_at - utc_now()).total_seconds())
            if remaining_s <= 0.0:
                break

            elapsed_s = max(0.0, (utc_now() - cycle_started).total_seconds())
            wait_s = max(1.0, float(interval_minutes) * 60.0 - elapsed_s)
            if rc != 0:
                wait_s = max(wait_s, float(args.failure_backoff_seconds))
            wait_s = min(wait_s, remaining_s)
            if wait_s > 0:
                logger.info("Sleeping %.1f seconds before next cycle", wait_s)
                time.sleep(wait_s)

    except KeyboardInterrupt:
        interrupted_at = utc_now().isoformat()
        state["status"] = "interrupted"
        state["completed_at_utc"] = interrupted_at
        state["updated_at_utc"] = interrupted_at
        persist_state(
            state=state,
            state_file=state_file,
            report_file=report_file,
            latest_report_file=latest_report_file,
        )
        msg = f"LT-11 canary interrupted run_id={run_id}"
        logger.warning(msg)
        notify(notifier, msg, logger=logger, channel="daily")
        return 130

    completed_at = utc_now().isoformat()
    state["status"] = "completed"
    state["completed_at_utc"] = completed_at
    state["updated_at_utc"] = completed_at
    persist_state(
        state=state,
        state_file=state_file,
        report_file=report_file,
        latest_report_file=latest_report_file,
    )

    completion_msg = (
        "LT-11 canary completed "
        f"run_id={run_id} cycles={int(state['cycles_total'])} "
        f"success={int(state['cycles_success'])} failed={int(state['cycles_failed'])}"
    )
    logger.info(completion_msg)
    notify(notifier, completion_msg, logger=logger, channel="daily")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
