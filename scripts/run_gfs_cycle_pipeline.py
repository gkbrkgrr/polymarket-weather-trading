#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
CYCLE_PATTERN = re.compile(r"^\d{10}$")
DEFAULT_TELEGRAM_CREDENTIALS_FILE = REPO_ROOT / ".secrets" / "telegram_bot.json"

# GFS issue cycle hour -> nominal publish time in UTC.
# Keep these in UTC regardless of server local timezone; cron should run at local equivalents.
PUBLISH_SCHEDULE_UTC: list[tuple[int, time]] = [
    (0, time(3, 30)),
    (6, time(9, 30)),
    (12, time(15, 30)),
    (18, time(21, 30)),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the per-cycle GFS pipeline in order: raw extraction, model predictions, "
            "and forecast progression report."
        )
    )
    parser.add_argument(
        "--cycle",
        default=None,
        help=(
            "Cycle token in YYYYMMDDHH. "
            "If omitted, cycle is inferred from current UTC time and publish schedule."
        ),
    )
    parser.add_argument(
        "--now-utc",
        default=None,
        help=(
            "Override current UTC clock for cycle inference, format YYYYMMDDHHMM "
            "(useful for testing). Ignored when --cycle is provided."
        ),
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to run downstream scripts.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root path.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not pass --overwrite to daily_station_tmax_predictor.py.",
    )
    parser.add_argument(
        "--skip-telegram",
        action="store_true",
        help="Skip Telegram report publishing step.",
    )
    parser.add_argument(
        "--telegram-credentials-file",
        type=Path,
        default=DEFAULT_TELEGRAM_CREDENTIALS_FILE,
        help=(
            "Path to Telegram credentials JSON for publishing reports "
            f"(default: {DEFAULT_TELEGRAM_CREDENTIALS_FILE})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved cycle and commands without executing them.",
    )
    return parser.parse_args(argv)


def parse_cycle_token(value: str) -> str:
    if CYCLE_PATTERN.fullmatch(value) is None:
        raise SystemExit(f"--cycle must be YYYYMMDDHH, got {value!r}")
    return value


def parse_now_utc_token(value: str) -> datetime:
    if re.fullmatch(r"\d{12}", value) is None:
        raise SystemExit(f"--now-utc must be YYYYMMDDHHMM, got {value!r}")
    return datetime.strptime(value, "%Y%m%d%H%M").replace(tzinfo=UTC)


def infer_cycle_from_time(now_utc: datetime) -> str:
    candidates: list[datetime] = []

    for day_offset in (-1, 0):
        day = (now_utc + timedelta(days=day_offset)).date()
        for cycle_hour, publish_time in PUBLISH_SCHEDULE_UTC:
            publish_dt = datetime.combine(day, publish_time, tzinfo=UTC)
            if publish_dt > now_utc:
                continue
            cycle_dt = datetime(day.year, day.month, day.day, cycle_hour, tzinfo=UTC)
            candidates.append(cycle_dt)

    if not candidates:
        raise SystemExit(
            "Could not infer cycle from UTC time; check system clock or provide --cycle explicitly."
        )
    return max(candidates).strftime("%Y%m%d%H")


def resolve_cycle_token(args: argparse.Namespace) -> str:
    if args.cycle:
        return parse_cycle_token(args.cycle)
    now_utc = parse_now_utc_token(args.now_utc) if args.now_utc else datetime.now(UTC)
    return infer_cycle_from_time(now_utc)


def build_commands(
    *,
    python_bin: Path,
    repo_root: Path,
    cycle: str,
    overwrite: bool,
    skip_telegram: bool,
    telegram_credentials_file: Path,
) -> list[list[str]]:
    scripts_dir = repo_root / "scripts"
    is_00z = cycle.endswith("00")

    commands: list[list[str]] = []
    commands.append(
        [
            str(python_bin),
            str(scripts_dir / "raw_point_extractor.py"),
            "--download",
            "--model",
            "gfs",
            "--cycle",
            cycle,
        ]
    )

    for model_name in ("city_extended", "xgb_opt_v1_100", "xgb_opt_v2_100"):
        cmd = [
            str(python_bin),
            str(scripts_dir / "daily_station_tmax_predictor.py"),
            "--model",
            model_name,
            "--cycle",
            cycle,
        ]
        if overwrite:
            cmd.append("--overwrite")
        commands.append(cmd)

    commands.append(
        [
            str(python_bin),
            str(scripts_dir / "forecast_progressions_reporter.py"),
            "--cycle",
            cycle,
        ]
    )

    # Full model performance heatmap report is generated only for 00Z cycles.
    if is_00z:
        commands.append(
            [
                str(python_bin),
                str(scripts_dir / "model_performances_reporter.py"),
            ]
        )

    if not skip_telegram:
        telegram_cmd = [
            str(python_bin),
            str(scripts_dir / "telegram_publish_report.py"),
            "--cycle",
            cycle,
            "--credentials-file",
            str(telegram_credentials_file),
        ]
        if is_00z:
            telegram_cmd.append("--include-heatmap")
        commands.append(telegram_cmd)
    return commands


def run_commands(commands: list[list[str]], dry_run: bool) -> None:
    for idx, cmd in enumerate(commands, start=1):
        print(f"[{idx}/{len(commands)}] {' '.join(cmd)}")
        if dry_run:
            continue
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cycle = resolve_cycle_token(args)
    overwrite = not bool(args.no_overwrite)

    commands = build_commands(
        python_bin=args.python_bin,
        repo_root=args.repo_root,
        cycle=cycle,
        overwrite=overwrite,
        skip_telegram=bool(args.skip_telegram),
        telegram_credentials_file=args.telegram_credentials_file,
    )

    print(f"Cycle selected: {cycle}")
    print(f"Repo root: {args.repo_root}")
    print(f"Python bin: {args.python_bin}")
    print(f"Overwrite predictions: {overwrite}")
    print(f"Skip Telegram: {bool(args.skip_telegram)}")
    if not args.skip_telegram:
        print(f"Telegram credentials file: {args.telegram_credentials_file}")
    print(f"Dry run: {bool(args.dry_run)}")
    run_commands(commands, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
