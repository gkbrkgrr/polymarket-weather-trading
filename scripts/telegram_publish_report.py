#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "forecast_progressions"
DEFAULT_CREDENTIALS_FILE = REPO_ROOT / ".secrets" / "telegram_bot.json"
CYCLE_PATTERN = re.compile(r"^\d{10}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish forecast progression PDF report to a Telegram forum topic."
    )
    parser.add_argument(
        "--cycle",
        required=True,
        help="Cycle token in YYYYMMDDHH used to locate report file.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Directory containing report PDFs (default: {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=DEFAULT_CREDENTIALS_FILE,
        help=f"Path to Telegram credentials JSON (default: {DEFAULT_CREDENTIALS_FILE})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="HTTP timeout for Telegram API requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print request details without sending anything.",
    )
    return parser.parse_args(argv)


def validate_cycle(cycle: str) -> str:
    if CYCLE_PATTERN.fullmatch(cycle) is None:
        raise SystemExit(f"--cycle must be YYYYMMDDHH, got {cycle!r}")
    return cycle


def resolve_report_path(reports_dir: Path, cycle: str) -> Path:
    path = reports_dir / f"xgb_optuna_forecasts_{cycle}.pdf"
    if not path.exists():
        raise SystemExit(f"Report file not found: {path}")
    return path


def load_credentials(path: Path) -> tuple[str, str, int]:
    if not path.exists():
        raise SystemExit(
            f"Credentials file not found: {path}. "
            "Create it with bot_token, chat_id, and message_thread_id."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in credentials file {path}: {exc}") from exc

    missing = [k for k in ("bot_token", "chat_id", "message_thread_id") if k not in raw]
    if missing:
        raise SystemExit(
            f"Credentials file {path} missing required keys: {', '.join(missing)}"
        )

    bot_token = str(raw["bot_token"]).strip()
    chat_id = str(raw["chat_id"]).strip()
    try:
        message_thread_id = int(raw["message_thread_id"])
    except (TypeError, ValueError) as exc:
        raise SystemExit("message_thread_id must be an integer.") from exc

    if not bot_token:
        raise SystemExit("bot_token in credentials file is empty.")
    if not chat_id:
        raise SystemExit("chat_id in credentials file is empty.")
    return bot_token, chat_id, message_thread_id


def send_document(
    *,
    bot_token: str,
    chat_id: str,
    message_thread_id: int,
    report_path: Path,
    cycle: str,
    timeout_seconds: float,
) -> int:
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    caption = f"xgb_optuna_forecasts_{cycle}"
    data = {
        "chat_id": chat_id,
        "message_thread_id": str(message_thread_id),
        "caption": caption,
    }

    with httpx.Client(timeout=timeout_seconds) as client:
        with report_path.open("rb") as fh:
            files = {"document": (report_path.name, fh, "application/pdf")}
            resp = client.post(url, data=data, files=files)
        resp.raise_for_status()
        payload = resp.json()

    if not payload.get("ok"):
        raise SystemExit(f"Telegram API error: {payload}")
    result = payload.get("result") or {}
    try:
        return int(result.get("message_id"))
    except (TypeError, ValueError):
        return -1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cycle = validate_cycle(args.cycle)
    report_path = resolve_report_path(args.reports_dir, cycle)
    bot_token, chat_id, thread_id = load_credentials(args.credentials_file)

    print(f"Report file: {report_path}")
    print(f"Chat ID: {chat_id}")
    print(f"Message thread ID: {thread_id}")
    print(f"Dry run: {bool(args.dry_run)}")

    if args.dry_run:
        return 0

    message_id = send_document(
        bot_token=bot_token,
        chat_id=chat_id,
        message_thread_id=thread_id,
        report_path=report_path,
        cycle=cycle,
        timeout_seconds=float(args.timeout_seconds),
    )
    print(f"Telegram publish successful. Message ID: {message_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
