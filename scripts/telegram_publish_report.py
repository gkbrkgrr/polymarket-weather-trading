#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECAST_REPORTS_DIR = REPO_ROOT / "reports" / "forecast_progressions"
DEFAULT_HEATMAP_REPORTS_DIR = REPO_ROOT / "reports" / "error_heatmaps"
DEFAULT_CREDENTIALS_FILE = REPO_ROOT / ".secrets" / "telegram_bot.json"
DEFAULT_HEATMAP_CHAT_ID = "-1003811684844"  # PROJECT PRETTY POLLY forum
DEFAULT_HEATMAP_THREAD_ID = 408  # Error Heatmaps topic
DEFAULT_HEATMAP_GLOB = "model_performances_*.pdf"
CYCLE_PATTERN = re.compile(r"^\d{10}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master Telegram uploader for forecast progression and heatmap PDF reports."
    )
    parser.add_argument(
        "--cycle",
        required=True,
        help="Cycle token in YYYYMMDDHH (used for forecast report path and caption).",
    )
    parser.add_argument(
        "--forecast-reports-dir",
        type=Path,
        default=DEFAULT_FORECAST_REPORTS_DIR,
        help=f"Directory containing forecast report PDFs (default: {DEFAULT_FORECAST_REPORTS_DIR})",
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
        "--no-forecast",
        action="store_true",
        help="Do not upload the forecast progression report.",
    )
    parser.add_argument(
        "--forecast-chat-id",
        type=str,
        default=None,
        help=(
            "Override forecast upload chat ID. "
            "Defaults to chat_id in credentials file."
        ),
    )
    parser.add_argument(
        "--forecast-message-thread-id",
        type=int,
        default=None,
        help=(
            "Override forecast upload thread ID. "
            "Defaults to message_thread_id in credentials file."
        ),
    )
    parser.add_argument(
        "--include-heatmap",
        action="store_true",
        help="Upload model performance heatmap PDF as well.",
    )
    parser.add_argument(
        "--heatmap-report-path",
        type=Path,
        default=None,
        help="Explicit heatmap PDF path. If omitted, latest match under --heatmap-reports-dir is used.",
    )
    parser.add_argument(
        "--heatmap-reports-dir",
        type=Path,
        default=DEFAULT_HEATMAP_REPORTS_DIR,
        help=f"Directory containing heatmap PDFs (default: {DEFAULT_HEATMAP_REPORTS_DIR})",
    )
    parser.add_argument(
        "--heatmap-report-glob",
        type=str,
        default=DEFAULT_HEATMAP_GLOB,
        help=f"Glob used when --heatmap-report-path is omitted (default: {DEFAULT_HEATMAP_GLOB})",
    )
    parser.add_argument(
        "--heatmap-chat-id",
        type=str,
        default=DEFAULT_HEATMAP_CHAT_ID,
        help=f"Heatmap upload chat ID (default: {DEFAULT_HEATMAP_CHAT_ID})",
    )
    parser.add_argument(
        "--heatmap-message-thread-id",
        type=int,
        default=DEFAULT_HEATMAP_THREAD_ID,
        help=f"Heatmap upload thread ID (default: {DEFAULT_HEATMAP_THREAD_ID})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print request details without sending.",
    )
    return parser.parse_args(argv)


def validate_cycle(cycle: str) -> str:
    if CYCLE_PATTERN.fullmatch(cycle) is None:
        raise SystemExit(f"--cycle must be YYYYMMDDHH, got {cycle!r}")
    return cycle


def resolve_forecast_report_path(reports_dir: Path, cycle: str) -> Path:
    path = reports_dir / f"xgb_optuna_forecasts_{cycle}.pdf"
    if not path.exists():
        raise SystemExit(f"Report file not found: {path}")
    return path


def resolve_heatmap_report_path(
    *,
    report_path: Path | None,
    reports_dir: Path,
    report_glob: str,
) -> Path:
    if report_path is not None:
        if not report_path.exists():
            raise SystemExit(f"Heatmap report file not found: {report_path}")
        return report_path
    if not reports_dir.exists():
        raise SystemExit(f"Heatmap reports directory not found: {reports_dir}")
    candidates = sorted(reports_dir.glob(report_glob))
    if not candidates:
        raise SystemExit(f"No heatmap report files matching {report_glob!r} under {reports_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def load_credentials(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(
            f"Credentials file not found: {path}. Create it with at least bot_token."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in credentials file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"Credentials file {path} must contain a JSON object.")
    bot_token = str(raw.get("bot_token", "")).strip()
    if not bot_token:
        raise SystemExit("bot_token in credentials file is missing or empty.")
    return raw


def send_document(
    *,
    bot_token: str,
    chat_id: str,
    message_thread_id: int,
    report_path: Path,
    caption: str,
    timeout_seconds: float,
) -> int:
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
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


def topic_message_link(*, chat_id: str, message_thread_id: int, message_id: int) -> str | None:
    if not chat_id.startswith("-100") or message_id <= 0:
        return None
    internal_chat_id = chat_id[4:]
    return f"https://t.me/c/{internal_chat_id}/{message_thread_id}/{message_id}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cycle = validate_cycle(args.cycle)
    creds = load_credentials(args.credentials_file)
    bot_token = str(creds["bot_token"]).strip()

    jobs: list[dict[str, object]] = []

    if not args.no_forecast:
        forecast_path = resolve_forecast_report_path(args.forecast_reports_dir, cycle)
        forecast_chat = (
            str(args.forecast_chat_id).strip()
            if args.forecast_chat_id is not None
            else str(creds.get("chat_id", "")).strip()
        )
        if not forecast_chat:
            raise SystemExit(
                "Forecast upload requires chat_id. Set it in credentials file or pass --forecast-chat-id."
            )
        if args.forecast_message_thread_id is not None:
            forecast_thread = int(args.forecast_message_thread_id)
        else:
            raw_thread = creds.get("message_thread_id")
            if raw_thread is None:
                raise SystemExit(
                    "Forecast upload requires message_thread_id. "
                    "Set it in credentials file or pass --forecast-message-thread-id."
                )
            try:
                forecast_thread = int(raw_thread)
            except (TypeError, ValueError) as exc:
                raise SystemExit("message_thread_id in credentials file must be an integer.") from exc

        jobs.append(
            {
                "name": "forecast_progression",
                "path": forecast_path,
                "chat_id": forecast_chat,
                "thread_id": forecast_thread,
                "caption": f"xgb_optuna_forecasts_{cycle}",
            }
        )

    if args.include_heatmap:
        heatmap_path = resolve_heatmap_report_path(
            report_path=args.heatmap_report_path,
            reports_dir=args.heatmap_reports_dir,
            report_glob=args.heatmap_report_glob,
        )
        jobs.append(
            {
                "name": "model_performances_heatmap",
                "path": heatmap_path,
                "chat_id": str(args.heatmap_chat_id).strip(),
                "thread_id": int(args.heatmap_message_thread_id),
                "caption": heatmap_path.stem,
            }
        )

    if not jobs:
        raise SystemExit("Nothing to upload. Enable at least one target (forecast and/or heatmap).")

    for job in jobs:
        print(f"Target: {job['name']}")
        print(f"Report file: {job['path']}")
        print(f"Chat ID: {job['chat_id']}")
        print(f"Message thread ID: {job['thread_id']}")
        print(f"Caption: {job['caption']}")
    print(f"Dry run: {bool(args.dry_run)}")

    if args.dry_run:
        return 0

    for job in jobs:
        message_id = send_document(
            bot_token=bot_token,
            chat_id=str(job["chat_id"]),
            message_thread_id=int(job["thread_id"]),
            report_path=Path(str(job["path"])),
            caption=str(job["caption"]),
            timeout_seconds=float(args.timeout_seconds),
        )
        print(f"Telegram publish successful: {job['name']} Message ID: {message_id}")
        link = topic_message_link(
            chat_id=str(job["chat_id"]),
            message_thread_id=int(job["thread_id"]),
            message_id=message_id,
        )
        if link is not None:
            print(f"Message link: {link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
