#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECAST_REPORTS_DIR = REPO_ROOT / "reports" / "forecast_progressions"
DEFAULT_HEATMAP_REPORTS_DIR = REPO_ROOT / "reports" / "error_heatmaps"
DEFAULT_CREDENTIALS_FILE = REPO_ROOT / ".secrets" / "telegram_bot.json"
DEFAULT_HEATMAP_CHAT_ID = "-1003811684844"  # PROJECT PRETTY POLLY forum
DEFAULT_HEATMAP_THREAD_ID = 408  # Error Heatmaps topic
DEFAULT_HEATMAP_GLOB = "model_performances_*.pdf"

DEFAULT_TRADES_LOGS_DIR = REPO_ROOT / "live_trading" / "logs"
DEFAULT_DAILY_REPORTS_DIR = REPO_ROOT / "live_trading" / "reports" / "daily"
DEFAULT_TELEGRAM_STATE_FILE = REPO_ROOT / "live_trading" / "state" / "telegram_publish_state.json"
DEFAULT_TRADES_CHAT_ID = "-1003811684844"
DEFAULT_TRADES_THREAD_ID = 467  # Trades topic
DEFAULT_DAILY_CHAT_ID = "-1003811684844"
DEFAULT_DAILY_THREAD_ID = 468  # Daily Z Reading topic
DEFAULT_TRADES_LOG_GLOB = "trades_*.jsonl"
DEFAULT_DAILY_REPORT_GLOB = "*_telegram.txt"

CYCLE_PATTERN = re.compile(r"^\d{10}$")
TOPIC_LINK_RE = re.compile(r"^https?://t\.me/c/(\d+)/(\d+)(?:/\d+)?/?$")

SLUG_EXACT_C_RE = re.compile(r"-(neg-\d+|\d+)c$")
SLUG_EXACT_F_RE = re.compile(r"-(\d+)f$")
SLUG_RANGE_F_RE = re.compile(r"-(\d+)-(\d+)f$")
SLUG_BELOW_F_RE = re.compile(r"-(\d+)forbelow$")
SLUG_ABOVE_F_RE = re.compile(r"-(\d+)forhigher$")
SLUG_STATION_RE = re.compile(r"^highest-temperature-in-([a-z0-9-]+)-on-")

STATION_TOKEN_TO_DISPLAY = {
    "ankara": "Ankara",
    "atlanta": "Atlanta",
    "buenos-aires": "Buenos Aires",
    "chicago": "Chicago",
    "dallas": "Dallas",
    "london": "London",
    "miami": "Miami",
    "nyc": "NYC",
    "paris": "Paris",
    "sao-paulo": "Sao Paulo",
    "seattle": "Seattle",
    "seoul": "Seoul",
    "toronto": "Toronto",
}
STATION_CANONICAL_DISPLAY = {
    "BuenosAires": "Buenos Aires",
    "SaoPaulo": "Sao Paulo",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master Telegram uploader for forecast PDFs + live trade/daily topic messages."
    )
    parser.add_argument(
        "--cycle",
        required=False,
        default=None,
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
        help="Override forecast upload chat ID. Defaults to chat_id in credentials file.",
    )
    parser.add_argument(
        "--forecast-message-thread-id",
        type=int,
        default=None,
        help="Override forecast upload thread ID. Defaults to message_thread_id in credentials file.",
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
        "--include-live-trades",
        action="store_true",
        help="Publish new live trading events from live_trading/logs/trades_*.jsonl to the Trades topic.",
    )
    parser.add_argument(
        "--trades-logs-dir",
        type=Path,
        default=DEFAULT_TRADES_LOGS_DIR,
        help=f"Directory containing live trade action JSONL logs (default: {DEFAULT_TRADES_LOGS_DIR})",
    )
    parser.add_argument(
        "--trades-log-glob",
        type=str,
        default=DEFAULT_TRADES_LOG_GLOB,
        help=f"Glob for live trade action logs (default: {DEFAULT_TRADES_LOG_GLOB})",
    )
    parser.add_argument(
        "--trades-chat-id",
        type=str,
        default=DEFAULT_TRADES_CHAT_ID,
        help=f"Trades topic chat ID (default: {DEFAULT_TRADES_CHAT_ID})",
    )
    parser.add_argument(
        "--trades-message-thread-id",
        type=int,
        default=DEFAULT_TRADES_THREAD_ID,
        help=f"Trades topic thread ID (default: {DEFAULT_TRADES_THREAD_ID})",
    )
    parser.add_argument(
        "--trades-topic-link",
        type=str,
        default=None,
        help="Optional full Trades topic link (e.g. https://t.me/c/<chat>/<topic>/<msg>) to derive chat/thread.",
    )
    parser.add_argument(
        "--max-trade-messages",
        type=int,
        default=200,
        help="Max trade/resolution messages to publish in one run (default: 200).",
    )

    parser.add_argument(
        "--include-live-daily-report",
        action="store_true",
        help="Publish new live_trading/reports/daily/*_telegram.txt to the daily report topic.",
    )
    parser.add_argument(
        "--daily-reports-dir",
        type=Path,
        default=DEFAULT_DAILY_REPORTS_DIR,
        help=f"Directory containing daily telegram text reports (default: {DEFAULT_DAILY_REPORTS_DIR})",
    )
    parser.add_argument(
        "--daily-report-glob",
        type=str,
        default=DEFAULT_DAILY_REPORT_GLOB,
        help=f"Glob for daily telegram report text files (default: {DEFAULT_DAILY_REPORT_GLOB})",
    )
    parser.add_argument(
        "--daily-chat-id",
        type=str,
        default=DEFAULT_DAILY_CHAT_ID,
        help=f"Daily report topic chat ID (default: {DEFAULT_DAILY_CHAT_ID})",
    )
    parser.add_argument(
        "--daily-message-thread-id",
        type=int,
        default=DEFAULT_DAILY_THREAD_ID,
        help=f"Daily report topic thread ID (default: {DEFAULT_DAILY_THREAD_ID})",
    )
    parser.add_argument(
        "--daily-topic-link",
        type=str,
        default=None,
        help="Optional full Daily topic link (e.g. https://t.me/c/<chat>/<topic>/<msg>) to derive chat/thread.",
    )

    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_TELEGRAM_STATE_FILE,
        help=f"State/checkpoint file for live topic publishing (default: {DEFAULT_TELEGRAM_STATE_FILE})",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll and publish live topics (trade events and/or daily report text files).",
    )
    parser.add_argument(
        "--watch-interval-seconds",
        type=float,
        default=5.0,
        help="Polling interval for --watch mode (default: 5.0).",
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
        raise SystemExit(f"Credentials file not found: {path}. Create it with at least bot_token.")
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


def send_text(
    *,
    bot_token: str,
    chat_id: str,
    message_thread_id: int,
    text: str,
    timeout_seconds: float,
) -> int:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "message_thread_id": str(message_thread_id),
        "text": text,
        "disable_web_page_preview": "true",
    }
    with httpx.Client(timeout=timeout_seconds) as client:
        resp = client.post(url, data=data)
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


def parse_topic_link(url: str) -> tuple[str, int]:
    m = TOPIC_LINK_RE.fullmatch(url.strip())
    if m is None:
        raise SystemExit(f"Invalid topic link: {url!r}")
    chat_id = f"-100{m.group(1)}"
    thread_id = int(m.group(2))
    return chat_id, thread_id


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "trade_offsets": {},
            "daily_sent_files": [],
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "trade_offsets": {},
            "daily_sent_files": [],
        }
    if not isinstance(raw, dict):
        return {
            "trade_offsets": {},
            "daily_sent_files": [],
        }
    raw.setdefault("trade_offsets", {})
    raw.setdefault("daily_sent_files", [])
    return raw


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def pick_first_number(record: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def display_station_name(record: dict[str, Any], slug: str | None) -> str:
    station = str(record.get("station") or "").strip()
    if station:
        return STATION_CANONICAL_DISPLAY.get(station, station)

    if slug:
        m = SLUG_STATION_RE.search(slug)
        if m:
            token = m.group(1)
            return STATION_TOKEN_TO_DISPLAY.get(token, token.replace("-", " ").title())
    return "Unknown"


def parse_market_suffix(slug: str | None, fallback_strike_k: Any) -> str:
    if not slug:
        if fallback_strike_k is None:
            return "Unknown strike"
        return f"{fallback_strike_k} \N{DEGREE SIGN}C"

    m_c = SLUG_EXACT_C_RE.search(slug)
    if m_c:
        token = m_c.group(1)
        if token.startswith("neg-"):
            strike = -int(token.split("-", 1)[1])
        else:
            strike = int(token)
        return f"{strike} \N{DEGREE SIGN}C"

    m_f = SLUG_EXACT_F_RE.search(slug)
    if m_f:
        return f"{int(m_f.group(1))} \N{DEGREE SIGN}F"

    m_range = SLUG_RANGE_F_RE.search(slug)
    if m_range:
        low = int(m_range.group(1))
        high = int(m_range.group(2))
        return f"{low}-{high} \N{DEGREE SIGN}F"

    m_below = SLUG_BELOW_F_RE.search(slug)
    if m_below:
        return f"\N{LESS-THAN OR EQUAL TO} {int(m_below.group(1))} \N{DEGREE SIGN}F"

    m_above = SLUG_ABOVE_F_RE.search(slug)
    if m_above:
        return f"\N{GREATER-THAN OR EQUAL TO} {int(m_above.group(1))} \N{DEGREE SIGN}F"

    if fallback_strike_k is not None:
        return f"{fallback_strike_k} \N{DEGREE SIGN}C"
    return "Unknown strike"


def format_market_name(record: dict[str, Any]) -> str:
    slug = str(record.get("slug") or "").strip() or None
    station = display_station_name(record, slug)
    strike_text = parse_market_suffix(slug, record.get("strike_k"))

    day_text = "Unknown date"
    day_raw = record.get("market_day_local")
    if day_raw is not None:
        parsed = None
        try:
            parsed = datetime.fromisoformat(str(day_raw)).date()
        except ValueError:
            parsed = None
        if parsed is not None:
            day_text = f"{parsed.day} {parsed.strftime('%B %Y')}"

    return f"{day_text} {station} {strike_text}".strip()


def infer_trade_action(record: dict[str, Any]) -> str | None:
    decision = str(record.get("decision") or "").strip().upper()
    side = str(record.get("side") or "").strip().upper()

    if decision in {"RESOLVE", "RESOLUTION"}:
        return "RESOLUTION"
    if side == "SELL" or decision == "SELL":
        return "SELL"
    if side == "BUY" or decision in {"TRADE", "BUY"}:
        return "BUY"
    return None


def format_float(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{ndigits}f}"


def format_money(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{abs(value):.2f}"


def format_trade_message(record: dict[str, Any]) -> str | None:
    action = infer_trade_action(record)
    if action is None:
        return None

    market = format_market_name(record)
    lot = pick_first_number(record, ["size", "lot", "quantity"])

    if action == "BUY":
        price = pick_first_number(record, ["chosen_no_ask", "price", "entry_price", "buy_price"])
        return (
            f"\N{SLOT MACHINE}Market: {market} | BUY | "
            f"Price: {format_float(price)} | Lot: {format_float(lot, ndigits=2)}"
        )

    if action == "SELL":
        buy_price = pick_first_number(record, ["buy_price", "entry_price", "avg_entry_price"])
        sell_price = pick_first_number(record, ["sell_price", "chosen_no_ask", "price", "exit_price"])
        pnl = pick_first_number(record, ["pnl_realized", "pnl"])
        if pnl is None and buy_price is not None and sell_price is not None and lot is not None:
            pnl = (sell_price - buy_price) * lot

        if pnl is None or pnl >= 0:
            return (
                f"\N{LARGE GREEN CIRCLE} Market: {market} | SELL | "
                f"Buy Price: {format_float(buy_price)} | Sell Price: {format_float(sell_price)} | "
                f"Lot: {format_float(lot, ndigits=2)} | Total Profit: {format_money(pnl)}"
            )
        return (
            f"\N{LARGE RED CIRCLE} Market: {market} | SELL | "
            f"Buy Price: {format_float(buy_price)} | Sell Price: {format_float(sell_price)} | "
            f"Lot: {format_float(lot, ndigits=2)} | Total Loss: {format_money(pnl)}"
        )

    buy_price = pick_first_number(record, ["buy_price", "entry_price", "chosen_no_ask"])
    pnl = pick_first_number(record, ["pnl_realized", "pnl"])
    if pnl is None or pnl >= 0:
        return (
            f"\N{LARGE GREEN CIRCLE} Market: {market} | Resolution | "
            f"Buy Price: {format_float(buy_price)} | Lot: {format_float(lot, ndigits=2)} | "
            f"Total Profit: {format_money(pnl)}"
        )
    return (
        f"\N{LARGE RED CIRCLE} Market: {market} | Resolution | "
        f"Buy Price: {format_float(buy_price)} | Lot: {format_float(lot, ndigits=2)} | "
        f"Total Loss: {format_money(pnl)}"
    )


def chunk_text(text: str, *, max_chars: int = 3900) -> list[str]:
    content = text.strip()
    if not content:
        return []
    if len(content) <= max_chars:
        return [content]

    chunks: list[str] = []
    remaining = content
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, max_chars)
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].lstrip("\n")
    return [c for c in chunks if c]


def resolve_live_topic(chat_id: str, thread_id: int, topic_link: str | None) -> tuple[str, int]:
    if topic_link:
        return parse_topic_link(topic_link)
    return chat_id, thread_id


def maybe_publish_live_trades(
    *,
    args: argparse.Namespace,
    bot_token: str,
    state: dict[str, Any],
) -> None:
    logs_dir = args.trades_logs_dir
    if not logs_dir.exists():
        print(f"Trades logs directory not found: {logs_dir}")
        return

    chat_id, thread_id = resolve_live_topic(
        str(args.trades_chat_id).strip(),
        int(args.trades_message_thread_id),
        args.trades_topic_link,
    )

    offsets_raw = state.setdefault("trade_offsets", {})
    if not isinstance(offsets_raw, dict):
        offsets_raw = {}
        state["trade_offsets"] = offsets_raw

    files = sorted(logs_dir.glob(args.trades_log_glob))
    if not files:
        print(f"No trade log files matching {args.trades_log_glob!r} under {logs_dir}")
        return

    sent_count = 0
    scanned_count = 0
    max_messages = max(1, int(args.max_trade_messages))

    for path in files:
        key = str(path.resolve())
        start_offset = int(offsets_raw.get(key, 0) or 0)

        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()

        if start_offset < 0:
            start_offset = 0
        if start_offset > len(lines):
            start_offset = len(lines)

        for line_idx in range(start_offset, len(lines)):
            scanned_count += 1
            line = lines[line_idx].strip()
            offsets_raw[key] = line_idx + 1

            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            message = format_trade_message(payload)
            if message is None:
                continue

            print(f"[trade-event] {message}")
            if not args.dry_run:
                message_id = send_text(
                    bot_token=bot_token,
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    text=message,
                    timeout_seconds=float(args.timeout_seconds),
                )
                link = topic_message_link(
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    message_id=message_id,
                )
                if link is not None:
                    print(f"Message link: {link}")

            sent_count += 1
            if sent_count >= max_messages:
                print(f"Reached --max-trade-messages={max_messages}; stopping this run.")
                print(f"Trades scanned={scanned_count} sent={sent_count}")
                return

        offsets_raw[key] = len(lines)

    print(f"Trades scanned={scanned_count} sent={sent_count}")


def maybe_publish_live_daily_reports(
    *,
    args: argparse.Namespace,
    bot_token: str,
    state: dict[str, Any],
) -> None:
    reports_dir = args.daily_reports_dir
    if not reports_dir.exists():
        print(f"Daily reports directory not found: {reports_dir}")
        return

    chat_id, thread_id = resolve_live_topic(
        str(args.daily_chat_id).strip(),
        int(args.daily_message_thread_id),
        args.daily_topic_link,
    )

    sent_files_raw = state.setdefault("daily_sent_files", [])
    if not isinstance(sent_files_raw, list):
        sent_files_raw = []
        state["daily_sent_files"] = sent_files_raw
    sent_files = set(str(x) for x in sent_files_raw)

    candidates = sorted(reports_dir.glob(args.daily_report_glob))
    if not candidates:
        print(f"No daily report files matching {args.daily_report_glob!r} under {reports_dir}")
        return

    published = 0
    for path in candidates:
        key = str(path.resolve())
        if key in sent_files:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            sent_files.add(key)
            continue

        chunks = chunk_text(text)
        print(f"[daily-report] {path.name} chunks={len(chunks)}")
        if not args.dry_run:
            for idx, chunk in enumerate(chunks, start=1):
                message_id = send_text(
                    bot_token=bot_token,
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    text=chunk,
                    timeout_seconds=float(args.timeout_seconds),
                )
                link = topic_message_link(
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    message_id=message_id,
                )
                if link is not None:
                    print(f"Message link ({idx}/{len(chunks)}): {link}")

        sent_files.add(key)
        published += 1

    state["daily_sent_files"] = sorted(sent_files)
    print(f"Daily reports published={published}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    creds = load_credentials(args.credentials_file)
    bot_token = str(creds["bot_token"]).strip()

    jobs: list[dict[str, object]] = []

    include_forecast = (not args.no_forecast) and bool(args.cycle)
    if not args.no_forecast and args.cycle is None:
        print("Forecast upload skipped because --cycle was not provided.")

    if include_forecast:
        cycle = validate_cycle(str(args.cycle))
        forecast_path = resolve_forecast_report_path(args.forecast_reports_dir, cycle)
        forecast_chat = (
            str(args.forecast_chat_id).strip()
            if args.forecast_chat_id is not None
            else str(creds.get("chat_id", "")).strip()
        )
        if not forecast_chat:
            raise SystemExit("Forecast upload requires chat_id in credentials or --forecast-chat-id.")

        if args.forecast_message_thread_id is not None:
            forecast_thread = int(args.forecast_message_thread_id)
        else:
            raw_thread = creds.get("message_thread_id")
            if raw_thread is None:
                raise SystemExit(
                    "Forecast upload requires message_thread_id in credentials or --forecast-message-thread-id."
                )
            try:
                forecast_thread = int(raw_thread)
            except (TypeError, ValueError) as exc:
                raise SystemExit("message_thread_id in credentials must be an integer.") from exc

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

    will_publish_live = bool(args.include_live_trades or args.include_live_daily_report)
    if args.watch and not will_publish_live:
        raise SystemExit("--watch requires --include-live-trades and/or --include-live-daily-report.")

    if not jobs and not will_publish_live:
        raise SystemExit(
            "Nothing to publish. Enable forecast/heatmap and/or --include-live-trades/--include-live-daily-report."
        )

    for job in jobs:
        print(f"Target: {job['name']}")
        print(f"Report file: {job['path']}")
        print(f"Chat ID: {job['chat_id']}")
        print(f"Message thread ID: {job['thread_id']}")
        print(f"Caption: {job['caption']}")

    if args.include_live_trades:
        trades_chat, trades_thread = resolve_live_topic(
            str(args.trades_chat_id).strip(),
            int(args.trades_message_thread_id),
            args.trades_topic_link,
        )
        print("Target: live_trades")
        print(f"Logs dir: {args.trades_logs_dir}")
        print(f"Glob: {args.trades_log_glob}")
        print(f"Chat ID: {trades_chat}")
        print(f"Message thread ID: {trades_thread}")

    if args.include_live_daily_report:
        daily_chat, daily_thread = resolve_live_topic(
            str(args.daily_chat_id).strip(),
            int(args.daily_message_thread_id),
            args.daily_topic_link,
        )
        print("Target: live_daily_report")
        print(f"Reports dir: {args.daily_reports_dir}")
        print(f"Glob: {args.daily_report_glob}")
        print(f"Chat ID: {daily_chat}")
        print(f"Message thread ID: {daily_thread}")

    print(f"State file: {args.state_file}")
    print(f"Dry run: {bool(args.dry_run)}")

    for job in jobs:
        if args.dry_run:
            continue
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

    if will_publish_live:
        def run_live_once() -> None:
            state = load_state(args.state_file)
            if args.include_live_trades:
                maybe_publish_live_trades(args=args, bot_token=bot_token, state=state)
            if args.include_live_daily_report:
                maybe_publish_live_daily_reports(args=args, bot_token=bot_token, state=state)
            if not args.dry_run:
                save_state(args.state_file, state)

        if args.watch:
            interval = max(0.5, float(args.watch_interval_seconds))
            print(f"Watch mode enabled. Poll interval: {interval:.1f}s")
            try:
                while True:
                    run_live_once()
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("Watch mode stopped.")
        else:
            run_live_once()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
