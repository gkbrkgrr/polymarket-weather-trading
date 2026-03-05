from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


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


def parse_topic_link(url: str) -> tuple[str, int]:
    m = TOPIC_LINK_RE.fullmatch(url.strip())
    if m is None:
        raise ValueError(f"Invalid topic link: {url!r}")
    chat_id = f"-100{m.group(1)}"
    thread_id = int(m.group(2))
    return chat_id, thread_id


def _pick_first_number(record: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _display_station_name(record: dict[str, Any], slug: str | None) -> str:
    station = str(record.get("station") or "").strip()
    if station:
        return STATION_CANONICAL_DISPLAY.get(station, station)

    if slug:
        m = SLUG_STATION_RE.search(slug)
        if m:
            token = m.group(1)
            return STATION_TOKEN_TO_DISPLAY.get(token, token.replace("-", " ").title())
    return "Unknown"


def _parse_market_suffix(slug: str | None, fallback_strike_k: Any) -> str:
    if not slug:
        if fallback_strike_k is None:
            return "Unknown strike"
        return f"{fallback_strike_k} \N{DEGREE SIGN}C"

    m_c = SLUG_EXACT_C_RE.search(slug)
    if m_c:
        token = m_c.group(1)
        strike = -int(token.split("-", 1)[1]) if token.startswith("neg-") else int(token)
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


def _format_market_name(record: dict[str, Any]) -> str:
    slug = str(record.get("slug") or "").strip() or None
    station = _display_station_name(record, slug)
    strike_text = _parse_market_suffix(slug, record.get("strike_k"))

    day_text = "Unknown date"
    day_raw = record.get("market_day_local")
    if day_raw is not None:
        try:
            parsed = datetime.fromisoformat(str(day_raw)).date()
            day_text = f"{parsed.day} {parsed.strftime('%B %Y')}"
        except ValueError:
            pass

    return f"{day_text} {station} {strike_text}".strip()


def _infer_trade_action(record: dict[str, Any]) -> str | None:
    decision = str(record.get("decision") or "").strip().upper()
    side = str(record.get("side") or "").strip().upper()

    if decision in {"RESOLVE", "RESOLUTION"}:
        return "RESOLUTION"
    if side == "SELL" or decision == "SELL":
        return "SELL"
    if side == "BUY" or decision in {"TRADE", "BUY"}:
        return "BUY"
    return None


def _format_float(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{ndigits}f}"


def _format_money(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{abs(value):.2f}"


def format_trade_message(record: dict[str, Any]) -> str | None:
    action = _infer_trade_action(record)
    if action is None:
        return None

    market = _format_market_name(record)
    lot = _pick_first_number(record, ["size", "lot", "quantity"])

    if action == "BUY":
        price = _pick_first_number(record, ["chosen_no_ask", "price", "entry_price", "buy_price"])
        return (
            f"\N{SLOT MACHINE}Market: {market} | BUY | "
            f"Price: {_format_float(price)} | Lot: {_format_float(lot, ndigits=2)}"
        )

    if action == "SELL":
        buy_price = _pick_first_number(record, ["buy_price", "entry_price", "avg_entry_price"])
        sell_price = _pick_first_number(record, ["sell_price", "chosen_no_ask", "price", "exit_price"])
        pnl = _pick_first_number(record, ["pnl_realized", "pnl"])
        if pnl is None and buy_price is not None and sell_price is not None and lot is not None:
            pnl = (sell_price - buy_price) * lot

        if pnl is None or pnl >= 0:
            return (
                f"\N{LARGE GREEN CIRCLE} Market: {market} | SELL | "
                f"Buy Price: {_format_float(buy_price)} | Sell Price: {_format_float(sell_price)} | "
                f"Lot: {_format_float(lot, ndigits=2)} | Total Profit: {_format_money(pnl)}"
            )
        return (
            f"\N{LARGE RED CIRCLE} Market: {market} | SELL | "
            f"Buy Price: {_format_float(buy_price)} | Sell Price: {_format_float(sell_price)} | "
            f"Lot: {_format_float(lot, ndigits=2)} | Total Loss: {_format_money(pnl)}"
        )

    buy_price = _pick_first_number(record, ["buy_price", "entry_price", "chosen_no_ask"])
    pnl = _pick_first_number(record, ["pnl_realized", "pnl"])
    if pnl is None or pnl >= 0:
        return (
            f"\N{LARGE GREEN CIRCLE} Market: {market} | Resolution | "
            f"Buy Price: {_format_float(buy_price)} | Lot: {_format_float(lot, ndigits=2)} | "
            f"Total Profit: {_format_money(pnl)}"
        )
    return (
        f"\N{LARGE RED CIRCLE} Market: {market} | Resolution | "
        f"Buy Price: {_format_float(buy_price)} | Lot: {_format_float(lot, ndigits=2)} | "
        f"Total Loss: {_format_money(pnl)}"
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


class TelegramNotifier:
    def __init__(
        self,
        *,
        enabled: bool,
        send_enabled: bool,
        bot_token: str | None,
        trades_chat_id: str | None,
        trades_thread_id: int | None,
        daily_chat_id: str | None,
        daily_thread_id: int | None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.send_enabled = bool(send_enabled)
        self.bot_token = bot_token
        self.trades_chat_id = trades_chat_id
        self.trades_thread_id = trades_thread_id
        self.daily_chat_id = daily_chat_id
        self.daily_thread_id = daily_thread_id
        self.timeout_seconds = float(timeout_seconds)

    @classmethod
    def from_config(
        cls,
        *,
        cfg: dict[str, Any],
        repo_root: Path,
        send_enabled: bool,
        logger: Any,
    ) -> "TelegramNotifier":
        tcfg = cfg.get("telegram_notifications", {})
        if not isinstance(tcfg, dict) or not bool(tcfg.get("enabled", False)):
            return cls(
                enabled=False,
                send_enabled=False,
                bot_token=None,
                trades_chat_id=None,
                trades_thread_id=None,
                daily_chat_id=None,
                daily_thread_id=None,
            )

        cred_path = Path(str(tcfg.get("credentials_file", ".secrets/telegram_bot.json")))
        if not cred_path.is_absolute():
            cred_path = (repo_root / cred_path).resolve()
        if not cred_path.exists():
            logger.warning("Telegram disabled: credentials file not found: %s", cred_path)
            return cls(
                enabled=False,
                send_enabled=False,
                bot_token=None,
                trades_chat_id=None,
                trades_thread_id=None,
                daily_chat_id=None,
                daily_thread_id=None,
            )

        try:
            creds = json.loads(cred_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Telegram disabled: invalid credentials file (%s): %s", cred_path, exc)
            return cls(
                enabled=False,
                send_enabled=False,
                bot_token=None,
                trades_chat_id=None,
                trades_thread_id=None,
                daily_chat_id=None,
                daily_thread_id=None,
            )

        bot_token = str(creds.get("bot_token", "")).strip()
        if not bot_token:
            logger.warning("Telegram disabled: bot_token missing in %s", cred_path)
            return cls(
                enabled=False,
                send_enabled=False,
                bot_token=None,
                trades_chat_id=None,
                trades_thread_id=None,
                daily_chat_id=None,
                daily_thread_id=None,
            )

        trades_link = str(tcfg.get("trades_topic_link", "")).strip()
        daily_link = str(tcfg.get("daily_topic_link", "")).strip()
        try:
            trades_chat_id, trades_thread_id = parse_topic_link(trades_link)
            daily_chat_id, daily_thread_id = parse_topic_link(daily_link)
        except ValueError as exc:
            logger.warning("Telegram disabled: %s", exc)
            return cls(
                enabled=False,
                send_enabled=False,
                bot_token=None,
                trades_chat_id=None,
                trades_thread_id=None,
                daily_chat_id=None,
                daily_thread_id=None,
            )

        logger.info(
            "Telegram notifier active (send_enabled=%s) trades=%s/%s daily=%s/%s",
            bool(send_enabled),
            trades_chat_id,
            trades_thread_id,
            daily_chat_id,
            daily_thread_id,
        )
        return cls(
            enabled=True,
            send_enabled=bool(send_enabled),
            bot_token=bot_token,
            trades_chat_id=trades_chat_id,
            trades_thread_id=trades_thread_id,
            daily_chat_id=daily_chat_id,
            daily_thread_id=daily_thread_id,
            timeout_seconds=float(tcfg.get("timeout_seconds", 20.0)),
        )

    def _send_text(self, *, chat_id: str, thread_id: int, text: str) -> int:
        if not self.enabled or not self.send_enabled or not self.bot_token:
            return -1
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "message_thread_id": str(thread_id),
            "text": text,
            "disable_web_page_preview": "true",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, data=data)
            resp.raise_for_status()
            payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API error: {payload}")
        result = payload.get("result") or {}
        try:
            return int(result.get("message_id"))
        except (TypeError, ValueError):
            return -1

    def notify_trade(self, payload: dict[str, Any], *, logger: Any) -> None:
        if not self.enabled:
            return
        msg = format_trade_message(payload)
        if not msg:
            return
        if not self.send_enabled:
            logger.info("Telegram trade (suppressed dry-run): %s", msg)
            return
        if self.trades_chat_id is None or self.trades_thread_id is None:
            return
        try:
            self._send_text(chat_id=self.trades_chat_id, thread_id=self.trades_thread_id, text=msg)
        except Exception:
            logger.exception("Failed to send trade telegram message")

    def notify_daily_report(self, *, telegram_text_path: Path, logger: Any) -> None:
        if not self.enabled:
            return
        if self.daily_chat_id is None or self.daily_thread_id is None:
            return
        if not telegram_text_path.exists():
            return
        content = telegram_text_path.read_text(encoding="utf-8").strip()
        chunks = chunk_text(content)
        if not chunks:
            return

        if not self.send_enabled:
            logger.info("Telegram daily report (suppressed dry-run): %s chunks=%d", telegram_text_path, len(chunks))
            return
        for chunk in chunks:
            try:
                self._send_text(chat_id=self.daily_chat_id, thread_id=self.daily_thread_id, text=chunk)
            except Exception:
                logger.exception("Failed to send daily report telegram message")
                return
