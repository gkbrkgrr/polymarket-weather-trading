from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def coerce_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


@dataclass(frozen=True)
class CursorState:
    last_ts: datetime
    last_tiebreak: str


def advance_cursor(current: CursorState, ts: datetime, tiebreak: str) -> CursorState:
    if ts > current.last_ts:
        return CursorState(ts, tiebreak)
    if ts == current.last_ts and tiebreak > current.last_tiebreak:
        return CursorState(ts, tiebreak)
    return current


def surrogate_trade_id(
    market_id: str,
    ts: datetime,
    price: Decimal,
    size: Decimal,
    outcome_id: str | None,
    side: str | None,
    tx_hash: str | None,
    outcome_index: int | None = None,
) -> str:
    payload = "|".join(
        [
            market_id,
            ts.astimezone(timezone.utc).isoformat(),
            f"{price}",
            f"{size}",
            outcome_id or "",
            str(outcome_index) if outcome_index is not None else "",
            side or "",
            tx_hash or "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
