from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class PricingDecision:
    chosen_no_ask: float | None
    price_source: str | None
    snapshot_ts_utc: datetime | None
    snapshot_age_minutes: float | None
    spread: float | None
    skipped_reason: str | None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v):
        return None
    return float(v)


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def compute_pricing_decision(
    *,
    snapshot: dict[str, Any] | None,
    now_utc: datetime,
    max_snapshot_age_minutes: float,
    slippage_buffer_yes_fallback: float,
    max_spread: float,
) -> PricingDecision:
    if not snapshot:
        return PricingDecision(
            chosen_no_ask=None,
            price_source=None,
            snapshot_ts_utc=None,
            snapshot_age_minutes=None,
            spread=None,
            skipped_reason="no_snapshot",
        )

    no_ts = _as_utc(snapshot.get("no_snapshot_ts_utc"))
    yes_ts = _as_utc(snapshot.get("yes_snapshot_ts_utc"))
    best_no_bid = _as_float(snapshot.get("best_no_bid"))
    best_no_ask = _as_float(snapshot.get("best_no_ask"))
    best_yes_bid = _as_float(snapshot.get("best_yes_bid"))
    best_yes_ask = _as_float(snapshot.get("best_yes_ask"))

    price_source: str | None = None
    chosen_price: float | None = None
    snapshot_ts: datetime | None = None

    if best_no_ask is not None:
        chosen_price = best_no_ask
        price_source = "NO_ask"
        snapshot_ts = no_ts or yes_ts
    elif best_yes_bid is not None:
        chosen_price = min(1.0, max(0.0, 1.0 - best_yes_bid + float(slippage_buffer_yes_fallback)))
        price_source = "YES_bid_fallback"
        snapshot_ts = yes_ts or no_ts
    else:
        return PricingDecision(
            chosen_no_ask=None,
            price_source=None,
            snapshot_ts_utc=None,
            snapshot_age_minutes=None,
            spread=None,
            skipped_reason="no_snapshot",
        )

    if snapshot_ts is None:
        return PricingDecision(
            chosen_no_ask=None,
            price_source=price_source,
            snapshot_ts_utc=None,
            snapshot_age_minutes=None,
            spread=None,
            skipped_reason="no_snapshot",
        )

    snapshot_age_minutes = max(0.0, (now_utc - snapshot_ts).total_seconds() / 60.0)
    if snapshot_age_minutes > float(max_snapshot_age_minutes):
        return PricingDecision(
            chosen_no_ask=chosen_price,
            price_source=price_source,
            snapshot_ts_utc=snapshot_ts,
            snapshot_age_minutes=snapshot_age_minutes,
            spread=None,
            skipped_reason="snapshot_too_old",
        )

    spread: float | None = None
    if best_no_bid is not None and best_no_ask is not None:
        spread = max(0.0, best_no_ask - best_no_bid)
    elif price_source == "YES_bid_fallback" and best_yes_bid is not None and best_yes_ask is not None:
        spread = max(0.0, best_yes_ask - best_yes_bid)

    if spread is None or spread > float(max_spread):
        return PricingDecision(
            chosen_no_ask=chosen_price,
            price_source=price_source,
            snapshot_ts_utc=snapshot_ts,
            snapshot_age_minutes=snapshot_age_minutes,
            spread=spread,
            skipped_reason="spread_too_wide",
        )

    return PricingDecision(
        chosen_no_ask=chosen_price,
        price_source=price_source,
        snapshot_ts_utc=snapshot_ts,
        snapshot_age_minutes=snapshot_age_minutes,
        spread=spread,
        skipped_reason=None,
    )
