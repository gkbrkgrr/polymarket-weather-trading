from __future__ import annotations

import hashlib
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo


@dataclass
class ExecutionResult:
    order_id: str
    order_status: str
    filled_price: float | None
    filled_size: float | None
    ts_utc: str


class ExecutionClient(ABC):
    @abstractmethod
    def place_order(
        self,
        market_id: str,
        side: str,
        outcome: str,
        price: float,
        size: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_open_orders(self, market_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError


class DummyExecutionClient(ExecutionClient):
    def __init__(
        self,
        *,
        price_tick: float = 0.001,
        conservative_fill: bool = True,
        realism_enabled: bool = True,
        deterministic_seed: int = 0,
        fill_probability_base: float = 0.9,
        partial_fill_probability: float = 0.2,
        max_slippage_ticks: int = 3,
    ) -> None:
        self.price_tick = float(price_tick)
        self.conservative_fill = bool(conservative_fill)
        self.realism_enabled = bool(realism_enabled)
        self.deterministic_seed = int(deterministic_seed)
        self.fill_probability_base = min(1.0, max(0.0, float(fill_probability_base)))
        self.partial_fill_probability = min(1.0, max(0.0, float(partial_fill_probability)))
        self.max_slippage_ticks = max(0, int(max_slippage_ticks))
        self._orders: dict[str, dict[str, Any]] = {}

    def _round_tick(self, price: float) -> float:
        ticks = round(float(price) / self.price_tick)
        return round(ticks * self.price_tick, 6)

    def _hash_uniform(self, key: str) -> float:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        n = int(digest[:16], 16)
        return n / float(0xFFFFFFFFFFFFFFFF)

    def _deterministic_uniform(self, order_key: str, channel: str) -> float:
        return self._hash_uniform(f"{self.deterministic_seed}|{order_key}|{channel}")

    def place_order(
        self,
        market_id: str,
        side: str,
        outcome: str,
        price: float,
        size: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        order_id = f"paper_{uuid.uuid4().hex[:16]}"
        requested_price = self._round_tick(price)
        now = datetime.now(tz=ZoneInfo("UTC")).isoformat()
        spread: float | None = None
        if metadata is not None:
            try:
                raw_spread = metadata.get("spread")
                if raw_spread is not None:
                    spread = max(0.0, float(raw_spread))
            except Exception:
                spread = None

        if not self.realism_enabled:
            if self.conservative_fill:
                filled_price = min(1.0, self._round_tick(requested_price + self.price_tick))
            else:
                filled_price = requested_price
            status = "filled"
            filled_size = float(size)
        else:
            order_key = "|".join(
                [
                    str(market_id),
                    str(side).lower(),
                    str(outcome).upper(),
                    f"{requested_price:.6f}",
                    f"{float(size):.6f}",
                    f"{spread if spread is not None else -1.0:.6f}",
                ]
            )
            spread_penalty = min(0.45, (float(spread) if spread is not None else 0.02) * 4.0)
            conservative_penalty = 0.04 if self.conservative_fill else 0.0
            fill_probability = min(0.995, max(0.05, self.fill_probability_base - spread_penalty - conservative_penalty))
            partial_probability = min(0.95, max(0.0, self.partial_fill_probability + spread_penalty * 0.35))

            roll_fill = self._deterministic_uniform(order_key, "fill")
            roll_partial = self._deterministic_uniform(order_key, "partial")
            roll_fraction = self._deterministic_uniform(order_key, "fraction")
            roll_slip = self._deterministic_uniform(order_key, "slippage")

            if roll_fill > fill_probability:
                status = "cancelled"
                filled_size = 0.0
                filled_price = None
            else:
                is_partial = roll_partial < partial_probability
                if is_partial:
                    # Keep partial fills meaningful and deterministic.
                    fraction = 0.35 + 0.55 * roll_fraction
                    filled_size = max(0.0, min(float(size), round(float(size) * fraction, 6)))
                    status = "partially_filled" if filled_size > 0 else "cancelled"
                else:
                    filled_size = float(size)
                    status = "filled"

                spread_ticks = 0
                if spread is not None:
                    spread_ticks = int(max(0, round(float(spread) / max(self.price_tick, 1e-9))))
                slip_base = min(self.max_slippage_ticks, max(0, int(math.ceil(spread_ticks * 0.3))))
                slip_extra_cap = max(0, self.max_slippage_ticks - slip_base)
                slip_extra = int(math.floor(roll_slip * (slip_extra_cap + 1)))
                slip_ticks = min(self.max_slippage_ticks, slip_base + slip_extra)
                if self.conservative_fill and filled_size > 0 and slip_ticks == 0:
                    slip_ticks = min(1, self.max_slippage_ticks)

                filled_price = min(1.0, self._round_tick(requested_price + slip_ticks * self.price_tick))

        self._orders[order_id] = {
            "order_id": order_id,
            "market_id": str(market_id),
            "side": str(side),
            "outcome": str(outcome),
            "requested_price": requested_price,
            "spread": spread,
            "size": float(size),
            "status": status,
            "filled_price": filled_price,
            "filled_size": float(filled_size),
            "created_ts_utc": now,
            "updated_ts_utc": now,
        }
        return order_id

    def cancel_order(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if not order:
            return
        if order.get("status") == "filled":
            return
        order["status"] = "cancelled"
        order["updated_ts_utc"] = datetime.now(tz=ZoneInfo("UTC")).isoformat()

    def get_open_orders(self, market_id: str | None = None) -> list[dict[str, Any]]:
        orders = [o for o in self._orders.values() if o.get("status") in {"open", "submitted"}]
        if market_id is not None:
            orders = [o for o in orders if str(o.get("market_id")) == str(market_id)]
        return orders

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        return self._orders.get(order_id)

    def execution_result(self, order_id: str) -> ExecutionResult | None:
        order = self._orders.get(order_id)
        if not order:
            return None
        return ExecutionResult(
            order_id=order_id,
            order_status=str(order.get("status")),
            filled_price=float(order["filled_price"]) if order.get("filled_price") is not None else None,
            filled_size=float(order["filled_size"]) if order.get("filled_size") is not None else None,
            ts_utc=str(order.get("updated_ts_utc") or order.get("created_ts_utc")),
        )


class RealExecutionClient(ExecutionClient):
    def place_order(
        self,
        market_id: str,
        side: str,
        outcome: str,
        price: float,
        size: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError("RealExecutionClient is intentionally stubbed. Integrate exchange API client here.")

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("RealExecutionClient is intentionally stubbed. Integrate exchange API client here.")

    def get_open_orders(self, market_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError("RealExecutionClient is intentionally stubbed. Integrate exchange API client here.")
