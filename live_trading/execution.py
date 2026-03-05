from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any
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
    def place_order(self, market_id: str, side: str, outcome: str, price: float, size: float) -> str:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_open_orders(self, market_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError


class DummyExecutionClient(ExecutionClient):
    def __init__(self, *, price_tick: float = 0.001, conservative_fill: bool = True) -> None:
        self.price_tick = float(price_tick)
        self.conservative_fill = bool(conservative_fill)
        self._orders: dict[str, dict[str, Any]] = {}

    def _round_tick(self, price: float) -> float:
        ticks = round(float(price) / self.price_tick)
        return round(ticks * self.price_tick, 6)

    def place_order(self, market_id: str, side: str, outcome: str, price: float, size: float) -> str:
        order_id = f"paper_{uuid.uuid4().hex[:16]}"
        requested_price = self._round_tick(price)
        if self.conservative_fill:
            filled_price = min(1.0, self._round_tick(requested_price + self.price_tick))
        else:
            filled_price = requested_price

        now = datetime.now(tz=ZoneInfo("UTC")).isoformat()
        self._orders[order_id] = {
            "order_id": order_id,
            "market_id": str(market_id),
            "side": str(side),
            "outcome": str(outcome),
            "requested_price": requested_price,
            "size": float(size),
            "status": "filled",
            "filled_price": filled_price,
            "filled_size": float(size),
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
    def place_order(self, market_id: str, side: str, outcome: str, price: float, size: float) -> str:
        raise NotImplementedError("RealExecutionClient is intentionally stubbed. Integrate exchange API client here.")

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("RealExecutionClient is intentionally stubbed. Integrate exchange API client here.")

    def get_open_orders(self, market_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError("RealExecutionClient is intentionally stubbed. Integrate exchange API client here.")
