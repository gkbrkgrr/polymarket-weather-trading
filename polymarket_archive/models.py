from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polymarket_archive.utils import coerce_decimal, parse_datetime


class GammaOutcome(BaseModel):
    model_config = ConfigDict(extra="allow")

    outcome_id: Optional[str] = None
    outcome_label: Optional[str] = None
    outcome_index: int = Field(...)
    raw: dict[str, Any]


class GammaMarket(BaseModel):
    model_config = ConfigDict(extra="allow")

    market_id: str
    slug: Optional[str] = None
    title: str
    status: Optional[str] = None
    event_start_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None
    raw: dict[str, Any]
    outcomes: List[GammaOutcome] = Field(default_factory=list)

    @field_validator("event_start_time", "resolution_time", mode="before")
    @classmethod
    def _parse_dt(cls, value: Any) -> Any:
        return parse_datetime(value)


class Trade(BaseModel):
    model_config = ConfigDict(extra="allow")

    trade_id: str
    market_id: str
    ts: datetime
    outcome_id: Optional[str] = None
    outcome_index: Optional[int] = None
    side: Optional[str] = None
    price: Decimal
    size: Decimal
    tx_hash: Optional[str] = None
    raw: dict[str, Any]

    @field_validator("ts", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> Any:
        return parse_datetime(value)

    @field_validator("price", "size", mode="before")
    @classmethod
    def _parse_decimal(cls, value: Any) -> Any:
        parsed = coerce_decimal(value)
        if parsed is None:
            raise ValueError("Invalid decimal")
        return parsed
