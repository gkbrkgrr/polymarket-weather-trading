from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import httpx
import orjson
from pydantic import ValidationError

from polymarket_archive.http import build_request_info, fetch_json, RequestLimiter
from polymarket_archive.models import GammaMarket, GammaOutcome
from polymarket_archive.raw_sink import RawSink


class GammaClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient,
        limiter: RequestLimiter,
        max_retries: int,
        raw_sink: RawSink,
        run_id: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.limiter = limiter
        self.max_retries = max_retries
        self.raw_sink = raw_sink
        self.run_id = run_id

    async def list_markets(
        self, search: str | None, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], datetime]:
        url = f"{self.base_url}/markets"
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        ts = datetime.now(timezone.utc)
        payload = await fetch_json(
            self.client, "GET", url, params, limiter=self.limiter, max_retries=self.max_retries
        )
        request_info = build_request_info(url, params, cursor=str(offset))
        self.raw_sink.write_record("gamma_discovery", ts, request_info, payload, self.run_id)
        markets = _extract_markets(payload)
        if markets:
            self.raw_sink.write_records("gamma_markets", ts, request_info, markets, self.run_id)
        return markets, ts


def _extract_markets(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("markets"), list):
            return payload["markets"]
    if isinstance(payload, list):
        return payload
    return []


def parse_market(payload: dict[str, Any]) -> GammaMarket | None:
    market_id = payload.get("id") or payload.get("market_id") or payload.get("marketId")
    title = payload.get("title") or payload.get("question")
    if not market_id or not title:
        return None
    resolution_time = (
        payload.get("resolution_time")
        or payload.get("resolutionTime")
        or payload.get("resolvedAt")
        or payload.get("resolved_at")
        or payload.get("resolutionDate")
        or payload.get("closedTime")
        or payload.get("closed_time")
    )
    status = _derive_status(payload, resolution_time)
    outcomes_payload = _normalize_outcomes(payload)
    outcomes: list[GammaOutcome] = []
    if isinstance(outcomes_payload, list):
        for idx, outcome in enumerate(outcomes_payload):
            if isinstance(outcome, dict):
                outcome_id = outcome.get("id") or outcome.get("outcome_id")
                outcome_label = outcome.get("label") or outcome.get("name") or outcome.get("outcome")
                outcome_index = outcome.get("index") if outcome.get("index") is not None else idx
                raw = outcome
            else:
                outcome_id = None
                outcome_label = str(outcome)
                outcome_index = idx
                raw = {"value": outcome}
            try:
                outcomes.append(
                    GammaOutcome(
                        outcome_id=outcome_id,
                        outcome_label=outcome_label,
                        outcome_index=outcome_index,
                        raw=raw,
                    )
                )
            except ValidationError:
                continue
    try:
        return GammaMarket(
            market_id=str(market_id),
            slug=payload.get("slug"),
            title=title,
            status=status,
            event_start_time=payload.get("event_start_time")
            or payload.get("eventStartTime")
            or payload.get("startDate")
            or payload.get("startDateIso")
            or payload.get("endDate")
            or payload.get("endDateIso"),
            resolution_time=resolution_time,
            raw=payload,
            outcomes=outcomes,
        )
    except ValidationError:
        return None


def _derive_status(payload: dict[str, Any], resolution_time: Any) -> str | None:
    status = payload.get("status")
    if isinstance(status, str) and status.strip():
        return status
    resolved_flag = payload.get("resolved") or payload.get("isResolved")
    if resolved_flag or resolution_time:
        return "resolved"
    if payload.get("closed") is True:
        return "closed"
    if payload.get("archived") is True:
        return "archived"
    if payload.get("active") is True:
        return "active"
    return None


def _normalize_outcomes(payload: dict[str, Any]) -> list[Any]:
    outcomes_payload = payload.get("outcomes") or payload.get("outcomeDescriptions") or []
    if isinstance(outcomes_payload, str):
        try:
            decoded = orjson.loads(outcomes_payload)
        except orjson.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            outcomes_payload = decoded
        else:
            outcomes_payload = [outcomes_payload]
    if isinstance(outcomes_payload, list):
        return outcomes_payload
    return []


def filter_markets(markets: Iterable[GammaMarket], title_filter: str) -> list[GammaMarket]:
    needle = title_filter.lower()
    return [market for market in markets if needle in market.title.lower()]
