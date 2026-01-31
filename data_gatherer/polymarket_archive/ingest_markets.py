from __future__ import annotations

from typing import Iterable

from polymarket_archive.db import Database
from polymarket_archive.models import GammaMarket


async def ingest_markets(db: Database, markets: Iterable[GammaMarket]) -> None:
    market_list = list(markets)
    if not market_list:
        return
    await db.upsert_markets(market_list)
    for market in market_list:
        await db.upsert_outcomes(market.outcomes, market.market_id)
