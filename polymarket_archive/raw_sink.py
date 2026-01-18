from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import orjson


@dataclass(frozen=True)
class RawRecord:
    ingest_ts: str
    source: str
    request: dict[str, Any]
    payload: Any
    run_id: str
    market_id: str | None = None


class RawSink:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def _build_path(self, source: str, ts: datetime, market_id: str | None = None) -> Path:
        ts = ts.astimezone(timezone.utc)
        stamp = ts.strftime("%Y%m%d_%H")
        folder = ts.strftime("%Y/%m/%d/%H")
        if source == "gamma_discovery":
            return self.base_dir / "gamma_discovery" / folder / f"discovery_{stamp}.jsonl.gz"
        if source == "gamma_markets":
            return self.base_dir / "gamma_markets" / folder / f"gamma_markets_{stamp}.jsonl.gz"
        if source == "data_trades":
            if not market_id:
                raise ValueError("market_id required for data_trades")
            return (
                self.base_dir
                / "data_trades"
                / folder
                / f"market_id={market_id}"
                / f"trades_{stamp}.jsonl.gz"
            )
        if source == "clob_book":
            if not market_id:
                raise ValueError("market_id required for clob_book")
            return (
                self.base_dir
                / "clob_book"
                / folder
                / f"market_id={market_id}"
                / f"book_{stamp}.jsonl.gz"
            )
        if source == "error":
            return self.base_dir / "errors" / folder / f"errors_{stamp}.jsonl.gz"
        raise ValueError(f"Unknown source: {source}")

    def _append(self, path: Path, records: Iterable[RawRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "ab") as handle:
            for record in records:
                line = orjson.dumps(record.__dict__) + b"\n"
                handle.write(line)

    def write_record(
        self,
        source: str,
        ts: datetime,
        request: dict[str, Any],
        payload: Any,
        run_id: str,
        market_id: str | None = None,
    ) -> None:
        ingest_ts = datetime.now(timezone.utc).isoformat()
        record = RawRecord(
            ingest_ts=ingest_ts,
            source=source,
            request=request,
            payload=payload,
            run_id=run_id,
            market_id=market_id,
        )
        path = self._build_path(source, ts, market_id=market_id)
        self._append(path, [record])

    def write_records(
        self,
        source: str,
        ts: datetime,
        request: dict[str, Any],
        payloads: Iterable[Any],
        run_id: str,
        market_id: str | None = None,
    ) -> None:
        ingest_ts = datetime.now(timezone.utc).isoformat()
        records = [
            RawRecord(
                ingest_ts=ingest_ts,
                source=source,
                request=request,
                payload=payload,
                run_id=run_id,
                market_id=market_id,
            )
            for payload in payloads
        ]
        path = self._build_path(source, ts, market_id=market_id)
        self._append(path, records)
