from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import orjson
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import yaml
from pydantic import BaseModel, Field

from polymarket_archive.utils import parse_datetime


class Settings(BaseModel):
    postgres_dsn: str = Field(..., description="PostgreSQL DSN")
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    data_base_url: str = "https://data-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    title_filter: str | None = "Highest temperature"
    market_title_contains: list[str] = Field(default_factory=list)
    market_filters: list[str] = Field(default_factory=list)
    market_tag_ids: list[int] = Field(default_factory=list)
    target_market_ids: list[str] = Field(default_factory=list)
    backfill_start: datetime = Field(
        default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    poll_interval_seconds: int = 30
    discovery_interval_seconds: int = 300
    concurrency: int = 10
    raw_dir: Path = Path("./data/raw")
    feature_clob: bool = False
    request_timeout_seconds: int = 20
    max_retries: int = 5
    markets_page_size: int = 200
    trades_page_size: int = 200
    book_snapshot_interval_seconds: int = 5
    rate_limit_per_second: int = 10
    log_level: str = "INFO"
    snapshot_interval_seconds: int | None = None


def _load_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(raw) or {}
    if path.suffix.lower() == ".json":
        return orjson.loads(raw)
    raise ValueError(f"Unsupported config format: {path}")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _coerce_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    return int(str(value))


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        if value.strip().startswith("["):
            try:
                parsed = orjson.loads(value)
            except orjson.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()]


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            out.append(_coerce_int(text))
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = orjson.loads(text)
            except orjson.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return _coerce_int_list(parsed)
        return [_coerce_int(part.strip()) for part in text.split(",") if part.strip()]
    return [_coerce_int(value)]


def _merge_settings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _dsn_with_master_db(dsn: str, dbname: str = "master_db") -> str:
    parts = conninfo_to_dict(dsn)
    if not parts:
        return dsn
    parts["dbname"] = dbname
    return make_conninfo(**parts)


def load_settings(config_path: str | Path | None = None) -> Settings:
    data: dict[str, Any] = {}
    if config_path:
        data = _load_file(Path(config_path))

    file_master_dsn = data.get("master_postgres_dsn") or data.get("MASTER_POSTGRES_DSN")
    file_base_dsn = data.get("postgres_dsn") or data.get("POSTGRES_DSN")

    file_overrides = {
        "postgres_dsn": (
            file_master_dsn
            if file_master_dsn
            else (_dsn_with_master_db(file_base_dsn) if file_base_dsn else None)
        ),
        "gamma_base_url": data.get("gamma_base_url") or data.get("GAMMA_BASE_URL"),
        "data_base_url": data.get("data_base_url") or data.get("DATA_BASE_URL"),
        "clob_base_url": data.get("clob_base_url") or data.get("CLOB_BASE_URL"),
        "clob_ws_url": data.get("clob_ws_url") or data.get("CLOB_WS_URL"),
        "title_filter": data.get("title_filter") or data.get("TITLE_FILTER"),
        "market_title_contains": data.get("market_title_contains")
        or data.get("MARKET_TITLE_CONTAINS"),
        "market_filters": data.get("market_filters") or data.get("MARKET_FILTERS"),
        "market_tag_ids": data.get("market_tag_ids") or data.get("MARKET_TAG_IDS"),
        "target_market_ids": data.get("target_market_ids")
        or data.get("TARGET_MARKET_IDS"),
        "backfill_start": data.get("backfill_start") or data.get("BACKFILL_START"),
        "poll_interval_seconds": data.get("poll_interval_seconds")
        or data.get("POLL_INTERVAL_SECONDS"),
        "discovery_interval_seconds": data.get("discovery_interval_seconds")
        or data.get("DISCOVERY_INTERVAL_SECONDS"),
        "concurrency": data.get("concurrency") or data.get("CONCURRENCY"),
        "raw_dir": data.get("raw_dir") or data.get("RAW_DIR"),
        "feature_clob": data.get("feature_clob") or data.get("FEATURE_CLOB"),
        "request_timeout_seconds": data.get("request_timeout_seconds"),
        "max_retries": data.get("max_retries"),
        "markets_page_size": data.get("markets_page_size"),
        "trades_page_size": data.get("trades_page_size"),
        "book_snapshot_interval_seconds": data.get("book_snapshot_interval_seconds"),
        "rate_limit_per_second": data.get("rate_limit_per_second"),
        "log_level": data.get("log_level"),
        "snapshot_interval_seconds": data.get("snapshot_interval_seconds"),
    }
    file_overrides = {key: value for key, value in file_overrides.items() if value is not None}

    env = os.environ
    env_overrides: dict[str, Any] = {}
    if env.get("MASTER_POSTGRES_DSN"):
        env_overrides["postgres_dsn"] = env.get("MASTER_POSTGRES_DSN")
    elif env.get("POSTGRES_DSN"):
        env_overrides["postgres_dsn"] = _dsn_with_master_db(env.get("POSTGRES_DSN") or "")
    if env.get("GAMMA_BASE_URL"):
        env_overrides["gamma_base_url"] = env.get("GAMMA_BASE_URL")
    if env.get("DATA_BASE_URL"):
        env_overrides["data_base_url"] = env.get("DATA_BASE_URL")
    if env.get("CLOB_BASE_URL"):
        env_overrides["clob_base_url"] = env.get("CLOB_BASE_URL")
    if env.get("CLOB_WS_URL"):
        env_overrides["clob_ws_url"] = env.get("CLOB_WS_URL")
    if env.get("TITLE_FILTER"):
        env_overrides["title_filter"] = env.get("TITLE_FILTER")
    if env.get("MARKET_TITLE_CONTAINS"):
        env_overrides["market_title_contains"] = env.get("MARKET_TITLE_CONTAINS")
    if env.get("MARKET_FILTERS"):
        env_overrides["market_filters"] = env.get("MARKET_FILTERS")
    if env.get("MARKET_TAG_IDS"):
        env_overrides["market_tag_ids"] = env.get("MARKET_TAG_IDS")
    if env.get("TARGET_MARKET_IDS"):
        env_overrides["target_market_ids"] = env.get("TARGET_MARKET_IDS")
    if env.get("BACKFILL_START"):
        env_overrides["backfill_start"] = env.get("BACKFILL_START")
    if env.get("POLL_INTERVAL_SECONDS"):
        env_overrides["poll_interval_seconds"] = env.get("POLL_INTERVAL_SECONDS")
    if env.get("DISCOVERY_INTERVAL_SECONDS"):
        env_overrides["discovery_interval_seconds"] = env.get("DISCOVERY_INTERVAL_SECONDS")
    if env.get("CONCURRENCY"):
        env_overrides["concurrency"] = env.get("CONCURRENCY")
    if env.get("RAW_DIR"):
        env_overrides["raw_dir"] = env.get("RAW_DIR")
    if env.get("FEATURE_CLOB"):
        env_overrides["feature_clob"] = env.get("FEATURE_CLOB")
    if env.get("REQUEST_TIMEOUT_SECONDS"):
        env_overrides["request_timeout_seconds"] = env.get("REQUEST_TIMEOUT_SECONDS")
    if env.get("MAX_RETRIES"):
        env_overrides["max_retries"] = env.get("MAX_RETRIES")
    if env.get("MARKETS_PAGE_SIZE"):
        env_overrides["markets_page_size"] = env.get("MARKETS_PAGE_SIZE")
    if env.get("TRADES_PAGE_SIZE"):
        env_overrides["trades_page_size"] = env.get("TRADES_PAGE_SIZE")
    if env.get("BOOK_SNAPSHOT_INTERVAL_SECONDS"):
        env_overrides["book_snapshot_interval_seconds"] = env.get("BOOK_SNAPSHOT_INTERVAL_SECONDS")
    if env.get("RATE_LIMIT_PER_SECOND"):
        env_overrides["rate_limit_per_second"] = env.get("RATE_LIMIT_PER_SECOND")
    if env.get("LOG_LEVEL"):
        env_overrides["log_level"] = env.get("LOG_LEVEL")
    if env.get("SNAPSHOT_INTERVAL_SECONDS"):
        env_overrides["snapshot_interval_seconds"] = env.get("SNAPSHOT_INTERVAL_SECONDS")

    merged = _merge_settings(file_overrides, env_overrides)

    if merged.get("raw_dir"):
        merged["raw_dir"] = Path(merged["raw_dir"])
    if merged.get("feature_clob") is not None:
        merged["feature_clob"] = _coerce_bool(merged["feature_clob"])
    for key in ("market_title_contains", "market_filters", "target_market_ids"):
        if merged.get(key) is not None:
            merged[key] = _coerce_str_list(merged.get(key))
    if merged.get("market_tag_ids") is not None:
        merged["market_tag_ids"] = _coerce_int_list(merged.get("market_tag_ids"))
    for key in (
        "poll_interval_seconds",
        "discovery_interval_seconds",
        "concurrency",
        "request_timeout_seconds",
        "max_retries",
        "markets_page_size",
        "trades_page_size",
        "book_snapshot_interval_seconds",
        "rate_limit_per_second",
        "snapshot_interval_seconds",
    ):
        if merged.get(key) is not None:
            merged[key] = _coerce_int(merged[key])

    if merged.get("backfill_start") is not None and not isinstance(
        merged["backfill_start"], datetime
    ):
        parsed = parse_datetime(merged["backfill_start"])
        if parsed is not None:
            merged["backfill_start"] = parsed

    if merged.get("snapshot_interval_seconds") is not None:
        merged["poll_interval_seconds"] = merged["snapshot_interval_seconds"]

    return Settings(**merged)
