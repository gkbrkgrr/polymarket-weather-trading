from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from master_db import resolve_master_postgres_dsn


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    color: str
    root: Path


DEFAULT_COUNTRY_ROOT = REPO_ROOT / "data" / "ml_predictions" / "city_extended"
DEFAULT_GLOBAL_V1_ROOT = REPO_ROOT / "data" / "ml_predictions" / "xgb_opt_v1_100"
DEFAULT_GLOBAL_V2_ROOT = REPO_ROOT / "data" / "ml_predictions" / "xgb_opt_v2_100"
DEFAULT_LOCATIONS_CSV = REPO_ROOT / "locations.csv"


def _resolve_path(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name)
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


def build_model_specs() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            key="country_based",
            label="Country Based",
            color="#0f766e",
            root=_resolve_path("TRADING_PANEL_COUNTRY_ROOT", DEFAULT_COUNTRY_ROOT),
        ),
        ModelSpec(
            key="global_v1",
            label="GlobalV1",
            color="#be123c",
            root=_resolve_path("TRADING_PANEL_GLOBAL_V1_ROOT", DEFAULT_GLOBAL_V1_ROOT),
        ),
        ModelSpec(
            key="global_v2",
            label="GlobalV2",
            color="#1d4ed8",
            root=_resolve_path("TRADING_PANEL_GLOBAL_V2_ROOT", DEFAULT_GLOBAL_V2_ROOT),
        ),
    )


MODEL_SPECS = build_model_specs()
LOCATIONS_CSV = _resolve_path("TRADING_PANEL_LOCATIONS_CSV", DEFAULT_LOCATIONS_CSV)
CACHE_TTL_SECONDS = int(os.getenv("TRADING_PANEL_CACHE_TTL_SECONDS", "120"))
DEFAULT_HISTORY_DAYS = int(os.getenv("TRADING_PANEL_HISTORY_DAYS", "16"))
_default_workers = max(4, min(12, os.cpu_count() or 8))
DEFAULT_PARQUET_READ_WORKERS = int(
    os.getenv("TRADING_PANEL_PARQUET_READ_WORKERS", str(_default_workers))
)
DEFAULT_HOST = os.getenv("TRADING_PANEL_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("TRADING_PANEL_PORT", "8787"))


def resolve_panel_master_dsn() -> str:
    explicit = os.getenv("MASTER_POSTGRES_DSN")
    return resolve_master_postgres_dsn(
        explicit_dsn=explicit,
        config_path=REPO_ROOT / "config.master_db.yaml",
    )
