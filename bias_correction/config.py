from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml


CYCLE_TOKEN_RE = re.compile(r"^\d{10}$")
MODEL_STRATEGIES = {
    "single_residual_with_model_feature",
    "per_model",
}


@dataclass(slots=True)
class BiasCorrectionConfig:
    predictions_root: Path
    train_start: str
    train_end: str
    backfill_start: str
    backfill_end: str
    rolling_window_days: int
    ewma_halflife_days: float
    min_history: int
    model_strategy: str
    n_jobs: int
    models: list[str] | None
    output_suffix: str
    obs_source_path: Path | None
    intermediate_dir: Path
    artifacts_root: Path
    reuse_intermediate: bool
    dry_run: bool
    log_level: str
    strict_schema: bool
    overwrite_output: bool

    @property
    def train_start_ts(self) -> pd.Timestamp:
        return parse_cycle_token(self.train_start, "train_start")

    @property
    def train_end_ts(self) -> pd.Timestamp:
        return parse_cycle_token(self.train_end, "train_end")

    @property
    def backfill_start_ts(self) -> pd.Timestamp:
        return parse_cycle_token(self.backfill_start, "backfill_start")

    def resolve_backfill_end(self, latest_cycle: pd.Timestamp) -> pd.Timestamp:
        if self.backfill_end.lower() == "latest":
            return latest_cycle
        return parse_cycle_token(self.backfill_end, "backfill_end")


DEFAULT_CONFIG: dict[str, Any] = {
    "predictions_root": "data/ml_predictions",
    "train_start": "2025010100",
    "train_end": "2025123118",
    "backfill_start": "2026010100",
    "backfill_end": "latest",
    "rolling_window_days": 45,
    "ewma_halflife_days": 14,
    "min_history": 30,
    "model_strategy": "single_residual_with_model_feature",
    "n_jobs": 8,
    "models": None,
    "output_suffix": "_biascorrected",
    "obs_source_path": None,
    "intermediate_dir": "bias_correction/intermediate",
    "artifacts_root": "bias_correction/artifacts",
    "reuse_intermediate": True,
    "dry_run": False,
    "log_level": "INFO",
    "strict_schema": False,
    "overwrite_output": True,
}


def parse_cycle_token(value: str, field_name: str) -> pd.Timestamp:
    if CYCLE_TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be YYYYMMDDHH, got {value!r}")
    return pd.to_datetime(value, format="%Y%m%d%H", utc=True)


def parse_models(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        return cleaned or None
    cleaned = [x.strip() for x in str(value).split(",") if x.strip()]
    return cleaned or None


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping at top-level: {path}")
    return loaded


def build_config(
    *,
    config_path: Path | None,
    cli_overrides: dict[str, Any],
    repo_root: Path,
) -> BiasCorrectionConfig:
    data: dict[str, Any] = dict(DEFAULT_CONFIG)
    data.update(_load_yaml(config_path))

    for key, value in cli_overrides.items():
        if value is not None:
            data[key] = value

    data["predictions_root"] = Path(data["predictions_root"]).expanduser()
    if not data["predictions_root"].is_absolute():
        data["predictions_root"] = (repo_root / data["predictions_root"]).resolve()

    data["intermediate_dir"] = Path(data["intermediate_dir"]).expanduser()
    if not data["intermediate_dir"].is_absolute():
        data["intermediate_dir"] = (repo_root / data["intermediate_dir"]).resolve()

    data["artifacts_root"] = Path(data["artifacts_root"]).expanduser()
    if not data["artifacts_root"].is_absolute():
        data["artifacts_root"] = (repo_root / data["artifacts_root"]).resolve()

    obs_source = data.get("obs_source_path")
    if obs_source is not None:
        obs_path = Path(obs_source).expanduser()
        if not obs_path.is_absolute():
            obs_path = (repo_root / obs_path).resolve()
        data["obs_source_path"] = obs_path

    data["models"] = parse_models(data.get("models"))
    data["n_jobs"] = int(data["n_jobs"])
    data["rolling_window_days"] = int(data["rolling_window_days"])
    data["min_history"] = int(data["min_history"])
    data["ewma_halflife_days"] = float(data["ewma_halflife_days"])
    data["reuse_intermediate"] = bool(data["reuse_intermediate"])
    data["dry_run"] = bool(data["dry_run"])
    data["strict_schema"] = bool(data["strict_schema"])
    data["overwrite_output"] = bool(data["overwrite_output"])

    if data["model_strategy"] not in MODEL_STRATEGIES:
        allowed = ", ".join(sorted(MODEL_STRATEGIES))
        raise ValueError(f"model_strategy must be one of [{allowed}], got {data['model_strategy']!r}")

    parse_cycle_token(str(data["train_start"]), "train_start")
    parse_cycle_token(str(data["train_end"]), "train_end")
    parse_cycle_token(str(data["backfill_start"]), "backfill_start")
    if str(data["backfill_end"]).lower() != "latest":
        parse_cycle_token(str(data["backfill_end"]), "backfill_end")

    if data["train_start"] > data["train_end"]:
        raise ValueError("train_start must be <= train_end")

    return BiasCorrectionConfig(
        predictions_root=data["predictions_root"],
        train_start=str(data["train_start"]),
        train_end=str(data["train_end"]),
        backfill_start=str(data["backfill_start"]),
        backfill_end=str(data["backfill_end"]),
        rolling_window_days=data["rolling_window_days"],
        ewma_halflife_days=data["ewma_halflife_days"],
        min_history=data["min_history"],
        model_strategy=str(data["model_strategy"]),
        n_jobs=data["n_jobs"],
        models=data["models"],
        output_suffix=str(data["output_suffix"]),
        obs_source_path=data.get("obs_source_path"),
        intermediate_dir=data["intermediate_dir"],
        artifacts_root=data["artifacts_root"],
        reuse_intermediate=data["reuse_intermediate"],
        dry_run=data["dry_run"],
        log_level=str(data["log_level"]).upper(),
        strict_schema=data["strict_schema"],
        overwrite_output=data["overwrite_output"],
    )
