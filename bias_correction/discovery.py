from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


CYCLE_FROM_NAME_RE = re.compile(r"(\d{10})(?=\.parquet$)")
ISSUE_TIME_ALIASES = (
    "issue_time_utc",
    "InitTimeUTC",
    "init_time",
    "issue_time",
    "cycle",
)


@dataclass(slots=True, frozen=True)
class PredictionFileInfo:
    model_name: str
    city: str
    path: Path
    issue_time_utc: pd.Timestamp


@dataclass(slots=True)
class DiscoveryResult:
    model_dirs: dict[str, Path]
    files: list[PredictionFileInfo]


def discover_model_dirs(
    *,
    predictions_root: Path,
    include_models: list[str] | None,
    output_suffix: str,
) -> dict[str, Path]:
    if not predictions_root.exists():
        raise FileNotFoundError(f"predictions_root not found: {predictions_root}")

    wanted = set(include_models or [])
    dirs: dict[str, Path] = {}
    for entry in sorted(predictions_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.endswith(output_suffix):
            continue
        if include_models is not None and name not in wanted:
            continue
        dirs[name] = entry

    if include_models is not None:
        missing = sorted(wanted - set(dirs))
        if missing:
            missing_text = ", ".join(missing)
            raise FileNotFoundError(
                f"Requested model directories not found under {predictions_root}: {missing_text}"
            )

    if not dirs:
        raise RuntimeError(
            f"No raw model directories found under {predictions_root} (excluding *{output_suffix})."
        )
    return dirs


def _issue_time_from_filename(path: Path) -> pd.Timestamp | None:
    match = CYCLE_FROM_NAME_RE.search(path.name)
    if match is None:
        return None
    token = match.group(1)
    return pd.to_datetime(token, format="%Y%m%d%H", utc=True)


def _issue_time_from_parquet(path: Path) -> pd.Timestamp | None:
    pf = pq.ParquetFile(path)
    schema_cols = set(pf.schema.names)
    for candidate in ISSUE_TIME_ALIASES:
        if candidate not in schema_cols:
            continue
        df = pf.read(columns=[candidate]).to_pandas()
        if df.empty:
            continue
        ts = pd.to_datetime(df[candidate], utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        return ts.iloc[0]
    return None


def _iter_candidate_files(model_dir: Path) -> Iterable[tuple[str, Path]]:
    for city_dir in sorted(model_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        for file_path in sorted(city_dir.glob("*.parquet")):
            yield city, file_path


def discover_prediction_files(
    *,
    model_dirs: dict[str, Path],
    strict_schema: bool,
    logger,
) -> DiscoveryResult:
    files: list[PredictionFileInfo] = []
    skipped = 0
    fallback_reads = 0

    for model_name, model_dir in sorted(model_dirs.items()):
        for city, file_path in _iter_candidate_files(model_dir):
            issue_time = _issue_time_from_filename(file_path)
            if issue_time is None:
                fallback_reads += 1
                issue_time = _issue_time_from_parquet(file_path)
            if issue_time is None:
                msg = (
                    f"Could not determine issue_time_utc for {file_path}; filename does not include YYYYMMDDHH"
                    " and no supported issue-time column exists."
                )
                if strict_schema:
                    raise ValueError(msg)
                logger.warning(msg)
                skipped += 1
                continue
            files.append(
                PredictionFileInfo(
                    model_name=model_name,
                    city=city,
                    path=file_path,
                    issue_time_utc=issue_time,
                )
            )

    files.sort(key=lambda x: (x.issue_time_utc, x.model_name, x.city, x.path.name))
    logger.info(
        "Discovered %d prediction files across %d models (skipped=%d, parquet-fallback=%d)",
        len(files),
        len(model_dirs),
        skipped,
        fallback_reads,
    )
    return DiscoveryResult(model_dirs=model_dirs, files=files)
