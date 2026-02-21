from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import groupby
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import BiasCorrectionConfig
from .discovery import DiscoveryResult, PredictionFileInfo, discover_model_dirs, discover_prediction_files
from .logging_utils import configure_logging
from .obs import attach_external_observations, load_observation_daily
from .schema import NormalizationResult, normalize_prediction_frame
from .stage_a import StageABiasEngine


@dataclass(slots=True)
class LoadStats:
    files_total: int = 0
    files_ok: int = 0
    files_failed: int = 0
    rows_total_raw: int = 0
    rows_total_normalized: int = 0
    rows_dropped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "files_total": int(self.files_total),
            "files_ok": int(self.files_ok),
            "files_failed": int(self.files_failed),
            "rows_total_raw": int(self.rows_total_raw),
            "rows_total_normalized": int(self.rows_total_normalized),
            "rows_dropped": int(self.rows_dropped),
        }


def _timestamp_tag() -> str:
    now = pd.Timestamp.utcnow()
    return now.strftime("%Y%m%d_%H%M%S")


def _filter_files(
    files: list[PredictionFileInfo],
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[PredictionFileInfo]:
    return [f for f in files if start_ts <= f.issue_time_utc <= end_ts]


def _output_path_for_file(cfg: BiasCorrectionConfig, info: PredictionFileInfo) -> Path:
    return cfg.predictions_root / f"{info.model_name}{cfg.output_suffix}" / info.city / info.path.name


def _log_warning_summary(
    *,
    warning_counter: Counter,
    logger,
    label: str,
    max_items: int = 10,
) -> None:
    if not warning_counter:
        return
    logger.info("%s normalization warnings (%d unique):", label, len(warning_counter))
    for warning_text, count in warning_counter.most_common(max_items):
        logger.info("  [%d] %s", count, warning_text)


def _normalize_single_file(
    *,
    info: PredictionFileInfo,
    strict_schema: bool,
    logger,
    warning_counter: Counter,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, int]:
    try:
        raw = pd.read_parquet(info.path)
        result: NormalizationResult = normalize_prediction_frame(
            df=raw,
            model_name=info.model_name,
            city_hint=info.city,
            source_path=info.path,
            issue_time_hint=info.issue_time_utc,
        )
        for warning in result.warnings:
            warning_counter[warning] += 1
        return raw, result.normalized, result.dropped_rows
    except Exception as exc:
        if strict_schema:
            raise
        logger.error("Skipping file due to schema/parse error: %s (%s)", info.path, exc)
        return None, None, 0


def _build_normalized_table(
    *,
    files: list[PredictionFileInfo],
    strict_schema: bool,
    logger,
) -> tuple[pd.DataFrame, LoadStats, Counter]:
    stats = LoadStats(files_total=len(files))
    warnings_counter: Counter = Counter()
    frames: list[pd.DataFrame] = []

    for info in files:
        raw_df, norm_df, dropped = _normalize_single_file(
            info=info,
            strict_schema=strict_schema,
            logger=logger,
            warning_counter=warnings_counter,
        )
        if raw_df is None or norm_df is None:
            stats.files_failed += 1
            continue
        stats.files_ok += 1
        stats.rows_total_raw += len(raw_df)
        stats.rows_total_normalized += len(norm_df)
        stats.rows_dropped += int(dropped)
        frames.append(norm_df)

    if not frames:
        raise RuntimeError("No rows were normalized from selected prediction files.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["issue_time_utc", "model_name", "city", "station_name", "lead_hours", "_row_idx"],
        kind="mergesort",
    ).reset_index(drop=True)
    return combined, stats, warnings_counter


def _apply_stage_a_online(
    *,
    df: pd.DataFrame,
    stage_a: StageABiasEngine,
    update_history: bool,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df.copy(), 0

    chunks: list[pd.DataFrame] = []
    updates = 0

    for _, issue_chunk in df.groupby("issue_time_utc", sort=True):
        scored = issue_chunk.copy()
        stage_scores = stage_a.score_rows(scored)
        scored = scored.join(stage_scores)
        scored["tmax_stageA"] = scored["tmax_pred"] + scored["bias_ewma"]
        scored["residual"] = scored["tmax_obs"] - scored["tmax_pred"]
        chunks.append(scored)

        if update_history:
            updates += stage_a.update_with_observed_residuals(scored, residual_col="residual")

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(
        ["issue_time_utc", "model_name", "city", "station_name", "lead_hours", "_row_idx"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out, updates


def _replay_stage_a_history(*, stage_a: StageABiasEngine, df_with_residual: pd.DataFrame) -> int:
    updates = 0
    for _, issue_chunk in df_with_residual.groupby("issue_time_utc", sort=True):
        updates += stage_a.update_with_observed_residuals(issue_chunk, residual_col="residual")
    return updates


def _write_corrected_file(
    *,
    raw_df: pd.DataFrame,
    corrected_rows: pd.DataFrame,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists and overwrite is disabled: {output_path}")

    out = raw_df.copy()
    new_cols = [
        "bias_ewma",
        "bias_level",
        "history_count",
        "tmax_stageA",
        "residual_stageB_hat",
        "tmax_corrected",
    ]
    for col in new_cols:
        out[col] = np.nan

    if not corrected_rows.empty:
        row_idx = corrected_rows["_row_idx"].astype(int).to_numpy()
        out.loc[row_idx, "bias_ewma"] = corrected_rows["bias_ewma"].to_numpy(dtype=float)
        out.loc[row_idx, "bias_level"] = corrected_rows["bias_level"].to_numpy(dtype=float)
        out.loc[row_idx, "history_count"] = corrected_rows["history_count"].to_numpy(dtype=float)
        out.loc[row_idx, "tmax_stageA"] = corrected_rows["tmax_stageA"].to_numpy(dtype=float)
        if "r2_hat" in corrected_rows.columns:
            out.loc[row_idx, "residual_stageB_hat"] = corrected_rows["r2_hat"].to_numpy(dtype=float)
        else:
            out.loc[row_idx, "residual_stageB_hat"] = 0.0
        out.loc[row_idx, "tmax_corrected"] = corrected_rows["tmax_corrected"].to_numpy(dtype=float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".parquet.part")
    if tmp.exists():
        tmp.unlink()
    out.to_parquet(tmp, index=False)
    tmp.replace(output_path)


def run_pipeline(cfg: BiasCorrectionConfig) -> dict[str, object]:
    run_tag = _timestamp_tag()
    run_dir = cfg.artifacts_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_file=run_dir / "run.log", level=cfg.log_level)

    logger.info("Starting bias-correction run %s", run_tag)
    logger.info("predictions_root=%s", cfg.predictions_root)

    model_dirs = discover_model_dirs(
        predictions_root=cfg.predictions_root,
        include_models=cfg.models,
        output_suffix=cfg.output_suffix,
    )
    discovery: DiscoveryResult = discover_prediction_files(
        model_dirs=model_dirs,
        strict_schema=cfg.strict_schema,
        logger=logger,
    )
    if not discovery.files:
        raise RuntimeError("No prediction parquet files were discovered.")

    latest_cycle = max(f.issue_time_utc for f in discovery.files)
    backfill_end_ts = cfg.resolve_backfill_end(latest_cycle)
    logger.info(
        "Date windows: train=%s..%s backfill=%s..%s",
        cfg.train_start,
        cfg.train_end,
        cfg.backfill_start,
        backfill_end_ts.strftime("%Y%m%d%H"),
    )

    train_files = _filter_files(
        discovery.files,
        start_ts=cfg.train_start_ts,
        end_ts=cfg.train_end_ts,
    )
    backfill_files = _filter_files(
        discovery.files,
        start_ts=cfg.backfill_start_ts,
        end_ts=backfill_end_ts,
    )

    if not backfill_files:
        raise RuntimeError("No backfill files found within requested backfill window")

    warmup_files = [f for f in train_files if f.issue_time_utc < cfg.backfill_start_ts]
    logger.info(
        "Selected %d warmup files (< backfill_start) and %d backfill files",
        len(warmup_files),
        len(backfill_files),
    )

    if cfg.dry_run:
        logger.info("Dry-run mode enabled; no files will be written and no model training will occur")
        preview = backfill_files[:15]
        for info in preview:
            logger.info("Would process: %s -> %s", info.path, _output_path_for_file(cfg, info))
        logger.info("Dry-run complete")
        return {
            "run_tag": run_tag,
            "dry_run": True,
            "models": sorted(model_dirs),
            "train_file_count": len(warmup_files),
            "backfill_file_count": len(backfill_files),
            "backfill_end": backfill_end_ts.strftime("%Y%m%d%H"),
        }

    obs_daily = load_observation_daily(obs_source_path=cfg.obs_source_path, logger=logger)

    stage_a = StageABiasEngine(
        rolling_window_days=cfg.rolling_window_days,
        ewma_halflife_days=cfg.ewma_halflife_days,
        min_history=cfg.min_history,
    )
    train_load_stats = LoadStats()
    train_norm_path: Path | None = None
    train_stagea_path: Path | None = None
    stagea_obs_rows = 0

    if warmup_files:
        warmup_end = warmup_files[-1].issue_time_utc.strftime("%Y%m%d%H")
        cfg.intermediate_dir.mkdir(parents=True, exist_ok=True)
        train_norm_path = cfg.intermediate_dir / (
            f"normalized_warmup_{cfg.train_start}_{warmup_end}_{'_'.join(sorted(model_dirs))}.parquet"
        )
        train_stagea_path = cfg.intermediate_dir / (
            f"stagea_warmup_{cfg.train_start}_{warmup_end}_{'_'.join(sorted(model_dirs))}.parquet"
        )

        if cfg.reuse_intermediate and train_norm_path.exists():
            logger.info("Loading normalized warmup table from %s", train_norm_path)
            train_norm = pd.read_parquet(train_norm_path)
            train_load_stats = LoadStats(
                files_total=len(warmup_files),
                files_ok=len(warmup_files),
                files_failed=0,
                rows_total_raw=int(len(train_norm)),
                rows_total_normalized=int(len(train_norm)),
                rows_dropped=0,
            )
        else:
            t0 = perf_counter()
            train_norm, train_load_stats, train_warnings = _build_normalized_table(
                files=warmup_files,
                strict_schema=cfg.strict_schema,
                logger=logger,
            )
            _log_warning_summary(
                warning_counter=train_warnings,
                logger=logger,
                label="Warmup",
            )
            train_norm = attach_external_observations(
                normalized_df=train_norm,
                obs_daily=obs_daily,
                logger=logger,
            )
            train_norm.to_parquet(train_norm_path, index=False)
            logger.info(
                "Built normalized warmup table in %.1fs and saved to %s",
                perf_counter() - t0,
                train_norm_path,
            )

        if cfg.reuse_intermediate and train_stagea_path.exists():
            logger.info("Loading Stage-A warmup table from %s", train_stagea_path)
            train_stagea = pd.read_parquet(train_stagea_path)
            updates = _replay_stage_a_history(stage_a=stage_a, df_with_residual=train_stagea)
            logger.info("Replayed %d residual updates into Stage-A state from cached warmup table", updates)
        else:
            t0 = perf_counter()
            train_stagea, updates = _apply_stage_a_online(
                df=train_norm,
                stage_a=stage_a,
                update_history=True,
            )
            train_stagea.to_parquet(train_stagea_path, index=False)
            logger.info(
                "Computed Stage-A warmup table in %.1fs (state updates=%d) and saved to %s",
                perf_counter() - t0,
                updates,
                train_stagea_path,
            )
        stagea_obs_rows = int(train_stagea["tmax_obs"].notna().sum())
    else:
        logger.info("No warmup files before backfill_start; Stage-A state starts empty.")

    logger.info(
        "Stage-A-only mode enabled (rolling_window_days=%d, ewma_halflife_days=%.1f). "
        "Skipping Stage-B model training. Warmup rows with observations: %d",
        cfg.rolling_window_days,
        cfg.ewma_halflife_days,
        stagea_obs_rows,
    )

    backfill_warning_counter: Counter = Counter()
    written_files = 0
    processed_files = 0
    failed_backfill_files = 0
    stage_a_updates_backfill = 0

    for issue_time, group_iter in groupby(backfill_files, key=lambda x: x.issue_time_utc):
        issue_files = list(group_iter)
        raw_by_source: dict[str, pd.DataFrame] = {}
        normalized_frames: list[pd.DataFrame] = []

        for info in issue_files:
            raw_df, norm_df, _ = _normalize_single_file(
                info=info,
                strict_schema=cfg.strict_schema,
                logger=logger,
                warning_counter=backfill_warning_counter,
            )
            processed_files += 1
            if raw_df is None or norm_df is None:
                failed_backfill_files += 1
                continue
            raw_by_source[str(info.path)] = raw_df
            normalized_frames.append(norm_df)

        if not normalized_frames:
            continue

        issue_norm = pd.concat(normalized_frames, ignore_index=True)
        issue_norm = attach_external_observations(
            normalized_df=issue_norm,
            obs_daily=obs_daily,
            logger=logger,
        )

        issue_scored = issue_norm.copy()
        stage_scores = stage_a.score_rows(issue_scored)
        issue_scored = issue_scored.join(stage_scores)
        issue_scored["tmax_stageA"] = issue_scored["tmax_pred"] + issue_scored["bias_ewma"]

        issue_scored["r2_hat"] = 0.0
        issue_scored["tmax_corrected"] = issue_scored["tmax_stageA"]

        for info in issue_files:
            source_key = str(info.path)
            raw_df = raw_by_source.get(source_key)
            if raw_df is None:
                continue

            corrected_rows = issue_scored[issue_scored["source_file"] == source_key].copy()
            output_path = _output_path_for_file(cfg, info)
            _write_corrected_file(
                raw_df=raw_df,
                corrected_rows=corrected_rows,
                output_path=output_path,
                overwrite=cfg.overwrite_output,
            )
            written_files += 1

        issue_scored["residual"] = issue_scored["tmax_obs"] - issue_scored["tmax_pred"]
        stage_a_updates_backfill += stage_a.update_with_observed_residuals(
            issue_scored,
            residual_col="residual",
        )

    _log_warning_summary(
        warning_counter=backfill_warning_counter,
        logger=logger,
        label="Backfill",
    )

    stage_a_stats = stage_a.summarize()
    if stage_a_stats.insufficient_primary:
        logger.info(
            "Top station/cycle/lead groups with insufficient Stage-A primary history (showing 20):"
        )
        for key, count in stage_a_stats.insufficient_primary.most_common(20):
            logger.info("  [%d] %s", count, key)

    summary = {
        "run_tag": run_tag,
        "predictions_root": str(cfg.predictions_root),
        "models": sorted(model_dirs),
        "train_window": {"start": cfg.train_start, "end": cfg.train_end},
        "backfill_window": {
            "start": cfg.backfill_start,
            "end": backfill_end_ts.strftime("%Y%m%d%H"),
        },
        "train_stats": train_load_stats.as_dict(),
        "backfill_stats": {
            "processed_files": int(processed_files),
            "written_files": int(written_files),
            "failed_files": int(failed_backfill_files),
            "stage_a_residual_updates": int(stage_a_updates_backfill),
        },
        "stage_a": {
            "fallback_usage": {str(k): int(v) for k, v in stage_a_stats.fallback_usage.items()},
            "insufficient_primary_groups": int(len(stage_a_stats.insufficient_primary)),
            "min_history": int(cfg.min_history),
            "rolling_window_days": int(cfg.rolling_window_days),
            "ewma_halflife_days": float(cfg.ewma_halflife_days),
        },
        "correction_mode": "stage_a_only",
        "stage_b_strategy": "disabled",
        "artifacts_dir": str(run_dir),
        "intermediate_train_norm": str(train_norm_path) if train_norm_path is not None else None,
        "intermediate_train_stagea": str(train_stagea_path) if train_stagea_path is not None else None,
    }

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Run complete. Summary written to %s", summary_path)

    return summary
