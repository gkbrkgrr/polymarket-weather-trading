#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_trading import db as dbmod
from live_trading.forecast_progression import update_progression_history
from live_trading.pricing import compute_pricing_decision
from live_trading.state import PilotStateStore
from live_trading.utils_time import utc_now
from master_db import resolve_master_postgres_dsn


CYCLE_TOKEN_RE = re.compile(r"^\d{10}$")
DEFAULT_LIVE_CONFIG = REPO_ROOT / "live_trading" / "config.live_pilot.yaml"
DEFAULT_PROBABILITIES_ROOT = REPO_ROOT / "reports" / "live_probabilities"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update live_state forecast_progression from a completed live probabilities cycle."
        )
    )
    parser.add_argument("--cycle", required=True, help="Cycle token YYYYMMDDHH.")
    parser.add_argument(
        "--live-config",
        type=Path,
        default=DEFAULT_LIVE_CONFIG,
        help=f"Live config yaml (default: {DEFAULT_LIVE_CONFIG})",
    )
    parser.add_argument(
        "--probabilities-root",
        type=Path,
        default=DEFAULT_PROBABILITIES_ROOT,
        help=f"Live probabilities root (default: {DEFAULT_PROBABILITIES_ROOT})",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Optional explicit state directory. Defaults to <live.output_dir>/state from live config.",
    )
    parser.add_argument(
        "--master-dsn",
        type=str,
        default=None,
        help="Optional explicit master DB DSN. Defaults to live config then master_db resolution.",
    )
    return parser.parse_args(argv)


def parse_cycle_token(value: str) -> str:
    text = str(value).strip()
    if CYCLE_TOKEN_RE.fullmatch(text) is None:
        raise SystemExit(f"--cycle must be YYYYMMDDHH, got {value!r}")
    return text


def resolve_path(path_value: Path, *, base: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    return (base / path_value).resolve()


def load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return loaded


def _load_json_dict(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"Manifest must be a JSON object: {path}")
    return loaded


def _manifest_status_is_success(manifest: dict[str, Any]) -> bool:
    status = str(manifest.get("status", "")).strip().lower()
    return (not status) or status == "success"


def _extract_cycle_token(manifest_path: Path, manifest: dict[str, Any]) -> str | None:
    cycle = str(manifest.get("cycle", "")).strip()
    if CYCLE_TOKEN_RE.fullmatch(cycle):
        return cycle

    parts = manifest_path.resolve().parts
    for idx, part in enumerate(parts):
        if part != "cycles":
            continue
        if idx + 1 >= len(parts):
            continue
        candidate = parts[idx + 1]
        if CYCLE_TOKEN_RE.fullmatch(candidate):
            return candidate
    return None


def _resolve_manifest_probability_file(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    probabilities_root: Path,
) -> Path:
    rel = manifest.get("probabilities_file")
    if rel is None or not str(rel).strip():
        raise SystemExit(f"Manifest missing probabilities_file: {manifest_path}")

    rel_path = Path(str(rel).strip())
    if rel_path.is_absolute():
        if rel_path.exists():
            return rel_path.resolve()
        raise SystemExit(f"Manifest probabilities file not found: {rel_path}")

    candidate_bases = [probabilities_root, manifest_path.parent]
    for base in candidate_bases:
        candidate = (base / rel_path).resolve()
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"Manifest probabilities file not found: {rel_path} (manifest={manifest_path})"
    )


def _manifest_generated_rank(manifest: dict[str, Any]) -> int:
    ts = pd.to_datetime(manifest.get("generated_at_utc"), utc=True, errors="coerce")
    if pd.isna(ts):
        return -1
    return int(ts.value)


def resolve_cycle_probability_file(*, probabilities_root: Path, cycle: str) -> tuple[Path, dict[str, Any], Path]:
    latest_manifest_path = probabilities_root / "latest_manifest.json"
    if latest_manifest_path.exists():
        latest_manifest = _load_json_dict(latest_manifest_path)
        latest_cycle = _extract_cycle_token(latest_manifest_path, latest_manifest)
        if _manifest_status_is_success(latest_manifest) and latest_cycle == cycle:
            prob_file = _resolve_manifest_probability_file(
                latest_manifest_path,
                latest_manifest,
                probabilities_root=probabilities_root,
            )
            return prob_file, latest_manifest, latest_manifest_path

    candidates: list[tuple[int, str, dict[str, Any], Path]] = []
    for manifest_path in sorted((probabilities_root / "cycles" / cycle).glob("*/manifest.json")):
        try:
            manifest = _load_json_dict(manifest_path)
        except Exception:
            continue
        manifest_cycle = _extract_cycle_token(manifest_path, manifest)
        if manifest_cycle != cycle or not _manifest_status_is_success(manifest):
            continue
        candidates.append((_manifest_generated_rank(manifest), str(manifest_path), manifest, manifest_path))

    legacy_manifest = probabilities_root / "cycles" / cycle / "manifest.json"
    if legacy_manifest.exists():
        try:
            manifest = _load_json_dict(legacy_manifest)
            manifest_cycle = _extract_cycle_token(legacy_manifest, manifest)
            if manifest_cycle == cycle and _manifest_status_is_success(manifest):
                candidates.append((_manifest_generated_rank(manifest), str(legacy_manifest), manifest, legacy_manifest))
        except Exception:
            pass

    if not candidates:
        raise SystemExit(
            f"Could not resolve successful manifest for cycle={cycle} under {probabilities_root}"
        )

    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    _, _, manifest, manifest_path = candidates[0]
    prob_file = _resolve_manifest_probability_file(manifest_path, manifest, probabilities_root=probabilities_root)
    return prob_file, manifest, manifest_path


def read_probability_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _clean_station_name(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def standardize_probabilities_for_progression(raw: pd.DataFrame) -> pd.DataFrame:
    station_col = "station_name" if "station_name" in raw.columns else ("station" if "station" in raw.columns else None)
    day_col = "market_day_local" if "market_day_local" in raw.columns else ("date" if "date" in raw.columns else None)
    if station_col is None or day_col is None:
        raise SystemExit("Probability rows must include station_name/station and market_day_local/date.")

    p_model_source = None
    for col_name in ("p_model", "p_model_adjusted", "p_model_raw"):
        if col_name in raw.columns:
            p_model_source = raw[col_name]
            break
    if p_model_source is None:
        p_model_source = pd.Series([pd.NA] * len(raw), dtype="float64")

    out = pd.DataFrame(
        {
            "station": _clean_station_name(raw[station_col]),
            "market_day_local": pd.to_datetime(raw[day_col], errors="coerce").dt.date,
            "market_id": raw["market_id"].astype("string").str.strip() if "market_id" in raw.columns else pd.Series([pd.NA] * len(raw), dtype="string"),
            "strike_k": pd.to_numeric(raw.get("strike_k"), errors="coerce"),
            "mode_k": pd.to_numeric(raw.get("mode_k"), errors="coerce"),
            "p_model": pd.to_numeric(p_model_source, errors="coerce"),
        }
    )
    out = out.dropna(subset=["station", "market_day_local", "strike_k", "mode_k", "p_model"]).copy()
    out = out.loc[out["station"] != ""].copy()
    out["strike_k"] = out["strike_k"].astype(int)
    out["mode_k"] = out["mode_k"].astype(int)
    out["market_day_local"] = out["market_day_local"].map(lambda d: d.isoformat())
    out["market_id"] = out["market_id"].astype("string")
    return out


def build_snapshot_map(
    *,
    conn,
    market_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], str]:
    snapshot_info = dbmod.detect_snapshot_table(conn)
    snapshot_df = dbmod.fetch_latest_snapshots(conn, snapshot_table=snapshot_info, market_ids=market_ids)
    snapshot_map: dict[str, dict[str, Any]] = {
        str(row.market_id): {
            "yes_snapshot_ts_utc": row.yes_snapshot_ts_utc,
            "no_snapshot_ts_utc": row.no_snapshot_ts_utc,
            "best_yes_bid": row.best_yes_bid,
            "best_yes_ask": row.best_yes_ask,
            "best_no_bid": row.best_no_bid,
            "best_no_ask": row.best_no_ask,
            "yes_bid_size": row.yes_bid_size,
            "yes_ask_size": row.yes_ask_size,
            "no_bid_size": row.no_bid_size,
            "no_ask_size": row.no_ask_size,
        }
        for row in snapshot_df.itertuples(index=False)
    }
    return snapshot_map, snapshot_info.table_name


def write_pending_batch(
    *,
    pending_dir: Path,
    cycle: str,
    cycle_time_utc: datetime,
    progression_input: pd.DataFrame,
) -> Path:
    pending_dir.mkdir(parents=True, exist_ok=True)
    out_path = pending_dir / f"forecast_progression_{cycle}.parquet"
    tmp_path = pending_dir / f".forecast_progression_{cycle}.{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.tmp.parquet"

    payload = progression_input.copy()
    payload["cycle_time_utc"] = cycle_time_utc.isoformat()
    payload.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cycle = parse_cycle_token(args.cycle)

    live_cfg_path = resolve_path(Path(args.live_config), base=REPO_ROOT)
    cfg = load_yaml_dict(live_cfg_path)

    probabilities_root = resolve_path(Path(args.probabilities_root), base=REPO_ROOT)
    prob_file, manifest, manifest_path = resolve_cycle_probability_file(
        probabilities_root=probabilities_root,
        cycle=cycle,
    )

    raw_prob = read_probability_file(prob_file)
    prob = standardize_probabilities_for_progression(raw_prob)
    if prob.empty:
        raise SystemExit(f"No standardized probability rows for cycle={cycle} from {prob_file}")

    mode_distance_min = int(cfg.get("mode_distance_min", 2))
    p_model_max = float(cfg.get("p_model_max", 0.12))
    edge_threshold = float(cfg.get("edge_threshold", 0.02))
    max_no_price = float(cfg.get("max_no_price", 0.92))
    max_snapshot_age_minutes = float(cfg.get("max_snapshot_age_minutes", 30))
    max_spread = float(cfg.get("max_spread", 0.05))
    slippage_buffer_yes_fallback = float(cfg.get("slippage_buffer_yes_fallback", 0.01))

    dsn_hint = args.master_dsn if args.master_dsn is not None else cfg.get("db_dsn")
    dsn = resolve_master_postgres_dsn(explicit_dsn=(str(dsn_hint).strip() if dsn_hint else None))

    market_ids = sorted(
        {
            str(mid)
            for mid in prob["market_id"].dropna().astype(str)
            if str(mid).strip() and str(mid).strip().lower() != "<na>"
        }
    )

    now_utc = utc_now()
    with dbmod.connect_db(dsn) as conn:
        snapshot_map, snapshot_table_name = build_snapshot_map(conn=conn, market_ids=market_ids)

    records: list[dict[str, Any]] = []
    for row in prob.itertuples(index=False):
        market_id = str(row.market_id).strip() if row.market_id is not None else ""
        snapshot = snapshot_map.get(market_id) if market_id else None
        pricing = compute_pricing_decision(
            snapshot=snapshot,
            now_utc=now_utc,
            max_snapshot_age_minutes=max_snapshot_age_minutes,
            slippage_buffer_yes_fallback=slippage_buffer_yes_fallback,
            max_spread=max_spread,
        )

        p_model = float(row.p_model)
        chosen_no_ask = None if pricing.chosen_no_ask is None else float(pricing.chosen_no_ask)
        edge = None if chosen_no_ask is None else float((1.0 - p_model) - chosen_no_ask)

        mode_distance_ok = abs(int(row.strike_k) - int(row.mode_k)) >= mode_distance_min
        is_candidate = (
            (pricing.skipped_reason is None)
            and (chosen_no_ask is not None)
            and mode_distance_ok
            and (p_model <= p_model_max)
            and (chosen_no_ask <= max_no_price)
            and (edge is not None and edge >= edge_threshold)
        )

        records.append(
            {
                "station": str(row.station),
                "market_day_local": str(row.market_day_local),
                "strike_k": int(row.strike_k),
                "mode_k": int(row.mode_k),
                "p_model": p_model,
                "chosen_no_ask": chosen_no_ask,
                "edge": edge,
                "is_candidate": bool(is_candidate),
            }
        )

    progression_input = pd.DataFrame.from_records(records)
    if progression_input.empty:
        raise SystemExit("No progression rows were built.")

    output_dir_raw = cfg.get("output_dir", "live_trading")
    output_dir = resolve_path(Path(str(output_dir_raw)), base=REPO_ROOT)
    state_dir = resolve_path(Path(args.state_dir), base=REPO_ROOT) if args.state_dir else (output_dir / "state")
    nav_seed = float(cfg.get("nav_usd", 10000.0))

    cycle_time_utc = datetime.strptime(cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)

    try:
        state_store = PilotStateStore(state_dir=state_dir, nav_usd=nav_seed)
    except RuntimeError as exc:
        if "State lock is already held" not in str(exc):
            raise
        pending_dir = state_dir / "forecast_progression_pending"
        queued_file = write_pending_batch(
            pending_dir=pending_dir,
            cycle=cycle,
            cycle_time_utc=cycle_time_utc,
            progression_input=progression_input,
        )
        candidate_count = int(progression_input["is_candidate"].sum())
        print(f"Cycle: {cycle}")
        print(f"Manifest: {manifest_path}")
        print(f"Manifest cycle/status: {manifest.get('cycle')} / {manifest.get('status')}")
        print(f"Probability file: {prob_file}")
        print(f"Snapshot table: {snapshot_table_name}")
        print(f"Rows processed: {len(progression_input)}")
        print(f"Rows flagged candidate: {candidate_count}")
        print("State lock busy; queued forecast progression batch for live ingestion.")
        print(f"Queued batch: {queued_file}")
        return 0

    try:
        before_keys = len(state_store.forecast_progression())
        update_progression_history(state_store.state, progression_input, cycle_time_utc)
        state_store.persist()
        after_keys = len(state_store.forecast_progression())
    finally:
        state_store.close()

    candidate_count = int(progression_input["is_candidate"].sum())
    print(f"Cycle: {cycle}")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest cycle/status: {manifest.get('cycle')} / {manifest.get('status')}")
    print(f"Probability file: {prob_file}")
    print(f"Snapshot table: {snapshot_table_name}")
    print(f"Rows processed: {len(progression_input)}")
    print(f"Rows flagged candidate: {candidate_count}")
    print(f"Progression keys before/after: {before_keys} -> {after_keys}")
    print(f"State path: {state_dir / 'live_state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
