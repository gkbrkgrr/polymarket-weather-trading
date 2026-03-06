#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_trading import db as dbmod
from live_trading.policy import PolicyContext, apply_policy
from live_trading.pricing import compute_pricing_decision
from live_trading.run_live_pilot import (
    build_station_timezone_map,
    load_config,
    read_probability_files,
    resolve_open_market_universe,
    resolve_path,
    resolve_probability_data_path,
    select_live_universe,
    standardize_probabilities,
)
from live_trading.state import PilotStateStore
from live_trading.utils_time import normalize_station_key, utc_now
from scripts.polymarket_trading_backtest import apply_filters_and_select


UTC = timezone.utc
DEFAULT_CONFIG_PATH = REPO_ROOT / "live_trading" / "config.live_pilot.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "selector_parity"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate selector parity between live policy filtering/ranking and backtest selector "
            "on identical candidate inputs."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help=f"Path to live pilot config (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--as-of-utc", type=str, default="", help="Optional as-of UTC timestamp (ISO8601). Defaults to now().")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output report directory (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--min-jaccard", type=float, default=0.98, help="Minimum required Jaccard similarity between selected sets.")
    parser.add_argument("--max-mismatch-rate", type=float, default=0.02, help="Maximum allowed symmetric-difference rate over union.")
    parser.add_argument("--max-rank-mismatch-groups", type=int, default=0, help="Maximum allowed group-level ranking mismatches.")
    parser.add_argument("--max-details", type=int, default=50, help="Maximum mismatch details recorded in report.")
    return parser.parse_args(argv)


def parse_as_of_utc(text: str) -> datetime:
    raw = str(text).strip()
    if not raw:
        return utc_now()
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise SystemExit(f"Invalid --as-of-utc: {text!r}")
    return parsed.to_pydatetime()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("selector_parity")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def build_candidates(
    *,
    universe: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    cfg: dict[str, Any],
    now_utc: datetime,
) -> pd.DataFrame:
    snapshot_map = {
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

    records: list[dict[str, Any]] = []
    for row in universe.itertuples(index=False):
        snapshot = None if pd.isna(row.market_id) else snapshot_map.get(str(row.market_id))
        pricing = compute_pricing_decision(
            snapshot=snapshot,
            now_utc=now_utc,
            max_snapshot_age_minutes=float(cfg.get("max_snapshot_age_minutes", 30)),
            slippage_buffer_yes_fallback=float(cfg.get("slippage_buffer_yes_fallback", 0.01)),
            max_spread=float(cfg.get("max_spread", 0.05)),
        )

        # Backtest selector has no explicit spread/staleness flags; emulate live gate by withholding price when pricing skipped.
        bt_price = pricing.chosen_no_ask if pricing.skipped_reason is None else None
        bt_age_hours = None if pricing.snapshot_age_minutes is None else float(pricing.snapshot_age_minutes) / 60.0

        market_day = pd.to_datetime(row.market_day_local, errors="coerce")
        if pd.isna(market_day):
            continue
        market_day_iso = market_day.date().isoformat()

        records.append(
            {
                "station": str(row.station),
                "station_key": normalize_station_key(str(row.station)),
                "market_day_local": market_day_iso,
                "date": pd.to_datetime(market_day_iso),
                "market_id": None if pd.isna(row.market_id) else str(row.market_id),
                "slug": str(row.slug),
                "asset_id": None if pd.isna(getattr(row, "asset_id", pd.NA)) else str(getattr(row, "asset_id")),
                "event_key": str(row.event_key),
                "strike_k": int(row.strike_k),
                "mode_k": int(row.mode_k),
                "p_model": float(row.p_model),
                "chosen_no_ask": pricing.chosen_no_ask,
                "snapshot_skip_reason": pricing.skipped_reason,
                "snapshot_age_minutes": pricing.snapshot_age_minutes,
                "best_no_ask": snapshot.get("best_no_ask") if snapshot else None,
                "best_yes_bid": snapshot.get("best_yes_bid") if snapshot else None,
                "no_trade_price": bt_price,
                "yes_trade_price": (None if bt_price is None else 1.0 - float(bt_price)),
                "no_trade_ts": now_utc,
                "yes_trade_ts": now_utc,
                "no_age_hours": bt_age_hours,
                "yes_age_hours": bt_age_hours,
                "no_lookback": 24.0,
                "yes_lookback": 24.0,
            }
        )

    return pd.DataFrame.from_records(records)


def run_live_selector(
    *,
    candidates: pd.DataFrame,
    cfg: dict[str, Any],
    station_tz: dict[str, str],
    now_utc: datetime,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="lt10_parity_state_") as td:
        state = PilotStateStore(Path(td), nav_usd=1_000_000_000.0)
        ctx = PolicyContext(
            nav_usd=1_000_000_000.0,
            mode_distance_min=int(cfg.get("mode_distance_min", 2)),
            p_model_max=float(cfg.get("p_model_max", 0.12)),
            edge_threshold=float(cfg.get("edge_threshold", 0.02)),
            max_no_price=float(cfg.get("max_no_price", 0.92)),
            top_n_per_event_day=int(cfg.get("top_n_per_event_day", 2)),
            stake_fraction=0.0,
            stake_cap_usd=0.0,
            min_order_size=0.0,
            station_daily_risk_fraction=1.0,
            portfolio_daily_risk_fraction=1.0,
            max_open_positions_per_station=1_000_000,
            max_open_positions_total=1_000_000,
            trade_window_start_local="00:00",
            trade_window_end_local="23:59",
        )
        return apply_policy(
            candidates=candidates.copy(),
            state_store=state,
            ctx=ctx,
            now_utc=now_utc,
            station_timezones=station_tz,
        )


def run_backtest_selector(candidates: pd.DataFrame, *, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected, full, _ = apply_filters_and_select(
        candidates.copy(),
        tail_prob_threshold=float(cfg.get("p_model_max", 0.12)),
        edge_threshold=float(cfg.get("edge_threshold", 0.02)),
        max_no_price=float(cfg.get("max_no_price", 0.92)),
        max_trades_per_day=int(cfg.get("top_n_per_event_day", 2)),
        yes_fallback_slippage_buffer=float(cfg.get("slippage_buffer_yes_fallback", 0.01)),
        london_min_edge_per_risk=-1e9,
        london_max_price=float(cfg.get("max_no_price", 0.92)),
        nyc_max_trade_age_hours=1e9,
        buenosaires_max_trade_age_hours=1e9,
        nyc_allow_yes_fallback=True,
        buenosaires_allow_yes_fallback=True,
        nyc_yes_fallback_slippage_buffer=float(cfg.get("slippage_buffer_yes_fallback", 0.01)),
        buenosaires_yes_fallback_slippage_buffer=float(cfg.get("slippage_buffer_yes_fallback", 0.01)),
        nyc_edge_threshold=float(cfg.get("edge_threshold", 0.02)),
        buenosaires_edge_threshold=float(cfg.get("edge_threshold", 0.02)),
    )
    return selected, full


def selection_key(row: pd.Series) -> tuple[str, str, str, int, str, str]:
    market_id = "" if pd.isna(row.get("market_id")) else str(row.get("market_id"))
    slug = "" if pd.isna(row.get("slug")) else str(row.get("slug"))
    station = "" if pd.isna(row.get("station")) else str(row.get("station"))
    day = "" if pd.isna(row.get("market_day_local")) else str(row.get("market_day_local"))
    event_key = "" if pd.isna(row.get("event_key")) else str(row.get("event_key"))
    strike_k = int(row.get("strike_k"))
    return (market_id, slug, station, day, event_key, strike_k)


def build_key_set(df: pd.DataFrame) -> set[tuple[str, str, str, int, str, str]]:
    if df.empty:
        return set()
    return {selection_key(r) for _, r in df.iterrows()}


def compare_rankings(live_selected: pd.DataFrame, bt_selected: pd.DataFrame) -> tuple[int, list[dict[str, Any]]]:
    group_cols = ["station", "market_day_local", "event_key"]

    live_ranks = (
        live_selected.sort_values(group_cols + ["edge", "strike_k"], ascending=[True, True, True, False, True], kind="mergesort")
        if not live_selected.empty
        else live_selected.copy()
    )
    bt_ranks = (
        bt_selected.sort_values(group_cols + ["edge", "strike_k"], ascending=[True, True, True, False, True], kind="mergesort")
        if not bt_selected.empty
        else bt_selected.copy()
    )

    live_groups = {
        tuple(k): [selection_key(r) for _, r in g.iterrows()]
        for k, g in live_ranks.groupby(group_cols, sort=False)
    }
    bt_groups = {
        tuple(k): [selection_key(r) for _, r in g.iterrows()]
        for k, g in bt_ranks.groupby(group_cols, sort=False)
    }

    mismatches: list[dict[str, Any]] = []
    all_groups = sorted(set(live_groups) | set(bt_groups))
    for group in all_groups:
        left = live_groups.get(group, [])
        right = bt_groups.get(group, [])
        if left != right:
            mismatches.append(
                {
                    "group": {
                        "station": group[0],
                        "market_day_local": group[1],
                        "event_key": group[2],
                    },
                    "live_order": left,
                    "backtest_order": right,
                }
            )
    return len(mismatches), mismatches


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logger()

    cfg_path = resolve_path(args.config)
    cfg = load_config(cfg_path)
    now_utc = parse_as_of_utc(args.as_of_utc)

    logger.info("Config: %s", cfg_path)
    logger.info("As-of UTC: %s", now_utc.isoformat())

    prob_root = resolve_path(str(cfg.get("probabilities_path")))
    prob_path, manifest = resolve_probability_data_path(prob_root)
    raw_prob = read_probability_files(prob_path)
    prob = standardize_probabilities(raw_prob)

    stations = [str(s).strip() for s in cfg.get("stations_allowlist", []) if str(s).strip()]
    station_tz = build_station_timezone_map(cfg, stations)
    universe = select_live_universe(prob=prob, cfg=cfg, station_tz=station_tz, now_utc=now_utc)
    if universe.empty:
        raise SystemExit("No live-universe rows after station/date/cutoff filters.")

    with dbmod.connect_db(str(cfg.get("db_dsn"))) as conn:
        snapshot_info = dbmod.detect_snapshot_table(conn)
        snapshot_table_name = snapshot_info.table_name
        mapped, unmapped = resolve_open_market_universe(
            universe=universe,
            conn=conn,
            station_tz=station_tz,
            cfg=cfg,
            logger=logger,
        )
        if mapped.empty:
            raise SystemExit("No mapped rows in open-market universe; parity harness cannot run.")

        market_ids = sorted(mapped["market_id"].dropna().astype(str).unique().tolist())
        snapshot_df = dbmod.fetch_latest_snapshots(conn, snapshot_table=snapshot_info, market_ids=market_ids)

    candidates = build_candidates(universe=mapped, snapshot_df=snapshot_df, cfg=cfg, now_utc=now_utc)
    if candidates.empty:
        raise SystemExit("No candidate rows were built for parity harness.")

    live_policy_out = run_live_selector(candidates=candidates, cfg=cfg, station_tz=station_tz, now_utc=now_utc)
    live_selected = live_policy_out.loc[live_policy_out["decision"].astype(str) == "TRADE"].copy()

    bt_selected, _bt_full = run_backtest_selector(candidates=candidates, cfg=cfg)

    live_set = build_key_set(live_selected)
    bt_set = build_key_set(bt_selected)
    inter = len(live_set & bt_set)
    union = len(live_set | bt_set)
    only_live = sorted(live_set - bt_set)
    only_bt = sorted(bt_set - live_set)

    jaccard = 1.0 if union == 0 else float(inter) / float(union)
    mismatch_rate = 0.0 if union == 0 else float(len(only_live) + len(only_bt)) / float(union)
    precision = 1.0 if not bt_set else float(inter) / float(len(bt_set))
    recall = 1.0 if not live_set else float(inter) / float(len(live_set))

    rank_mismatch_count, rank_mismatches = compare_rankings(live_selected, bt_selected)

    pass_jaccard = jaccard >= float(args.min_jaccard)
    pass_mismatch_rate = mismatch_rate <= float(args.max_mismatch_rate)
    pass_rank = rank_mismatch_count <= int(args.max_rank_mismatch_groups)
    passed = pass_jaccard and pass_mismatch_rate and pass_rank

    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"selector_parity_{ts}.json"

    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now().isoformat(),
        "as_of_utc": now_utc.isoformat(),
        "config_path": str(cfg_path),
        "probabilities_path": str(prob_path),
        "manifest_cycle": None if manifest is None else manifest.get("cycle"),
        "snapshot_table": snapshot_table_name,
        "tolerances": {
            "min_jaccard": float(args.min_jaccard),
            "max_mismatch_rate": float(args.max_mismatch_rate),
            "max_rank_mismatch_groups": int(args.max_rank_mismatch_groups),
        },
        "counts": {
            "prob_rows": int(len(prob)),
            "universe_rows": int(len(universe)),
            "mapped_rows": int(len(mapped)),
            "unmapped_rows": int(len(unmapped)),
            "candidate_rows": int(len(candidates)),
            "live_selected": int(len(live_set)),
            "backtest_selected": int(len(bt_set)),
            "intersection": int(inter),
            "union": int(union),
            "only_live": int(len(only_live)),
            "only_backtest": int(len(only_bt)),
            "rank_mismatch_groups": int(rank_mismatch_count),
        },
        "metrics": {
            "jaccard": jaccard,
            "mismatch_rate": mismatch_rate,
            "precision_vs_live": precision,
            "recall_vs_live": recall,
        },
        "status": {
            "pass_jaccard": bool(pass_jaccard),
            "pass_mismatch_rate": bool(pass_mismatch_rate),
            "pass_rank": bool(pass_rank),
            "passed": bool(passed),
        },
        "details": {
            "only_live_sample": only_live[: max(0, int(args.max_details))],
            "only_backtest_sample": only_bt[: max(0, int(args.max_details))],
            "rank_mismatch_sample": rank_mismatches[: max(0, int(args.max_details))],
        },
        "notes": {
            "backtest_alignment": (
                "Backtest selector receives identical candidate rows; prices are withheld when live pricing emits a snapshot skip "
                "reason to align spread/staleness gating semantics."
            )
        },
    }

    atomic_write_json(report_path, report)

    logger.info("Candidate rows: %d", len(candidates))
    logger.info("Selections | live=%d backtest=%d inter=%d union=%d", len(live_set), len(bt_set), inter, union)
    logger.info("Metrics | jaccard=%.6f mismatch_rate=%.6f precision=%.6f recall=%.6f", jaccard, mismatch_rate, precision, recall)
    logger.info("Rank mismatch groups: %d", rank_mismatch_count)
    logger.info("Report: %s", report_path)

    if not passed:
        logger.error("Selector parity FAILED tolerance checks")
        return 1

    logger.info("Selector parity PASSED tolerance checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
