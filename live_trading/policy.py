from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .state import PilotStateStore
from .utils_time import is_within_trade_window


SKIP_REASONS = {
    "no_snapshot",
    "snapshot_too_old",
    "spread_too_wide",
    "health_gate_blocked",
    "p_model_too_high",
    "mode_distance_fail",
    "edge_too_low",
    "price_too_high",
    "risk_limit_hit",
    "max_positions_hit",
    "already_open_position",
    "trade_cooldown",
    "outside_trade_window",
    "kill_switch_active",
}


@dataclass
class PolicyContext:
    nav_usd: float
    nav_peak_usd: float
    mode_distance_min: int
    p_model_max: float
    edge_threshold: float
    max_no_price: float
    top_n_per_event_day: int
    stake_fraction: float
    stake_cap_usd: float
    min_order_size: float
    station_daily_risk_fraction: float
    portfolio_daily_risk_fraction: float
    max_open_positions_per_station: int
    max_open_positions_total: int
    trade_cooldown_minutes: float
    drawdown_position_scaling: bool
    max_drawdown_fraction: float
    min_drawdown_scale: float
    trade_window_start_local: str
    trade_window_end_local: str


def compute_size(
    *,
    nav_usd: float,
    nav_peak_usd: float,
    stake_fraction: float,
    stake_cap_usd: float,
    min_order_size: float,
    drawdown_position_scaling: bool,
    max_drawdown_fraction: float,
    min_drawdown_scale: float,
) -> float:
    target = min(float(nav_usd) * float(stake_fraction), float(stake_cap_usd))
    scale = 1.0
    if bool(drawdown_position_scaling) and float(max_drawdown_fraction) > 0:
        nav_now = max(0.0, float(nav_usd))
        nav_peak = max(float(nav_peak_usd), nav_now, 1e-9)
        drawdown = max(0.0, (nav_peak - nav_now) / nav_peak)
        progress = min(1.0, drawdown / float(max_drawdown_fraction))
        floor = min(1.0, max(0.0, float(min_drawdown_scale)))
        scale = 1.0 - progress * (1.0 - floor)
    return max(float(min_order_size), float(target) * float(scale))


def _initial_skip_reason(row: pd.Series, ctx: PolicyContext, now_local: datetime, kill_switch_active: bool) -> str | None:
    if kill_switch_active:
        return "kill_switch_active"
    if not is_within_trade_window(
        now_local=now_local,
        start_local_hhmm=ctx.trade_window_start_local,
        end_local_hhmm=ctx.trade_window_end_local,
    ):
        return "outside_trade_window"

    if not (abs(float(row["strike_k"]) - float(row["mode_k"])) >= float(ctx.mode_distance_min)):
        return "mode_distance_fail"
    if not (float(row["p_model"]) <= float(ctx.p_model_max)):
        return "p_model_too_high"

    snapshot_reason = row.get("snapshot_skip_reason")
    if isinstance(snapshot_reason, str) and snapshot_reason in SKIP_REASONS:
        return snapshot_reason

    chosen_price = row.get("chosen_no_ask")
    if chosen_price is None or pd.isna(chosen_price):
        return "no_snapshot"

    edge = float(row["edge"])
    if edge < float(ctx.edge_threshold):
        return "edge_too_low"
    if float(chosen_price) > float(ctx.max_no_price):
        return "price_too_high"
    return None


def apply_policy(
    *,
    candidates: pd.DataFrame,
    state_store: PilotStateStore,
    ctx: PolicyContext,
    now_utc: datetime,
    station_timezones: dict[str, str],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    df = candidates.copy()

    df["NO_true"] = 1.0 - pd.to_numeric(df["p_model"], errors="coerce")
    df["edge"] = df["NO_true"] - pd.to_numeric(df["chosen_no_ask"], errors="coerce")
    df["decision"] = "SKIP"
    df["skipped_reason"] = "edge_too_low"

    df["risk_used_station_today"] = 0.0
    df["risk_used_station_daily"] = 0.0
    df["risk_used_station_open"] = 0.0
    df["risk_limit_station_today"] = float(ctx.nav_usd) * float(ctx.station_daily_risk_fraction)
    df["risk_used_portfolio_today"] = 0.0
    df["risk_used_portfolio_daily"] = 0.0
    df["risk_used_portfolio_open"] = 0.0
    df["risk_limit_portfolio_today"] = float(ctx.nav_usd) * float(ctx.portfolio_daily_risk_fraction)

    df["size"] = compute_size(
        nav_usd=ctx.nav_usd,
        nav_peak_usd=ctx.nav_peak_usd,
        stake_fraction=ctx.stake_fraction,
        stake_cap_usd=ctx.stake_cap_usd,
        min_order_size=ctx.min_order_size,
        drawdown_position_scaling=ctx.drawdown_position_scaling,
        max_drawdown_fraction=ctx.max_drawdown_fraction,
        min_drawdown_scale=ctx.min_drawdown_scale,
    )
    df["stake_usd"] = pd.to_numeric(df["chosen_no_ask"], errors="coerce") * pd.to_numeric(df["size"], errors="coerce")

    pre_pass = np.zeros(len(df), dtype=bool)

    for pos, (idx, row) in enumerate(df.iterrows()):
        station = str(row["station"])
        tz = station_timezones.get(station, "UTC")
        try:
            now_local = now_utc.astimezone(ZoneInfo(tz))
        except Exception:
            now_local = now_utc.astimezone(ZoneInfo("UTC"))

        kill_switch = state_store.is_global_kill() or state_store.is_station_paused(station)
        reason = _initial_skip_reason(row, ctx, now_local, kill_switch)
        if reason is None:
            pre_pass[pos] = True
            df.at[idx, "skipped_reason"] = ""
        else:
            df.at[idx, "skipped_reason"] = reason

    df["_pre_pass"] = pre_pass

    passed_pool = df.loc[df["_pre_pass"]].copy()
    selected_idx: set[int] = set()
    if not passed_pool.empty:
        ranked = passed_pool.sort_values(
            ["station", "market_day_local", "event_key", "edge", "strike_k"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        ranked["_rank"] = ranked.groupby(["station", "market_day_local", "event_key"], sort=False).cumcount() + 1
        selected_idx = set(ranked.loc[ranked["_rank"] <= int(ctx.top_n_per_event_day)].index.tolist())

    mask_pre_but_not_selected = df["_pre_pass"] & (~df.index.isin(selected_idx))
    df.loc[mask_pre_but_not_selected, "skipped_reason"] = "edge_too_low"

    station_added_risk: dict[tuple[str, str], float] = {}
    portfolio_added_risk: dict[str, float] = {}
    station_added_positions: dict[tuple[str, str], int] = {}
    added_positions_total_by_day: dict[str, int] = {}
    open_position_keys: set[str] = set()
    open_position_identity_keys: set[str] = set()
    cycle_added_position_keys: set[str] = set()
    cycle_added_position_identity_keys: set[str] = set()

    for pos in state_store.open_positions():
        if str(pos.get("status", "open")) != "open":
            continue
        position_identity_key = state_store.position_identity_key(
            station=pos.get("station"),
            market_day_local=pos.get("market_day_local"),
            strike_k=pos.get("strike_k"),
        )
        if position_identity_key:
            open_position_identity_keys.add(position_identity_key)

        market_id = str(pos.get("market_id") or "").strip()
        slug = str(pos.get("slug") or "").strip()
        if market_id:
            open_position_keys.add(f"mid:{market_id}")
        elif slug:
            open_position_keys.add(f"slug:{slug}")

    for idx in sorted(selected_idx, key=lambda i: float(df.at[i, "edge"]), reverse=True):
        row = df.loc[idx]
        station = str(row["station"])
        day_local = str(pd.to_datetime(row["market_day_local"]).date())
        stake_usd = float(row["stake_usd"])
        market_id = str(row.get("market_id") or "").strip()
        slug = str(row.get("slug") or "").strip()
        position_identity_key = state_store.position_identity_key(
            station=station,
            market_day_local=row.get("market_day_local"),
            strike_k=row.get("strike_k"),
        )
        position_key = f"mid:{market_id}" if market_id else (f"slug:{slug}" if slug else "")

        if position_identity_key and (
            position_identity_key in open_position_identity_keys
            or position_identity_key in cycle_added_position_identity_keys
        ):
            df.at[idx, "decision"] = "SKIP"
            df.at[idx, "skipped_reason"] = "already_open_position"
            continue

        if position_key and (position_key in open_position_keys or position_key in cycle_added_position_keys):
            df.at[idx, "decision"] = "SKIP"
            df.at[idx, "skipped_reason"] = "already_open_position"
            continue

        station_key = (day_local, station)
        added_station_risk = station_added_risk.get(station_key, 0.0)
        current_station_risk_daily = state_store.station_risk_used(day_local=day_local, station=station)
        current_station_risk_open = state_store.station_open_risk(day_local=day_local, station=station)
        current_station_risk = max(current_station_risk_daily, current_station_risk_open)
        station_risk_used_total = current_station_risk + added_station_risk

        added_portfolio = portfolio_added_risk.get(day_local, 0.0)
        current_portfolio_risk_daily = state_store.portfolio_risk_used(day_local=day_local)
        current_portfolio_risk_open = state_store.portfolio_open_risk(day_local=day_local)
        current_portfolio_risk = max(current_portfolio_risk_daily, current_portfolio_risk_open)
        portfolio_risk_used_total = current_portfolio_risk + added_portfolio

        station_risk_limit = float(ctx.nav_usd) * float(ctx.station_daily_risk_fraction)
        portfolio_risk_limit = float(ctx.nav_usd) * float(ctx.portfolio_daily_risk_fraction)

        df.at[idx, "risk_used_station_today"] = station_risk_used_total
        df.at[idx, "risk_used_station_daily"] = current_station_risk_daily
        df.at[idx, "risk_used_station_open"] = current_station_risk_open
        df.at[idx, "risk_limit_station_today"] = station_risk_limit
        df.at[idx, "risk_used_portfolio_today"] = portfolio_risk_used_total
        df.at[idx, "risk_used_portfolio_daily"] = current_portfolio_risk_daily
        df.at[idx, "risk_used_portfolio_open"] = current_portfolio_risk_open
        df.at[idx, "risk_limit_portfolio_today"] = portfolio_risk_limit

        if (station_risk_used_total + stake_usd > station_risk_limit) or (
            portfolio_risk_used_total + stake_usd > portfolio_risk_limit
        ):
            df.at[idx, "decision"] = "SKIP"
            df.at[idx, "skipped_reason"] = "risk_limit_hit"
            continue

        if position_identity_key and state_store.is_trade_cooldown_active(
            position_identity_key,
            cooldown_minutes=float(ctx.trade_cooldown_minutes),
            now_utc=now_utc,
        ):
            df.at[idx, "decision"] = "SKIP"
            df.at[idx, "skipped_reason"] = "trade_cooldown"
            continue

        open_station = state_store.open_position_count_for_station(station)
        open_total = state_store.open_position_count()
        added_station_positions = station_added_positions.get(station_key, 0)
        added_total_positions = added_positions_total_by_day.get(day_local, 0)

        if open_station + added_station_positions >= int(ctx.max_open_positions_per_station):
            df.at[idx, "decision"] = "SKIP"
            df.at[idx, "skipped_reason"] = "max_positions_hit"
            continue
        if open_total + added_total_positions >= int(ctx.max_open_positions_total):
            df.at[idx, "decision"] = "SKIP"
            df.at[idx, "skipped_reason"] = "max_positions_hit"
            continue

        df.at[idx, "decision"] = "TRADE"
        df.at[idx, "skipped_reason"] = ""

        station_added_risk[station_key] = added_station_risk + stake_usd
        portfolio_added_risk[day_local] = added_portfolio + stake_usd
        station_added_positions[station_key] = added_station_positions + 1
        added_positions_total_by_day[day_local] = added_total_positions + 1
        if position_identity_key:
            cycle_added_position_identity_keys.add(position_identity_key)
        if position_key:
            cycle_added_position_keys.add(position_key)

    df = df.drop(columns=["_pre_pass"], errors="ignore")
    return df
