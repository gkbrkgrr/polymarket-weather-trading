from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .forecast_progression import PROGRESSION_FEATURE_COLUMNS, evaluate_progression_controls
from .state import PilotStateStore
from .utils_time import is_within_trade_window

LOGGER = logging.getLogger("live_pilot")


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
    "ensemble_high_std",
    "ensemble_high_range",
    "ensemble_low_same_side_ratio",
    "ensemble_strike_disagreement",
    "ensemble_invalid_data",
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
    use_progression_confidence: bool = True
    progression_enable_gate: bool = True
    progression_min_cycles_seen: int = 3
    progression_min_consecutive_candidate_cycles: int = 2
    progression_enable_negative_veto: bool = True
    progression_negative_edge_trend_threshold: float = -0.01
    progression_min_mode_consistency_ratio: float = 0.40
    progression_negative_p_model_trend_threshold: float = 0.01
    progression_weight_consecutive: float = 0.30
    progression_weight_candidate_ratio: float = 0.20
    progression_weight_edge_trend: float = 0.20
    progression_weight_mode_consistency: float = 0.15
    progression_weight_low_p_model: float = 0.10
    progression_weight_low_edge_volatility: float = 0.05
    progression_edge_trend_cap: float = 0.05
    progression_enable_size_multiplier: bool = True
    progression_min_size_multiplier: float = 0.85
    progression_max_size_multiplier: float = 1.35
    use_ensemble_confidence: bool = True
    ensemble_probability_adjustment_enabled: bool = True
    ensemble_trade_size_adjustment_enabled: bool = True
    ensemble_disagreement_neutral_shrink_cap: float = 0.25
    ensemble_std_cap_c: float = 2.0
    ensemble_range_cap_c: float = 4.0
    ensemble_enable_gate: bool = True
    ensemble_min_same_side_ratio: float = 0.67
    ensemble_max_std_c_for_trade: float = 2.5
    ensemble_max_range_c_for_trade: float = 5.0
    ensemble_enable_strike_disagreement_veto: bool = True
    ensemble_min_size_multiplier: float = 0.75
    ensemble_max_size_multiplier: float = 1.15


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


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return bool(value)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _normalize_disagreement(
    *,
    std_c: float | None,
    range_c: float | None,
    iqr_c: float | None,
    std_cap_c: float,
    range_cap_c: float,
) -> float:
    std_cap = max(1e-9, float(std_cap_c))
    range_cap = max(1e-9, float(range_cap_c))
    iqr_cap = max(1e-9, float(range_cap_c) * 0.5)

    std_n = _clip01(0.0 if std_c is None else max(0.0, float(std_c)) / std_cap)
    range_n = _clip01(0.0 if range_c is None else max(0.0, float(range_c)) / range_cap)
    iqr_n = _clip01(0.0 if iqr_c is None else max(0.0, float(iqr_c)) / iqr_cap)
    return _clip01((std_n + range_n + iqr_n) / 3.0)


def _size_multiplier_from_agreement(agreement_score: float, ctx: PolicyContext) -> float:
    agreement = _clip01(agreement_score)
    min_mult = max(0.0, float(ctx.ensemble_min_size_multiplier))
    max_mult = max(min_mult, float(ctx.ensemble_max_size_multiplier))
    if agreement >= 0.5:
        up = (agreement - 0.5) / 0.5
        mult = 1.0 + up * (max_mult - 1.0)
    else:
        down = (0.5 - agreement) / 0.5
        mult = 1.0 - down * (1.0 - min_mult)
    return min(max_mult, max(min_mult, float(mult)))


def _evaluate_ensemble_controls(row: pd.Series, ctx: PolicyContext) -> dict[str, Any]:
    std_c = _as_float_or_none(row.get("ensemble_pred_std"))
    range_c = _as_float_or_none(row.get("ensemble_pred_range"))
    iqr_c = _as_float_or_none(row.get("ensemble_pred_iqr"))

    yes_count = _as_float_or_none(row.get("ensemble_models_yes_count"))
    no_count = _as_float_or_none(row.get("ensemble_models_no_count"))
    same_side_ratio = _as_float_or_none(row.get("ensemble_same_side_ratio"))
    if same_side_ratio is None and yes_count is not None and no_count is not None and (yes_count + no_count) > 0:
        same_side_ratio = max(yes_count, no_count) / (yes_count + no_count)
    if same_side_ratio is not None:
        same_side_ratio = _clip01(same_side_ratio)

    strike_disagreement = _as_bool(
        row.get("ensemble_strike_disagreement_flag"),
        default=_as_bool(row.get("ensemble_cross_strike_disagreement"), default=False),
    )
    if not strike_disagreement and yes_count is not None and no_count is not None:
        strike_disagreement = yes_count > 0 and no_count > 0

    disagreement = _as_float_or_none(row.get("ensemble_disagreement_score"))
    if disagreement is None:
        disagreement = _normalize_disagreement(
            std_c=std_c,
            range_c=range_c,
            iqr_c=iqr_c,
            std_cap_c=float(ctx.ensemble_std_cap_c),
            range_cap_c=float(ctx.ensemble_range_cap_c),
        )
    disagreement = _clip01(disagreement)

    agreement = _as_float_or_none(row.get("ensemble_agreement_score"))
    if agreement is None:
        agreement = 1.0 - disagreement
    agreement = _clip01(agreement)

    sign_agreement_ratio = _as_float_or_none(row.get("ensemble_sign_agreement_ratio"))
    if sign_agreement_ratio is None:
        sign_agreement_ratio = 0.5
    sign_agreement_ratio = _clip01(sign_agreement_ratio)

    confidence_multiplier = _as_float_or_none(row.get("ensemble_confidence_multiplier"))
    if confidence_multiplier is None:
        shrink_cap = _clip01(float(ctx.ensemble_disagreement_neutral_shrink_cap))
        confidence_multiplier = 1.0 - disagreement * shrink_cap
    confidence_multiplier = max(0.0, float(confidence_multiplier))

    uncertainty_penalty = _as_float_or_none(row.get("ensemble_uncertainty_penalty"))
    if uncertainty_penalty is None:
        uncertainty_penalty = max(0.0, 1.0 - confidence_multiplier)
    uncertainty_penalty = max(0.0, float(uncertainty_penalty))

    fallback_raw = row.get("ensemble_fallback_marker")
    try:
        if fallback_raw is None or pd.isna(fallback_raw):
            fallback_marker = ""
        else:
            fallback_marker = str(fallback_raw).strip()
    except Exception:
        fallback_marker = str(fallback_raw).strip() if fallback_raw is not None else ""
    has_any_ensemble_metric = any(v is not None for v in (std_c, range_c, iqr_c, same_side_ratio))

    gate_pass = True
    gate_reason = "ensemble_gate_disabled"
    if bool(ctx.ensemble_enable_gate):
        gate_reason = "ensemble_gate_pass"
        if std_c is not None and float(std_c) > float(ctx.ensemble_max_std_c_for_trade):
            gate_pass = False
            gate_reason = "ensemble_high_std"
        elif range_c is not None and float(range_c) > float(ctx.ensemble_max_range_c_for_trade):
            gate_pass = False
            gate_reason = "ensemble_high_range"
        elif same_side_ratio is not None and float(same_side_ratio) < float(ctx.ensemble_min_same_side_ratio):
            gate_pass = False
            gate_reason = "ensemble_low_same_side_ratio"
        elif bool(ctx.ensemble_enable_strike_disagreement_veto) and strike_disagreement:
            gate_pass = False
            gate_reason = "ensemble_strike_disagreement"
        elif not has_any_ensemble_metric and not fallback_marker:
            gate_reason = "ensemble_gate_neutral_pass_missing_data"

    if bool(ctx.ensemble_trade_size_adjustment_enabled) and bool(ctx.use_ensemble_confidence):
        ensemble_size_multiplier = _size_multiplier_from_agreement(agreement, ctx)
    else:
        ensemble_size_multiplier = 1.0

    return {
        "ensemble_agreement_score": agreement,
        "ensemble_disagreement_score": disagreement,
        "ensemble_sign_agreement_ratio": sign_agreement_ratio,
        "ensemble_cross_strike_disagreement": bool(strike_disagreement),
        "ensemble_same_side_ratio": 1.0 if same_side_ratio is None else _clip01(same_side_ratio),
        "ensemble_models_yes_count": int(max(0.0, yes_count)) if yes_count is not None else 0,
        "ensemble_models_no_count": int(max(0.0, no_count)) if no_count is not None else 0,
        "ensemble_strike_disagreement_flag": bool(strike_disagreement),
        "ensemble_confidence_multiplier": float(confidence_multiplier),
        "ensemble_uncertainty_penalty": float(uncertainty_penalty),
        "ensemble_size_multiplier": float(ensemble_size_multiplier),
        "ensemble_gate_pass": bool(gate_pass),
        "ensemble_gate_reason": str(gate_reason),
        "ensemble_fallback_marker": fallback_marker,
    }


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
    df["base_size_before_progression"] = pd.to_numeric(df["size"], errors="coerce")
    df["base_stake_usd_before_progression"] = pd.to_numeric(df["stake_usd"], errors="coerce")
    df["final_size_after_progression"] = pd.to_numeric(df["size"], errors="coerce")
    df["final_stake_usd_after_progression"] = pd.to_numeric(df["stake_usd"], errors="coerce")
    df["final_size_after_ensemble"] = pd.to_numeric(df["size"], errors="coerce")
    df["final_stake_usd_after_ensemble"] = pd.to_numeric(df["stake_usd"], errors="coerce")
    df["progression_gate_pass"] = True
    df["progression_gate_reason"] = "progression_not_evaluated"
    df["progression_confidence_score"] = 0.5
    df["progression_confidence_multiplier"] = 1.0
    if "p_model_raw" not in df.columns:
        df["p_model_raw"] = pd.to_numeric(df["p_model"], errors="coerce")
    else:
        df["p_model_raw"] = pd.to_numeric(df["p_model_raw"], errors="coerce").fillna(
            pd.to_numeric(df["p_model"], errors="coerce")
        )
    if "p_model_adjusted" not in df.columns:
        df["p_model_adjusted"] = pd.to_numeric(df["p_model"], errors="coerce")
    else:
        df["p_model_adjusted"] = pd.to_numeric(df["p_model_adjusted"], errors="coerce").fillna(
            pd.to_numeric(df["p_model"], errors="coerce")
        )

    for col_name, default_val in (
        ("ensemble_agreement_score", 0.5),
        ("ensemble_disagreement_score", 0.5),
        ("ensemble_sign_agreement_ratio", 0.5),
        ("ensemble_same_side_ratio", 1.0),
        ("ensemble_confidence_multiplier", 1.0),
        ("ensemble_uncertainty_penalty", 0.0),
        ("ensemble_size_multiplier", 1.0),
    ):
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(default_val)
        else:
            df[col_name] = default_val
    for col_name, default_val in (
        ("ensemble_models_yes_count", 0),
        ("ensemble_models_no_count", 0),
    ):
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(default_val).astype(int)
        else:
            df[col_name] = int(default_val)
    for col_name, default_val in (
        ("ensemble_cross_strike_disagreement", False),
        ("ensemble_strike_disagreement_flag", False),
    ):
        if col_name in df.columns:
            df[col_name] = df[col_name].map(lambda x: _as_bool(x, default=default_val))
        else:
            df[col_name] = bool(default_val)
    if "ensemble_gate_pass" not in df.columns:
        df["ensemble_gate_pass"] = True
    if "ensemble_gate_reason" not in df.columns:
        df["ensemble_gate_reason"] = "ensemble_not_evaluated"
    if "ensemble_fallback_marker" not in df.columns:
        df["ensemble_fallback_marker"] = ""

    pre_pass = np.zeros(len(df), dtype=bool)

    for pos, (idx, row) in enumerate(df.iterrows()):
        station = str(row["station"])
        tz = station_timezones.get(station, "UTC")
        try:
            now_local = now_utc.astimezone(ZoneInfo(tz))
        except Exception:
            now_local = now_utc.astimezone(ZoneInfo("UTC"))

        kill_switch = state_store.is_global_kill()
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

    ordered_selected_idx = sorted(selected_idx, key=lambda i: float(df.at[i, "edge"]), reverse=True)
    final_eligible_idx: list[int] = []
    progression_cfg = ctx.__dict__
    missing_progression_columns = [
        col_name
        for col_name in PROGRESSION_FEATURE_COLUMNS.values()
        if col_name not in df.columns
    ]
    if missing_progression_columns:
        LOGGER.warning(
            "progression_neutral_fallback reason=missing_progression_columns missing=%s",
            ",".join(sorted(missing_progression_columns)),
        )
    missing_ensemble_columns = [
        col_name
        for col_name in (
            "ensemble_pred_std",
            "ensemble_pred_range",
            "ensemble_same_side_ratio",
            "ensemble_disagreement_score",
        )
        if col_name not in df.columns
    ]
    if missing_ensemble_columns:
        LOGGER.warning(
            "ensemble_neutral_fallback reason=missing_ensemble_columns missing=%s",
            ",".join(sorted(missing_ensemble_columns)),
        )
    for idx in ordered_selected_idx:
        row = df.loc[idx]
        progression = evaluate_progression_controls(
            row,
            cfg=progression_cfg,
            p_model_max=float(ctx.p_model_max),
            logger=None if missing_progression_columns else LOGGER,
        )

        gate_pass = bool(progression.get("progression_gate_pass", True))
        gate_reason = str(progression.get("progression_gate_reason", "progression_gate_pass"))
        confidence_score = float(progression.get("progression_confidence_score", 0.5))
        confidence_multiplier = float(progression.get("progression_confidence_multiplier", 1.0))
        ensemble = _evaluate_ensemble_controls(row, ctx)
        ensemble_gate_pass = bool(ensemble.get("ensemble_gate_pass", True))
        ensemble_gate_reason = str(ensemble.get("ensemble_gate_reason", "ensemble_gate_pass"))
        ensemble_size_multiplier = float(ensemble.get("ensemble_size_multiplier", 1.0))

        base_size = pd.to_numeric(df.at[idx, "base_size_before_progression"], errors="coerce")
        if pd.isna(base_size):
            base_size = 0.0
        base_stake = pd.to_numeric(df.at[idx, "base_stake_usd_before_progression"], errors="coerce")
        if pd.isna(base_stake):
            price = pd.to_numeric(row.get("chosen_no_ask"), errors="coerce")
            base_stake = float(price) * float(base_size) if not pd.isna(price) else 0.0

        final_size = float(base_size)
        final_stake = float(base_stake)
        combined_multiplier = float(confidence_multiplier) * float(ensemble_size_multiplier)
        combined_gate_pass = bool(gate_pass and ensemble_gate_pass)
        if combined_gate_pass:
            final_size = float(base_size) * float(combined_multiplier)
            final_stake = float(base_stake) * float(combined_multiplier)
            final_eligible_idx.append(idx)
        else:
            df.at[idx, "decision"] = "SKIP"
            df.at[idx, "skipped_reason"] = gate_reason if not gate_pass else ensemble_gate_reason

        df.at[idx, "progression_gate_pass"] = gate_pass
        df.at[idx, "progression_gate_reason"] = gate_reason
        df.at[idx, "progression_confidence_score"] = confidence_score
        df.at[idx, "progression_confidence_multiplier"] = confidence_multiplier
        df.at[idx, "ensemble_agreement_score"] = float(ensemble.get("ensemble_agreement_score", 0.5))
        df.at[idx, "ensemble_disagreement_score"] = float(ensemble.get("ensemble_disagreement_score", 0.5))
        df.at[idx, "ensemble_sign_agreement_ratio"] = float(ensemble.get("ensemble_sign_agreement_ratio", 0.5))
        df.at[idx, "ensemble_cross_strike_disagreement"] = bool(
            ensemble.get("ensemble_cross_strike_disagreement", False)
        )
        df.at[idx, "ensemble_models_yes_count"] = int(ensemble.get("ensemble_models_yes_count", 0))
        df.at[idx, "ensemble_models_no_count"] = int(ensemble.get("ensemble_models_no_count", 0))
        df.at[idx, "ensemble_same_side_ratio"] = float(ensemble.get("ensemble_same_side_ratio", 0.5))
        df.at[idx, "ensemble_strike_disagreement_flag"] = bool(
            ensemble.get("ensemble_strike_disagreement_flag", False)
        )
        df.at[idx, "ensemble_confidence_multiplier"] = float(
            ensemble.get("ensemble_confidence_multiplier", 1.0)
        )
        df.at[idx, "ensemble_uncertainty_penalty"] = float(
            ensemble.get("ensemble_uncertainty_penalty", 0.0)
        )
        df.at[idx, "ensemble_size_multiplier"] = float(ensemble_size_multiplier)
        df.at[idx, "ensemble_gate_pass"] = bool(ensemble_gate_pass)
        df.at[idx, "ensemble_gate_reason"] = str(ensemble_gate_reason)
        if str(ensemble.get("ensemble_fallback_marker", "")).strip():
            df.at[idx, "ensemble_fallback_marker"] = str(ensemble.get("ensemble_fallback_marker"))
        df.at[idx, "size"] = final_size
        df.at[idx, "stake_usd"] = final_stake
        df.at[idx, "final_size_after_progression"] = final_size
        df.at[idx, "final_stake_usd_after_progression"] = final_stake
        df.at[idx, "final_size_after_ensemble"] = final_size
        df.at[idx, "final_stake_usd_after_ensemble"] = final_stake

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

    for idx in final_eligible_idx:
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
