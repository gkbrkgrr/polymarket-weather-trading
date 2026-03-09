from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .state import build_logical_market_key


MAX_HISTORY_PER_LOGICAL_KEY = 15
LOGGER = logging.getLogger("live_pilot")

PROGRESSION_FEATURE_COLUMNS: dict[str, str] = {
    "cycles_seen": "progression_cycles_seen",
    "consecutive_candidate_cycles": "progression_consecutive_candidate_cycles",
    "candidate_ratio_15": "progression_candidate_ratio_15",
    "edge_mean_last3": "progression_edge_mean_last3",
    "edge_mean_last5": "progression_edge_mean_last5",
    "edge_std_last5": "progression_edge_std_last5",
    "edge_trend_last3": "progression_edge_trend_last3",
    "p_model_mean_last3": "progression_p_model_mean_last3",
    "p_model_trend_last3": "progression_p_model_trend_last3",
    "mode_consistency_ratio": "progression_mode_consistency_ratio",
}

DEFAULT_PROGRESSION_CONFIDENCE_SCORE = 0.5
DEFAULT_PROGRESSION_CONFIDENCE_MULTIPLIER = 1.0
DEFAULT_PROGRESSIVE_MULTIPLIER_BANDS: tuple[tuple[float, float], ...] = (
    (0.30, 0.90),
    (0.50, 1.00),
    (0.70, 1.10),
    (float("inf"), 1.20),
)

LogicalKey = tuple[str, str, int]


def build_logical_key(*, station: Any, market_day_local: Any, strike_k: Any) -> LogicalKey | None:
    key = build_logical_market_key(station=station, market_day_local=market_day_local, strike_k=strike_k)
    if key is None:
        return None
    station_key, day_local, strike_txt = key.split("|", 2)
    return station_key, day_local, int(strike_txt)


def logical_key_to_state_key(logical_key: LogicalKey) -> str:
    station_key, day_local, strike_k = logical_key
    return f"{station_key}|{day_local}|{int(strike_k)}"


def _normalize_cycle_time_utc(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return datetime.now(tz=timezone.utc).isoformat()
    return ts.to_pydatetime().astimezone(timezone.utc).isoformat()


def _ensure_progression_state(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    progression = state.get("forecast_progression")
    if not isinstance(progression, dict):
        progression = {}
        state["forecast_progression"] = progression
    return progression


def _to_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        value_f = float(value)
        if not pd.isna(value_f):
            return value_f
    except Exception:
        pass
    return default


def _to_int(value: Any, *, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        value_i = int(value)
        return value_i
    except Exception:
        return default


def _to_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "t", "yes", "y"}:
            return True
        if norm in {"0", "false", "f", "no", "n"}:
            return False
    return bool(value)


def _safe_history_list(
    progression: dict[str, list[dict[str, Any]]],
    logical_key_state: str,
) -> list[dict[str, Any]]:
    history = progression.get(logical_key_state)
    if not isinstance(history, list):
        history = []
        progression[logical_key_state] = history
    return history


def _history_tail_numeric(history: Sequence[Mapping[str, Any]], field: str, size: int) -> list[float]:
    vals: list[float] = []
    for row in history[-max(0, int(size)) :]:
        value = _to_float(row.get(field))
        if value is not None:
            vals.append(value)
    return vals


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = _mean(values)
    var = sum((x - mu) * (x - mu) for x in values) / n
    return float(var**0.5)


def _linear_slope(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0

    mean_x = (n - 1) / 2.0
    mean_y = _mean(values)
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        dx = float(i) - mean_x
        num += dx * (float(y) - mean_y)
        den += dx * dx
    if den <= 0.0:
        return 0.0
    return float(num / den)


def _upsert_history_entry(history: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    cycle_time = str(entry.get("cycle_time_utc") or "")
    replaced = False
    for i, row in enumerate(history):
        if str(row.get("cycle_time_utc") or "") == cycle_time:
            history[i] = entry
            replaced = True
            break
    if not replaced:
        history.append(entry)

    history.sort(key=lambda row: str(row.get("cycle_time_utc") or ""))
    if len(history) > MAX_HISTORY_PER_LOGICAL_KEY:
        del history[:-MAX_HISTORY_PER_LOGICAL_KEY]


def compute_progression_features(history: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    history_size = len(history)
    if history_size <= 0:
        return {
            "cycles_seen": 0,
            "consecutive_candidate_cycles": 0,
            "candidate_ratio_15": 0.0,
            "edge_mean_last3": 0.0,
            "edge_mean_last5": 0.0,
            "edge_std_last5": 0.0,
            "edge_trend_last3": 0.0,
            "p_model_mean_last3": 0.0,
            "p_model_trend_last3": 0.0,
            "mode_consistency_ratio": 0.0,
        }

    candidate_count = 0
    consecutive_candidates = 0
    for row in history:
        if _to_bool(row.get("is_candidate"), default=False):
            candidate_count += 1

    for row in reversed(history):
        if _to_bool(row.get("is_candidate"), default=False):
            consecutive_candidates += 1
        else:
            break

    edge_last3 = _history_tail_numeric(history, "edge", 3)
    edge_last5 = _history_tail_numeric(history, "edge", 5)
    p_model_last3 = _history_tail_numeric(history, "p_model", 3)

    modes_last5 = [_to_int(row.get("mode_k")) for row in history[-5:]]
    if not modes_last5:
        mode_consistency_ratio = 0.0
    elif len(modes_last5) == 1:
        mode_consistency_ratio = 1.0
    else:
        stable_count = 0
        for i in range(1, len(modes_last5)):
            prev_mode = modes_last5[i - 1]
            curr_mode = modes_last5[i]
            if prev_mode is not None and curr_mode is not None and prev_mode == curr_mode:
                stable_count += 1
        mode_consistency_ratio = stable_count / float(len(modes_last5) - 1)

    return {
        "cycles_seen": int(history_size),
        "consecutive_candidate_cycles": int(consecutive_candidates),
        "candidate_ratio_15": float(candidate_count / float(history_size)),
        "edge_mean_last3": _mean(edge_last3),
        "edge_mean_last5": _mean(edge_last5),
        "edge_std_last5": _std(edge_last5),
        "edge_trend_last3": _linear_slope(edge_last3),
        "p_model_mean_last3": _mean(p_model_last3),
        "p_model_trend_last3": _linear_slope(p_model_last3),
        "mode_consistency_ratio": float(mode_consistency_ratio),
    }


def update_progression_history(
    state: dict[str, Any],
    candidates_df: pd.DataFrame,
    cycle_time_utc: Any,
) -> None:
    if candidates_df.empty:
        _ensure_progression_state(state)
        return

    cycle_time_iso = _normalize_cycle_time_utc(cycle_time_utc)
    progression = _ensure_progression_state(state)

    for row in candidates_df.itertuples(index=False):
        record = row._asdict()
        logical_key = build_logical_key(
            station=record.get("station"),
            market_day_local=record.get("market_day_local"),
            strike_k=record.get("strike_k"),
        )
        if logical_key is None:
            continue

        p_model = _to_float(record.get("p_model"))
        chosen_no_ask = _to_float(record.get("chosen_no_ask"))
        edge = _to_float(record.get("edge"))
        if edge is None and p_model is not None and chosen_no_ask is not None:
            edge = float((1.0 - float(p_model)) - float(chosen_no_ask))

        is_candidate_default = True
        if "decision" in record:
            is_candidate_default = str(record.get("decision") or "").upper() in {"TRADE", "SKIP"}

        entry = {
            "cycle_time_utc": cycle_time_iso,
            "p_model": float(p_model) if p_model is not None else float("nan"),
            "edge": float(edge) if edge is not None else float("nan"),
            "chosen_no_ask": float(chosen_no_ask) if chosen_no_ask is not None else float("nan"),
            "mode_k": int(_to_int(record.get("mode_k"), default=0) or 0),
            "is_candidate": bool(_to_bool(record.get("is_candidate"), default=is_candidate_default)),
            "decision": "NONE",
        }

        logical_key_state = logical_key_to_state_key(logical_key)
        history = _safe_history_list(progression, logical_key_state)
        _upsert_history_entry(history, entry)

        if LOGGER.isEnabledFor(logging.DEBUG):
            features = compute_progression_features(history)
            LOGGER.debug(
                "forecast_progression logical_key=%s history_length=%d consecutive_candidate_cycles=%d",
                logical_key,
                len(history),
                features["consecutive_candidate_cycles"],
            )


def attach_progression_features(state: dict[str, Any], candidates_df: pd.DataFrame) -> pd.DataFrame:
    out = candidates_df.copy()
    progression = _ensure_progression_state(state)

    if out.empty:
        for feature_name, column_name in PROGRESSION_FEATURE_COLUMNS.items():
            default_val: int | float = 0 if feature_name in {"cycles_seen", "consecutive_candidate_cycles"} else 0.0
            out[column_name] = default_val
        return out

    features_cache: dict[str, dict[str, float | int]] = {}
    feature_values: dict[str, list[float | int]] = {
        feature_name: []
        for feature_name in PROGRESSION_FEATURE_COLUMNS
    }

    for row in out.itertuples(index=False):
        record = row._asdict()
        logical_key = build_logical_key(
            station=record.get("station"),
            market_day_local=record.get("market_day_local"),
            strike_k=record.get("strike_k"),
        )

        if logical_key is None:
            features = compute_progression_features([])
        else:
            logical_key_state = logical_key_to_state_key(logical_key)
            features = features_cache.get(logical_key_state)
            if features is None:
                history = _safe_history_list(progression, logical_key_state)
                features = compute_progression_features(history)
                features_cache[logical_key_state] = features

        for feature_name in PROGRESSION_FEATURE_COLUMNS:
            feature_values[feature_name].append(features[feature_name])

    for feature_name, column_name in PROGRESSION_FEATURE_COLUMNS.items():
        out[column_name] = feature_values[feature_name]

    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, float(value)))


def neutral_progression_result(*, gate_pass: bool, gate_reason: str) -> dict[str, Any]:
    return {
        "progression_gate_pass": bool(gate_pass),
        "progression_gate_reason": str(gate_reason),
        "progression_confidence_score": float(DEFAULT_PROGRESSION_CONFIDENCE_SCORE),
        "progression_confidence_multiplier": float(DEFAULT_PROGRESSION_CONFIDENCE_MULTIPLIER),
    }


def score_to_progression_multiplier(score: float) -> float:
    score_f = _clamp(float(score), 0.0, 1.0)
    for threshold, multiplier in DEFAULT_PROGRESSIVE_MULTIPLIER_BANDS:
        if score_f < float(threshold):
            return float(multiplier)
    return float(DEFAULT_PROGRESSION_CONFIDENCE_MULTIPLIER)


def evaluate_progression_controls(
    row: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    p_model_max: float,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    try:
        gate_enabled = _to_bool(cfg.get("progression_enable_gate"), default=True)
        use_progression_confidence = _to_bool(cfg.get("use_progression_confidence"), default=True)
        enable_size_multiplier = _to_bool(cfg.get("progression_enable_size_multiplier"), default=True)

        raw_cycles_seen = _to_int(row.get(PROGRESSION_FEATURE_COLUMNS["cycles_seen"]))
        raw_consecutive = _to_int(row.get(PROGRESSION_FEATURE_COLUMNS["consecutive_candidate_cycles"]))
        raw_candidate_ratio = _to_float(row.get(PROGRESSION_FEATURE_COLUMNS["candidate_ratio_15"]))
        raw_edge_trend = _to_float(row.get(PROGRESSION_FEATURE_COLUMNS["edge_trend_last3"]))
        raw_p_model_mean = _to_float(row.get(PROGRESSION_FEATURE_COLUMNS["p_model_mean_last3"]))
        raw_p_model_trend = _to_float(row.get(PROGRESSION_FEATURE_COLUMNS["p_model_trend_last3"]))
        raw_mode_consistency = _to_float(row.get(PROGRESSION_FEATURE_COLUMNS["mode_consistency_ratio"]))
        raw_edge_std = _to_float(row.get(PROGRESSION_FEATURE_COLUMNS["edge_std_last5"]))

        missing_cols: list[str] = []
        required_pairs = [
            (PROGRESSION_FEATURE_COLUMNS["cycles_seen"], raw_cycles_seen),
            (PROGRESSION_FEATURE_COLUMNS["consecutive_candidate_cycles"], raw_consecutive),
            (PROGRESSION_FEATURE_COLUMNS["candidate_ratio_15"], raw_candidate_ratio),
            (PROGRESSION_FEATURE_COLUMNS["edge_trend_last3"], raw_edge_trend),
            (PROGRESSION_FEATURE_COLUMNS["p_model_mean_last3"], raw_p_model_mean),
            (PROGRESSION_FEATURE_COLUMNS["p_model_trend_last3"], raw_p_model_trend),
            (PROGRESSION_FEATURE_COLUMNS["mode_consistency_ratio"], raw_mode_consistency),
            (PROGRESSION_FEATURE_COLUMNS["edge_std_last5"], raw_edge_std),
        ]
        for col_name, value in required_pairs:
            if value is None:
                missing_cols.append(col_name)

        if missing_cols:
            if logger is not None:
                logger.warning(
                    "progression_neutral_fallback reason=missing_or_malformed_data station=%s day=%s strike_k=%s missing=%s",
                    row.get("station"),
                    row.get("market_day_local"),
                    row.get("strike_k"),
                    ",".join(sorted(missing_cols)),
                )
            return neutral_progression_result(
                gate_pass=True,
                gate_reason="progression_neutral_missing_data",
            )

        cycles_seen = int(raw_cycles_seen or 0)
        consecutive_cycles = int(raw_consecutive or 0)
        candidate_ratio = float(raw_candidate_ratio or 0.0)
        edge_trend = float(raw_edge_trend or 0.0)
        p_model_mean = float(raw_p_model_mean or 0.0)
        p_model_trend = float(raw_p_model_trend or 0.0)
        mode_consistency = float(raw_mode_consistency or 0.0)
        edge_std = float(raw_edge_std or 0.0)

        gate_pass = True
        gate_reason = "progression_gate_disabled" if not gate_enabled else ""

        if gate_enabled:
            min_cycles_seen = max(0, int(_to_int(cfg.get("progression_min_cycles_seen"), default=3) or 3))
            min_consecutive = max(
                0,
                int(_to_int(cfg.get("progression_min_consecutive_candidate_cycles"), default=2) or 2),
            )

            if cycles_seen < min_cycles_seen:
                gate_pass = False
                gate_reason = "progression_insufficient_history"
            elif consecutive_cycles < min_consecutive:
                gate_pass = False
                gate_reason = "progression_not_persistent"
            elif _to_bool(cfg.get("progression_enable_negative_veto"), default=True):
                negative_edge_trend_threshold = float(
                    _to_float(cfg.get("progression_negative_edge_trend_threshold"), default=-0.01) or -0.01
                )
                min_mode_consistency_ratio = float(
                    _to_float(cfg.get("progression_min_mode_consistency_ratio"), default=0.40) or 0.40
                )
                negative_p_model_trend_threshold = float(
                    _to_float(cfg.get("progression_negative_p_model_trend_threshold"), default=0.01) or 0.01
                )
                p_model_max_ref = max(0.0, float(p_model_max))
                near_p_model_limit = p_model_max_ref * 0.90

                if edge_trend < negative_edge_trend_threshold:
                    gate_pass = False
                    gate_reason = "progression_negative_edge_trend"
                elif mode_consistency < min_mode_consistency_ratio:
                    gate_pass = False
                    gate_reason = "progression_low_mode_consistency"
                elif (
                    p_model_max_ref > 0.0
                    and p_model_trend > negative_p_model_trend_threshold
                    and p_model_mean >= near_p_model_limit
                ):
                    gate_pass = False
                    gate_reason = "progression_worsening_p_model"
                else:
                    gate_reason = "progression_gate_pass"
            else:
                gate_reason = "progression_gate_pass"

        score = float(DEFAULT_PROGRESSION_CONFIDENCE_SCORE)
        if use_progression_confidence:
            weight_consecutive = max(
                0.0,
                float(_to_float(cfg.get("progression_weight_consecutive"), default=0.30) or 0.30),
            )
            weight_candidate_ratio = max(
                0.0,
                float(_to_float(cfg.get("progression_weight_candidate_ratio"), default=0.20) or 0.20),
            )
            weight_edge_trend = max(
                0.0,
                float(_to_float(cfg.get("progression_weight_edge_trend"), default=0.20) or 0.20),
            )
            weight_mode_consistency = max(
                0.0,
                float(_to_float(cfg.get("progression_weight_mode_consistency"), default=0.15) or 0.15),
            )
            weight_low_p_model = max(
                0.0,
                float(_to_float(cfg.get("progression_weight_low_p_model"), default=0.10) or 0.10),
            )
            weight_low_edge_volatility = max(
                0.0,
                float(_to_float(cfg.get("progression_weight_low_edge_volatility"), default=0.05) or 0.05),
            )

            weight_total = (
                weight_consecutive
                + weight_candidate_ratio
                + weight_edge_trend
                + weight_mode_consistency
                + weight_low_p_model
                + weight_low_edge_volatility
            )
            if weight_total > 0.0:
                edge_trend_cap = abs(float(_to_float(cfg.get("progression_edge_trend_cap"), default=0.05) or 0.05))
                if edge_trend_cap <= 0.0:
                    edge_trend_cap = 0.05

                consecutive_norm = _clamp(float(consecutive_cycles) / 4.0, 0.0, 1.0)
                candidate_ratio_norm = _clamp(candidate_ratio, 0.0, 1.0)
                edge_trend_norm = (
                    _clamp(edge_trend, -edge_trend_cap, edge_trend_cap) + edge_trend_cap
                ) / (2.0 * edge_trend_cap)
                mode_consistency_norm = _clamp(mode_consistency, 0.0, 1.0)
                p_model_ref = max(1e-6, float(p_model_max))
                low_p_model_norm = 1.0 - _clamp(float(p_model_mean) / p_model_ref, 0.0, 1.0)
                low_edge_volatility_norm = 1.0 - _clamp(float(edge_std) / edge_trend_cap, 0.0, 1.0)

                raw_score = (
                    (weight_consecutive * consecutive_norm)
                    + (weight_candidate_ratio * candidate_ratio_norm)
                    + (weight_edge_trend * edge_trend_norm)
                    + (weight_mode_consistency * mode_consistency_norm)
                    + (weight_low_p_model * low_p_model_norm)
                    + (weight_low_edge_volatility * low_edge_volatility_norm)
                ) / weight_total

                min_cycles_for_full_score = max(
                    1,
                    int(_to_int(cfg.get("progression_min_cycles_seen"), default=3) or 3),
                )
                history_penalty = 0.15 * (
                    1.0 - _clamp(float(cycles_seen) / float(min_cycles_for_full_score), 0.0, 1.0)
                )
                raw_score -= history_penalty

                worsening_p_model_threshold = abs(
                    float(
                        _to_float(
                            cfg.get("progression_negative_p_model_trend_threshold"),
                            default=0.01,
                        )
                        or 0.01
                    )
                )
                trend_scale = max(1e-6, worsening_p_model_threshold)
                if p_model_trend > 0.0:
                    raw_score -= min(0.10, 0.05 * (p_model_trend / trend_scale))

                score = _clamp(raw_score, 0.0, 1.0)

        multiplier = float(DEFAULT_PROGRESSION_CONFIDENCE_MULTIPLIER)
        if use_progression_confidence and enable_size_multiplier:
            min_multiplier = float(
                _to_float(cfg.get("progression_min_size_multiplier"), default=0.85) or 0.85
            )
            max_multiplier = float(
                _to_float(cfg.get("progression_max_size_multiplier"), default=1.35) or 1.35
            )
            if min_multiplier > max_multiplier:
                min_multiplier, max_multiplier = max_multiplier, min_multiplier
            multiplier = _clamp(
                score_to_progression_multiplier(score),
                min_multiplier,
                max_multiplier,
            )

        return {
            "progression_gate_pass": bool(gate_pass),
            "progression_gate_reason": str(gate_reason),
            "progression_confidence_score": _clamp(float(score), 0.0, 1.0),
            "progression_confidence_multiplier": float(multiplier),
        }
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "progression_neutral_fallback reason=evaluation_exception station=%s day=%s strike_k=%s error=%s: %s",
                row.get("station"),
                row.get("market_day_local"),
                row.get("strike_k"),
                exc.__class__.__name__,
                exc,
            )
        return neutral_progression_result(
            gate_pass=True,
            gate_reason="progression_neutral_exception",
        )


def apply_pending_progression_updates(
    state: dict[str, Any],
    pending_dir: Path,
    *,
    logger: logging.Logger | None = None,
) -> list[Path]:
    if not pending_dir.exists():
        return []

    processed: list[Path] = []
    for path in sorted(pending_dir.glob("*.parquet")):
        try:
            batch_df = pd.read_parquet(path)
            if batch_df.empty:
                processed.append(path)
                continue
            if "cycle_time_utc" not in batch_df.columns:
                raise ValueError("missing cycle_time_utc column")

            for cycle_time_utc, grp in batch_df.groupby("cycle_time_utc", sort=True):
                payload_df = grp.drop(columns=["cycle_time_utc"], errors="ignore")
                update_progression_history(state, payload_df, cycle_time_utc)
            processed.append(path)
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Failed to apply pending forecast progression batch path=%s error=%s: %s",
                    path,
                    exc.__class__.__name__,
                    exc,
                )
    return processed
