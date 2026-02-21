from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Deque

import numpy as np
import pandas as pd


@dataclass(slots=True)
class StageAStats:
    fallback_usage: Counter
    insufficient_primary: Counter


class _EWMAGroupState:
    def __init__(self, *, window_days: float, halflife_days: float) -> None:
        self.window_seconds = float(window_days) * 86400.0
        self.halflife_seconds = float(halflife_days) * 86400.0
        self.samples: Deque[tuple[float, float]] = deque()

    def _prune(self, current_time_s: float) -> None:
        cutoff = current_time_s - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def bias_and_count(self, current_time_s: float) -> tuple[float, int]:
        self._prune(current_time_s)
        n = len(self.samples)
        if n == 0:
            return 0.0, 0

        numer = 0.0
        denom = 0.0
        decay = math.log(2.0) / self.halflife_seconds
        for ts_s, residual in self.samples:
            weight = math.exp(-decay * (current_time_s - ts_s))
            numer += weight * residual
            denom += weight
        if denom <= 0:
            return 0.0, n
        return numer / denom, n

    def add(self, *, issue_time_s: float, residual: float) -> None:
        self.samples.append((issue_time_s, float(residual)))


class StageABiasEngine:
    """
    Leakage-safe online Stage-A bias estimator.

    Primary group: (model_name, station_name, cycle, lead_hours)
    Fallback 1:   (model_name, cycle, lead_hours)        # drop station
    Fallback 2:   (model_name, lead_hours)               # drop cycle
    """

    def __init__(
        self,
        *,
        rolling_window_days: int,
        ewma_halflife_days: float,
        min_history: int,
    ) -> None:
        self.window_days = int(rolling_window_days)
        self.halflife_days = float(ewma_halflife_days)
        self.min_history = int(min_history)

        self._states_l0: dict[tuple[object, ...], _EWMAGroupState] = {}
        self._states_l1: dict[tuple[object, ...], _EWMAGroupState] = {}
        self._states_l2: dict[tuple[object, ...], _EWMAGroupState] = {}

        self.fallback_usage: Counter = Counter()
        self.insufficient_primary: Counter = Counter()

    @staticmethod
    def _lead_key(lead_hours: float) -> int | float:
        if pd.isna(lead_hours):
            return float("nan")
        rounded = int(round(float(lead_hours)))
        if abs(float(lead_hours) - rounded) < 1e-6:
            return rounded
        return round(float(lead_hours), 3)

    def _key_l0(self, row) -> tuple[object, ...]:
        return (
            row.model_name,
            row.station_name,
            int(row.cycle),
            self._lead_key(row.lead_hours),
        )

    def _key_l1(self, row) -> tuple[object, ...]:
        return (
            row.model_name,
            int(row.cycle),
            self._lead_key(row.lead_hours),
        )

    def _key_l2(self, row) -> tuple[object, ...]:
        return (
            row.model_name,
            self._lead_key(row.lead_hours),
        )

    def _get_state(self, level: int, key: tuple[object, ...]) -> _EWMAGroupState | None:
        if level == 0:
            return self._states_l0.get(key)
        if level == 1:
            return self._states_l1.get(key)
        if level == 2:
            return self._states_l2.get(key)
        raise ValueError(f"Unsupported level={level}")

    def _ensure_state(self, level: int, key: tuple[object, ...]) -> _EWMAGroupState:
        if level == 0:
            state = self._states_l0.get(key)
            if state is None:
                state = _EWMAGroupState(
                    window_days=self.window_days,
                    halflife_days=self.halflife_days,
                )
                self._states_l0[key] = state
            return state
        if level == 1:
            state = self._states_l1.get(key)
            if state is None:
                state = _EWMAGroupState(
                    window_days=self.window_days,
                    halflife_days=self.halflife_days,
                )
                self._states_l1[key] = state
            return state
        if level == 2:
            state = self._states_l2.get(key)
            if state is None:
                state = _EWMAGroupState(
                    window_days=self.window_days,
                    halflife_days=self.halflife_days,
                )
                self._states_l2[key] = state
            return state
        raise ValueError(f"Unsupported level={level}")

    def score_rows(self, df_issue_time: pd.DataFrame) -> pd.DataFrame:
        if df_issue_time.empty:
            return pd.DataFrame(
                columns=["bias_ewma", "bias_level", "history_count"],
                index=df_issue_time.index,
            )

        issue_time = pd.to_datetime(df_issue_time["issue_time_utc"], utc=True, errors="coerce")
        if issue_time.isna().any():
            raise ValueError("issue_time_utc contains nulls while scoring Stage A")

        biases = np.zeros(len(df_issue_time), dtype=float)
        levels = np.full(len(df_issue_time), -1, dtype=int)
        counts = np.zeros(len(df_issue_time), dtype=int)

        for i, row in enumerate(df_issue_time.itertuples(index=False)):
            current_time_s = float(pd.Timestamp(row.issue_time_utc).value) / 1e9
            k0 = self._key_l0(row)
            k1 = self._key_l1(row)
            k2 = self._key_l2(row)

            chosen_bias = 0.0
            chosen_level = -1
            primary_count = 0

            for level, key in ((0, k0), (1, k1), (2, k2)):
                state = self._get_state(level, key)
                if state is None:
                    continue
                bias, n_hist = state.bias_and_count(current_time_s)
                if level == 0:
                    primary_count = n_hist
                if n_hist >= self.min_history:
                    chosen_bias = bias
                    chosen_level = level
                    break

            if chosen_level == -1:
                self.insufficient_primary[k0] += 1

            self.fallback_usage[chosen_level] += 1
            biases[i] = chosen_bias
            levels[i] = chosen_level
            counts[i] = primary_count

        return pd.DataFrame(
            {
                "bias_ewma": biases,
                "bias_level": levels,
                "history_count": counts,
            },
            index=df_issue_time.index,
        )

    def update_with_observed_residuals(
        self,
        df_issue_time: pd.DataFrame,
        *,
        residual_col: str,
    ) -> int:
        if df_issue_time.empty:
            return 0

        updated = 0
        for row in df_issue_time.itertuples(index=False):
            residual = getattr(row, residual_col)
            if pd.isna(residual):
                continue
            current_time_s = float(pd.Timestamp(row.issue_time_utc).value) / 1e9

            k0 = self._key_l0(row)
            k1 = self._key_l1(row)
            k2 = self._key_l2(row)

            self._ensure_state(0, k0).add(issue_time_s=current_time_s, residual=float(residual))
            self._ensure_state(1, k1).add(issue_time_s=current_time_s, residual=float(residual))
            self._ensure_state(2, k2).add(issue_time_s=current_time_s, residual=float(residual))
            updated += 1

        return updated

    def summarize(self) -> StageAStats:
        return StageAStats(
            fallback_usage=self.fallback_usage.copy(),
            insufficient_primary=self.insufficient_primary.copy(),
        )
