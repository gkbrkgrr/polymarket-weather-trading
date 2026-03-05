from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .utils_time import utc_now


STATE_FILE_NAME = "live_state.json"


@dataclass
class PositionRecord:
    position_id: str
    opened_ts_utc: str
    station: str
    market_day_local: str
    market_id: str
    slug: str | None
    asset_id: str | None
    strike_k: int
    mode_k: int
    p_model: float
    entry_price: float
    size: float
    stake_usd: float
    edge_at_entry: float
    price_source: str
    status: str = "open"


class PilotStateStore:
    def __init__(self, state_dir: Path, nav_usd: float) -> None:
        self.state_dir = state_dir
        self.state_path = state_dir / STATE_FILE_NAME
        self.nav_seed = float(nav_usd)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_init()

    def _default_state(self) -> dict[str, Any]:
        ts = utc_now().isoformat()
        return {
            "version": 1,
            "created_at_utc": ts,
            "updated_at_utc": ts,
            "nav_usd": self.nav_seed,
            "kill_switch": {
                "global": False,
                "station_paused": {},
            },
            "daily": {},
            "open_positions": [],
            "last_report_date_local": None,
            "stoploss": {
                "consecutive_days_hit": 0,
                "history": [],
            },
        }

    def _load_or_init(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._default_state()
            self._atomic_write(state)
            return state

        with self.state_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        if not isinstance(loaded, dict):
            loaded = self._default_state()

        loaded.setdefault("version", 1)
        loaded.setdefault("created_at_utc", utc_now().isoformat())
        loaded.setdefault("updated_at_utc", utc_now().isoformat())
        loaded.setdefault("nav_usd", self.nav_seed)
        loaded.setdefault("kill_switch", {"global": False, "station_paused": {}})
        loaded.setdefault("daily", {})
        loaded.setdefault("open_positions", [])
        loaded.setdefault("last_report_date_local", None)
        loaded.setdefault("stoploss", {"consecutive_days_hit": 0, "history": []})
        return loaded

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")
        tmp.replace(self.state_path)

    def persist(self) -> None:
        self.state["updated_at_utc"] = utc_now().isoformat()
        self._atomic_write(self.state)

    @property
    def nav_usd(self) -> float:
        return float(self.state.get("nav_usd", self.nav_seed))

    def set_nav_usd(self, nav: float) -> None:
        self.state["nav_usd"] = float(nav)

    def is_global_kill(self) -> bool:
        return bool(self.state.get("kill_switch", {}).get("global", False))

    def set_global_kill(self, value: bool) -> None:
        self.state.setdefault("kill_switch", {}).update({"global": bool(value)})

    def is_station_paused(self, station: str) -> bool:
        station_paused = self.state.get("kill_switch", {}).get("station_paused", {})
        if not isinstance(station_paused, dict):
            return False
        return bool(station_paused.get(station, False))

    def set_station_paused(self, station: str, paused: bool) -> None:
        ks = self.state.setdefault("kill_switch", {})
        station_paused = ks.setdefault("station_paused", {})
        station_paused[station] = bool(paused)

    def _day_bucket(self, day_local: date | str) -> dict[str, Any]:
        key = day_local if isinstance(day_local, str) else day_local.isoformat()
        daily = self.state.setdefault("daily", {})
        bucket = daily.setdefault(
            key,
            {
                "portfolio_risk_used": 0.0,
                "portfolio_pnl_realized": 0.0,
                "stations": {},
                "trade_count": 0,
                "skip_count": 0,
            },
        )
        stations = bucket.setdefault("stations", {})
        if not isinstance(stations, dict):
            bucket["stations"] = {}
        return bucket

    def _station_bucket(self, day_local: date | str, station: str) -> dict[str, Any]:
        day_bucket = self._day_bucket(day_local)
        stations = day_bucket.setdefault("stations", {})
        station_bucket = stations.setdefault(
            station,
            {
                "risk_used": 0.0,
                "pnl_realized": 0.0,
                "trades": 0,
                "skips": 0,
                "stoploss_triggered": False,
            },
        )
        return station_bucket

    def add_skip(self, *, day_local: date | str, station: str) -> None:
        day_bucket = self._day_bucket(day_local)
        day_bucket["skip_count"] = int(day_bucket.get("skip_count", 0)) + 1
        st = self._station_bucket(day_local, station)
        st["skips"] = int(st.get("skips", 0)) + 1

    def add_trade(self, *, day_local: date | str, station: str, risk_used: float) -> None:
        day_bucket = self._day_bucket(day_local)
        day_bucket["trade_count"] = int(day_bucket.get("trade_count", 0)) + 1
        day_bucket["portfolio_risk_used"] = float(day_bucket.get("portfolio_risk_used", 0.0)) + float(risk_used)

        st = self._station_bucket(day_local, station)
        st["trades"] = int(st.get("trades", 0)) + 1
        st["risk_used"] = float(st.get("risk_used", 0.0)) + float(risk_used)

    def add_realized_pnl(self, *, day_local: date | str, station: str, pnl: float) -> None:
        day_bucket = self._day_bucket(day_local)
        day_bucket["portfolio_pnl_realized"] = float(day_bucket.get("portfolio_pnl_realized", 0.0)) + float(pnl)

        st = self._station_bucket(day_local, station)
        st["pnl_realized"] = float(st.get("pnl_realized", 0.0)) + float(pnl)

    def station_risk_used(self, *, day_local: date | str, station: str) -> float:
        return float(self._station_bucket(day_local, station).get("risk_used", 0.0))

    def portfolio_risk_used(self, *, day_local: date | str) -> float:
        return float(self._day_bucket(day_local).get("portfolio_risk_used", 0.0))

    def station_trade_count(self, *, station: str) -> int:
        return sum(
            int(day_info.get("stations", {}).get(station, {}).get("trades", 0))
            for day_info in self.state.get("daily", {}).values()
            if isinstance(day_info, dict)
        )

    def open_positions(self) -> list[dict[str, Any]]:
        rows = self.state.get("open_positions", [])
        return rows if isinstance(rows, list) else []

    def open_positions_for_station(self, station: str) -> list[dict[str, Any]]:
        return [p for p in self.open_positions() if str(p.get("station")) == station and str(p.get("status", "open")) == "open"]

    def add_open_position(self, record: dict[str, Any]) -> str:
        rid = str(record.get("position_id") or uuid.uuid4().hex)
        payload = dict(record)
        payload["position_id"] = rid
        payload.setdefault("opened_ts_utc", utc_now().isoformat())
        payload.setdefault("status", "open")
        self.open_positions().append(payload)
        return rid

    def close_position(self, position_id: str, *, close_ts_utc: datetime, pnl: float, resolution: str) -> dict[str, Any] | None:
        for pos in self.open_positions():
            if str(pos.get("position_id")) != str(position_id):
                continue
            if str(pos.get("status", "open")) != "open":
                return None
            pos["status"] = "closed"
            pos["closed_ts_utc"] = close_ts_utc.isoformat()
            pos["pnl_realized"] = float(pnl)
            pos["resolution"] = str(resolution)
            return pos
        return None

    def cleanup_closed_positions(self) -> None:
        open_only = [p for p in self.open_positions() if str(p.get("status", "open")) == "open"]
        self.state["open_positions"] = open_only

    def daily_realized_pnl(self, day_local: date | str) -> float:
        return float(self._day_bucket(day_local).get("portfolio_pnl_realized", 0.0))

    def station_daily_realized_pnl(self, day_local: date | str, station: str) -> float:
        return float(self._station_bucket(day_local, station).get("pnl_realized", 0.0))

    def mark_station_stoploss(self, *, day_local: date | str, station: str, triggered: bool) -> None:
        st = self._station_bucket(day_local, station)
        st["stoploss_triggered"] = bool(triggered)

    def update_stoploss_streak(self, *, day_local: date | str, stoploss_hit: bool) -> int:
        stoploss = self.state.setdefault("stoploss", {"consecutive_days_hit": 0, "history": []})
        streak = int(stoploss.get("consecutive_days_hit", 0))
        if stoploss_hit:
            streak += 1
        else:
            streak = 0
        stoploss["consecutive_days_hit"] = streak
        history = stoploss.setdefault("history", [])
        history.append({
            "day_local": day_local if isinstance(day_local, str) else day_local.isoformat(),
            "stoploss_hit": bool(stoploss_hit),
            "updated_at_utc": utc_now().isoformat(),
        })
        if len(history) > 90:
            del history[:-90]
        return streak

    def stoploss_streak(self) -> int:
        return int(self.state.get("stoploss", {}).get("consecutive_days_hit", 0))

    def last_report_date_local(self) -> str | None:
        value = self.state.get("last_report_date_local")
        return str(value) if value else None

    def set_last_report_date_local(self, day_local: date | str) -> None:
        self.state["last_report_date_local"] = day_local if isinstance(day_local, str) else day_local.isoformat()
