from __future__ import annotations

import fcntl
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .utils_time import normalize_station_key, parse_local_day, utc_now


STATE_FILE_NAME = "live_state.json"
STATE_LOCK_FILE_NAME = "live_state.lock"


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
        self.lock_path = state_dir / STATE_LOCK_FILE_NAME
        self.nav_seed = float(nav_usd)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = self._acquire_lock()
        try:
            self.state = self._load_or_init()
        except Exception:
            self.close()
            raise

    def _acquire_lock(self) -> Any:
        lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            lock_file.close()
            raise RuntimeError(f"State lock is already held by another process: {self.lock_path}")
        return lock_file

    def _assert_lock_held(self) -> None:
        if getattr(self, "_lock_file", None) is None:
            raise RuntimeError("State lock is not initialized.")

    def close(self) -> None:
        lock_file = getattr(self, "_lock_file", None)
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            lock_file.close()
        except Exception:
            pass
        self._lock_file = None

    def __del__(self) -> None:
        self.close()

    def _default_state(self) -> dict[str, Any]:
        ts = utc_now().isoformat()
        return {
            "version": 1,
            "created_at_utc": ts,
            "updated_at_utc": ts,
            "nav_usd": self.nav_seed,
            "nav_peak_usd": self.nav_seed,
            "kill_switch": {
                "global": False,
                "station_paused": {},
            },
            "daily": {},
            "open_positions": [],
            "recent_order_keys": [],
            "trade_cooldowns": {},
            "last_report_date_local": None,
            "stoploss": {
                "consecutive_days_hit": 0,
                "history": [],
            },
        }

    def _load_or_init(self) -> dict[str, Any]:
        self._assert_lock_held()
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
        loaded.setdefault("nav_peak_usd", max(float(loaded.get("nav_usd", self.nav_seed)), self.nav_seed))
        loaded["nav_peak_usd"] = max(float(loaded.get("nav_peak_usd", self.nav_seed)), float(loaded.get("nav_usd", self.nav_seed)))
        loaded.setdefault("kill_switch", {"global": False, "station_paused": {}})
        loaded.setdefault("daily", {})
        loaded.setdefault("open_positions", [])
        loaded.setdefault("recent_order_keys", [])
        loaded.setdefault("trade_cooldowns", {})
        loaded.setdefault("last_report_date_local", None)
        loaded.setdefault("stoploss", {"consecutive_days_hit": 0, "history": []})
        return loaded

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self._assert_lock_held()
        tmp = self.state_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")
        tmp.replace(self.state_path)

    def persist(self) -> None:
        self._assert_lock_held()
        self.state["updated_at_utc"] = utc_now().isoformat()
        self._atomic_write(self.state)

    @property
    def nav_usd(self) -> float:
        return float(self.state.get("nav_usd", self.nav_seed))

    @property
    def nav_peak_usd(self) -> float:
        return max(float(self.state.get("nav_peak_usd", self.nav_seed)), self.nav_usd)

    def set_nav_usd(self, nav: float) -> None:
        nav_val = float(nav)
        self.state["nav_usd"] = nav_val
        self.state["nav_peak_usd"] = max(float(self.state.get("nav_peak_usd", self.nav_seed)), nav_val)

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

    def trade_cooldowns(self) -> dict[str, str]:
        rows = self.state.get("trade_cooldowns", {})
        if not isinstance(rows, dict):
            rows = {}
            self.state["trade_cooldowns"] = rows
        return rows

    def recent_order_keys(self) -> list[dict[str, Any]]:
        rows = self.state.get("recent_order_keys", [])
        if not isinstance(rows, list):
            rows = []
            self.state["recent_order_keys"] = rows
        return rows

    def _parse_timestamp_utc(self, value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            ts = datetime.fromisoformat(str(value))
        except Exception:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def prune_recent_order_keys(self, *, retention_hours: float = 24.0, now_utc: datetime | None = None) -> None:
        now_ref = (now_utc or utc_now()).astimezone(timezone.utc)
        ttl = timedelta(hours=max(0.0, float(retention_hours)))
        cutoff = now_ref - ttl

        latest_by_key: dict[str, datetime] = {}
        for row in self.recent_order_keys():
            if not isinstance(row, dict):
                continue
            order_key = str(row.get("order_key") or "").strip()
            if not order_key:
                continue
            created_ts = self._parse_timestamp_utc(row.get("created_at_utc"))
            if created_ts is None or created_ts < cutoff:
                continue
            prev = latest_by_key.get(order_key)
            if prev is None or created_ts > prev:
                latest_by_key[order_key] = created_ts

        cleaned = [
            {"order_key": key, "created_at_utc": ts.isoformat()}
            for key, ts in sorted(latest_by_key.items(), key=lambda item: item[1], reverse=True)
        ]
        self.state["recent_order_keys"] = cleaned

    def recent_order_key_set(self, *, retention_hours: float = 24.0, now_utc: datetime | None = None) -> set[str]:
        self.prune_recent_order_keys(retention_hours=retention_hours, now_utc=now_utc)
        return {str(row.get("order_key")) for row in self.recent_order_keys() if isinstance(row, dict) and row.get("order_key")}

    def has_recent_order_key(self, order_key: str, *, retention_hours: float = 24.0, now_utc: datetime | None = None) -> bool:
        key = str(order_key or "").strip()
        if not key:
            return False
        return key in self.recent_order_key_set(retention_hours=retention_hours, now_utc=now_utc)

    def record_recent_order_key(self, order_key: str, *, retention_hours: float = 24.0, now_utc: datetime | None = None) -> None:
        key = str(order_key or "").strip()
        if not key:
            return
        created_ts = (now_utc or utc_now()).astimezone(timezone.utc)
        self.prune_recent_order_keys(retention_hours=retention_hours, now_utc=created_ts)
        self.recent_order_keys().append({"order_key": key, "created_at_utc": created_ts.isoformat()})
        self.prune_recent_order_keys(retention_hours=retention_hours, now_utc=created_ts)

    def prune_trade_cooldowns(self, *, cooldown_minutes: float, now_utc: datetime | None = None) -> None:
        now_ref = (now_utc or utc_now()).astimezone(timezone.utc)
        retention_minutes = max(60.0, float(cooldown_minutes) * 4.0)
        cutoff = now_ref - timedelta(minutes=retention_minutes)
        cooldowns = self.trade_cooldowns()
        remove_keys: list[str] = []
        for key, ts_raw in cooldowns.items():
            ts = self._parse_timestamp_utc(ts_raw)
            if ts is None or ts < cutoff:
                remove_keys.append(str(key))
        for key in remove_keys:
            cooldowns.pop(key, None)

    def active_trade_cooldown_keys(self, *, cooldown_minutes: float, now_utc: datetime | None = None) -> set[str]:
        self.prune_trade_cooldowns(cooldown_minutes=cooldown_minutes, now_utc=now_utc)
        if float(cooldown_minutes) <= 0:
            return set()
        now_ref = (now_utc or utc_now()).astimezone(timezone.utc)
        cooldown_delta = timedelta(minutes=float(cooldown_minutes))
        out: set[str] = set()
        for key, ts_raw in self.trade_cooldowns().items():
            ts = self._parse_timestamp_utc(ts_raw)
            if ts is None:
                continue
            if now_ref - ts < cooldown_delta:
                out.add(str(key))
        return out

    def is_trade_cooldown_active(
        self,
        identity_key: str,
        *,
        cooldown_minutes: float,
        now_utc: datetime | None = None,
    ) -> bool:
        key = str(identity_key or "").strip()
        if not key:
            return False
        return key in self.active_trade_cooldown_keys(cooldown_minutes=cooldown_minutes, now_utc=now_utc)

    def record_trade_cooldown(
        self,
        identity_key: str,
        *,
        cooldown_minutes: float,
        now_utc: datetime | None = None,
    ) -> None:
        key = str(identity_key or "").strip()
        if not key:
            return
        now_ref = (now_utc or utc_now()).astimezone(timezone.utc)
        self.prune_trade_cooldowns(cooldown_minutes=cooldown_minutes, now_utc=now_ref)
        self.trade_cooldowns()[key] = now_ref.isoformat()

    def position_identity_key(self, *, station: Any, market_day_local: Any, strike_k: Any) -> str | None:
        station_key = normalize_station_key(str(station or ""))
        day_local = parse_local_day(market_day_local)
        try:
            strike = int(strike_k)
        except Exception:
            strike = None

        if not station_key or day_local is None or strike is None:
            return None
        return f"{station_key}|{day_local.isoformat()}|{strike}"

    def open_position_identity_keys(self) -> set[str]:
        keys: set[str] = set()
        for pos in self.open_positions():
            if str(pos.get("status", "open")) != "open":
                continue
            key = self.position_identity_key(
                station=pos.get("station"),
                market_day_local=pos.get("market_day_local"),
                strike_k=pos.get("strike_k"),
            )
            if key:
                keys.add(key)
        return keys

    def has_open_position_identity(self, *, station: Any, market_day_local: Any, strike_k: Any) -> bool:
        key = self.position_identity_key(station=station, market_day_local=market_day_local, strike_k=strike_k)
        if key is None:
            return False
        return key in self.open_position_identity_keys()

    def _to_day_key(self, value: date | str | Any) -> str | None:
        parsed = parse_local_day(value)
        if parsed is None:
            return None
        return parsed.isoformat()

    def _position_stake(self, pos: dict[str, Any]) -> float:
        try:
            return max(0.0, float(pos.get("stake_usd") or 0.0))
        except Exception:
            return 0.0

    def open_positions_for_station(self, station: str) -> list[dict[str, Any]]:
        return [p for p in self.open_positions() if str(p.get("station")) == station and str(p.get("status", "open")) == "open"]

    def open_position_count_for_station(self, station: str) -> int:
        return len(self.open_positions_for_station(station))

    def open_position_count(self) -> int:
        return sum(1 for p in self.open_positions() if str(p.get("status", "open")) == "open")

    def station_open_risk(self, *, day_local: date | str, station: str) -> float:
        day_key = self._to_day_key(day_local)
        if day_key is None:
            return 0.0
        total = 0.0
        for pos in self.open_positions_for_station(station):
            pos_day = self._to_day_key(pos.get("market_day_local"))
            if pos_day != day_key:
                continue
            total += self._position_stake(pos)
        return total

    def portfolio_open_risk(self, *, day_local: date | str) -> float:
        day_key = self._to_day_key(day_local)
        if day_key is None:
            return 0.0
        total = 0.0
        for pos in self.open_positions():
            if str(pos.get("status", "open")) != "open":
                continue
            pos_day = self._to_day_key(pos.get("market_day_local"))
            if pos_day != day_key:
                continue
            total += self._position_stake(pos)
        return total

    def station_conservative_risk_used(self, *, day_local: date | str, station: str) -> float:
        return max(
            self.station_risk_used(day_local=day_local, station=station),
            self.station_open_risk(day_local=day_local, station=station),
        )

    def portfolio_conservative_risk_used(self, *, day_local: date | str) -> float:
        return max(
            self.portfolio_risk_used(day_local=day_local),
            self.portfolio_open_risk(day_local=day_local),
        )

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
