#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "live_trading" / "config.live_pilot.yaml"

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_trading.state import PilotStateStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and operate live pilot kill switch state."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Live pilot YAML config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Optional explicit state directory override. Defaults to <output_dir>/state from config.",
    )
    parser.add_argument(
        "--nav-seed",
        type=float,
        default=10000.0,
        help="Seed NAV used only if state file does not exist yet.",
    )
    parser.add_argument(
        "--actor",
        type=str,
        default=getpass.getuser(),
        help="Operator identifier recorded in kill-switch event history.",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Print current kill-switch status and recent events.")

    p_enable = sub.add_parser("enable-global", help="Enable global kill switch.")
    p_enable.add_argument("--reason", type=str, default="", help="Reason for the change.")

    p_disable = sub.add_parser("disable-global", help="Disable global kill switch.")
    p_disable.add_argument("--reason", type=str, default="", help="Reason for the change.")

    p_pause = sub.add_parser("pause-station", help="Pause a station-level kill switch.")
    p_pause.add_argument("station", type=str, help="Station name.")
    p_pause.add_argument("--reason", type=str, default="", help="Reason for the change.")

    p_unpause = sub.add_parser("unpause-station", help="Unpause a station-level kill switch.")
    p_unpause.add_argument("station", type=str, help="Station name.")
    p_unpause.add_argument("--reason", type=str, default="", help="Reason for the change.")

    p_clear = sub.add_parser("clear-station-pauses", help="Unpause all stations.")
    p_clear.add_argument("--reason", type=str, default="", help="Reason for the change.")
    return parser.parse_args(argv)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return dict(loaded)


def _resolve_state_dir(args: argparse.Namespace) -> Path:
    if args.state_dir is not None:
        return _resolve_path(Path(args.state_dir), REPO_ROOT)
    cfg_path = _resolve_path(Path(args.config), REPO_ROOT)
    cfg = _load_config(cfg_path)
    output_dir = _resolve_path(Path(str(cfg.get("output_dir", "live_trading"))), REPO_ROOT)
    return (output_dir / "state").resolve()


def _append_event(
    *,
    state_store: PilotStateStore,
    event_type: str,
    actor: str,
    reason: str,
    station: str | None = None,
) -> None:
    kill_switch = state_store.state.setdefault("kill_switch", {})
    events = kill_switch.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        kill_switch["events"] = events
    rec = {
        "ts_utc": utc_now().isoformat(),
        "event": str(event_type),
        "actor": str(actor).strip() or "unknown",
        "reason": str(reason).strip(),
    }
    if station is not None:
        rec["station"] = str(station)
    events.append(rec)
    if len(events) > 500:
        del events[:-500]


def _status_payload(state_store: PilotStateStore) -> dict[str, Any]:
    ks = state_store.state.get("kill_switch", {})
    station_paused = ks.get("station_paused", {})
    paused = {}
    if isinstance(station_paused, dict):
        paused = {str(k): bool(v) for k, v in station_paused.items() if bool(v)}

    events = ks.get("events", [])
    recent_events = []
    if isinstance(events, list):
        recent_events = [e for e in events[-20:] if isinstance(e, dict)]

    return {
        "state_path": str(state_store.state_path),
        "updated_at_utc": state_store.state.get("updated_at_utc"),
        "global_kill": bool(ks.get("global", False)),
        "station_paused": paused,
        "recent_events": recent_events,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state_dir = _resolve_state_dir(args)
    state_store = PilotStateStore(state_dir=state_dir, nav_usd=float(args.nav_seed))

    cmd = str(args.command)
    if cmd == "status":
        print(json.dumps(_status_payload(state_store), ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    actor = str(args.actor).strip() or "unknown"
    reason = str(getattr(args, "reason", "")).strip()

    if cmd == "enable-global":
        state_store.set_global_kill(True)
        _append_event(state_store=state_store, event_type="enable_global", actor=actor, reason=reason)
        state_store.persist()
    elif cmd == "disable-global":
        state_store.set_global_kill(False)
        _append_event(state_store=state_store, event_type="disable_global", actor=actor, reason=reason)
        state_store.persist()
    elif cmd == "pause-station":
        station = str(args.station).strip()
        if not station:
            raise SystemExit("station must be non-empty")
        state_store.set_station_paused(station, True)
        _append_event(
            state_store=state_store,
            event_type="pause_station",
            actor=actor,
            reason=reason,
            station=station,
        )
        state_store.persist()
    elif cmd == "unpause-station":
        station = str(args.station).strip()
        if not station:
            raise SystemExit("station must be non-empty")
        state_store.set_station_paused(station, False)
        _append_event(
            state_store=state_store,
            event_type="unpause_station",
            actor=actor,
            reason=reason,
            station=station,
        )
        state_store.persist()
    elif cmd == "clear-station-pauses":
        station_paused = state_store.state.setdefault("kill_switch", {}).setdefault("station_paused", {})
        paused_stations = []
        if isinstance(station_paused, dict):
            paused_stations = [str(st) for st, paused in station_paused.items() if bool(paused)]
            for st in paused_stations:
                station_paused[st] = False
        _append_event(
            state_store=state_store,
            event_type="clear_station_pauses",
            actor=actor,
            reason=reason,
        )
        if paused_stations:
            _append_event(
                state_store=state_store,
                event_type="clear_station_pauses_detail",
                actor=actor,
                reason=f"stations={','.join(paused_stations)}",
            )
        state_store.persist()
    else:
        raise SystemExit(f"Unknown command: {cmd}")

    print(json.dumps(_status_payload(state_store), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
