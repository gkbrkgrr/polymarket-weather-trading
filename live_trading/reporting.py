from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .state import PilotStateStore
from .utils_time import to_yyyymmdd


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_daily_actions(*, logs_dir: Path, day_local: date) -> list[dict[str, Any]]:
    ymd = to_yyyymmdd(day_local)
    preferred = logs_dir / f"trades_{ymd}.jsonl"
    rows = _read_jsonl(preferred)
    if rows:
        return [r for r in rows if str(r.get("market_day_local", "")) == day_local.isoformat()]

    all_rows: list[dict[str, Any]] = []
    for path in sorted(logs_dir.glob("trades_*.jsonl")):
        all_rows.extend(_read_jsonl(path))
    return [r for r in all_rows if str(r.get("market_day_local", "")) == day_local.isoformat()]


def _build_station_breakdown(
    *,
    stations: list[str],
    actions: list[dict[str, Any]],
    state_store: PilotStateStore,
    day_local: date,
) -> list[dict[str, Any]]:
    day_key = day_local.isoformat()
    out: list[dict[str, Any]] = []

    for station in stations:
        station_actions = [a for a in actions if str(a.get("station")) == station]
        station_trades = [a for a in station_actions if str(a.get("decision")) == "TRADE"]
        station_skips = [a for a in station_actions if str(a.get("decision")) == "SKIP"]

        skip_reasons = Counter(str(a.get("skipped_reason")) for a in station_skips if str(a.get("skipped_reason")))
        top_edges = sorted(
            (
                {
                    "slug": t.get("slug"),
                    "edge": _safe_float(t.get("edge")),
                    "price": _safe_float(t.get("chosen_no_ask")),
                    "size": _safe_float(t.get("size")),
                }
                for t in station_trades
            ),
            key=lambda x: (x["edge"] if x["edge"] is not None else -999.0),
            reverse=True,
        )[:3]

        out.append(
            {
                "station": station,
                "trades": int(len(station_trades)),
                "skips": int(len(station_skips)),
                "pnl": float(state_store.station_daily_realized_pnl(day_local=day_key, station=station)),
                "exposure": float(sum(_safe_float(t.get("stake_usd")) or 0.0 for t in station_trades)),
                "risk_used": float(state_store.station_risk_used(day_local=day_key, station=station)),
                "top_edges": top_edges,
                "main_skip_reasons": skip_reasons.most_common(5),
            }
        )

    return out


def _build_open_positions_section(
    *,
    state_store: PilotStateStore,
    now_utc: datetime,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pos in state_store.open_positions():
        if str(pos.get("status", "open")) != "open":
            continue
        opened_raw = pos.get("opened_ts_utc")
        opened = pd.to_datetime(opened_raw, utc=True, errors="coerce")
        age_hours = None
        if not pd.isna(opened):
            age_hours = max(0.0, (now_utc - opened.to_pydatetime()).total_seconds() / 3600.0)

        out.append(
            {
                "station": pos.get("station"),
                "slug": pos.get("slug"),
                "strike": pos.get("strike_k"),
                "entry_price": _safe_float(pos.get("entry_price")),
                "size": _safe_float(pos.get("size")),
                "current_mid": _safe_float(pos.get("current_mid")),
                "edge_at_entry": _safe_float(pos.get("edge_at_entry")),
                "age_hours": age_hours,
            }
        )
    return out


def _compute_unrealized_pnl(open_positions: list[dict[str, Any]]) -> float | None:
    total = 0.0
    has_any = False
    for pos in open_positions:
        entry = _safe_float(pos.get("entry_price"))
        current_mid = _safe_float(pos.get("current_mid"))
        size = _safe_float(pos.get("size"))
        if entry is None or current_mid is None or size is None:
            continue
        # NO-position mark-to-market proxy: value ~= (1 - current_mid)
        mtm_entry = (1.0 - entry) * size
        mtm_now = (1.0 - current_mid) * size
        total += mtm_now - mtm_entry
        has_any = True
    return total if has_any else None


def generate_daily_report(
    *,
    output_dir: Path,
    logs_dir: Path,
    state_store: PilotStateStore,
    day_local: date,
    stations: list[str],
    nav_seed: float,
    now_utc: datetime,
) -> dict[str, Any]:
    reports_daily_dir = output_dir / "reports" / "daily"
    reports_daily_dir.mkdir(parents=True, exist_ok=True)

    actions = load_daily_actions(logs_dir=logs_dir, day_local=day_local)
    trades = [a for a in actions if str(a.get("decision")) == "TRADE"]
    skips = [a for a in actions if str(a.get("decision")) == "SKIP"]

    day_key = day_local.isoformat()
    pnl_realized = float(state_store.daily_realized_pnl(day_local=day_key))
    nav_end = float(state_store.nav_usd)
    nav_start = float(nav_end - pnl_realized)

    open_positions = _build_open_positions_section(state_store=state_store, now_utc=now_utc)
    pnl_unrealized = _compute_unrealized_pnl(open_positions)
    station_breakdown = _build_station_breakdown(
        stations=stations,
        actions=actions,
        state_store=state_store,
        day_local=day_local,
    )

    resolve_actions = [a for a in actions if str(a.get("decision")) == "RESOLVE"]
    win_rate = None
    if resolve_actions:
        wins = sum(1 for a in resolve_actions if (_safe_float(a.get("pnl_realized")) or 0.0) > 0.0)
        win_rate = wins / float(len(resolve_actions))

    skip_counts = Counter(str(a.get("skipped_reason")) for a in skips if str(a.get("skipped_reason")))
    alerts: list[str] = []
    if skip_counts.get("no_snapshot", 0) > 0:
        alerts.append("missing snapshots detected")
    if skip_counts.get("spread_too_wide", 0) > 0:
        alerts.append("spread anomalies detected")
    if state_store.is_global_kill():
        alerts.append("global kill switch active")

    summary: dict[str, Any] = {
        "date_local": day_key,
        "NAV_start": nav_start,
        "NAV_end": nav_end,
        "total_trades": int(len(trades)),
        "total_skips": int(len(skips)),
        "pnl_realized": pnl_realized,
        "station_breakdown": station_breakdown,
        "open_positions": open_positions,
        "risk_flags": {
            "station_paused": [
                s
                for s in stations
                if state_store.is_station_paused(s)
            ],
            "global_kill_switch": state_store.is_global_kill(),
        },
    }

    if pnl_unrealized is not None:
        summary["pnl_unrealized"] = float(pnl_unrealized)
    if win_rate is not None:
        summary["win_rate"] = float(win_rate)

    ymd = to_yyyymmdd(day_local)
    json_path = reports_daily_dir / f"{ymd}_summary.json"
    csv_path = reports_daily_dir / f"{ymd}_summary.csv"
    telegram_path = reports_daily_dir / f"{ymd}_telegram.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")

    station_df = pd.DataFrame(
        [
            {
                "date_local": day_key,
                "station": row["station"],
                "trades": row["trades"],
                "skips": row["skips"],
                "pnl": row["pnl"],
                "exposure": row["exposure"],
                "risk_used": row["risk_used"],
                "paused": row["station"] in summary["risk_flags"]["station_paused"],
                "top_edges": json.dumps(row["top_edges"], ensure_ascii=True),
                "main_skip_reasons": json.dumps(row["main_skip_reasons"], ensure_ascii=True),
            }
            for row in station_breakdown
        ]
    )
    station_df.to_csv(csv_path, index=False)

    top_trades = sorted(
        (
            {
                "station": t.get("station"),
                "slug": t.get("slug"),
                "edge": _safe_float(t.get("edge")) or 0.0,
                "price": _safe_float(t.get("chosen_no_ask")) or 0.0,
                "size": _safe_float(t.get("size")) or 0.0,
            }
            for t in trades
        ),
        key=lambda x: x["edge"],
        reverse=True,
    )[:5]

    lines: list[str] = []
    lines.append(f"Live Pilot {day_key} | NAV {nav_end:.2f} | PnL {pnl_realized:+.2f}")
    lines.append("")
    lines.append("Stations:")
    for row in station_breakdown:
        paused = "yes" if row["station"] in summary["risk_flags"]["station_paused"] else "no"
        lines.append(
            f"- {row['station']}: trades={row['trades']} pnl={row['pnl']:+.2f} exposure={row['exposure']:.2f} paused={paused}"
        )

    lines.append("")
    lines.append("Top 5 trades by edge:")
    if top_trades:
        for t in top_trades:
            lines.append(
                f"- {t['station']} | edge={t['edge']:.4f} | price={t['price']:.3f} | size={t['size']:.2f} | {t['slug']}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Top 5 skip reasons:")
    for reason, count in skip_counts.most_common(5):
        lines.append(f"- {reason}: {count}")
    if not skip_counts:
        lines.append("- none")

    lines.append("")
    lines.append("Alerts:")
    if alerts:
        for alert in alerts:
            lines.append(f"- {alert}")
    else:
        lines.append("- none")

    telegram_text = "\n".join(lines) + "\n"
    telegram_path.write_text(telegram_text, encoding="utf-8")

    return {
        "summary": summary,
        "json_path": json_path,
        "csv_path": csv_path,
        "telegram_path": telegram_path,
    }
