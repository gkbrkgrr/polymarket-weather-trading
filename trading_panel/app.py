from __future__ import annotations

import datetime as dt

from flask import Flask, abort, jsonify, render_template, request

from .config import (
    CACHE_TTL_SECONDS,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LOCATIONS_CSV,
    MODEL_SPECS,
    resolve_panel_master_dsn,
)
from .data_service import PanelDataService

app = Flask(__name__, template_folder="templates", static_folder="static")

service = PanelDataService(
    model_specs=MODEL_SPECS,
    locations_csv=LOCATIONS_CSV,
    master_dsn=resolve_panel_master_dsn(),
    cache_ttl_seconds=CACHE_TTL_SECONDS,
    default_history_days=DEFAULT_HISTORY_DAYS,
)


@app.get("/")
def index() -> str:
    default_date = dt.date.today().isoformat()
    return render_template("index.html", default_date=default_date)


@app.get("/api/panel-data")
def panel_data():
    requested = request.args.get("date")
    day = _parse_day(requested)
    payload = service.get_panel_payload(day)
    return jsonify(payload)


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


def _parse_day(raw: str | None) -> dt.date:
    if raw is None or not raw.strip():
        return dt.date.today()

    try:
        return dt.date.fromisoformat(raw)
    except ValueError:
        abort(400, description=f"Invalid date {raw!r}. Use YYYY-MM-DD.")


def main() -> None:
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=False)


if __name__ == "__main__":
    main()
