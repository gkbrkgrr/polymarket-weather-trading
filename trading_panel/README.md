# Trading Panel

Local web panel for tracking station-level ML forecast progressions and Wunderground observations.

## What it shows

- Main panel title: selected date (`YYYY-MM-DD`)
- One facet per station
- Interactive time-series progression for each model:
  - `Country Based` (`data/ml_predictions/city_extended`)
  - `GlobalV1` (`data/ml_predictions/xgb_opt_v1_100`)
  - `GlobalV2` (`data/ml_predictions/xgb_opt_v2_100`)
- Facet subtitles below station title:
  - Left: `Current Local Time: HH:MM:SS` (`23:59:59` when selected day is already over for that station)
  - Right: `Last Observation: <rounded_value> at <HH:MM:SS>` from `master_db.station_observations`

## Run

From repository root:

```bash
conda run -n mto python -m trading_panel.app
```

App URL:

- `http://127.0.0.1:8787/`

## Environment variables

- `TRADING_PANEL_PORT` (default `8787`)
- `TRADING_PANEL_HOST` (default `127.0.0.1`)
- `TRADING_PANEL_CACHE_TTL_SECONDS` (default `120`)
- `TRADING_PANEL_HISTORY_DAYS` (default `16`)
- `TRADING_PANEL_COUNTRY_ROOT`
- `TRADING_PANEL_GLOBAL_V1_ROOT`
- `TRADING_PANEL_GLOBAL_V2_ROOT`
- `TRADING_PANEL_LOCATIONS_CSV`
- `MASTER_POSTGRES_DSN` (optional override)

`MASTER_POSTGRES_DSN` resolution otherwise follows repository config via `config.master_db.yaml`.
