## Observation scraper (Wunderground / weather.com)

This repo's `locations.csv` contains Wunderground daily-history URLs for each station. The scraper uses those URLs to:

- discover the `apiKey` embedded in the page HTML
- fetch historical observations from `api.weather.com`
- store per-station archives under `data/observations/` (one file per station)

Defaults are computed relative to the repo root, so you can run the script from any working directory.

### Output format

- Default: `jsonl` (no extra dependencies, works in constrained environments).
- Optional: `parquet` (requires a working `pyarrow` installation).

### Parquet dependency

```bash
python -m pip install "pyarrow>=14"
```

### Backfill an archive

```bash
python data_gatherer/observations_scraper.py --start-date 2026-01-01
```

To force parquet output:

```bash
python data_gatherer/observations_scraper.py --start-date 2026-01-01 --format parquet
```

### Incremental updates (run every 30 minutes)

The scraper is idempotent: if a station archive already exists (`.jsonl` or `.parquet`), it automatically re-scrapes from the last stored timestamp (with a 1-day safety window) and appends any new observations.

```bash
python data_gatherer/observations_scraper.py
```

To run continuously:

```bash
python data_gatherer/observations_scraper.py --loop-minutes 30
```

If you see occasional network timeouts, increase retries/backoff:

```bash
python data_gatherer/observations_scraper.py --retries 5 --retry-backoff-s 2
```

### Notes

- Temperature is stored for all stations.
- Precipitation columns are only populated for the `NYC` station (others are null).
- Observation spacing depends on the upstream station cadence (best-effort toward 30-minute intervals).
