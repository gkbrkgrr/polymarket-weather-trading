# Polymarket Weather Data Archiver

Stores Polymarket market metadata + price snapshots for a set of weather-related contracts in a local SQLite database.

## Quickstart

1. Create a config:
   - Copy `config.example.json` to `config.json` and edit the `targets` list if needed.
2. Initialize the database:
   - `python polymarket_archive.py --config config.json init-db`
3. Discover relevant markets (fills `targets`/`target_markets`):
   - `python polymarket_archive.py --config config.json discover --print`
   - If daily markets (e.g. `... on January 6`) are missing, increase `discovery.max_pages` in `config.json` (they can be many pages deep).
   - If this is too slow, reduce `discovery.max_pages`.
4. Write one snapshot:
   - `python polymarket_archive.py --config config.json snapshot`
5. Run continuously:
   - `python polymarket_archive.py --config config.json run --interval-s 60 --discover-every-s 3600`
   - This periodically re-runs discovery, so it automatically picks up new daily markets like `Highest temperature in Seoul January 7`.
   - Or start it in the background on Windows: `powershell -ExecutionPolicy Bypass -File .\run_archiver.ps1` (stop with `powershell -ExecutionPolicy Bypass -File .\stop_archiver.ps1`)
   - On Linux/VPS: `bash run_archiver.sh`

## Tests

- `python -m unittest discover -s tests -p "test_*.py"`

## VPS Notes

- Install as a CLI (optional): `pip install .` then run `polymarket-archiver --config /path/to/config.json run ...`
- Environment overrides:
  - `POLYMARKET_ARCHIVE_DB_PATH` (SQLite file path)
  - `POLYMARKET_GAMMA_BASE_URL`
  - `POLYMARKET_DISCOVERY_MAX_PAGES`
  - `POLYMARKET_DISCOVERY_STOP_AFTER_MATCHES_PER_TARGET`
- Systemd template: `deploy/polymarket-archiver.service`
- Step-by-step Linux guide: `VPS_SETUP.md`

## Database

The schema lives in `schema.sql`. The main tables are:

- `targets`: your contract patterns.
- `markets`: market metadata from Gamma.
- `target_markets`: mapping from targets → discovered markets.
- `market_snapshots`: per-market snapshots (raw JSON included).
- `outcome_prices`: per-outcome price snapshots.

Notes:
- `outcome_prices.price` comes from Gamma `outcomePrices` (a “mark”/probability-style price).
- The Polymarket UI “Yes” quote often matches `market_snapshots.best_ask` (and “Sell Yes” matches `market_snapshots.best_bid`).
