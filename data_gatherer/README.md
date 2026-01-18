# Polymarket Weather Data Archiver

Stores Polymarket market metadata + price snapshots for a set of weather-related contracts in a local parquet dataset (partitioned by day).

## Quickstart

1. Create a config:
   - Copy `config.example.json` to `config.json` and edit the `targets` list if needed.
2. Initialize the archive directory:
   - `python3 polymarket_archive.py --config config.json init-db`
3. Discover relevant markets (fills `target_markets` + updates `markets`/`market_outcomes`):
   - `python3 polymarket_archive.py --config config.json discover --print`
   - If daily markets (e.g. `... on January 6`) are missing, increase `discovery.max_pages` in `config.json` (they can be many pages deep).
   - If this is too slow, reduce `discovery.max_pages`.
4. Write one snapshot:
   - `python3 polymarket_archive.py --config config.json snapshot`
5. Run continuously:
   - `python3 polymarket_archive.py --config config.json run --interval-s 60 --discover-every-s 3600`
   - This periodically re-runs discovery, so it automatically picks up new daily markets like `Highest temperature in Seoul January 7`.
   - Or start it in the background on Windows: `powershell -ExecutionPolicy Bypass -File .\run_archiver.ps1` (stop with `powershell -ExecutionPolicy Bypass -File .\stop_archiver.ps1`)
   - On Linux/VPS: `bash run_archiver.sh`

## Tests

- `python3 -m unittest discover -s tests -p "test_*.py"`

## VPS Notes

- Install as a CLI (optional): `pip install .` then run `polymarket-archiver --config /path/to/config.json run ...`
- Environment overrides:
  - `POLYMARKET_ARCHIVE_DIR` (archive directory path; `POLYMARKET_ARCHIVE_DB_PATH` is accepted as a legacy alias)
  - `POLYMARKET_GAMMA_BASE_URL`
  - `POLYMARKET_DISCOVERY_MAX_PAGES`
  - `POLYMARKET_DISCOVERY_STOP_AFTER_MATCHES_PER_TARGET`
- Systemd template: `deploy/polymarket-archiver.service`
- Step-by-step Linux guide: `VPS_SETUP.md`

## Archive Layout

The archive directory contains:

- `targets.parquet`: your contract patterns.
- `markets.parquet`: market metadata from Gamma.
- `market_outcomes.parquet`: per-market outcome labels/token ids.
- `target_markets.parquet`: mapping from target name → discovered markets.
- `market_snapshots/`: partitioned parquet dataset (`snapshot_date=YYYY-MM-DD`) with per-market snapshots (raw JSON included).
- `outcome_prices/`: partitioned parquet dataset (`snapshot_date=YYYY-MM-DD`) with per-outcome prices.
- `market_trades/`: partitioned parquet dataset (`trade_date=YYYY-MM-DD`) with matched trade prints from Polymarket’s public data API.
- `token_price_history/`: partitioned parquet dataset (`price_date=YYYY-MM-DD`) with historical per-token prices from CLOB `/prices-history`.
- `orderbook_snapshots/`: partitioned parquet dataset (`snapshot_date=YYYY-MM-DD`) with CLOB best/worst bid/ask prices + sizes (plus optional depth as JSON).
- `orderbook_levels/`: partitioned parquet dataset (`snapshot_date=YYYY-MM-DD`) with one row per bid/ask price level (when `--with-orderbook` and `--orderbook-levels != 0`).

Notes:
- `outcome_prices.price` comes from Gamma `outcomePrices` (a “mark”/probability-style price).
- The Polymarket UI “Yes” quote often matches `market_snapshots.best_ask` (and “Sell Yes” matches `market_snapshots.best_bid`).

## Backtesting: Load A Day

- Python:
  - `from polymarket_archive import load_day, iter_day, load_day_trades, load_reference_tables`
  - `day = load_day("/path/to/polymarket_archive", "2026-01-13")`
  - `day["market_trades"]` (list of trades for that UTC day)
  - `day["token_price_history"]` (historical per-token prices for that UTC day)
  - `day["orderbook_snapshots"]` (best/worst bid/ask prices + sizes per token at snapshot times)
  - `day["orderbook_levels"]` (per-level bid/ask depth rows, when captured)
  - `trades = load_day_trades("/path/to/polymarket_archive", "2026-01-13")`
  - `for tick in iter_day("/path/to/polymarket_archive", "2026-01-13"): ...`
  - `refs = load_reference_tables("/path/to/polymarket_archive")`

## Backfill Trades (Historical)

If you started archiving recently (so earlier days are empty), you can backfill *transactions* via the public Polymarket data API:

- `python3 polymarket_archive.py --config config.json backfill-trades --start-date 2026-01-01 --end-date 2026-01-13`

This writes matched rows into `market_trades/trade_date=YYYY-MM-DD/` so you can replay older days even if you didn’t run the live snapshotter back then.

To sanity-check what was backfilled for a day:
- `python3 polymarket_archive.py --config config.json load-day-trades --date 2026-01-01 --print | head`

## Backfill Price History (Recommended)

To backfill older days reliably, use the CLOB price history endpoint (does not require the CLOB API key):

- `python3 polymarket_archive.py --config config.json backfill-price-history --start-date 2026-01-01 --end-date 2026-01-13 --fidelity-min 60`
- `python3 polymarket_archive.py --config config.json load-day-price-history --date 2026-01-01 --print | head`

To also populate the main `market_snapshots/` + `outcome_prices/` tables for past days (so `load_day()` isn’t empty), add:
- `--materialize-snapshots`

For live operation (to avoid snapshotting thousands of expired markets every minute), use:
- `python3 polymarket_archive.py --config config.json run --rolling-days 0 --interval-s 60 --discover-every-s 3600`

To also capture the “shares available at the best price” (order book size), add:
- `--with-orderbook --orderbook-levels 1`

To capture *all* open prices + sizes (full depth returned by the public CLOB endpoint), use:
- `--with-orderbook --orderbook-levels -1`

To inspect captured depth for a specific day:
- `python3 polymarket_archive.py --config config.json load-day-orderbook-levels --date 2026-01-14 --print | head`

## Recreate From Scratch (Backfill + Live)

- `bash recreate_market_archive.sh` rebuilds the archive from a start date using minutely `token_price_history` (`--fidelity-min 1`) and then (optionally) runs the live archiver with full-depth order books.
- Historical full-depth order books are not backfillable with the current public endpoints; depth capture starts when you begin running `--with-orderbook`.
  - Tip: for large backfills, use smaller chunks via `PRICE_CHUNK_DAYS=1` (default) or `PRICE_CHUNK_DAYS=7` to make it easier to resume.
  - Note: for historical discovery, use `--closed true --active any` (some closed markets still report `active=true`).

## Legacy Migration (SQLite → Parquet)

- `python3 polymarket_archive.py migrate-sqlite --sqlite polymarket_archive.sqlite3 --archive-dir polymarket_archive`

## Default Location

With the default `config.example.json` (stored in `data_gatherer/`), the archive is written under `data/market_data/` via `archive_dir: "../data/market_data/polymarket_archive"`.
