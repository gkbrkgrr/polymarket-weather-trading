# VPS Setup (Linux)

This repo runs a small Python process that periodically:

1) discovers matching Polymarket markets (to catch new daily contracts), then  
2) snapshots prices/metadata into a local parquet dataset (partitioned by day).

## 1) Prereqs

- Ubuntu/Debian example packages:
  - `sudo apt-get update`
  - `sudo apt-get install -y python3 python3-venv python3-pip git`

## 2) Copy the project to the VPS

Example target path used by the systemd unit:

- `/opt/polymarket-weather-trading/data_gatherer`

## 3) Create a virtualenv + install

From the `data_gatherer` directory:

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -U pip`
- `pip install .`

## 4) Create config + initialize the archive

- Copy `config.example.json` → `config.json` and adjust targets if needed.
- Initialize the archive directory:
  - `polymarket-archiver --config ./config.json init-db`

Optional: verify discovery finds markets:
- `polymarket-archiver --config ./config.json discover --print | head`

## 5) Run (simple)

Foreground:
- `bash run_archiver.sh`

Background (screen/tmux):
- `tmux new -s archiver`
- `bash run_archiver.sh`

## 6) Run (systemd, recommended)

1. Create a log directory:
   - `sudo mkdir -p /var/log/polymarket-archiver`
   - `sudo chown -R $USER:$USER /var/log/polymarket-archiver`
2. Copy the unit file and edit paths if you installed elsewhere:
   - `sudo cp deploy/polymarket-archiver.service /etc/systemd/system/polymarket-archiver.service`
3. Ensure the service uses your venv Python.
   - Easiest approach: edit `run_archiver.sh` to call the venv python:
     - replace `python` with `/opt/polymarket-weather-trading/data_gatherer/.venv/bin/python`
4. Enable + start:
   - `sudo systemctl daemon-reload`
   - `sudo systemctl enable --now polymarket-archiver`
5. Check status/logs:
   - `systemctl status polymarket-archiver --no-pager`
   - `tail -f /var/log/polymarket-archiver/archiver.log`

## Notes

- The archive directory defaults to `../data/market_data/polymarket_archive` relative to `config.json`.
- If you don’t want to store `config.json` in the repo folder, set:
  - `POLYMARKET_ARCHIVE_DIR=/path/to/polymarket_archive`
  - `POLYMARKET_DISCOVERY_MAX_PAGES=200` (or higher if needed)
