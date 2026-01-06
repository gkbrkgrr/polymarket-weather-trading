#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config.json}"
INTERVAL_S="${INTERVAL_S:-60}"
DISCOVER_EVERY_S="${DISCOVER_EVERY_S:-3600}"
LOG_FILE="${LOG_FILE:-archiver.log}"

exec python -u polymarket_archive.py --config "$CONFIG" run \
  --interval-s "$INTERVAL_S" \
  --discover-every-s "$DISCOVER_EVERY_S" \
  --log-file "$LOG_FILE"

