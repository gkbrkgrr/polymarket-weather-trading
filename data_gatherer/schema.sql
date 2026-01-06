PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS targets (
  target_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  match_any_json TEXT NOT NULL,
  match_all_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS markets (
  market_id TEXT PRIMARY KEY,
  question TEXT NOT NULL,
  slug TEXT,
  condition_id TEXT,
  category TEXT,
  end_date TEXT,
  created_at TEXT,
  last_seen_ts INTEGER NOT NULL,
  raw_first_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_outcomes (
  market_id TEXT NOT NULL,
  outcome_index INTEGER NOT NULL,
  outcome TEXT NOT NULL,
  clob_token_id TEXT,
  PRIMARY KEY (market_id, outcome_index),
  FOREIGN KEY (market_id) REFERENCES markets(market_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS target_markets (
  target_id INTEGER NOT NULL,
  market_id TEXT NOT NULL,
  first_seen_ts INTEGER NOT NULL,
  last_seen_ts INTEGER NOT NULL,
  PRIMARY KEY (target_id, market_id),
  FOREIGN KEY (target_id) REFERENCES targets(target_id) ON DELETE CASCADE,
  FOREIGN KEY (market_id) REFERENCES markets(market_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS market_snapshots (
  snapshot_ts INTEGER NOT NULL,
  market_id TEXT NOT NULL,
  active INTEGER,
  closed INTEGER,
  archived INTEGER,
  volume_num REAL,
  liquidity_num REAL,
  last_trade_price REAL,
  best_bid REAL,
  best_ask REAL,
  spread REAL,
  one_day_price_change REAL,
  one_hour_price_change REAL,
  one_week_price_change REAL,
  one_month_price_change REAL,
  one_year_price_change REAL,
  raw_json TEXT NOT NULL,
  PRIMARY KEY (snapshot_ts, market_id),
  FOREIGN KEY (market_id) REFERENCES markets(market_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS outcome_prices (
  snapshot_ts INTEGER NOT NULL,
  market_id TEXT NOT NULL,
  outcome_index INTEGER NOT NULL,
  outcome TEXT NOT NULL,
  price REAL,
  PRIMARY KEY (snapshot_ts, market_id, outcome_index),
  FOREIGN KEY (snapshot_ts, market_id) REFERENCES market_snapshots(snapshot_ts, market_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_market_ts ON market_snapshots(market_id, snapshot_ts DESC);
CREATE INDEX IF NOT EXISTS idx_target_markets_target ON target_markets(target_id);
