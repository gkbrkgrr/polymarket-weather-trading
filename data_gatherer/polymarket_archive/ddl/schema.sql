CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    slug TEXT NULL,
    title TEXT NOT NULL,
    status TEXT NULL,
    event_start_time TIMESTAMPTZ NULL,
    resolution_time TIMESTAMPTZ NULL,
    raw JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS markets_updated_at_idx ON markets(updated_at);
CREATE INDEX IF NOT EXISTS markets_status_idx ON markets(status);
CREATE INDEX IF NOT EXISTS markets_status_resolution_idx ON markets(status, resolution_time);

CREATE TABLE IF NOT EXISTS outcomes (
    market_id TEXT NOT NULL REFERENCES markets(market_id) ON DELETE CASCADE,
    outcome_id TEXT NULL,
    outcome_label TEXT NULL,
    outcome_index INT NOT NULL,
    raw JSONB NOT NULL,
    PRIMARY KEY (market_id, outcome_index)
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES markets(market_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    outcome_id TEXT NULL,
    outcome_index INT NULL,
    side TEXT NULL,
    price NUMERIC(10,6) NOT NULL,
    size NUMERIC(18,8) NOT NULL,
    tx_hash TEXT NULL,
    raw JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS trades_market_ts_idx ON trades(market_id, ts DESC);
CREATE INDEX IF NOT EXISTS trades_ts_idx ON trades(ts DESC);

CREATE TABLE IF NOT EXISTS cursors (
    market_id TEXT PRIMARY KEY REFERENCES markets(market_id) ON DELETE CASCADE,
    last_ts TIMESTAMPTZ NOT NULL,
    last_tiebreak TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS book_snapshots (
    market_id TEXT NOT NULL REFERENCES markets(market_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    outcome_index INT NULL,
    best_bid NUMERIC(10,6) NULL,
    best_ask NUMERIC(10,6) NULL,
    bid_size NUMERIC(18,8) NULL,
    ask_size NUMERIC(18,8) NULL,
    raw JSONB NOT NULL,
    PRIMARY KEY (market_id, ts, outcome_index)
);
CREATE INDEX IF NOT EXISTS book_snapshots_market_outcome_ts_idx
    ON book_snapshots(market_id, outcome_index, ts DESC);
CREATE INDEX IF NOT EXISTS book_snapshots_ts_idx ON book_snapshots(ts DESC);
