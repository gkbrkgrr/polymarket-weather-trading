import argparse
import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_DB_PATH = "polymarket_archive.sqlite3"


DEFAULT_TARGETS: list[dict[str, Any]] = [
    {"name": "Highest temperature in Seoul", "match_any": ["Highest temperature in Seoul"], "match_all": []},
    {
        "name": "Highest temperature in NYC",
        "match_any": ["Highest temperature in NYC", "Highest temperature in New York City"],
        "match_all": [],
    },
    {"name": "Highest temperature in London", "match_any": ["Highest temperature in London"], "match_all": []},
    {"name": "Highest temperature in Atlanta", "match_any": ["Highest temperature in Atlanta"], "match_all": []},
    {"name": "Highest temperature in Dallas", "match_any": ["Highest temperature in Dallas"], "match_all": []},
    {"name": "Highest temperature in Toronto", "match_any": ["Highest temperature in Toronto"], "match_all": []},
    {"name": "Highest temperature in Seattle", "match_any": ["Highest temperature in Seattle"], "match_all": []},
    {"name": "Highest temperature in Buenos Aires", "match_any": ["Highest temperature in Buenos Aires"], "match_all": []},
    {
        "name": "Will the highest temperature in New York City be between",
        "match_any": ["Will the highest temperature in New York City be between", "Will the highest temperature in NYC be between"],
        "match_all": [],
    },
    {
        "name": "Will the highest temperature in Toronto be between",
        "match_any": ["Will the highest temperature in Toronto be between"],
        "match_all": [],
    },
    {
        "name": "Will the highest temperature in Seoul be between",
        "match_any": ["Will the highest temperature in Seoul be between"],
        "match_all": [],
    },
    {
        "name": "Precipitation in NYC (monthly inches)",
        "match_any": ["Precipitation in NYC"],
        "match_all": ["monthly total precipitation in inches"],
    },
]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json_load_maybe(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
        if value.startswith("[") or value.startswith("{"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
    return default


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _to_int01(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "t", "1", "yes", "y"}:
            return 1
        if value in {"false", "f", "0", "no", "n"}:
            return 0
    return None


def http_get_json(url: str, timeout_s: int = 30) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "polymarket-weather-archiver/1.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def gamma_list_markets(
    gamma_base_url: str,
    *,
    limit: int = 200,
    offset: int = 0,
    active: bool | None = True,
    closed: bool | None = False,
    archived: bool | None = False,
    max_pages: int = 50,
    timeout_s: int = 30,
    control: dict[str, Any] | None = None,
) -> Iterable[dict[str, Any]]:
    page = 0
    while page < max_pages:
        if control is not None:
            control["pages_scanned"] = page
        params: dict[str, str] = {"limit": str(limit), "offset": str(offset)}
        if active is not None:
            params["active"] = "true" if active else "false"
        if closed is not None:
            params["closed"] = "true" if closed else "false"
        if archived is not None:
            params["archived"] = "true" if archived else "false"

        url = f"{gamma_base_url.rstrip('/')}/markets?{urllib.parse.urlencode(params)}"
        payload = http_get_json(url, timeout_s=timeout_s)
        if isinstance(payload, list):
            value = payload
            returned = len(value)
        elif isinstance(payload, dict):
            value = payload.get("value", [])
            returned = payload.get("Count", len(value) if isinstance(value, list) else 0)
        else:
            break

        if not isinstance(value, list):
            break
        for market in value:
            if control is not None and control.get("stop"):
                return
            if isinstance(market, dict):
                yield market

        if not isinstance(returned, int):
            returned = len(value)
        if returned < limit or len(value) == 0:
            break

        offset += limit
        page += 1

    if control is not None and page >= max_pages:
        control["hit_max_pages"] = True


def gamma_get_market(gamma_base_url: str, market_id: str, *, timeout_s: int = 30) -> dict[str, Any]:
    url = f"{gamma_base_url.rstrip('/')}/markets/{urllib.parse.quote(str(market_id))}"
    payload = http_get_json(url, timeout_s=timeout_s)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected response for /markets/{market_id}")
    return payload


@dataclass(frozen=True)
class Target:
    name: str
    match_any: list[str]
    match_all: list[str]

    def matches(self, question: str) -> bool:
        q = question.casefold()
        if self.match_all and not all(s.casefold() in q for s in self.match_all):
            return False
        if self.match_any and not any(s.casefold() in q for s in self.match_any):
            return False
        return True


def load_config(path: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {}
    config_dir = os.getcwd()
    if path is None:
        if os.path.exists("config.json"):
            path = "config.json"
        else:
            path = None

    if path is not None:
        config_dir = os.path.dirname(os.path.abspath(path)) or os.getcwd()
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

    config.setdefault("db_path", DEFAULT_DB_PATH)
    config.setdefault("gamma_base_url", DEFAULT_GAMMA_BASE_URL)
    config.setdefault(
        "discovery",
        {
            "active": True,
            "closed": False,
            "archived": False,
            "limit": 200,
            "max_pages": 50,
            "timeout_s": 30,
        },
    )
    config.setdefault("snapshot_timeout_s", 30)
    config.setdefault("targets", DEFAULT_TARGETS)

    db_path = os.environ.get("POLYMARKET_ARCHIVE_DB_PATH")
    if db_path:
        config["db_path"] = db_path
    gamma_base_url = os.environ.get("POLYMARKET_GAMMA_BASE_URL")
    if gamma_base_url:
        config["gamma_base_url"] = gamma_base_url
    max_pages = os.environ.get("POLYMARKET_DISCOVERY_MAX_PAGES")
    if max_pages and max_pages.isdigit():
        config.setdefault("discovery", {})
        config["discovery"]["max_pages"] = int(max_pages)
    stop_after = os.environ.get("POLYMARKET_DISCOVERY_STOP_AFTER_MATCHES_PER_TARGET")
    if stop_after and stop_after.isdigit():
        config.setdefault("discovery", {})
        config["discovery"]["stop_after_matches_per_target"] = int(stop_after)

    if isinstance(config.get("db_path"), str) and not os.path.isabs(config["db_path"]):
        config["db_path"] = os.path.abspath(os.path.join(config_dir, config["db_path"]))
    return config


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    schema_path = os.path.join(os.path.dirname(__file__) or ".", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()


def upsert_targets(conn: sqlite3.Connection, targets: list[Target]) -> None:
    for t in targets:
        conn.execute(
            """
            INSERT INTO targets (name, match_any_json, match_all_json)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              match_any_json=excluded.match_any_json,
              match_all_json=excluded.match_all_json
            """,
            (t.name, json.dumps(t.match_any), json.dumps(t.match_all)),
        )
    conn.commit()


def load_targets_from_config(config: dict[str, Any]) -> list[Target]:
    raw_targets = config.get("targets", [])
    targets: list[Target] = []
    for raw in raw_targets:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        match_any = raw.get("match_any", [])
        match_all = raw.get("match_all", [])
        if not isinstance(match_any, list):
            match_any = []
        if not isinstance(match_all, list):
            match_all = []
        targets.append(
            Target(
                name=name,
                match_any=[str(s) for s in match_any if str(s).strip()],
                match_all=[str(s) for s in match_all if str(s).strip()],
            )
        )
    return targets


def upsert_market_and_outcomes(conn: sqlite3.Connection, market: dict[str, Any], seen_ts: int) -> None:
    market_id = str(market.get("id", "")).strip()
    if not market_id:
        return

    conn.execute(
        """
        INSERT INTO markets (
          market_id, question, slug, condition_id, category, end_date, created_at, last_seen_ts, raw_first_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(market_id) DO UPDATE SET
          question=excluded.question,
          slug=excluded.slug,
          condition_id=excluded.condition_id,
          category=excluded.category,
          end_date=excluded.end_date,
          created_at=excluded.created_at,
          last_seen_ts=excluded.last_seen_ts
        """,
        (
            market_id,
            str(market.get("question", "")),
            market.get("slug"),
            market.get("conditionId"),
            market.get("category"),
            market.get("endDate"),
            market.get("createdAt"),
            int(seen_ts),
            json.dumps(market, separators=(",", ":"), ensure_ascii=False),
        ),
    )

    outcomes = _json_load_maybe(market.get("outcomes"), default=[])
    outcome_prices = _json_load_maybe(market.get("outcomePrices"), default=[])
    clob_token_ids = _json_load_maybe(market.get("clobTokenIds"), default=[])

    if isinstance(outcomes, list):
        for i, outcome in enumerate(outcomes):
            clob_token_id = None
            if isinstance(clob_token_ids, list) and i < len(clob_token_ids):
                clob_token_id = str(clob_token_ids[i])
            conn.execute(
                """
                INSERT INTO market_outcomes (market_id, outcome_index, outcome, clob_token_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(market_id, outcome_index) DO UPDATE SET
                  outcome=excluded.outcome,
                  clob_token_id=excluded.clob_token_id
                """,
                (market_id, int(i), str(outcome), clob_token_id),
            )

    conn.commit()


def upsert_target_market_link(conn: sqlite3.Connection, target_name: str, market_id: str, seen_ts: int) -> None:
    row = conn.execute("SELECT target_id FROM targets WHERE name = ?", (target_name,)).fetchone()
    if row is None:
        return
    target_id = int(row["target_id"])

    conn.execute(
        """
        INSERT INTO target_markets (target_id, market_id, first_seen_ts, last_seen_ts)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(target_id, market_id) DO UPDATE SET
          last_seen_ts=excluded.last_seen_ts
        """,
        (target_id, market_id, int(seen_ts), int(seen_ts)),
    )
    conn.commit()


def discover(conn: sqlite3.Connection, config: dict[str, Any]) -> list[dict[str, Any]]:
    targets = load_targets_from_config(config)
    upsert_targets(conn, targets)

    discovery_cfg = config.get("discovery", {})
    gamma_base_url = str(config.get("gamma_base_url", DEFAULT_GAMMA_BASE_URL))
    stop_after_matches_per_target = int(discovery_cfg.get("stop_after_matches_per_target", 0) or 0)

    seen_ts = _now_ms()
    matches: list[dict[str, Any]] = []
    matched_counts: dict[str, int] = {t.name: 0 for t in targets}
    control: dict[str, Any] = {}

    for market in gamma_list_markets(
        gamma_base_url,
        limit=int(discovery_cfg.get("limit", 200)),
        active=discovery_cfg.get("active", True),
        closed=discovery_cfg.get("closed", False),
        archived=discovery_cfg.get("archived", False),
        max_pages=int(discovery_cfg.get("max_pages", 50)),
        timeout_s=int(discovery_cfg.get("timeout_s", 30)),
        control=control,
    ):
        question = str(market.get("question", ""))
        market_id = str(market.get("id", "")).strip()
        if not market_id:
            continue
        matched = [t for t in targets if t.matches(question)]
        if not matched:
            continue

        upsert_market_and_outcomes(conn, market, seen_ts)
        for t in matched:
            upsert_target_market_link(conn, t.name, market_id, seen_ts)
            matched_counts[t.name] = matched_counts.get(t.name, 0) + 1
        matches.append({"market_id": market_id, "question": question, "targets": [t.name for t in matched]})

        if stop_after_matches_per_target > 0:
            if all(count >= stop_after_matches_per_target for count in matched_counts.values()):
                control["stop"] = True

    return matches


def snapshot(conn: sqlite3.Connection, config: dict[str, Any], *, only_target: str | None = None) -> int:
    gamma_base_url = str(config.get("gamma_base_url", DEFAULT_GAMMA_BASE_URL))
    snapshot_ts = _now_ms()
    timeout_s = int(config.get("snapshot_timeout_s", 30))

    if only_target is None:
        rows = conn.execute(
            """
            SELECT DISTINCT tm.market_id
            FROM target_markets tm
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT tm.market_id
            FROM target_markets tm
            JOIN targets t ON t.target_id = tm.target_id
            WHERE t.name = ?
            """,
            (only_target,),
        ).fetchall()

    market_ids = [str(r["market_id"]) for r in rows]
    if not market_ids:
        return 0

    inserted = 0
    for market_id in market_ids:
        market = gamma_get_market(gamma_base_url, market_id, timeout_s=timeout_s)
        upsert_market_and_outcomes(conn, market, snapshot_ts)

        conn.execute(
            """
            INSERT INTO market_snapshots (
              snapshot_ts,
              market_id,
              active,
              closed,
              archived,
              volume_num,
              liquidity_num,
              last_trade_price,
              best_bid,
              best_ask,
              spread,
              one_day_price_change,
              one_hour_price_change,
              one_week_price_change,
              one_month_price_change,
              one_year_price_change,
              raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_ts, market_id) DO NOTHING
            """,
            (
                int(snapshot_ts),
                market_id,
                _to_int01(market.get("active")),
                _to_int01(market.get("closed")),
                _to_int01(market.get("archived")),
                _to_float(market.get("volumeNum", market.get("volume"))),
                _to_float(market.get("liquidityNum", market.get("liquidity"))),
                _to_float(market.get("lastTradePrice")),
                _to_float(market.get("bestBid")),
                _to_float(market.get("bestAsk")),
                _to_float(market.get("spread")),
                _to_float(market.get("oneDayPriceChange")),
                _to_float(market.get("oneHourPriceChange")),
                _to_float(market.get("oneWeekPriceChange")),
                _to_float(market.get("oneMonthPriceChange")),
                _to_float(market.get("oneYearPriceChange")),
                json.dumps(market, separators=(",", ":"), ensure_ascii=False),
            ),
        )

        outcomes = _json_load_maybe(market.get("outcomes"), default=[])
        outcome_prices = _json_load_maybe(market.get("outcomePrices"), default=[])

        if not isinstance(outcomes, list) or not isinstance(outcome_prices, list):
            outcomes = []
            outcome_prices = []

        if outcomes and outcome_prices and len(outcomes) == len(outcome_prices):
            for i, (outcome, price) in enumerate(zip(outcomes, outcome_prices, strict=True)):
                conn.execute(
                    """
                    INSERT INTO outcome_prices (snapshot_ts, market_id, outcome_index, outcome, price)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_ts, market_id, outcome_index) DO NOTHING
                    """,
                    (int(snapshot_ts), market_id, int(i), str(outcome), _to_float(price)),
                )

        conn.commit()
        inserted += 1

    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive Polymarket weather contracts into SQLite.")
    parser.add_argument("--config", default=None, help="Path to config JSON (default: ./config.json if present)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init-db", help="Create tables in the SQLite database.")
    init.add_argument("--db", default=None, help=f"SQLite DB path (default from config or {DEFAULT_DB_PATH})")

    disc = sub.add_parser("discover", help="Discover matching markets and store them in the DB.")
    disc.add_argument("--print", action="store_true", help="Print matches as JSON lines.")

    snap = sub.add_parser("snapshot", help="Fetch current prices for tracked markets and store snapshots.")
    snap.add_argument("--only-target", default=None, help="Only snapshot markets linked to this target name.")

    run = sub.add_parser("run", help="Continuously snapshot markets on an interval.")
    run.add_argument("--interval-s", type=int, default=60, help="Snapshot interval in seconds.")
    run.add_argument("--discover-every-s", type=int, default=3600, help="Run discovery every N seconds.")
    run.add_argument("--log-file", default=None, help="Append logs to this file (also prints to stdout).")

    return parser.parse_args()


def _log(line: str, *, log_file: str | None) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] {line}"
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    db_path = args.db if getattr(args, "db", None) else str(config.get("db_path", DEFAULT_DB_PATH))
    conn = connect_db(db_path)
    init_db(conn)

    if args.cmd == "init-db":
        return 0

    if args.cmd == "discover":
        matches = discover(conn, config)
        if args.print:
            for m in matches:
                print(json.dumps(m, ensure_ascii=False))
        return 0

    if args.cmd == "snapshot":
        count = snapshot(conn, config, only_target=args.only_target)
        print(f"Snapshotted {count} markets into {db_path}")
        return 0

    if args.cmd == "run":
        last_discover = 0.0
        failures = 0
        while True:
            now = time.time()
            try:
                if now - last_discover >= args.discover_every_s:
                    matches = discover(conn, config)
                    if matches:
                        _log(f"Discovered {len(matches)} matching markets", log_file=args.log_file)
                    last_discover = now

                snap_count = snapshot(conn, config)
                _log(f"Snapshotted {snap_count} markets", log_file=args.log_file)
                failures = 0
                time.sleep(max(1, args.interval_s))
            except KeyboardInterrupt:
                _log("Stopping (KeyboardInterrupt)", log_file=args.log_file)
                return 0
            except Exception as e:
                failures += 1
                _log(f"Error (failure #{failures}): {e.__class__.__name__}: {e}", log_file=args.log_file)
                time.sleep(min(300, max(5, failures * 5)))

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
