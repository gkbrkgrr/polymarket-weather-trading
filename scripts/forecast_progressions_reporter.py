#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from master_db import (
    get_historical_daily_tmax_bounds as fetch_historical_daily_tmax_bounds,
    resolve_master_postgres_dsn,
)

DEFAULT_COUNTRY_ROOT = REPO_ROOT / "data" / "ml_predictions" / "city_extended"
DEFAULT_GLOBAL_V1_ROOT = REPO_ROOT / "data" / "ml_predictions" / "xgb_opt_v1_100"
DEFAULT_GLOBAL_V2_ROOT = REPO_ROOT / "data" / "ml_predictions" / "xgb_opt_v2_100"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "forecast_progressions"
DEFAULT_LOCATIONS_CSV = REPO_ROOT / "locations.csv"
DEFAULT_OBS_ROOT = REPO_ROOT / "data" / "observations"
REQUIRED_COLUMNS = ["city_name", "issue_time_utc", "target_date_local", "Forecast"]
CYCLE_TOKEN_PATTERN = re.compile(r"(\d{10})$")

MODEL_SPECS = [
    {
        "key": "country_based",
        "label": "Country Based",
        "color": "green",
        "root_arg": "country_root",
    },
    {
        "key": "global_v1",
        "label": "GlobalV1",
        "color": "red",
        "root_arg": "global_v1_root",
    },
    {
        "key": "global_v2",
        "label": "GlobalV2",
        "color": "blue",
        "root_arg": "global_v2_root",
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a multipage PDF showing forecast progression by local day "
            "across city facets, comparing Country Based vs GlobalV1 vs GlobalV2."
        )
    )
    parser.add_argument(
        "--country-root",
        type=Path,
        default=DEFAULT_COUNTRY_ROOT,
        help=f"Country-based prediction root (default: {DEFAULT_COUNTRY_ROOT})",
    )
    parser.add_argument(
        "--global-v1-root",
        type=Path,
        default=DEFAULT_GLOBAL_V1_ROOT,
        help=f"GlobalV1 prediction root (default: {DEFAULT_GLOBAL_V1_ROOT})",
    )
    parser.add_argument(
        "--global-v2-root",
        type=Path,
        default=DEFAULT_GLOBAL_V2_ROOT,
        help=f"GlobalV2 prediction root (default: {DEFAULT_GLOBAL_V2_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for PDF reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cycle",
        type=str,
        default=None,
        help=(
            "GFS cycle token (YYYYMMDDHH). "
            "If omitted, the latest common cycle across all model roots is used."
        ),
    )
    parser.add_argument(
        "--locations-csv",
        type=Path,
        default=DEFAULT_LOCATIONS_CSV,
        help=f"Locations CSV used to detect USA cities (default: {DEFAULT_LOCATIONS_CSV})",
    )
    parser.add_argument(
        "--obs-root",
        type=Path,
        default=DEFAULT_OBS_ROOT,
        help=f"Deprecated (observations now read from DB); kept for compatibility (default: {DEFAULT_OBS_ROOT})",
    )
    parser.add_argument(
        "--obs-dsn",
        type=str,
        default=None,
        help="Optional DSN for master_db observation reads (defaults to MASTER_POSTGRES_DSN/config-derived value).",
    )
    parser.add_argument(
        "--cities-per-page",
        type=int,
        default=9,
        help="Number of city facets per page (default: 9 for 3x3).",
    )
    return parser.parse_args(argv)


def discover_city_dirs(input_root: Path) -> dict[str, Path]:
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    city_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    if not city_dirs:
        raise SystemExit(f"No city directories found under: {input_root}")
    return {p.name: p for p in city_dirs}


def extract_cycle_token(path: Path) -> str | None:
    match = CYCLE_TOKEN_PATTERN.search(path.stem)
    if match is None:
        return None
    return match.group(1)


def discover_cycle_tokens(city_dirs: dict[str, Path]) -> set[str]:
    tokens: set[str] = set()
    for city_dir in city_dirs.values():
        for file_path in city_dir.glob("*.parquet"):
            token = extract_cycle_token(file_path)
            if token is not None:
                tokens.add(token)
    if not tokens:
        raise SystemExit("No cycle-tokenized parquet files found under input root.")
    return tokens


def resolve_cycle_token(requested_cycle: str | None, available_tokens: set[str]) -> str:
    latest_cycle = max(available_tokens)
    if requested_cycle is None:
        return latest_cycle
    if re.fullmatch(r"\d{10}", requested_cycle) is None:
        raise SystemExit(
            f"--cycle must be in YYYYMMDDHH format, got: {requested_cycle!r}"
        )
    if requested_cycle not in available_tokens:
        raise SystemExit(
            f"Requested cycle {requested_cycle} not found under input roots. "
            f"Latest available common cycle is {latest_cycle}."
        )
    return requested_cycle


def discover_usa_cities(locations_csv: Path) -> set[str]:
    if not locations_csv.exists():
        raise SystemExit(f"Locations CSV not found: {locations_csv}")
    loc_df = pd.read_csv(locations_csv)
    required = {"name", "url"}
    missing = required - set(loc_df.columns)
    if missing:
        raise SystemExit(
            f"Locations CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    names = loc_df["name"].astype(str).str.strip()
    urls = loc_df["url"].astype(str).str.strip().str.lower()
    usa_mask = urls.str.contains("/daily/us/", regex=False)
    return set(names[usa_mask].tolist())


def load_city_predictions(city_dir: Path) -> pd.DataFrame:
    files = sorted(city_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    parts: list[pd.DataFrame] = []
    for file_path in files:
        part = pd.read_parquet(file_path, columns=REQUIRED_COLUMNS)
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df["issue_time_utc"] = pd.to_datetime(df["issue_time_utc"], utc=True, errors="coerce")
    df["target_date_local"] = pd.to_datetime(df["target_date_local"], errors="coerce").dt.date
    df["Forecast"] = pd.to_numeric(df["Forecast"], errors="coerce")
    df = df.dropna(subset=["city_name", "issue_time_utc", "target_date_local", "Forecast"]).copy()
    return df


def collect_model_predictions(
    *,
    model_key: str,
    city_dir_map: dict[str, Path],
    selected_cities: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for city in selected_cities:
        city_dir = city_dir_map[city]
        city_df = load_city_predictions(city_dir)
        if city_df.empty:
            continue
        city_df["city_name"] = city
        city_df["model_key"] = model_key
        frames.append(city_df)
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["model_key"])
    return pd.concat(frames, ignore_index=True)


def load_historical_daily_tmax_bounds(obs_root: Path, cities: list[str], obs_dsn: str | None) -> pd.DataFrame:
    del obs_root
    columns = [
        "city_name",
        "month_day",
        "hist_min_c",
        "hist_max_c",
        "hist_min_f",
        "hist_max_f",
    ]
    dsn = resolve_master_postgres_dsn(explicit_dsn=obs_dsn)
    out = fetch_historical_daily_tmax_bounds(
        stations=cities,
        start_date=pd.Timestamp("2000-01-01").date(),
        master_dsn=dsn,
    )
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out


def select_local_days_from_cycle(df: pd.DataFrame, cycle_token: str) -> list:
    cycle_utc = pd.to_datetime(cycle_token, format="%Y%m%d%H", utc=True)
    cycle_rows = df[df["issue_time_utc"] == cycle_utc]
    if cycle_rows.empty:
        raise SystemExit(
            "Could not find rows for cycle "
            f"{cycle_token} in loaded predictions."
        )
    local_days = sorted(pd.unique(cycle_rows["target_date_local"]))
    if not local_days:
        raise SystemExit(
            f"Cycle {cycle_token} has no target local days."
        )
    return local_days


def build_output_pdf_path(output_dir: Path, cycle_token: str) -> Path:
    return output_dir / f"xgb_optuna_forecasts_{cycle_token}.pdf"


def chunked(values: list[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [values[i : i + size] for i in range(0, len(values), size)]


def render_pdf(
    *,
    df: pd.DataFrame,
    cities: list[str],
    output_pdf: Path,
    local_days: list,
    historical_bounds: pd.DataFrame,
    cities_per_page: int,
) -> int:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    page_count = 0
    bounds_lookup: dict[tuple[str, str], tuple[float, float]] = {}

    if not historical_bounds.empty:
        for row in historical_bounds.itertuples(index=False):
            if pd.notna(row.hist_min_plot) and pd.notna(row.hist_max_plot):
                bounds_lookup[(row.city_name, row.month_day)] = (
                    float(row.hist_min_plot),
                    float(row.hist_max_plot),
                )

    model_label = {spec["key"]: spec["label"] for spec in MODEL_SPECS}
    model_color = {spec["key"]: spec["color"] for spec in MODEL_SPECS}
    model_order = [spec["key"] for spec in MODEL_SPECS]

    cities_pages = chunked(cities, cities_per_page)
    n_rows, n_cols = 3, 3
    facet_count = n_rows * n_cols

    with PdfPages(output_pdf) as pdf:
        for local_day in local_days:
            day_df = df[df["target_date_local"] == local_day]
            day_issue_times = sorted(pd.unique(day_df["issue_time_utc"]))
            issue_time_to_x = {ts: i for i, ts in enumerate(day_issue_times)}
            x_ticks = list(range(len(day_issue_times)))
            x_labels = [pd.Timestamp(ts).strftime("%Y%m%d%H") for ts in day_issue_times]
            month_day = pd.Timestamp(local_day).strftime("%m-%d")

            for city_page_idx, city_chunk in enumerate(cities_pages, start=1):
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
                day_token = pd.Timestamp(local_day).strftime("%Y%m%d")
                if len(cities_pages) > 1:
                    title = f"{day_token}  |  Cities Page {city_page_idx}/{len(cities_pages)}"
                else:
                    title = day_token
                fig.suptitle(title, fontsize=20, y=0.99)

                axes_flat = axes.ravel()
                legend_handles: dict[str, object] = {}

                for i in range(facet_count):
                    ax = axes_flat[i]
                    if i >= len(city_chunk):
                        ax.axis("off")
                        continue

                    city = city_chunk[i]
                    city_day = (
                        day_df[day_df["city_name"] == city]
                        .sort_values("issue_time_utc", kind="mergesort")
                    )
                    ax.set_title(city, fontsize=12)

                    historical_line = bounds_lookup.get((city, month_day))
                    if historical_line is not None:
                        hist_min_plot, hist_max_plot = historical_line
                        ax.axhline(hist_min_plot, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)
                        ax.axhline(hist_max_plot, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)

                    if city_day.empty:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    else:
                        for model_key in model_order:
                            model_city_day = (
                                city_day[city_day["model_key"] == model_key]
                                .drop_duplicates(subset=["issue_time_utc"], keep="last")
                            )
                            if model_city_day.empty:
                                continue
                            x_vals = [issue_time_to_x[ts] for ts in model_city_day["issue_time_utc"]]
                            y_vals = model_city_day["forecast_rounded"].tolist()
                            (line,) = ax.plot(
                                x_vals,
                                y_vals,
                                marker="o",
                                linewidth=1.5,
                                markersize=3.8,
                                color=model_color[model_key],
                            )
                            legend_handles.setdefault(model_key, line)

                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
                    if x_ticks:
                        ax.set_xlim(-0.5, len(x_ticks) - 0.5)
                    ax.tick_params(axis="y", labelsize=8)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                legend_keys = [k for k in model_order if k in legend_handles]
                if legend_keys:
                    handles = [legend_handles[k] for k in legend_keys]
                    labels = [model_label[k] for k in legend_keys]
                    fig.legend(
                        handles,
                        labels,
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0.015),
                        ncol=3,
                        frameon=False,
                        fontsize=11,
                    )

                plt.tight_layout(rect=[0, 0.07, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)
                page_count += 1

    return page_count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    root_map = {
        "country_based": args.country_root,
        "global_v1": args.global_v1_root,
        "global_v2": args.global_v2_root,
    }

    city_dirs_by_model: dict[str, dict[str, Path]] = {}
    cycle_tokens_by_model: dict[str, set[str]] = {}
    for spec in MODEL_SPECS:
        key = spec["key"]
        city_map = discover_city_dirs(root_map[key])
        city_dirs_by_model[key] = city_map
        cycle_tokens_by_model[key] = discover_cycle_tokens(city_map)

    common_cities = sorted(
        set(city_dirs_by_model["country_based"]).intersection(
            city_dirs_by_model["global_v1"],
            city_dirs_by_model["global_v2"],
        )
    )
    if not common_cities:
        raise SystemExit("No common city directories found across country/global_v1/global_v2 roots.")

    common_cycles = (
        cycle_tokens_by_model["country_based"]
        & cycle_tokens_by_model["global_v1"]
        & cycle_tokens_by_model["global_v2"]
    )
    if not common_cycles:
        raise SystemExit("No common cycle tokens found across country/global_v1/global_v2 roots.")

    latest_cycle_token = max(common_cycles)
    cycle_token = resolve_cycle_token(args.cycle, common_cycles)
    cycle_utc = pd.to_datetime(cycle_token, format="%Y%m%d%H", utc=True)

    usa_cities = discover_usa_cities(args.locations_csv)

    model_frames: list[pd.DataFrame] = []
    for spec in MODEL_SPECS:
        key = spec["key"]
        frame = collect_model_predictions(
            model_key=key,
            city_dir_map=city_dirs_by_model[key],
            selected_cities=common_cities,
        )
        model_frames.append(frame)

    df = pd.concat(model_frames, ignore_index=True)
    if df.empty:
        raise SystemExit("Loaded prediction data is empty.")

    historical_bounds = load_historical_daily_tmax_bounds(args.obs_root, common_cities, args.obs_dsn)

    usa_mask = df["city_name"].isin(usa_cities)
    df.loc[usa_mask, "Forecast"] = (df.loc[usa_mask, "Forecast"] * 9.0 / 5.0) + 32.0
    df["forecast_rounded"] = df["Forecast"].round().astype(int)

    if not historical_bounds.empty:
        historical_bounds["hist_min_plot"] = historical_bounds["hist_min_c"]
        historical_bounds["hist_max_plot"] = historical_bounds["hist_max_c"]
        usa_hist_mask = historical_bounds["city_name"].isin(usa_cities)
        historical_bounds.loc[usa_hist_mask, "hist_min_plot"] = historical_bounds.loc[
            usa_hist_mask, "hist_min_f"
        ]
        historical_bounds.loc[usa_hist_mask, "hist_max_plot"] = historical_bounds.loc[
            usa_hist_mask, "hist_max_f"
        ]
    else:
        historical_bounds = pd.DataFrame(
            columns=["city_name", "month_day", "hist_min_plot", "hist_max_plot"]
        )

    local_days = select_local_days_from_cycle(df=df, cycle_token=cycle_token)
    df_filtered = df[
        (df["target_date_local"].isin(local_days)) & (df["issue_time_utc"] <= cycle_utc)
    ].copy()

    output_pdf = build_output_pdf_path(args.output_dir, cycle_token)
    pages = render_pdf(
        df=df_filtered,
        cities=common_cities,
        output_pdf=output_pdf,
        local_days=local_days,
        historical_bounds=historical_bounds,
        cities_per_page=int(args.cities_per_page),
    )

    print(f"Country root: {args.country_root}")
    print(f"GlobalV1 root: {args.global_v1_root}")
    print(f"GlobalV2 root: {args.global_v2_root}")
    print(f"Common cities: {', '.join(common_cities)}")
    print(f"Cycle used: {cycle_token}")
    print(f"Latest common cycle: {latest_cycle_token}")
    print(f"USA cities converted to Fahrenheit: {', '.join(sorted(usa_cities))}")
    print(f"Observation bounds DSN: {resolve_master_postgres_dsn(explicit_dsn=args.obs_dsn)}")
    print(f"Historical month-day bounds loaded: {len(historical_bounds)}")
    print(f"Local days from cycle: {len(local_days)}")
    print(f"Pages written: {pages}")
    print(f"Wrote PDF: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
