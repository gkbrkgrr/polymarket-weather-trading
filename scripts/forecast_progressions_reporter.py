#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "ml_predictions" / "xgb_opt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "forecast_progressions"
REQUIRED_COLUMNS = ["city_name", "issue_time_utc", "target_date_local", "Forecast"]
CYCLE_TOKEN_PATTERN = re.compile(r"(\d{10})$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a multipage PDF showing forecast progression by local day "
            "across 9 city facets (3x3), using local days from the latest cycle."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root directory containing per-city parquet files (default: {DEFAULT_INPUT_ROOT})",
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
            "If omitted, the latest available cycle is used."
        ),
    )
    return parser.parse_args(argv)


def discover_city_dirs(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    city_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    if not city_dirs:
        raise SystemExit(f"No city directories found under: {input_root}")
    if len(city_dirs) != 9:
        raise SystemExit(
            f"Expected exactly 9 city directories for 3x3 facets, found {len(city_dirs)} under {input_root}"
        )
    return city_dirs


def extract_cycle_token(path: Path) -> str | None:
    match = CYCLE_TOKEN_PATTERN.search(path.stem)
    if match is None:
        return None
    return match.group(1)


def discover_cycle_tokens(city_dirs: list[Path]) -> list[str]:
    tokens: list[str] = []
    for city_dir in city_dirs:
        for file_path in city_dir.glob("*.parquet"):
            token = extract_cycle_token(file_path)
            if token is not None:
                tokens.append(token)
    if not tokens:
        raise SystemExit("No cycle-tokenized parquet files found under input root.")
    return sorted(set(tokens))


def resolve_cycle_token(requested_cycle: str | None, available_tokens: list[str]) -> str:
    latest_cycle = max(available_tokens)
    if requested_cycle is None:
        return latest_cycle
    if re.fullmatch(r"\d{10}", requested_cycle) is None:
        raise SystemExit(
            f"--cycle must be in YYYYMMDDHH format, got: {requested_cycle!r}"
        )
    if requested_cycle not in set(available_tokens):
        raise SystemExit(
            f"Requested cycle {requested_cycle} not found under input root. "
            f"Latest available cycle is {latest_cycle}."
        )
    return requested_cycle


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
    df["forecast_rounded"] = df["Forecast"].round().astype(int)
    return df


def collect_predictions(city_dirs: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    cities: list[str] = []

    for city_dir in city_dirs:
        city_name = city_dir.name
        city_df = load_city_predictions(city_dir)
        if city_df.empty:
            city_df = pd.DataFrame(
                columns=["city_name", "issue_time_utc", "target_date_local", "Forecast", "forecast_rounded"]
            )
        else:
            city_df["city_name"] = city_name
        frames.append(city_df)
        cities.append(city_name)

    all_df = pd.concat(frames, ignore_index=True)
    return all_df, cities


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


def render_pdf(df: pd.DataFrame, cities: list[str], output_pdf: Path, local_days: list) -> int:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    page_count = 0

    with PdfPages(output_pdf) as pdf:
        for local_day in local_days:
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            fig.suptitle(pd.Timestamp(local_day).strftime("%Y%m%d"), fontsize=20, y=0.99)

            day_df = df[df["target_date_local"] == local_day]
            day_issue_times = sorted(pd.unique(day_df["issue_time_utc"]))
            issue_time_to_x = {ts: i for i, ts in enumerate(day_issue_times)}
            x_ticks = list(range(len(day_issue_times)))
            x_labels = [pd.Timestamp(ts).strftime("%Y%m%d%H") for ts in day_issue_times]
            axes_flat = axes.ravel()

            for ax, city in zip(axes_flat, cities):
                city_day = (
                    day_df[day_df["city_name"] == city]
                    .sort_values("issue_time_utc", kind="mergesort")
                    .drop_duplicates(subset=["issue_time_utc"], keep="last")
                )
                ax.set_title(city, fontsize=12)

                if city_day.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                else:
                    x_vals = [issue_time_to_x[ts] for ts in city_day["issue_time_utc"]]
                    y_vals = city_day["forecast_rounded"].tolist()
                    ax.plot(x_vals, y_vals, marker="o", linewidth=1.2, markersize=3.5)

                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
                if x_ticks:
                    ax.set_xlim(-0.5, len(x_ticks) - 0.5)
                ax.tick_params(axis="y", labelsize=8)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(True, axis="y", linestyle=":", alpha=0.35)
                ax.set_xlabel("")
                ax.set_ylabel("")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)
            page_count += 1

    return page_count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    city_dirs = discover_city_dirs(args.input_root)
    available_cycles = discover_cycle_tokens(city_dirs)
    latest_cycle_token = max(available_cycles)
    cycle_token = resolve_cycle_token(args.cycle, available_cycles)
    df, cities = collect_predictions(city_dirs)
    local_days = select_local_days_from_cycle(df=df, cycle_token=cycle_token)
    df_filtered = df[df["target_date_local"].isin(local_days)].copy()
    output_pdf = build_output_pdf_path(args.output_dir, cycle_token)
    pages = render_pdf(df=df_filtered, cities=cities, output_pdf=output_pdf, local_days=local_days)

    print(f"Input root: {args.input_root}")
    print(f"Cities: {', '.join(cities)}")
    print(f"Cycle used: {cycle_token}")
    print(f"Latest cycle: {latest_cycle_token}")
    print(f"Local days from cycle: {len(local_days)}")
    print(f"Pages written: {pages}")
    print(f"Wrote PDF: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
