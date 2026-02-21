#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "ml_predictions"
DEFAULT_OBS_ROOT = REPO_ROOT / "data" / "observations"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "error_heatmaps"
MODEL_ORDER = ["xgb", "xgb_opt", "xgb_biascorrected", "xgb_opt_biascorrected"]
TOKEN_RE = re.compile(r"(\d{10})(?=\.parquet$)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a per-city PDF report with two heatmaps: mean BIAS and mean MAE "
            "by lead time and model."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root containing model directories (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--obs-root",
        type=Path,
        default=DEFAULT_OBS_ROOT,
        help=f"Directory containing city observation parquet files (default: {DEFAULT_OBS_ROOT})",
    )
    parser.add_argument(
        "--date-prefix",
        type=str,
        default=None,
        help="Optional cycle prefix filter (YYYYMMDD or YYYYMMDDHH). If omitted, all files are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for PDF reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def _cycle_token_from_name(path: Path) -> str | None:
    match = TOKEN_RE.search(path.name)
    if match is None:
        return None
    return match.group(1)


def _resolve_prediction_column(model_name: str, columns: pd.Index) -> str:
    available = set(columns)
    if model_name.endswith("_biascorrected") and "tmax_corrected" in available:
        return "tmax_corrected"
    if "Forecast" in available:
        return "Forecast"
    raise ValueError(
        f"Could not find prediction column in model={model_name}. "
        "Expected one of: tmax_corrected, Forecast"
    )


def load_model_predictions(
    *,
    input_root: Path,
    model_name: str,
    date_prefix: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    model_dir = input_root / model_name
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    files: list[Path] = []
    tokens: list[str] = []
    for file_path in sorted(model_dir.glob("*/*.parquet")):
        token = _cycle_token_from_name(file_path)
        if token is None:
            continue
        if date_prefix is not None and not token.startswith(date_prefix):
            continue
        files.append(file_path)
        tokens.append(token)

    if not files:
        if date_prefix is None:
            raise SystemExit(f"No parquet files found for model={model_name} under {model_dir}")
        raise SystemExit(
            f"No parquet files found for model={model_name} with date prefix {date_prefix} under {model_dir}"
        )

    parts: list[pd.DataFrame] = []
    for file_path in files:
        raw = pd.read_parquet(file_path)
        pred_col = _resolve_prediction_column(model_name, raw.columns)

        city_col = "city_name" if "city_name" in raw.columns else None
        if city_col is None:
            raw["city_name"] = file_path.parent.name

        required = ["city_name", "issue_time_utc", "target_date_local", "lead_time_hours", pred_col]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise SystemExit(f"Missing required columns in {file_path}: {', '.join(missing)}")

        part = raw[required].copy()
        part = part.rename(columns={pred_col: "prediction"})
        part["model_name"] = model_name
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df["issue_time_utc"] = pd.to_datetime(df["issue_time_utc"], utc=True, errors="coerce")
    df["target_date_local"] = pd.to_datetime(df["target_date_local"], errors="coerce").dt.normalize()
    df["lead_time_hours"] = pd.to_numeric(df["lead_time_hours"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(
        subset=["city_name", "issue_time_utc", "target_date_local", "lead_time_hours", "prediction"]
    ).copy()
    return df, tokens


def load_daily_observations(obs_root: Path) -> pd.DataFrame:
    if not obs_root.exists():
        raise SystemExit(f"Observation directory not found: {obs_root}")

    obs_files = sorted(obs_root.glob("*.parquet"))
    if not obs_files:
        raise SystemExit(f"No observation parquet files found under: {obs_root}")

    parts: list[pd.DataFrame] = []
    for obs_path in obs_files:
        city_name = obs_path.stem
        raw = pd.read_parquet(obs_path)
        cols = {str(c).lower(): str(c) for c in raw.columns}

        if "temperature_c" in cols:
            temp_c = pd.to_numeric(raw[cols["temperature_c"]], errors="coerce")
        elif "temp_c" in cols:
            temp_c = pd.to_numeric(raw[cols["temp_c"]], errors="coerce")
        elif "temperature_f" in cols:
            temp_f = pd.to_numeric(raw[cols["temperature_f"]], errors="coerce")
            temp_c = (temp_f - 32.0) * (5.0 / 9.0)
        elif "temp_f" in cols:
            temp_f = pd.to_numeric(raw[cols["temp_f"]], errors="coerce")
            temp_c = (temp_f - 32.0) * (5.0 / 9.0)
        else:
            continue

        if "target_date_local" in cols:
            target_date = pd.to_datetime(raw[cols["target_date_local"]], errors="coerce").dt.normalize()
        elif "target_date" in cols:
            target_date = pd.to_datetime(raw[cols["target_date"]], errors="coerce").dt.normalize()
        elif "observed_at_local" in cols:
            text = raw[cols["observed_at_local"]].astype("string")
            target_date = pd.to_datetime(text.str.slice(0, 10), errors="coerce").dt.normalize()
        else:
            continue

        city_obs = pd.DataFrame(
            {
                "city_name": city_name,
                "target_date_local": target_date,
                "observation": temp_c,
            }
        )
        city_obs = city_obs.dropna(subset=["target_date_local", "observation"])
        if city_obs.empty:
            continue
        daily = city_obs.groupby(["city_name", "target_date_local"], as_index=False)["observation"].max()
        parts.append(daily)

    if not parts:
        raise SystemExit(f"No usable observation rows were parsed from: {obs_root}")

    out = pd.concat(parts, ignore_index=True)
    out = out.groupby(["city_name", "target_date_local"], as_index=False)["observation"].max()
    return out


def _render_single_heatmap(
    *,
    ax,
    metric_grid: pd.DataFrame,
    count_grid: pd.DataFrame,
    title: str,
    cmap: str,
    center_zero: bool,
) -> None:
    y_labels = [str(int(x)) for x in metric_grid.index.tolist()]
    x_labels = metric_grid.columns.tolist()
    values = metric_grid.to_numpy(dtype=float)
    counts = count_grid.to_numpy(dtype=float)
    finite = np.isfinite(values)

    if center_zero:
        max_abs = float(np.nanmax(np.abs(values[finite]))) if finite.any() else 1.0
        if max_abs <= 0.0:
            max_abs = 1.0
        ax.imshow(values, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect="auto")
    else:
        vmax = float(np.nanmax(values[finite])) if finite.any() else 1.0
        if vmax <= 0.0:
            vmax = 1.0
        ax.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            count = counts[row_idx, col_idx]
            if not np.isfinite(value) or not np.isfinite(count):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f},{int(count)}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )


def render_city_page(
    *,
    bias_grid: pd.DataFrame,
    bias_count_grid: pd.DataFrame,
    mae_grid: pd.DataFrame,
    mae_count_grid: pd.DataFrame,
    city_name: str,
    range_start: str,
    range_end: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    bias_title = f"Mean BIAS by Lead Times for {city_name}\nRange: {range_start} - {range_end}"
    mae_title = f"Mean MAE by Lead Times for {city_name}\nRange: {range_start} - {range_end}"
    _render_single_heatmap(
        ax=axes[0],
        metric_grid=bias_grid,
        count_grid=bias_count_grid,
        title=bias_title,
        cmap="RdBu_r",
        center_zero=True,
    )
    _render_single_heatmap(
        ax=axes[1],
        metric_grid=mae_grid,
        count_grid=mae_count_grid,
        title=mae_title,
        cmap="YlOrRd",
        center_zero=False,
    )
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.date_prefix is not None and re.fullmatch(r"\d{8,10}", args.date_prefix) is None:
        raise SystemExit(
            f"--date-prefix must be digits in YYYYMMDD or YYYYMMDDHH style, got: {args.date_prefix!r}"
        )

    all_frames: list[pd.DataFrame] = []
    all_tokens: list[str] = []
    for model_name in MODEL_ORDER:
        model_df, model_tokens = load_model_predictions(
            input_root=args.input_root,
            model_name=model_name,
            date_prefix=args.date_prefix,
        )
        all_frames.append(model_df)
        all_tokens.extend(model_tokens)

    pred_df = pd.concat(all_frames, ignore_index=True)
    obs_df = load_daily_observations(args.obs_root)

    merged = pred_df.merge(obs_df, on=["city_name", "target_date_local"], how="left")
    merged = merged.dropna(subset=["observation"]).copy()
    if merged.empty:
        raise SystemExit("No joined prediction/observation rows available after merge.")

    merged["error"] = merged["prediction"] - merged["observation"]
    merged["abs_error"] = merged["error"].abs()
    merged["lead_time_hours"] = merged["lead_time_hours"].round().astype(int)

    issue_times = pd.to_datetime(merged["issue_time_utc"], utc=True, errors="coerce").dropna()
    if issue_times.empty:
        if not all_tokens:
            raise SystemExit("Could not determine cycle range for output naming/title.")
        range_start = min(all_tokens)
        range_end = max(all_tokens)
    else:
        range_start = issue_times.min().strftime("%Y%m%d%H")
        range_end = issue_times.max().strftime("%Y%m%d%H")

    output_path = args.output_dir / f"model_performances_{range_start}{range_end}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages_written = 0
    cities = sorted(merged["city_name"].astype(str).dropna().unique().tolist())
    with PdfPages(output_path) as pdf:
        for city_name in cities:
            city_rows = merged[merged["city_name"] == city_name].copy()
            if city_rows.empty:
                continue

            grouped = (
                city_rows.groupby(["lead_time_hours", "model_name"], as_index=False)
                .agg(
                    mean_bias=("error", "mean"),
                    mean_mae=("abs_error", "mean"),
                    sample_count=("error", "count"),
                )
                .sort_values(["lead_time_hours", "model_name"], kind="mergesort")
            )
            bias_grid = grouped.pivot(index="lead_time_hours", columns="model_name", values="mean_bias")
            bias_grid = bias_grid.reindex(columns=MODEL_ORDER).sort_index()
            mae_grid = grouped.pivot(index="lead_time_hours", columns="model_name", values="mean_mae")
            mae_grid = mae_grid.reindex(columns=MODEL_ORDER).sort_index()
            count_grid = grouped.pivot(index="lead_time_hours", columns="model_name", values="sample_count")
            count_grid = count_grid.reindex(columns=MODEL_ORDER).sort_index()
            bias_count_grid = count_grid.reindex(index=bias_grid.index, columns=bias_grid.columns)
            mae_count_grid = count_grid.reindex(index=mae_grid.index, columns=mae_grid.columns)

            if bias_grid.empty or mae_grid.empty:
                continue

            fig = render_city_page(
                bias_grid=bias_grid,
                bias_count_grid=bias_count_grid,
                mae_grid=mae_grid,
                mae_count_grid=mae_count_grid,
                city_name=city_name,
                range_start=range_start,
                range_end=range_end,
            )
            pdf.savefig(fig)
            plt.close(fig)
            pages_written += 1

    print(f"Date prefix: {args.date_prefix if args.date_prefix is not None else 'ALL'}")
    print(f"Rows used: {len(merged)}")
    print(f"Cities: {len(cities)}")
    print(f"Pages written: {pages_written}")
    print(f"Range: {range_start} - {range_end}")
    print(f"Wrote PDF: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
