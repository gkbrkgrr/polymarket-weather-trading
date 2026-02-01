import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_forecast_daily_max(model_path: Path, us_cities: set[str]) -> pd.DataFrame:
    files = sorted(model_path.glob("*.parquet"))
    if not files:
        return pd.DataFrame(
            columns=["City", "date_str", "horizon_hours", "forecast_max"]
        )

    frames = []
    for f in files:
        df = pd.read_parquet(
            f,
            columns=["City", "InitTimeUTC", "ValidTimeLocal", "TemperatureStation"],
        )
        frames.append(df)

    forecast = pd.concat(frames, ignore_index=True)
    forecast["InitTimeUTC"] = pd.to_datetime(forecast["InitTimeUTC"], utc=True)

    forecast["local_date"] = pd.to_datetime(forecast["ValidTimeLocal"].str.slice(0, 10))
    offset_str = forecast["ValidTimeLocal"].str.slice(-6)
    offset_sign = offset_str.str[0].map({"+": 1, "-": -1})
    offset_hours = offset_str.str.slice(1, 3).astype(int)
    offset_minutes = offset_str.str.slice(4, 6).astype(int)
    forecast["offset_minutes"] = offset_sign * (offset_hours * 60 + offset_minutes)

    daily_fcst = (
        forecast.groupby(
            ["City", "InitTimeUTC", "local_date", "offset_minutes"], as_index=False
        )["TemperatureStation"]
        .max()
    )

    local_day_end = (
        daily_fcst["local_date"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )
    local_day_end_utc = local_day_end - pd.to_timedelta(
        daily_fcst["offset_minutes"], unit="m"
    )
    local_day_end_utc = pd.to_datetime(local_day_end_utc, utc=True)

    daily_fcst["horizon_hours"] = (
        (local_day_end_utc - daily_fcst["InitTimeUTC"])
        .dt.total_seconds()
        .div(3600)
        .round()
        .astype(int)
    )

    def convert_temp_k(row: pd.Series) -> float:
        if row["City"] in us_cities:
            return (row["TemperatureStation"] - 273.15) * 9 / 5 + 32
        return row["TemperatureStation"] - 273.15

    daily_fcst["forecast_max"] = daily_fcst.apply(convert_temp_k, axis=1)
    daily_fcst["date_str"] = daily_fcst["local_date"].dt.strftime("%Y%m%d")

    return daily_fcst[["City", "date_str", "horizon_hours", "forecast_max"]]


def main() -> None:
    locations = pd.read_csv("locations.csv")
    cities = locations["name"].tolist()
    us_cities = set(locations[locations["url"].str.contains("/us/")]["name"])

    models = {
        "gfs": Path("data/point_data/gfs/raw"),
        "ecmwf-hres": Path("data/point_data/ecmwf-hres/raw"),
        "ecmwf-aifs-single": Path("data/point_data/ecmwf-aifs-single/raw"),
    }

    obs_frames = []
    for city in cities:
        path = Path("data/observations") / f"{city}.parquet"
        if not path.exists():
            continue
        obs = pd.read_parquet(path)
        obs["local_date"] = pd.to_datetime(obs["observed_at_local"].str.slice(0, 10))

        temp_col = "temperature_f" if city in us_cities else "temperature_c"
        obs = obs[["local_date", temp_col]].rename(columns={temp_col: "observed_temp"})
        daily_obs = obs.groupby("local_date", as_index=False)["observed_temp"].max()
        daily_obs["City"] = city
        daily_obs["date_str"] = daily_obs["local_date"].dt.strftime("%Y%m%d")
        obs_frames.append(daily_obs[["City", "date_str", "observed_temp"]])

    daily_obs = pd.concat(obs_frames, ignore_index=True)

    all_bias_frames = []
    model_bias = {}
    for model_name, model_path in models.items():
        daily_fcst = load_forecast_daily_max(model_path, us_cities)
        merged = daily_fcst.merge(daily_obs, on=["City", "date_str"], how="inner")
        merged["bias"] = merged["forecast_max"] - merged["observed_temp"]
        merged["model"] = model_name
        model_bias[model_name] = merged
        all_bias_frames.append(merged)

    all_bias = pd.concat(all_bias_frames, ignore_index=True)
    if all_bias.empty:
        raise RuntimeError("No bias data available to plot.")

    min_date = all_bias["date_str"].min()
    max_date = all_bias["date_str"].max()

    analysis_dir = Path("data/analysis_data/daily_bias_raw")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = (
        analysis_dir
        / f"temperature_station_daily_biases_raw_{min_date}_{max_date}.parquet"
    )
    all_bias.to_parquet(parquet_path, index=False)

    output_dir = Path("outputs/daily_biases_raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"temperature_station_daily_biases_raw_{max_date}.pdf"

    with PdfPages(pdf_path) as pdf:
        for city in cities:
            pivots = []
            max_cols = 0
            max_rows = 0
            for model_name in models:
                city_data = model_bias[model_name]
                city_data = city_data[city_data["City"] == city]
                if city_data.empty:
                    pivots.append(None)
                    continue
                pivot = city_data.pivot_table(
                    index="horizon_hours",
                    columns="date_str",
                    values="bias",
                    aggfunc="mean",
                ).sort_index(ascending=False)
                pivots.append(pivot)
                max_cols = max(max_cols, pivot.shape[1])
                max_rows = max(max_rows, pivot.shape[0])

            fig_width = max(10, 0.4 * max_cols)
            fig_height = max(8, 0.35 * max_rows * 3)
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(fig_width, fig_height))

            for ax, (model_name, pivot) in zip(axes, zip(models.keys(), pivots)):
                if pivot is None or pivot.empty:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {city} ({model_name})",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    continue

                data = pivot.values
                data_masked = np.ma.masked_invalid(data)
                abs_max = np.nanmax(np.abs(data))
                if not np.isfinite(abs_max) or abs_max == 0:
                    abs_max = 1.0

                cmap = plt.get_cmap("coolwarm").copy()
                cmap.set_bad(color="lightgrey")

                im = ax.imshow(
                    data_masked,
                    aspect="auto",
                    cmap=cmap,
                    vmin=-abs_max,
                    vmax=abs_max,
                )

                ax.set_title(f"{city} {model_name} raw")
                ax.set_xlabel("")
                ax.set_ylabel("Horizon (hours)")

                ax.set_xticks(np.arange(pivot.shape[1]))
                ax.set_xticklabels(pivot.columns, rotation=90)
                ax.set_yticks(np.arange(pivot.shape[0]))
                ax.set_yticklabels(pivot.index)
                ax.tick_params(axis="x", labelsize=7)
                ax.tick_params(axis="y", labelsize=7)

                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        value = pivot.iat[i, j]
                        if pd.isna(value):
                            continue
                        label = f"{value:.0f}"
                        ax.text(
                            j,
                            i,
                            label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="black",
                        )

                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved PDF to {pdf_path}")
    print(f"Saved bias data to {parquet_path}")


if __name__ == "__main__":
    main()
