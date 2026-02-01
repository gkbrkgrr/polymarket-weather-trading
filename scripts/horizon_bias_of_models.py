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
    model_labels = {
        "gfs": "GFS",
        "ecmwf-hres": "HRES",
        "ecmwf-aifs-single": "AIFS-SINGLE",
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

    if not obs_frames:
        raise RuntimeError("No observation data available to compute bias.")

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

    max_date = all_bias["date_str"].max()

    output_dir = Path("outputs/horizon_biases_raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"temperature_station_horizon_biases_raw_{max_date}.pdf"

    city_pivots = []
    for city in cities:
        city_frames = []
        for model_name, model_data in model_bias.items():
            city_data = model_data[model_data["City"] == city]
            if city_data.empty:
                continue
            summary = city_data.groupby("horizon_hours", as_index=False)["bias"].mean()
            summary["model"] = model_name
            city_frames.append(summary)

        if not city_frames:
            city_pivots.append((city, None))
            continue

        city_summary = pd.concat(city_frames, ignore_index=True)
        pivot = city_summary.pivot(index="horizon_hours", columns="model", values="bias")
        pivot = pivot.reindex(columns=list(models.keys()))
        pivot = pivot.sort_index(ascending=False)
        city_pivots.append((city, pivot))

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="lightgrey")

    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(city_pivots), 9):
            chunk = city_pivots[start : start + 9]
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 14))
            axes_list = axes.ravel().tolist()

            abs_values = []
            for _, pivot in chunk:
                if pivot is None or pivot.empty:
                    continue
                abs_values.append(np.nanmax(np.abs(pivot.values)))
            abs_max = np.nanmax(abs_values) if abs_values else np.nan
            if not np.isfinite(abs_max) or abs_max == 0:
                abs_max = 1.0

            im = None
            for ax, (city, pivot) in zip(axes_list, chunk):
                if pivot is None or pivot.empty:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {city}",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    continue

                data_masked = np.ma.masked_invalid(pivot.values)
                im = ax.imshow(
                    data_masked,
                    aspect="auto",
                    cmap=cmap,
                    vmin=-abs_max,
                    vmax=abs_max,
                )

                ax.set_title(f"{city}")
                ax.set_ylabel("Horizon (hours)")

                ax.set_xticks(np.arange(pivot.shape[1]))
                ax.set_xticklabels([model_labels[m] for m in pivot.columns], rotation=0)
                ax.set_yticks(np.arange(pivot.shape[0]))
                ax.set_yticklabels(pivot.index)
                ax.tick_params(axis="x", labelsize=9)
                ax.tick_params(axis="y", labelsize=8)

                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        value = pivot.iat[i, j]
                        if pd.isna(value):
                            continue
                        ax.text(
                            j,
                            i,
                            f"{value:.0f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black",
                        )

            for ax in axes_list[len(chunk) :]:
                ax.axis("off")

            if im is not None:
                fig.colorbar(im, ax=axes_list, shrink=0.9)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    main()
