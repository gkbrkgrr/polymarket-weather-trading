#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class Location:
    city: str
    lat: float
    lon: float
    timezone: str
    elevation_m: float


def _parse_lat_lon(value: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*([0-9.]+)\s*([NS])\s*([0-9.]+)\s*([EW])\s*", value)
    if not match:
        raise ValueError(f"Invalid lat_lon value: {value!r}")
    lat = float(match.group(1)) * (1 if match.group(2) == "N" else -1)
    lon = float(match.group(3)) * (1 if match.group(4) == "E" else -1)
    return lat, lon


def load_locations(path: Path) -> list[Location]:
    locations: list[Location] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            city = (row.get("name") or "").strip()
            lat_lon = (row.get("lat_lon") or "").strip()
            tz = (row.get("timezone") or "").strip()
            elev_ft_str = (row.get("elevation") or "").strip()
            if not city or not lat_lon or not tz or not elev_ft_str:
                continue
            lat, lon = _parse_lat_lon(lat_lon)
            elev_ft = float(elev_ft_str)
            elev_m = elev_ft * 0.3048
            locations.append(
                Location(
                    city=city,
                    lat=lat,
                    lon=lon,
                    timezone=tz,
                    elevation_m=elev_m,
                )
            )
    if not locations:
        raise ValueError(f"No usable rows found in {path}")
    return locations


def round_half_up(value: float) -> int:
    if math.isnan(value):
        raise ValueError("round_half_up received NaN")
    return int(math.floor(value + 0.5 + 1e-12))


def _normalize_lon_for_grid(lon: float, grid_lons: np.ndarray) -> float:
    lon_min = float(np.nanmin(grid_lons))
    lon_max = float(np.nanmax(grid_lons))
    if lon_min >= 0.0 and lon_max > 180.0:
        out = lon % 360.0
        if out < lon_min:
            out += 360.0
        return out
    return lon


def _ensure_ascending(
    coord: np.ndarray, values: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    if coord.size < 2:
        return coord, values
    if coord[0] <= coord[-1]:
        return coord, values
    coord2 = coord[::-1].copy()
    values2 = np.flip(values, axis=axis)
    return coord2, values2


def bilinear_interpolate(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    *,
    lat: float,
    lon: float,
) -> float:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D values, got shape {values.shape}")

    lats, values = _ensure_ascending(lats, values, axis=0)
    lons, values = _ensure_ascending(lons, values, axis=1)

    lon = _normalize_lon_for_grid(lon, lons)

    lat = float(np.clip(lat, lats[0], lats[-1]))
    lon = float(np.clip(lon, lons[0], lons[-1]))

    i1 = int(np.searchsorted(lats, lat, side="right"))
    j1 = int(np.searchsorted(lons, lon, side="right"))
    i0 = max(i1 - 1, 0)
    j0 = max(j1 - 1, 0)
    i1 = min(i1, lats.size - 1)
    j1 = min(j1, lons.size - 1)

    lat0 = float(lats[i0])
    lat1 = float(lats[i1])
    lon0 = float(lons[j0])
    lon1 = float(lons[j1])

    denom_lat = lat1 - lat0
    denom_lon = lon1 - lon0
    w_lat = 0.0 if denom_lat == 0.0 else (lat - lat0) / denom_lat
    w_lon = 0.0 if denom_lon == 0.0 else (lon - lon0) / denom_lon
    w_lat = float(np.clip(w_lat, 0.0, 1.0))
    w_lon = float(np.clip(w_lon, 0.0, 1.0))

    q00 = float(values[i0, j0])
    q01 = float(values[i0, j1])
    q10 = float(values[i1, j0])
    q11 = float(values[i1, j1])

    corners = np.array([q00, q01, q10, q11], dtype=float)
    if np.any(np.isnan(corners)):
        pts = np.array(
            [
                (lat0, lon0),
                (lat0, lon1),
                (lat1, lon0),
                (lat1, lon1),
            ],
            dtype=float,
        )
        d2 = (pts[:, 0] - lat) ** 2 + (pts[:, 1] - lon) ** 2
        for idx in np.argsort(d2):
            if not math.isnan(corners[idx]):
                return float(corners[idx])
        return float("nan")

    return float(
        q00 * (1 - w_lat) * (1 - w_lon)
        + q01 * (1 - w_lat) * w_lon
        + q10 * w_lat * (1 - w_lon)
        + q11 * w_lat * w_lon
    )


def open_field(
    grib_path: Path,
    *,
    short_name: str,
    type_of_level: str,
    level: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filter_by_keys: dict[str, object] = {
        "shortName": short_name,
        "typeOfLevel": type_of_level,
    }
    if level is not None:
        filter_by_keys["level"] = int(level)

    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": filter_by_keys,
            "indexpath": "",
        },
    )
    try:
        if not ds.data_vars:
            raise ValueError(f"No data variables found for {filter_by_keys} in {grib_path}")
        var_name = next(iter(ds.data_vars))
        da = ds[var_name]
        lats = np.asarray(ds["latitude"].values, dtype=float)
        lons = np.asarray(ds["longitude"].values, dtype=float)
        values = np.asarray(da.values, dtype=float)
        return lats, lons, values
    finally:
        ds.close()


def open_pressure_stack(
    grib_path: Path,
    *,
    short_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": short_name, "typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
        },
    )
    try:
        if not ds.data_vars:
            raise ValueError(f"No data variables found for shortName={short_name} in {grib_path}")
        var_name = next(iter(ds.data_vars))
        da = ds[var_name]
        lats = np.asarray(ds["latitude"].values, dtype=float)
        lons = np.asarray(ds["longitude"].values, dtype=float)

        levels = [int(x) for x in np.asarray(ds["isobaricInhPa"].values).tolist()]
        stack: dict[int, np.ndarray] = {}
        for idx, lvl in enumerate(levels):
            stack[lvl] = np.asarray(da.isel(isobaricInhPa=idx).values, dtype=float)
        return lats, lons, stack
    finally:
        ds.close()


def parse_init_and_lead_from_filename(model: str, path: Path) -> tuple[datetime, int]:
    name = path.name
    if model == "gfs":
        match = re.search(r"gfs_(\d{10})_f(\d{3})\.grib2$", name)
        if match:
            init = datetime.strptime(match.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
            lead = int(match.group(2))
            return init, lead
    if model == "ecmwf-hres":
        match = re.search(r"ecmwf-hres-[^_]+_(\d{10})_(\d{3})\.grib2$", name)
        if match:
            init = datetime.strptime(match.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
            lead = int(match.group(2))
            return init, lead
    if model == "ecmwf-aifs-single":
        match = re.search(r"ecmwf-aifs-single-[^_]+_(\d{10})_(\d{3})\.grib2$", name)
        if match:
            init = datetime.strptime(match.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
            lead = int(match.group(2))
            return init, lead

    match_init = re.search(r"(\d{10})", name)
    match_lead = re.search(r"(?:_f|_)(\d{3})(?=\\.grib2$)", name)
    if not match_init or not match_lead:
        raise ValueError(f"Could not parse init/lead from filename: {name}")
    init = datetime.strptime(match_init.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
    lead = int(match_lead.group(1))
    return init, lead


def station_temperature(
    *,
    t2m: float,
    station_elev_m: float,
    t_by_level: dict[int, float],
    gh_by_level: dict[int, float],
) -> float:
    if math.isnan(t2m):
        return float("nan")

    z0 = 2.0
    if station_elev_m <= z0:
        return t2m

    for lvl in (1000, 925, 850, 700):
        t = t_by_level.get(lvl, float("nan"))
        gh = gh_by_level.get(lvl, float("nan"))
        if math.isnan(t) or math.isnan(gh):
            continue
        if gh <= station_elev_m:
            continue
        if gh <= z0:
            continue
        ratio = (station_elev_m - z0) / (gh - z0)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return t2m + (t - t2m) * ratio

    return t2m


def download_cycle(*, model: str, cycle: str, max_step_hours: int, step_interval_hours: int) -> None:
    if model == "ecmwf-hres":
        script = REPO_ROOT / "data_gatherer" / "ecmwf_forecast_gatherer" / "ecmwf_forecast_download.py"
        cmd = [
            sys.executable,
            str(script),
            "--cycle",
            cycle,
            "--model",
            "ecmwf-hres",
            "--max-step-hours",
            str(max_step_hours),
            "--step-interval-hours",
            str(step_interval_hours),
        ]
    elif model == "ecmwf-aifs-single":
        script = REPO_ROOT / "data_gatherer" / "ecmwf_forecast_gatherer" / "ecmwf_forecast_download.py"
        cmd = [
            sys.executable,
            str(script),
            "--cycle",
            cycle,
            "--model",
            "ecmwf-aifs-single",
            "--max-step-hours",
            str(max_step_hours),
            "--step-interval-hours",
            str(step_interval_hours),
        ]
    elif model == "gfs":
        script = REPO_ROOT / "data_gatherer" / "gfs_forecast_gatherer" / "gfs_forecast_download.py"
        cmd = [
            sys.executable,
            str(script),
            "--cycle",
            cycle,
            "--max-step-hours",
            str(max_step_hours),
            "--step-interval-hours",
            str(step_interval_hours),
        ]
    else:
        raise ValueError(f"Unsupported model: {model}")

    subprocess.run(cmd, check=True)


def build_point_records(
    *,
    model: str,
    grib_files: list[Path],
    locations: list[Location],
) -> dict[str, pd.DataFrame]:
    groups: dict[str, list[Path]] = {}
    for path in grib_files:
        init_dt, _lead = parse_init_and_lead_from_filename(model, path)
        init_str = init_dt.strftime("%Y%m%d%H")
        groups.setdefault(init_str, []).append(path)

    out: dict[str, pd.DataFrame] = {}
    for init_str, paths in groups.items():
        rows: list[dict[str, object]] = []
        for grib_path in sorted(paths):
            init_dt, lead_hour = parse_init_and_lead_from_filename(model, grib_path)
            valid_dt_utc = init_dt + timedelta(hours=lead_hour)

            lats2, lons2, t2_field = open_field(
                grib_path, short_name="2t", type_of_level="heightAboveGround", level=2
            )
            lats_t, lons_t, t_stack = open_pressure_stack(grib_path, short_name="t")
            gh_short = "z" if model == "ecmwf-aifs-single" else "gh"
            lats_gh, lons_gh, gh_stack_raw = open_pressure_stack(grib_path, short_name=gh_short)

            if model == "ecmwf-aifs-single":
                gh_stack: dict[int, np.ndarray] = {}
                for lvl, z_grid in gh_stack_raw.items():
                    gh_stack[lvl] = np.divide(z_grid, GRAVITY_M_S2, dtype=float)
            else:
                gh_stack = gh_stack_raw

            if not (np.array_equal(lats2, lats_t) and np.array_equal(lats2, lats_gh)):
                raise ValueError(f"Latitude grids differ in {grib_path}")
            if not (np.array_equal(lons2, lons_t) and np.array_equal(lons2, lons_gh)):
                raise ValueError(f"Longitude grids differ in {grib_path}")

            for loc in locations:
                t2m = bilinear_interpolate(lats2, lons2, t2_field, lat=loc.lat, lon=loc.lon)
                t_by_level: dict[int, float] = {}
                gh_by_level: dict[int, float] = {}
                for lvl in (1000, 925, 850, 700):
                    t_grid = t_stack.get(lvl)
                    if t_grid is None:
                        t_grid = np.full_like(t2_field, np.nan, dtype=float)
                    gh_grid = gh_stack.get(lvl)
                    if gh_grid is None:
                        gh_grid = np.full_like(t2_field, np.nan, dtype=float)
                    t_by_level[lvl] = bilinear_interpolate(
                        lats2, lons2, t_grid, lat=loc.lat, lon=loc.lon
                    )
                    gh_by_level[lvl] = bilinear_interpolate(
                        lats2, lons2, gh_grid, lat=loc.lat, lon=loc.lon
                    )

                t_station = station_temperature(
                    t2m=t2m,
                    station_elev_m=loc.elevation_m,
                    t_by_level=t_by_level,
                    gh_by_level=gh_by_level,
                )

                tzinfo = ZoneInfo(loc.timezone)
                valid_local = valid_dt_utc.astimezone(tzinfo).isoformat()

                def _rounded_or_none(v: float) -> int | None:
                    if math.isnan(v):
                        return None
                    return round_half_up(v)

                rows.append(
                    {
                        "City": loc.city,
                        "InitTimeUTC": pd.Timestamp(init_dt),
                        "ValidTimeUTC": pd.Timestamp(valid_dt_utc),
                        "ValidTimeLocal": valid_local,
                        "Timezone": loc.timezone,
                        "LeadHour": int(lead_hour),
                        "StationElevation": float(loc.elevation_m),
                        "Temperature2m": _rounded_or_none(t2m),
                        "Temperature1000mb": _rounded_or_none(t_by_level[1000]),
                        "Temperature925mb": _rounded_or_none(t_by_level[925]),
                        "Temperature850mb": _rounded_or_none(t_by_level[850]),
                        "Temperature700mb": _rounded_or_none(t_by_level[700]),
                        "TemperatureStation": _rounded_or_none(t_station),
                    }
                )

        df = pd.DataFrame.from_records(rows)
        df["InitTimeUTC"] = pd.to_datetime(df["InitTimeUTC"], utc=True)
        df["ValidTimeUTC"] = pd.to_datetime(df["ValidTimeUTC"], utc=True)

        for col in (
            "Temperature2m",
            "Temperature1000mb",
            "Temperature925mb",
            "Temperature850mb",
            "Temperature700mb",
            "TemperatureStation",
        ):
            df[col] = df[col].astype("Int64")

        df["LeadHour"] = df["LeadHour"].astype("int32")
        df = df[
            [
                "City",
                "InitTimeUTC",
                "ValidTimeUTC",
                "ValidTimeLocal",
                "Timezone",
                "LeadHour",
                "StationElevation",
                "Temperature2m",
                "Temperature1000mb",
                "Temperature925mb",
                "Temperature850mb",
                "Temperature700mb",
                "TemperatureStation",
            ]
        ].sort_values(["City", "ValidTimeUTC"], kind="mergesort")
        out[init_str] = df
    return out


def find_existing_parquet(out_dir: Path, *, model: str, init_str: str) -> Path | None:
    candidates = list(out_dir.glob(f"{model}_{init_str}_*.parquet"))
    if not candidates:
        return None
    best: tuple[str, Path] | None = None
    for path in candidates:
        match = re.fullmatch(rf"{re.escape(model)}_{init_str}_(\d{{10}})\.parquet", path.name)
        if not match:
            continue
        end = match.group(1)
        if best is None or end > best[0]:
            best = (end, path)
    return best[1] if best else None


def write_or_append_parquet(
    *,
    model: str,
    init_str: str,
    df_new: pd.DataFrame,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = find_existing_parquet(out_dir, model=model, init_str=init_str)
    if existing:
        df_old = pd.read_parquet(existing)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df["InitTimeUTC"] = pd.to_datetime(df["InitTimeUTC"], utc=True)
        df["ValidTimeUTC"] = pd.to_datetime(df["ValidTimeUTC"], utc=True)
    else:
        df = df_new

    df = df.drop_duplicates(subset=["City", "ValidTimeUTC"], keep="last")
    df = df.sort_values(["City", "ValidTimeUTC"], kind="mergesort")

    for col in (
        "Temperature2m",
        "Temperature1000mb",
        "Temperature925mb",
        "Temperature850mb",
        "Temperature700mb",
        "TemperatureStation",
    ):
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    end_str = df["ValidTimeUTC"].max().to_pydatetime().strftime("%Y%m%d%H")
    out_path = out_dir / f"{model}_{init_str}_{end_str}.parquet"
    tmp_path = out_path.with_suffix(".parquet.part")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)

    if existing and existing != out_path:
        existing.unlink(missing_ok=True)
    return out_path


def default_grib_dir(model: str, cycle: str) -> Path:
    if model not in {"ecmwf-hres", "ecmwf-aifs-single", "gfs"}:
        raise ValueError("--model must be one of: ecmwf-hres, ecmwf-aifs-single, gfs")
    return REPO_ROOT / "data" / "raster_data" / model / cycle


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Bilinearly interpolate GRIB2 fields (2t/t/gh or 2t/t/z) onto locations.csv points, "
            "then write/append a parquet in data/point_data/<model>/raw/."
        )
    )
    p.add_argument("--model", required=True, choices=["ecmwf-hres", "ecmwf-aifs-single", "gfs"])
    p.add_argument("--cycle", help="Init time (UTC) in YYYYMMDDHH; also used for default GRIB dir")
    p.add_argument("--download", action="store_true", help="Download GRIB2 files for --cycle first")
    p.add_argument("--max-step-hours", type=int, default=96)
    p.add_argument("--step-interval-hours", type=int, default=3)
    p.add_argument("--locations-csv", default=str(REPO_ROOT / "locations.csv"))
    p.add_argument("--grib-dir", help="Directory containing per-step .grib2 files")
    p.add_argument("--grib-files", nargs="*", help="Explicit list of .grib2 files")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.model in {"ecmwf-hres", "ecmwf-aifs-single", "gfs"}:
        conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        running_in_mto = Path(sys.prefix).name == "mto" or "/envs/mto" in sys.prefix
        if conda_env and conda_env != "mto" and not running_in_mto:
            print(
                f"Warning: expected conda env 'mto', got {conda_env!r} (sys.executable={sys.executable})",
                file=sys.stderr,
            )

    if args.download:
        if not args.cycle:
            raise SystemExit("--download requires --cycle")
        download_cycle(
            model=args.model,
            cycle=args.cycle,
            max_step_hours=int(args.max_step_hours),
            step_interval_hours=int(args.step_interval_hours),
        )

    grib_files: list[Path] = []
    if args.grib_files:
        grib_files.extend(Path(p).expanduser().resolve() for p in args.grib_files)
    if args.grib_dir:
        grib_dir = Path(args.grib_dir).expanduser().resolve()
        grib_files.extend(sorted(grib_dir.glob("*.grib2")))
    if not grib_files:
        if not args.cycle:
            raise SystemExit("Provide --cycle, or --grib-dir, or --grib-files")
        grib_dir = default_grib_dir(args.model, args.cycle)
        grib_files = sorted(grib_dir.glob("*.grib2"))

    grib_files = [p for p in grib_files if p.exists()]
    if not grib_files:
        raise SystemExit("No GRIB2 files found")

    if args.verbose:
        init_leads: list[tuple[str, int, str]] = []
        for pth in grib_files:
            init_dt, lead = parse_init_and_lead_from_filename(args.model, pth)
            init_leads.append((init_dt.strftime("%Y%m%d%H"), lead, pth.name))
        inits = sorted({x[0] for x in init_leads})
        leads = sorted({x[1] for x in init_leads})
        print(f"Found {len(grib_files)} GRIB2 files")
        print(f"Init cycles: {', '.join(inits)}")
        if leads:
            print(f"LeadHour min/max: {leads[0]}/{leads[-1]} (count={len(leads)})")

    locations = load_locations(Path(args.locations_csv).expanduser().resolve())
    dfs = build_point_records(model=args.model, grib_files=grib_files, locations=locations)

    out_root = REPO_ROOT / "data" / "point_data" / args.model / "raw"
    for init_str, df in sorted(dfs.items()):
        out_path = write_or_append_parquet(
            model=args.model,
            init_str=init_str,
            df_new=df,
            out_dir=out_root,
        )
        if args.verbose:
            steps = df["ValidTimeUTC"].nunique()
            cities = df["City"].nunique()
            print(f"Init {init_str}: steps={steps}, cities={cities}, rows={len(df)}")
        print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
