from __future__ import annotations

import argparse
import re
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""} and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from weather_data.grib_reader import GribXarrayReader, derive_rh2m, group_grib2_paths_by_step, normalize_param


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError("This pipeline requires pandas. Install it with: pip install pandas") from e
    return pd


_LAT_LON_RE = re.compile(
    r"^\s*(?P<lat>\d+(?:\.\d+)?)(?P<lat_hem>[NS])\s*(?P<lon>\d+(?:\.\d+)?)(?P<lon_hem>[EW])\s*$",
    re.IGNORECASE,
)

_PARQUET_WRITE_LOCK = threading.Lock()


def parse_lat_lon(lat_lon: str) -> tuple[float, float]:
    """
    Parse coordinates like "51.51N0.03E" into (lat, lon) floats.
    """
    m = _LAT_LON_RE.match(str(lat_lon))
    if not m:
        raise ValueError(f"Unsupported lat_lon format: {lat_lon!r} (expected like '51.51N0.03E')")
    lat = float(m.group("lat")) * (1.0 if m.group("lat_hem").upper() == "N" else -1.0)
    lon = float(m.group("lon")) * (1.0 if m.group("lon_hem").upper() == "E" else -1.0)
    return lat, lon


def load_locations_coords(locations_csv: str | Path) -> Any:
    pd = _require_pandas()
    locations_csv = Path(locations_csv).expanduser()
    df = pd.read_csv(locations_csv)
    if "name" not in df.columns or "lat_lon" not in df.columns:
        raise ValueError(f"{str(locations_csv)!r} must have columns: name, lat_lon")

    latlons = df["lat_lon"].map(parse_lat_lon)
    coords = pd.DataFrame(
        {
            "name": df["name"].astype(str),
            "latitude": [x[0] for x in latlons],
            "longitude": [x[1] for x in latlons],
        }
    )
    return coords


_FORECASTED_AT_RE = re.compile(r"(?P<ts>\d{14})")


def forecasted_at_from_grib_filename(path: str | Path) -> "np.datetime64":
    m = _FORECASTED_AT_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"Could not find YYYYMMDDhhmmss timestamp in filename: {str(path)!r}")
    ts = m.group("ts")
    return np.datetime64(f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}")


def _dataset_uses_lon_360(ds: xr.Dataset) -> bool:
    if "longitude" not in ds.coords:
        raise KeyError("Dataset missing 'longitude' coord")
    lon_vals = np.asarray(ds.coords["longitude"].values, dtype=float)
    if lon_vals.size == 0:
        return False
    return bool(np.nanmax(lon_vals) > 180.0)


def _wrap_lon_360(lon: float) -> float:
    return lon % 360.0


def _wrap_lon_180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _reconcile_lons(ds: xr.Dataset, lons: np.ndarray) -> np.ndarray:
    uses_360 = _dataset_uses_lon_360(ds)
    if uses_360:
        return np.asarray([_wrap_lon_360(float(x)) for x in lons], dtype=float)
    return np.asarray([_wrap_lon_180(float(x)) for x in lons], dtype=float)


def _valid_time_from_dataset(ds: xr.Dataset) -> "np.datetime64":
    for key in ("valid_time", "validTime"):
        if key in ds.coords or key in ds:
            v = np.asarray(ds[key].values).reshape(-1)[0]
            return np.datetime64(v)
    if "time" in ds.coords and "step" in ds.coords:
        v = np.asarray(ds.coords["time"].values + ds.coords["step"].values).reshape(-1)[0]
        return np.datetime64(v)
    if "time" in ds.coords:
        v = np.asarray(ds.coords["time"].values).reshape(-1)[0]
        return np.datetime64(v)
    raise KeyError("Could not determine valid time from dataset (expected valid_time or time+step)")


def read_grib2_timestep(
    grib_paths: Sequence[str | Path],
    *,
    params: Sequence[str] = ("t2m", "u10", "v10", "tp"),
    member_dim: str = "number",
    data_types: Sequence[str] | None = None,
    filter_by_keys: dict[str, Any] | None = None,
    engine: str = "cfgrib",
) -> xr.Dataset:
    """
    Read one forecast timestep across one or more GRIB2 files and return an xarray.Dataset.
    """
    if not grib_paths:
        raise ValueError("grib_paths cannot be empty")
    if engine != "cfgrib":
        raise ValueError("Only engine='cfgrib' is supported")

    reader = GribXarrayReader()
    paths = [Path(p).expanduser().resolve() for p in grib_paths]

    if data_types is None:
        products: set[str] = set()
        meta_re = re.compile(
            r"-(?P<step>\d+)h-[a-z0-9-]+-(?P<type>[a-z0-9]+)(?:__params-[a-z0-9-]+)?\.grib2$",
            re.IGNORECASE,
        )
        for p in paths:
            m = meta_re.search(p.name)
            if m:
                products.add(m.group("type").lower())
        if "ef" in products:
            data_types = ("cf", "pf")
        elif "fc" in products:
            data_types = ("fc",)
        elif "ep" in products:
            data_types = ("ep",)
        elif products & {"cf", "pf"}:
            data_types = tuple(sorted(products & {"cf", "pf"}))
        else:
            data_types = ("cf", "pf")

    base = dict(filter_by_keys or {})

    try:
        return reader.read_ensemble_params_from_files(
            paths,
            list(params),
            data_types=tuple(str(x) for x in data_types),
            filter_by_keys=base,
            engine=engine,
            member_dim=member_dim,
        )
    except Exception:
        # Fall back to scanning each file for each param without assuming ensemble dataTypes.
        def _ensure_member_dim(da: xr.DataArray) -> xr.DataArray:
            if member_dim in da.dims:
                return da
            if member_dim in da.coords:
                value = da.coords[member_dim].values
                try:
                    value_i = int(value)
                except Exception:
                    value_i = 0
                da = da.reset_coords(member_dim, drop=True)
                return da.expand_dims({member_dim: [value_i]})
            return da.expand_dims({member_dim: [0]})

        out: dict[str, xr.DataArray] = {}
        for display_name in params:
            display_name_s = str(display_name)
            if display_name_s.strip().lower() == "rh2m":
                t2m = None
                d2m = None
                for file in paths:
                    try:
                        t2m = reader.read_param(file, "t2m", filter_by_keys=base, engine=engine)
                        d2m = reader.read_param(file, "d2m", filter_by_keys=base, engine=engine)
                        break
                    except Exception:
                        continue
                if t2m is None or d2m is None:
                    raise KeyError(
                        "rh2m not present as a GRIB param and could not be derived; requires both t2m and d2m."
                    )
                out["rh2m"] = _ensure_member_dim(derive_rh2m(_ensure_member_dim(t2m), _ensure_member_dim(d2m)))
                continue

            short = normalize_param(display_name_s)
            da = None
            for file in paths:
                try:
                    da = reader.read_param(file, short, filter_by_keys=base, engine=engine)
                    break
                except Exception:
                    continue
            if da is None:
                raise KeyError(f"Param {display_name_s!r} not found in any file for timestep")
            out[display_name_s] = _ensure_member_dim(da)
        return xr.Dataset(out)


def build_station_table(
    ds: xr.Dataset,
    coords: Any,
    *,
    source_filename: str | Path,
    value_columns: Sequence[str] | None = None,
    tp_name: str = "tp",
) -> Any:
    """
    Interpolate a timestep dataset onto point locations and return a flat pandas DataFrame.
    """
    pd = _require_pandas()

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise KeyError("Dataset missing latitude/longitude coords")

    required_cols = {"name", "latitude", "longitude"}
    missing = required_cols - set(coords.columns)
    if missing:
        raise ValueError(f"coords is missing required columns: {sorted(missing)}")

    lats = np.asarray(coords["latitude"].values, dtype=float)
    lons = np.asarray(coords["longitude"].values, dtype=float)
    lons = _reconcile_lons(ds, lons)

    lat_da = xr.DataArray(lats, dims=("point",), name="latitude")
    lon_da = xr.DataArray(lons, dims=("point",), name="longitude")

    requested = list(value_columns) if value_columns is not None else list(ds.data_vars.keys())
    requested = [str(c) for c in requested]

    member_dim = "number"
    if member_dim in ds.coords:
        member_vals = np.asarray(ds.coords[member_dim].values).reshape(-1)
    elif member_dim in ds.dims:
        member_vals = np.arange(int(ds.sizes[member_dim]), dtype=int)
    else:
        member_vals = np.asarray([0], dtype=int)
    try:
        member_vals = member_vals.astype(int, copy=False)
    except Exception:
        member_vals = np.asarray([int(x) for x in member_vals], dtype=int)

    point_vals = np.arange(len(lats), dtype=int)

    def _squeeze_singletons(da: xr.DataArray) -> xr.DataArray:
        for dim in list(da.dims):
            if dim in {member_dim, "point"}:
                continue
            if int(da.sizes.get(dim, 0)) == 1:
                da = da.squeeze(dim, drop=True)
        return da

    def _ensure_2d(da: xr.DataArray) -> np.ndarray:
        da = _squeeze_singletons(da)

        if "point" not in da.dims:
            if da.ndim == 0:
                da = da.expand_dims({"point": point_vals})
            else:
                raise ValueError(f"Expected interpolated DataArray to have dim 'point', got dims={da.dims!r}")

        if member_dim not in da.dims:
            da = da.expand_dims({member_dim: [0]})

        if member_dim not in da.coords:
            da = da.assign_coords({member_dim: np.arange(int(da.sizes[member_dim]), dtype=int)})
        if "point" not in da.coords:
            da = da.assign_coords({"point": np.arange(int(da.sizes["point"]), dtype=int)})

        da = da.reindex({member_dim: member_vals, "point": point_vals}, fill_value=np.nan)
        da = da.transpose(member_dim, "point")
        return np.asarray(da.values)

    out = pd.DataFrame(
        {
            member_dim: np.repeat(member_vals, len(point_vals)),
            "point": np.tile(point_vals, len(member_vals)),
        }
    )
    out["city_name"] = out["point"].map(dict(enumerate(coords["name"].tolist())))

    def _select_cell_containing_tp(da_tp: xr.DataArray) -> xr.DataArray:
        lat_vals = np.asarray(da_tp["latitude"].values, dtype=float)
        lon_vals = np.asarray(da_tp["longitude"].values, dtype=float)
        if lat_vals.size < 2 or lon_vals.size < 2:
            return da_tp.sel(latitude=lat_da, longitude=lon_da, method="nearest")

        dlat = float(np.abs(lat_vals[1] - lat_vals[0]))
        dlon = float(np.abs(lon_vals[1] - lon_vals[0]))
        half_dlat = dlat / 2.0
        half_dlon = dlon / 2.0

        lat_desc = bool(lat_vals[0] > lat_vals[-1])
        if lat_desc:
            lat_asc = lat_vals[::-1]
            lat_edges = np.concatenate(
                [
                    [lat_asc[0] - half_dlat],
                    (lat_asc[:-1] + lat_asc[1:]) / 2.0,
                    [lat_asc[-1] + half_dlat],
                ]
            )
            lat_bin = np.searchsorted(lat_edges, lats, side="right") - 1
            lat_bin = np.clip(lat_bin, 0, len(lat_asc) - 1)
            lat_idx = (len(lat_vals) - 1) - lat_bin
        else:
            lat_edges = np.concatenate(
                [
                    [lat_vals[0] - half_dlat],
                    (lat_vals[:-1] + lat_vals[1:]) / 2.0,
                    [lat_vals[-1] + half_dlat],
                ]
            )
            lat_idx = np.searchsorted(lat_edges, lats, side="right") - 1
            lat_idx = np.clip(lat_idx, 0, len(lat_vals) - 1)

        lon_edges = np.concatenate(
            [
                [lon_vals[0] - half_dlon],
                (lon_vals[:-1] + lon_vals[1:]) / 2.0,
                [lon_vals[-1] + half_dlon],
            ]
        )
        lon_idx = np.searchsorted(lon_edges, lons, side="right") - 1
        lon_idx = np.clip(lon_idx, 0, len(lon_vals) - 1)

        lat_centers = xr.DataArray(lat_vals[lat_idx], dims="point")
        lon_centers = xr.DataArray(lon_vals[lon_idx], dims="point")
        return da_tp.sel(latitude=lat_centers, longitude=lon_centers)

    for col in requested:
        if col not in ds.data_vars:
            out[col] = np.nan
            continue
        if col == tp_name:
            da = _select_cell_containing_tp(ds[col]).rename(col)
        else:
            da = ds[col].interp(latitude=lat_da, longitude=lon_da, method="linear").rename(col)
        out[col] = _ensure_2d(da).reshape(-1)

    out["forecasted_at_utc"] = pd.to_datetime(forecasted_at_from_grib_filename(source_filename))
    out["datetime_utc"] = pd.to_datetime(_valid_time_from_dataset(ds))

    for col in requested:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    cols = ["forecasted_at_utc", "datetime_utc", "city_name", member_dim] + [c for c in requested if c in out.columns]
    return out[cols]


def _safe_city_dir_name(city_name: str) -> str:
    city_name = str(city_name).strip()
    if not city_name:
        return "unknown"
    return re.sub(r"[<>:\"/\\\\|?*]", "_", city_name)


def _fmt_yyyymmddhhmm(dt: Any) -> str:
    pd = _require_pandas()
    return pd.Timestamp(dt).strftime("%Y%m%d%H%M")


def write_city_parquets(
    result: Any,
    *,
    base_dir: str | Path = Path("data") / "point_data" / "ecmwf-ensemble",
) -> list[Path]:
    pd = _require_pandas()
    base_dir = Path(base_dir)
    if result is None or len(result) == 0:
        return []

    required = {"city_name", "forecasted_at_utc", "datetime_utc"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"result is missing required columns: {sorted(missing)}")

    df = result.copy()
    df["forecasted_at_utc"] = pd.to_datetime(df["forecasted_at_utc"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    written: list[Path] = []
    for (city_name, forecasted_at_utc), g in df.groupby(["city_name", "forecasted_at_utc"], dropna=False):
        with _PARQUET_WRITE_LOCK:
            city_dir = base_dir / _safe_city_dir_name(str(city_name)) / "raw"
            city_dir.mkdir(parents=True, exist_ok=True)

            forecasted_s = _fmt_yyyymmddhhmm(forecasted_at_utc)
            pattern = f"ecmwf-ensemble_indexes_{forecasted_s}_*.parquet"
            existing = list(city_dir.glob(pattern))

            def _end_key(p: Path) -> str:
                m = re.search(rf"ecmwf-ensemble_indexes_{forecasted_s}_(\\d{{12}})\\.parquet$", p.name)
                return m.group(1) if m else ""

            existing_sorted = sorted(existing, key=_end_key)
            existing_df = None
            if existing_sorted:
                existing_df = pd.read_parquet(existing_sorted[-1])

            g2 = g.copy()
            g2.columns = [str(c) for c in g2.columns]
            g2 = g2.loc[:, ~g2.columns.duplicated()]

            if existing_df is not None:
                existing_df = existing_df.copy()
                existing_df.columns = [str(c) for c in existing_df.columns]
                existing_df = existing_df.loc[:, ~existing_df.columns.duplicated()]
                existing_df = existing_df.reindex(columns=g2.columns)

            if existing_df is None:
                combined = g2
            else:
                # Avoid pandas' internal concat edge cases by concatenating column-wise via numpy.
                combined_cols: dict[str, Any] = {}
                for col in g2.columns:
                    a = existing_df[col].to_numpy()
                    b = g2[col].to_numpy()
                    combined_cols[str(col)] = np.concatenate([a, b], axis=0)
                combined = pd.DataFrame(combined_cols)
            combined["forecasted_at_utc"] = pd.to_datetime(combined["forecasted_at_utc"])
            combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"])

            if "number" in combined.columns:
                combined["number"] = pd.to_numeric(combined["number"], errors="coerce").astype("Int64")

            dedup_subset = [c for c in ["forecasted_at_utc", "datetime_utc", "number"] if c in combined.columns]
            if dedup_subset:
                combined = combined.drop_duplicates(subset=dedup_subset, keep="last")

            sort_cols = [c for c in ["forecasted_at_utc", "datetime_utc", "number"] if c in combined.columns]
            if sort_cols:
                combined = combined.sort_values(sort_cols, kind="mergesort")

            preferred_cols = [c for c in result.columns if c in combined.columns] + [
                c for c in combined.columns if c not in result.columns
            ]
            combined = combined[preferred_cols].reset_index(drop=True)

            end_dt = combined["datetime_utc"].max()
            end_s = _fmt_yyyymmddhhmm(end_dt)
            out_path = city_dir / f"ecmwf-ensemble_indexes_{forecasted_s}_{end_s}.parquet"

            for p in existing:
                if p.resolve() != out_path.resolve() and p.exists():
                    p.unlink()

            combined.to_parquet(out_path, index=False)
            written.append(out_path)

    return written


def process_timestep_gribs_to_parquets(
    grib_paths: Sequence[str | Path],
    *,
    coords: Any,
    params: Sequence[str] = ("t2m", "u10", "v10", "tp"),
    point_output_dir: str | Path = Path("data") / "point_data" / "ecmwf-ensemble",
    data_types: Sequence[str] | None = None,
    member_dim: str = "number",
) -> list[Path]:
    ds = read_grib2_timestep(
        grib_paths,
        params=params,
        data_types=data_types,
        member_dim=member_dim,
    )
    table = build_station_table(ds, coords, source_filename=Path(grib_paths[0]), value_columns=list(params))
    return write_city_parquets(table, base_dir=point_output_dir)


def _parse_steps(steps: str | None) -> set[int] | None:
    if steps is None:
        return None
    steps = str(steps).strip()
    if not steps:
        return None
    out: set[int] = set()
    for part in steps.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?", part)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            stride = int(m.group(3) or 1)
            if stride <= 0 or end < start:
                raise ValueError(f"Invalid step range: {part!r}")
            out.update(range(start, end + 1, stride))
            continue
        if part.isdigit():
            out.add(int(part))
            continue
        raise ValueError(f"Unsupported --steps token: {part!r} (use e.g. 0,6,12 or 0-240:6)")
    return out or None


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process local GRIB2 files (grouped by forecast step) into per-city parquet outputs."
    )
    p.add_argument(
        "--grib-dir",
        required=True,
        help="Directory containing GRIB2 files for a single run (e.g. data/raster_data/ecmwf-ensemble/18z/20260105)",
    )
    p.add_argument("--recursive", action="store_true", help="Search for GRIB2 files recursively under --grib-dir")
    p.add_argument(
        "--locations",
        default="locations.csv",
        help="Path to locations.csv (expects columns: name, lat_lon)",
    )
    p.add_argument(
        "--params",
        default="t2m,u10,v10,tp",
        help="Comma-separated vars to interpolate/write (e.g. 't2m,tp,u10,v10,rh2m')",
    )
    p.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step hours and/or ranges, e.g. '0,6,12' or '0-240:6' (default: all found)",
    )
    p.add_argument(
        "--point-output-dir",
        default=str(Path("data") / "point_data" / "ecmwf-ensemble"),
        help="Base output directory for per-city parquet files",
    )
    p.add_argument("--dry-run", action="store_true", help="List planned work without reading GRIBs/writing parquet")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)

    grib_dir = Path(args.grib_dir).expanduser()
    if not grib_dir.exists():
        raise SystemExit(f"--grib-dir does not exist: {str(grib_dir)!r}")

    paths = list(grib_dir.rglob("*.grib2") if args.recursive else grib_dir.glob("*.grib2"))
    if not paths:
        raise SystemExit(f"No .grib2 files found under: {str(grib_dir)!r}")

    requested_params = [p.strip() for p in str(args.params).split(",") if p.strip()]
    if not requested_params:
        raise SystemExit("--params cannot be empty")

    step_filter = _parse_steps(args.steps)
    by_step = group_grib2_paths_by_step(paths)
    if step_filter is not None:
        by_step = {s: ps for s, ps in by_step.items() if s in step_filter}
    if not by_step:
        raise SystemExit("No timesteps matched.")

    if args.dry_run:
        for step, ps in by_step.items():
            print(f"{step}h: {len(ps)} file(s)")
        return 0

    coords = load_locations_coords(args.locations)
    for step, ps in by_step.items():
        print(f"Processing step {step}h ({len(ps)} file(s))", flush=True)
        written = process_timestep_gribs_to_parquets(
            ps,
            coords=coords,
            params=requested_params,
            point_output_dir=args.point_output_dir,
        )
        print(f"  wrote {len(written)} parquet(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
