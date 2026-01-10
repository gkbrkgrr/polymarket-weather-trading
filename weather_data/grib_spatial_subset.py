from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from weather_data.grib_reader import BBox

_ECCODES_LOCK = threading.Lock()


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError("This script requires pandas. Install it with: pip install pandas") from e
    return pd


def _require_eccodes() -> Any:
    try:
        import eccodes  # type: ignore
    except ImportError as e:
        raise ImportError("GRIB2 subsetting requires 'eccodes'. Install it with: pip install eccodes") from e
    return eccodes


def _wrap_lon_180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _wrap_lon_360(lon: float) -> float:
    return lon % 360.0


def _lon_interval_min_arc(lons: Sequence[float], *, margin_deg: float) -> tuple[float, float]:
    """
    Return (west, east) in [-180, 180) representing the smallest arc that covers all lons,
    expanded by `margin_deg` on both sides. If expanded coverage would exceed the full circle,
    returns (-180, 180).

    `west > east` means the interval crosses the dateline.
    """
    if not lons:
        raise ValueError("lons cannot be empty")
    if margin_deg < 0:
        raise ValueError("margin_deg must be >= 0")

    vals = np.asarray([_wrap_lon_180(float(x)) for x in lons], dtype=float)
    vals.sort()
    if vals.size == 1:
        w = _wrap_lon_180(float(vals[0]) - margin_deg)
        e = _wrap_lon_180(float(vals[0]) + margin_deg)
        if 2.0 * margin_deg >= 360.0:
            return (-180.0, 180.0)
        return (w, e)

    extended = np.concatenate([vals, [float(vals[0]) + 360.0]])
    gaps = np.diff(extended)
    gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[gap_idx])

    arc_len = 360.0 - max_gap
    if arc_len + 2.0 * margin_deg >= 360.0:
        return (-180.0, 180.0)

    west = float(vals[(gap_idx + 1) % vals.size]) - margin_deg
    east = float(vals[gap_idx]) + margin_deg
    return (_wrap_lon_180(west), _wrap_lon_180(east))


def bbox_from_locations_csv(locations_csv: str | Path, *, margin_deg: float = 2.0) -> BBox:
    pd = _require_pandas()
    from weather_data.grib_points import parse_lat_lon

    locations_csv = Path(locations_csv).expanduser()
    df = pd.read_csv(locations_csv)
    if "lat_lon" not in df.columns:
        raise ValueError(f"{str(locations_csv)!r} must have a 'lat_lon' column")

    lat_lons = [parse_lat_lon(x) for x in df["lat_lon"].astype(str)]
    lats = [x[0] for x in lat_lons]
    lons = [x[1] for x in lat_lons]

    if not lats or not lons:
        raise ValueError(f"No coordinates found in {str(locations_csv)!r}")

    north = min(90.0, float(max(lats)) + margin_deg)
    south = max(-90.0, float(min(lats)) - margin_deg)
    west = max(-180.0, float(min(lons)) - margin_deg)
    east = min(180.0, float(max(lons)) + margin_deg)

    return BBox(north=north, west=west, south=south, east=east)


@dataclass(frozen=True)
class SpatialSubsetResult:
    messages_written: int
    output_bytes: int
    bbox: BBox
    source: Path
    dest: Path


def _dataset_lon_convention(lons: np.ndarray) -> str:
    if lons.size == 0:
        return "unknown"
    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))
    if lon_min >= 0.0 and lon_max > 180.0:
        return "0_360"
    return "-180_180"


def _coerce_lon_bounds(west: float, east: float, *, convention: str) -> tuple[float, float]:
    if convention == "0_360":
        return (_wrap_lon_360(west), _wrap_lon_360(east))
    return (_wrap_lon_180(west), _wrap_lon_180(east))


def _select_indices_1d(values: np.ndarray, *, low: float, high: float) -> tuple[int, int]:
    if values.size == 0:
        raise ValueError("values cannot be empty")
    lo = float(min(low, high))
    hi = float(max(low, high))
    mask = np.isfinite(values) & (values >= lo) & (values <= hi)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"No grid values within bounds [{lo}, {hi}]")
    return (int(idx.min()), int(idx.max()))


def _select_lon_indices(values: np.ndarray, *, west: float, east: float) -> np.ndarray:
    if values.size == 0:
        raise ValueError("values cannot be empty")
    west = float(west)
    east = float(east)
    if west <= east:
        mask = np.isfinite(values) & (values >= west) & (values <= east)
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise ValueError(f"No longitudes within bounds [{west}, {east}]")
        return idx

    # Dateline-crossing: take the tail + head in scanning order.
    left = np.where(np.isfinite(values) & (values >= west))[0]
    right = np.where(np.isfinite(values) & (values <= east))[0]
    idx = np.concatenate([left, right])
    if idx.size == 0:
        raise ValueError(f"No longitudes within wrap bounds west={west}, east={east}")
    return idx


def _subset_one_message(
    gid: int,
    *,
    bbox: BBox,
    cached_grid: dict[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    ecc = _require_eccodes()

    grid_type = str(ecc.codes_get(gid, "gridType"))
    if grid_type != "regular_ll":
        raise ValueError(f"Unsupported gridType={grid_type!r}; only 'regular_ll' is supported")

    ni = int(ecc.codes_get(gid, "Ni"))
    nj = int(ecc.codes_get(gid, "Nj"))
    j_consecutive = int(ecc.codes_get(gid, "jPointsAreConsecutive"))

    if cached_grid is not None:
        if cached_grid.get("Ni") == ni and cached_grid.get("Nj") == nj and cached_grid.get("jPointsAreConsecutive") == j_consecutive:
            lats = cached_grid["lats"]
            lons = cached_grid["lons"]
            j0, j1 = cached_grid["j0"], cached_grid["j1"]
            lon_idx = cached_grid["lon_idx"]
            convention = cached_grid["lon_convention"]
        else:
            cached_grid = None

    if cached_grid is None:
        lats = np.asarray(ecc.codes_get_array(gid, "distinctLatitudes"), dtype=float)
        lons = np.asarray(ecc.codes_get_array(gid, "distinctLongitudes"), dtype=float)
        if lats.size != nj or lons.size != ni:
            raise ValueError(f"Unexpected grid coord lengths: Nj={nj} vs {lats.size}, Ni={ni} vs {lons.size}")

        convention = _dataset_lon_convention(lons)
        west, east = _coerce_lon_bounds(bbox.west, bbox.east, convention=convention)

        j0, j1 = _select_indices_1d(lats, low=bbox.south, high=bbox.north)
        lon_idx = _select_lon_indices(lons, west=west, east=east)

        cached_grid = {
            "gridType": grid_type,
            "Ni": ni,
            "Nj": nj,
            "jPointsAreConsecutive": j_consecutive,
            "lats": lats,
            "lons": lons,
            "j0": j0,
            "j1": j1,
            "lon_idx": lon_idx,
            "lon_convention": convention,
        }

    if j0 == 0 and j1 == nj - 1 and len(lon_idx) == ni:
        # No-op: entire globe selected.
        return (gid, cached_grid)

    values = np.asarray(ecc.codes_get_array(gid, "values"), dtype=float)
    if values.size != ni * nj:
        raise ValueError(f"Unexpected values length: got {values.size}, expected {ni*nj}")

    if j_consecutive == 0:
        grid = values.reshape((nj, ni))
        sub = grid[j0 : j1 + 1, :][:, lon_idx]
        out_vals = sub.reshape(-1)
    else:
        # values order: i-major with j consecutive
        grid = values.reshape((ni, nj)).T
        sub = grid[j0 : j1 + 1, :][:, lon_idx]
        out_vals = sub.T.reshape(-1)

    new = ecc.codes_clone(gid)
    try:
        new_nj = int(j1 - j0 + 1)
        new_ni = int(len(lon_idx))
        ecc.codes_set(new, "Nj", new_nj)
        ecc.codes_set(new, "Ni", new_ni)

        ecc.codes_set(new, "latitudeOfFirstGridPointInDegrees", float(lats[j0]))
        ecc.codes_set(new, "latitudeOfLastGridPointInDegrees", float(lats[j1]))
        ecc.codes_set(new, "longitudeOfFirstGridPointInDegrees", float(lons[int(lon_idx[0])]))
        ecc.codes_set(new, "longitudeOfLastGridPointInDegrees", float(lons[int(lon_idx[-1])]))

        ecc.codes_set_values(new, out_vals)
        return (new, cached_grid)
    except Exception:
        ecc.codes_release(new)
        raise


def subset_grib2_file(
    source: str | Path,
    dest: str | Path,
    *,
    bbox: BBox,
    overwrite: bool = False,
    write_marker: bool = True,
) -> SpatialSubsetResult:
    """
    Write a spatially subsetted GRIB2 file limited to `bbox`.

    Notes:
    - Currently supports only `gridType=regular_ll` messages.
    - Uses ecCodes to decode/encode; output packing may differ from input.
    """
    ecc = _require_eccodes()
    source = Path(source).expanduser().resolve()
    dest = Path(dest).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    if dest.exists() and not overwrite:
        raise FileExistsError(str(dest))

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)

    cached_grid: dict[str, Any] | None = None
    messages_written = 0

    # ecCodes uses some global state; treat subsetting as a critical section to avoid
    # hard-to-debug crashes when this code is invoked from multi-threaded pipelines.
    with _ECCODES_LOCK:
        with source.open("rb") as src_f, tmp.open("wb") as out_f:
            while True:
                gid = ecc.codes_grib_new_from_file(src_f)
                if gid is None:
                    break
                new_gid: int | None = None
                try:
                    new_gid, cached_grid = _subset_one_message(gid, bbox=bbox, cached_grid=cached_grid)
                    ecc.codes_write(new_gid, out_f)
                    messages_written += 1
                finally:
                    if new_gid is not None and new_gid != gid:
                        ecc.codes_release(new_gid)
                    ecc.codes_release(gid)

    tmp.replace(dest)

    if write_marker:
        marker = dest.with_suffix(dest.suffix + ".bbox.json")
        marker.write_text(
            json.dumps(
                {
                    "bbox": {"north": bbox.north, "west": bbox.west, "south": bbox.south, "east": bbox.east},
                    "source": str(source),
                    "dest": str(dest),
                    "messages_written": messages_written,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    return SpatialSubsetResult(
        messages_written=messages_written,
        output_bytes=dest.stat().st_size,
        bbox=bbox,
        source=source,
        dest=dest,
    )


def subset_grib2_inplace(
    path: str | Path,
    *,
    bbox: BBox,
    backup_ext: str | None = None,
) -> SpatialSubsetResult:
    """
    Spatially subset a GRIB2 file and replace it atomically.

    If `backup_ext` is set, the original is preserved at `<path><backup_ext>`.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() != ".grib2":
        raise ValueError(f"Expected a .grib2 file, got {str(path)!r}")

    tmp_out = path.with_suffix(path.suffix + ".subset.part")
    tmp_out.unlink(missing_ok=True)

    res = subset_grib2_file(path, tmp_out, bbox=bbox, overwrite=True, write_marker=False)

    if backup_ext:
        backup = Path(str(path) + str(backup_ext))
        backup.unlink(missing_ok=True)
        path.replace(backup)
    else:
        path.unlink(missing_ok=True)

    tmp_out.replace(path)

    marker = path.with_suffix(path.suffix + ".bbox.json")
    marker.write_text(
        json.dumps(
            {
                "bbox": {"north": bbox.north, "west": bbox.west, "south": bbox.south, "east": bbox.east},
                "source": str(res.source),
                "dest": str(path),
                "messages_written": res.messages_written,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    return SpatialSubsetResult(
        messages_written=res.messages_written,
        output_bytes=path.stat().st_size,
        bbox=bbox,
        source=Path(res.source),
        dest=path,
    )
