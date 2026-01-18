#!/usr/bin/env python3
"""
Download ECMWF HRES GRIB2 steps via the ECMWF open data portal.

Variables retrieved:
- 2m temperature (2t)
- temperature at 1000/925/850/700 hPa (t)
- geopotential height at 1000/925/850/700 hPa (gh)

Each step is saved as:
  ecmwf-hres-<stream>_<yyyymmddhh>_<step>.grib2
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    from ecmwf.opendata import Client
except Exception as exc:  # pragma: no cover - import check only
    Client = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from eccodes import codes_get, codes_grib_new_from_file, codes_release
except Exception:
    codes_get = None
    codes_grib_new_from_file = None
    codes_release = None


SURFACE_PARAMS = "2t"
PRESSURE_PARAMS = "t/gh"
PRESSURE_LEVELS = "1000/925/850/700"
DEFAULT_BUFFER_DEG = 1.0


@dataclass(frozen=True)
class ECMWFStepRequest:
    cycle: datetime
    step_hours: int
    stream: str

    @property
    def cycle_str(self) -> str:
        return self.cycle.strftime("%Y%m%d%H")

    @property
    def step_str(self) -> str:
        return f"{self.step_hours:03d}"

    @property
    def filename(self) -> str:
        return f"ecmwf-hres-{self.stream}_{self.cycle_str}_{self.step_str}.grib2"

    def filename_for_stream(self, stream: str) -> str:
        return f"ecmwf-hres-{stream}_{self.cycle_str}_{self.step_str}.grib2"


class AreaNotSupportedError(RuntimeError):
    pass


def parse_cycle(cycle_str: str) -> datetime:
    try:
        cycle = datetime.strptime(cycle_str, "%Y%m%d%H")
    except ValueError as exc:
        raise ValueError("cycle must be in YYYYMMDDHH format") from exc
    return cycle


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raster_data" / "ecmwf-hres"


def parse_area(area: Optional[str]) -> Optional[str]:
    if not area:
        return None
    parts = area.replace("/", ",").split(",")
    if len(parts) != 4:
        raise ValueError("area must be N/W/S/E (use commas or slashes)")
    n, w, s, e = (float(p.strip()) for p in parts)
    return f"{n}/{w}/{s}/{e}"


def _parse_lat_lon(value: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*([0-9.]+)\s*([NS])\s*([0-9.]+)\s*([EW])\s*", value)
    if not match:
        raise ValueError(f"Invalid lat_lon value: {value!r}")
    lat = float(match.group(1)) * (1 if match.group(2) == "N" else -1)
    lon = float(match.group(3)) * (1 if match.group(4) == "E" else -1)
    return lat, lon


def default_area_from_locations(
    locations_csv: Path,
    *,
    buffer_deg: float,
) -> str:
    if not locations_csv.exists():
        raise FileNotFoundError(f"locations.csv not found at {locations_csv}")

    lat_min = 90.0
    lat_max = -90.0
    lon_min = 180.0
    lon_max = -180.0

    with open(locations_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = (row.get("lat_lon") or "").strip()
            if not value:
                continue
            lat, lon = _parse_lat_lon(value)
            lat_min = min(lat_min, lat)
            lat_max = max(lat_max, lat)
            lon_min = min(lon_min, lon)
            lon_max = max(lon_max, lon)

    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError("No usable lat_lon values in locations.csv")

    lat_min = max(-90.0, lat_min - buffer_deg)
    lat_max = min(90.0, lat_max + buffer_deg)
    lon_min = max(-180.0, lon_min - buffer_deg)
    lon_max = min(180.0, lon_max + buffer_deg)

    return f"{lat_max}/{lon_min}/{lat_min}/{lon_max}"


def is_area_unsupported_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "area" in msg and "not supported" in msg


def read_grib_stream(path: Path) -> Optional[str]:
    if not codes_grib_new_from_file:
        return None
    try:
        with open(path, "rb") as f:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                return None
            try:
                return str(codes_get(gid, "stream"))
            finally:
                codes_release(gid)
    except Exception:
        return None


def build_common_kwargs(
    *,
    cycle: datetime,
    step_hours: int,
    area: Optional[str],
    grid: Optional[str],
    stream: str,
) -> dict:
    kwargs: dict[str, object] = {
        "date": cycle.strftime("%Y%m%d"),
        "time": f"{cycle.hour:02d}",
        "step": step_hours,
        "type": "fc",
        "stream": stream,
    }
    if area:
        kwargs["area"] = area
    if grid:
        kwargs["grid"] = grid
    return kwargs


def retrieve_step(
    *,
    client: "Client",
    req: ECMWFStepRequest,
    out_path: Path,
    area: Optional[str],
    grid: Optional[str],
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        print(f"Exists, skipping: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_surface = out_path.with_suffix(".surface.grib2.part")
    tmp_pressure = out_path.with_suffix(".pl.grib2.part")
    tmp_final = out_path.with_suffix(out_path.suffix + ".part")

    for tmp in (tmp_surface, tmp_pressure, tmp_final):
        if tmp.exists():
            tmp.unlink()

    common_kwargs = build_common_kwargs(
        cycle=req.cycle,
        step_hours=req.step_hours,
        area=area,
        grid=grid,
        stream=req.stream,
    )

    try:
        client.retrieve(
            **common_kwargs,
            param=SURFACE_PARAMS,
            levtype="sfc",
            target=str(tmp_surface),
        )
        client.retrieve(
            **common_kwargs,
            param=PRESSURE_PARAMS,
            levtype="pl",
            levelist=PRESSURE_LEVELS,
            target=str(tmp_pressure),
        )

        with open(tmp_final, "wb") as w:
            for src in (tmp_surface, tmp_pressure):
                with open(src, "rb") as r:
                    shutil.copyfileobj(r, w)

        actual_stream = read_grib_stream(tmp_final)
        final_path = out_path
        if actual_stream:
            final_path = out_path.with_name(req.filename_for_stream(actual_stream))
            if final_path != out_path:
                print(
                    "Stream in GRIB differs from request; "
                    f"renaming to {final_path.name}."
                )
        if final_path.exists() and not overwrite:
            print(f"Exists, skipping: {final_path}")
            return
        tmp_final.replace(final_path)
        print(f"Saved: {final_path}")
    except Exception as exc:
        for tmp in (tmp_surface, tmp_pressure, tmp_final):
            tmp.unlink(missing_ok=True)
        if area and is_area_unsupported_error(exc):
            raise AreaNotSupportedError(str(exc)) from exc
        raise
    finally:
        for tmp in (tmp_surface, tmp_pressure):
            tmp.unlink(missing_ok=True)


def iter_steps(explicit: Optional[Iterable[int]], max_step: int, interval: int) -> list[int]:
    if explicit:
        steps = sorted({int(s) for s in explicit})
    else:
        if max_step < 0:
            raise ValueError("max-step-hours must be >= 0")
        if interval <= 0:
            raise ValueError("step-interval-hours must be > 0")
        steps = list(range(0, max_step + 1, interval))
    return steps


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", required=True, help="UTC cycle in YYYYMMDDHH format")
    p.add_argument(
        "--output-root",
        default=str(default_output_root()),
        help="Root output directory (default: data/raster_data/ecmwf-hres)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--steps", type=int, nargs="+", help="Explicit step hours to download")
    p.add_argument("--max-step-hours", type=int, default=96)
    p.add_argument("--step-interval-hours", type=int, default=3)
    p.add_argument(
        "--area",
        help=(
            "Subset area as N/W/S/E (commas or slashes). "
            "Example: 60/10/30/40. Default uses locations.csv with 1 deg buffer "
            "(falls back to full grid if the server rejects area)."
        ),
    )
    p.add_argument("--grid", help="Output grid, e.g. 0.25/0.25")
    p.add_argument("--stream", default="oper", help="ECMWF stream (default: oper)")
    p.add_argument("--resol", default="0p25", help="Model resolution (default: 0p25)")

    args = p.parse_args(argv)

    if Client is None:
        print(
            f"Missing dependency: ecmwf-opendata ({_IMPORT_ERROR})",
            file=sys.stderr,
        )
        return 2

    try:
        cycle = parse_cycle(args.cycle)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    try:
        area = parse_area(args.area)
        if area is None:
            locations_csv = Path(__file__).resolve().parents[2] / "locations.csv"
            area = default_area_from_locations(
                locations_csv,
                buffer_deg=DEFAULT_BUFFER_DEG,
            )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    try:
        steps = iter_steps(args.steps, args.max_step_hours, args.step_interval_hours)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    output_root = Path(args.output_root).expanduser().resolve()
    cycle_root = output_root / cycle.strftime("%Y%m%d%H")

    client = Client(source="ecmwf", model="ifs", resol=args.resol)
    area_supported = area is not None

    for step in steps:
        req = ECMWFStepRequest(cycle=cycle, step_hours=step, stream=args.stream)
        out_path = cycle_root / req.filename
        for attempt in range(2):
            try:
                retrieve_step(
                    client=client,
                    req=req,
                    out_path=out_path,
                    area=area if area_supported else None,
                    grid=args.grid,
                    overwrite=args.overwrite,
                )
                break
            except AreaNotSupportedError as exc:
                if not area_supported or attempt == 1:
                    print(f"Failed step {step:03d}: {exc}", file=sys.stderr)
                    break
                print(
                    "Area subsetting not supported by ECMWF open data; "
                    "retrying without area.",
                    file=sys.stderr,
                )
                area_supported = False
                continue
            except Exception as exc:
                print(f"Failed step {step:03d}: {exc}", file=sys.stderr)
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
