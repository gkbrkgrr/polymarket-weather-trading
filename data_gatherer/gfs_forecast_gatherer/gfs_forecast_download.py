#!/usr/bin/env python3
"""
Download GFS GRIB2 files from NOAA NOMADS (filter_gfs_0p25.pl).

Defaults:
- 2m temperature
- temperature at 1000/925/850/700 hPa
- geopotential height at 1000/925/850/700 hPa

Files are saved as:
  gfs_<yyyymmddhh>_f<fff>.grib2
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
SURFACE_LEVELS = ("2_m_above_ground",)
PRESSURE_LEVELS = ("1000_mb", "925_mb", "850_mb", "700_mb")
VARIABLES = ("TMP", "HGT")
DEFAULT_BUFFER_DEG = 1.0
DEFAULT_WAIT_INTERVAL_SECONDS = 60.0
DEFAULT_MAX_WAIT_SECONDS = 7200.0


class NotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class GFSRequest:
    cycle: datetime
    step_hours: int

    @property
    def cycle_str(self) -> str:
        return self.cycle.strftime("%Y%m%d%H")

    @property
    def step_str(self) -> str:
        return f"{self.step_hours:03d}"

    @property
    def filename(self) -> str:
        return f"gfs_{self.cycle_str}_f{self.step_str}.grib2"

    @property
    def nomads_file(self) -> str:
        return f"gfs.t{self.cycle.hour:02d}z.pgrb2.0p25.f{self.step_str}"

    @property
    def nomads_dir(self) -> str:
        return f"/gfs.{self.cycle.strftime('%Y%m%d')}/{self.cycle.hour:02d}/atmos"


def parse_cycle(cycle_str: str) -> datetime:
    try:
        return datetime.strptime(cycle_str, "%Y%m%d%H")
    except ValueError as exc:
        raise ValueError("cycle must be in YYYYMMDDHH format") from exc


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raster_data" / "gfs"


def _parse_lat_lon(value: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*([0-9.]+)\s*([NS])\s*([0-9.]+)\s*([EW])\s*", value)
    if not match:
        raise ValueError(f"Invalid lat_lon value: {value!r}")
    lat = float(match.group(1)) * (1 if match.group(2) == "N" else -1)
    lon = float(match.group(3)) * (1 if match.group(4) == "E" else -1)
    return lat, lon


def parse_area(area: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    if not area:
        return None
    parts = area.replace("/", ",").split(",")
    if len(parts) != 4:
        raise ValueError("area must be N/W/S/E (use commas or slashes)")
    top, left, bottom, right = (float(p.strip()) for p in parts)
    if top < bottom:
        raise ValueError("area north must be >= south")
    if left > right:
        raise ValueError("area west must be <= east")
    return top, left, bottom, right


def default_area_from_locations(
    locations_csv: Path,
    *,
    buffer_deg: float,
) -> tuple[float, float, float, float]:
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

    return lat_max, lon_min, lat_min, lon_max


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


def build_url(req: GFSRequest, area: tuple[float, float, float, float]) -> str:
    top, left, bottom, right = area
    params: dict[str, str] = {
        "file": req.nomads_file,
        "subregion": "",
        "leftlon": f"{left:.2f}",
        "rightlon": f"{right:.2f}",
        "toplat": f"{top:.2f}",
        "bottomlat": f"{bottom:.2f}",
        "dir": req.nomads_dir,
    }
    for level in SURFACE_LEVELS:
        params[f"lev_{level}"] = "on"
    for level in PRESSURE_LEVELS:
        params[f"lev_{level}"] = "on"
    for var in VARIABLES:
        params[f"var_{var}"] = "on"
    return f"{BASE_URL}?{urlencode(params)}"


def download_step(
    *,
    req: GFSRequest,
    out_path: Path,
    area: tuple[float, float, float, float],
    overwrite: bool,
    timeout: float,
) -> None:
    if out_path.exists() and not overwrite:
        print(f"Exists, skipping: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    url = build_url(req, area)

    if tmp_path.exists():
        tmp_path.unlink()

    request = Request(url, headers={"User-Agent": "gfs-forecast-download"})
    try:
        with urlopen(request, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status != 200:
                raise RuntimeError(f"HTTP {status} for {url}")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(resp, f)
        if tmp_path.stat().st_size == 0:
            raise NotReadyError("Downloaded file is empty")
        tmp_path.replace(out_path)
        print(f"Saved: {out_path}")
    except HTTPError as exc:
        if exc.code in {403, 404, 500, 502, 503, 504}:
            raise NotReadyError(f"HTTP {exc.code} for {url}") from exc
        raise
    except URLError as exc:
        raise NotReadyError(f"Network error for {url}: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def wait_for_step(
    *,
    req: GFSRequest,
    out_path: Path,
    area: tuple[float, float, float, float],
    overwrite: bool,
    timeout: float,
    wait_interval: float,
    max_wait: float,
    wait_enabled: bool,
) -> bool:
    start = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            download_step(
                req=req,
                out_path=out_path,
                area=area,
                overwrite=overwrite,
                timeout=timeout,
            )
            return True
        except NotReadyError as exc:
            if not wait_enabled:
                print(f"Not ready for step {req.step_str}: {exc}", file=sys.stderr)
                return False
            elapsed = time.monotonic() - start
            if elapsed >= max_wait:
                print(
                    f"Timed out waiting for step {req.step_str} after {elapsed:.0f}s "
                    f"({attempts} attempts): {exc}",
                    file=sys.stderr,
                )
                return False
            print(
                f"Waiting for step {req.step_str} (attempt {attempts}): {exc}",
                file=sys.stderr,
            )
            time.sleep(wait_interval)
        except (HTTPError, URLError, RuntimeError) as exc:
            print(f"Failed step {req.step_str}: {exc}", file=sys.stderr)
            return False
        except Exception as exc:
            print(f"Failed step {req.step_str}: {exc}", file=sys.stderr)
            return False


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", required=True, help="UTC cycle in YYYYMMDDHH format")
    p.add_argument(
        "--output-root",
        default=str(default_output_root()),
        help="Root output directory (default: data/raster_data/gfs)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--steps", type=int, nargs="+", help="Explicit step hours to download")
    p.add_argument("--max-step-hours", type=int, default=96)
    p.add_argument("--step-interval-hours", type=int, default=3)
    p.add_argument(
        "--area",
        help=(
            "Subset area as N/W/S/E (commas or slashes). "
            "Example: 60/10/30/40. Default uses locations.csv with 1 deg buffer."
        ),
    )
    p.add_argument("--buffer-deg", type=float, default=DEFAULT_BUFFER_DEG)
    p.add_argument("--timeout", type=float, default=120.0)
    wait_group = p.add_mutually_exclusive_group()
    wait_group.add_argument(
        "--wait",
        dest="wait_enabled",
        action="store_true",
        help="Wait for steps to be published (default)",
    )
    wait_group.add_argument(
        "--no-wait",
        dest="wait_enabled",
        action="store_false",
        help="Do not wait for steps to be published",
    )
    p.set_defaults(wait_enabled=True)
    p.add_argument(
        "--wait-interval-seconds",
        type=float,
        default=DEFAULT_WAIT_INTERVAL_SECONDS,
        help="Seconds between availability checks (default: 60)",
    )
    p.add_argument(
        "--max-wait-seconds",
        type=float,
        default=DEFAULT_MAX_WAIT_SECONDS,
        help="Maximum seconds to wait per step (default: 7200)",
    )

    args = p.parse_args(argv)

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
                buffer_deg=args.buffer_deg,
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

    for step in steps:
        req = GFSRequest(cycle=cycle, step_hours=step)
        out_path = cycle_root / req.filename
        ok = wait_for_step(
            req=req,
            out_path=out_path,
            area=area,
            overwrite=args.overwrite,
            timeout=args.timeout,
            wait_interval=args.wait_interval_seconds,
            max_wait=args.max_wait_seconds,
            wait_enabled=args.wait_enabled,
        )
        if not ok:
            print(f"Failed step {step:03d}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
