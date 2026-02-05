#!/usr/bin/env python3
"""
Download GFS archive GRIB2 files from an AWS S3 bucket and subset with wgrib2.

Defaults:
- 2m temperature (t2m)
- 2m dew point (d2m)
- 2m relative humidity (rh2m)
- 10m winds (u10/v10)
- total cloud cover (tcc)
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
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BUCKET = "noaa-gfs-bdp-pds"
DEFAULT_BUFFER_DEG = 1.5
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 15.0

SURFACE_PATTERNS = (
    "TMP:2 m above ground",
    "DPT:2 m above ground",
    "RH:2 m above ground",
    "UGRD:10 m above ground",
    "VGRD:10 m above ground",
    "TCDC:entire atmosphere",
)
PRESSURE_LEVELS = (1000, 925, 850, 700)


class DownloadError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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
    def s3_keys(self) -> tuple[str, str]:
        base = f"gfs.{self.cycle.strftime('%Y%m%d')}/{self.cycle.hour:02d}"
        name = f"gfs.t{self.cycle.hour:02d}z.pgrb2.0p25.f{self.step_str}"
        return (
            f"{base}/atmos/{name}",
            f"{base}/{name}",
        )


def parse_cycle(cycle_str: str) -> datetime:
    try:
        return datetime.strptime(cycle_str, "%Y%m%d%H")
    except ValueError as exc:
        raise ValueError("cycle must be in YYYYMMDDHH format") from exc


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raster_data" / "gfs_archive"


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


def build_match_regex() -> str:
    parts = list(SURFACE_PATTERNS)
    for level in PRESSURE_LEVELS:
        parts.append(f"TMP:{level} mb")
    for level in PRESSURE_LEVELS:
        parts.append(f"HGT:{level} mb")
    joined = "|".join(re.escape(p) for p in parts)
    return rf":({joined}):"


def wgrib2_bbox(area: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    north, west, south, east = area
    lat_s = south
    lat_n = north
    lon_w = west
    lon_e = east
    if lat_s > lat_n:
        raise ValueError("Area south must be <= north")
    if lon_e == lon_w:
        raise ValueError("Area west/east span is zero")
    if lon_e < lon_w:
        lon_e += 360.0
    return lon_w, lon_e, lat_s, lat_n


def ensure_wgrib2(wgrib2_bin: str) -> str:
    resolved = shutil.which(wgrib2_bin)
    if not resolved:
        raise FileNotFoundError(f"wgrib2 not found on PATH: {wgrib2_bin}")
    return resolved


def download_to_path(
    *,
    url: str,
    out_path: Path,
    timeout: float,
    retries: int,
    retry_delay: float,
    user_agent: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    if tmp_path.exists():
        tmp_path.unlink()

    request = Request(url, headers={"User-Agent": user_agent})
    attempt = 0
    while True:
        attempt += 1
        try:
            with urlopen(request, timeout=timeout) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if status != 200:
                    raise DownloadError(f"HTTP {status} for {url}", status_code=status)
                with open(tmp_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
            if tmp_path.stat().st_size == 0:
                raise DownloadError("Downloaded file is empty")
            tmp_path.replace(out_path)
            return
        except HTTPError as exc:
            if attempt > retries:
                tmp_path.unlink(missing_ok=True)
                raise DownloadError(
                    f"HTTP {exc.code} for {url}",
                    status_code=exc.code,
                ) from exc
            time.sleep(retry_delay)
        except (URLError, DownloadError) as exc:
            if attempt > retries:
                tmp_path.unlink(missing_ok=True)
                raise DownloadError(str(exc)) from exc
            time.sleep(retry_delay)
        finally:
            tmp_path.unlink(missing_ok=True)


def download_with_fallback(
    *,
    urls: Iterable[str],
    out_path: Path,
    timeout: float,
    retries: int,
    retry_delay: float,
    user_agent: str,
) -> str:
    last_error: Optional[DownloadError] = None
    urls_list = list(urls)
    if not urls_list:
        raise DownloadError("No candidate URLs provided")
    for idx, url in enumerate(urls_list):
        try:
            download_to_path(
                url=url,
                out_path=out_path,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                user_agent=user_agent,
            )
            return url
        except DownloadError as exc:
            last_error = exc
            if exc.status_code == 404 and idx < len(urls_list) - 1:
                continue
            raise
    raise last_error or DownloadError("All candidate URLs failed")


def subset_with_wgrib2(
    *,
    wgrib2_bin: str,
    input_path: Path,
    output_path: Path,
    bbox: tuple[float, float, float, float],
    match_regex: str,
    verbose: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_suffix(output_path.suffix + ".part")
    lon_w, lon_e, lat_s, lat_n = bbox

    if tmp_out.exists():
        tmp_out.unlink()

    cmd = [
        wgrib2_bin,
        str(input_path),
        "-match",
        match_regex,
        "-set_grib_type",
        "same",
        "-small_grib",
        f"{lon_w}:{lon_e}",
        f"{lat_s}:{lat_n}",
        str(tmp_out),
    ]

    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.PIPE

    try:
        subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr, text=True)
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            raise RuntimeError(exc.stderr.strip()) from exc
        raise

    if not tmp_out.exists() or tmp_out.stat().st_size == 0:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError("wgrib2 produced empty output")

    tmp_out.replace(output_path)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", required=True, help="UTC cycle in YYYYMMDDHH format")
    p.add_argument(
        "--output-root",
        default=str(default_output_root()),
        help="Root output directory (default: data/raster_data/gfs_archive)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--steps", type=int, nargs="+", help="Explicit step hours to download")
    p.add_argument("--max-step-hours", type=int, default=96)
    p.add_argument("--step-interval-hours", type=int, default=3)
    p.add_argument(
        "--area",
        help=(
            "Subset area as N/W/S/E (commas or slashes). "
            "Example: 60/10/30/40. Default uses locations.csv with 1.5 deg buffer."
        ),
    )
    p.add_argument("--buffer-deg", type=float, default=DEFAULT_BUFFER_DEG)
    p.add_argument(
        "--locations-csv",
        default=str(Path(__file__).resolve().parents[2] / "locations.csv"),
        help="Path to locations.csv (default: repo root locations.csv)",
    )
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument(
        "--base-url",
        help="Override base URL (default: https://<bucket>.s3.amazonaws.com)",
    )
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument("--retry-delay-seconds", type=float, default=DEFAULT_RETRY_DELAY_SECONDS)
    p.add_argument("--wgrib2-bin", default="wgrib2")
    p.add_argument("--keep-source", action="store_true")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)

    try:
        cycle = parse_cycle(args.cycle)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    try:
        area = parse_area(args.area)
        if area is None:
            area = default_area_from_locations(
                Path(args.locations_csv).expanduser().resolve(),
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

    try:
        wgrib2_path = ensure_wgrib2(args.wgrib2_bin)
    except FileNotFoundError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 2

    try:
        bbox = wgrib2_bbox(area)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    base_url = args.base_url or f"https://{args.bucket}.s3.amazonaws.com"
    output_root = Path(args.output_root).expanduser().resolve()
    cycle_root = output_root / cycle.strftime("%Y%m%d%H")
    tmp_root = cycle_root / ".tmp"
    match_regex = build_match_regex()
    failures = 0

    for step in steps:
        req = GFSRequest(cycle=cycle, step_hours=step)
        out_path = cycle_root / req.filename

        if out_path.exists() and not args.overwrite:
            print(f"Exists, skipping: {out_path}")
            continue

        source_path = tmp_root / req.filename
        urls = [f"{base_url}/{key}" for key in req.s3_keys]

        try:
            download_with_fallback(
                urls=urls,
                out_path=source_path,
                timeout=args.timeout,
                retries=args.retries,
                retry_delay=args.retry_delay_seconds,
                user_agent="gfs-forecast-archiver",
            )
            subset_with_wgrib2(
                wgrib2_bin=wgrib2_path,
                input_path=source_path,
                output_path=out_path,
                bbox=bbox,
                match_regex=match_regex,
                verbose=args.verbose,
            )
            print(f"Saved: {out_path}")
        except (DownloadError, RuntimeError, OSError) as exc:
            failures += 1
            print(f"Failed step {step:03d}: {exc}", file=sys.stderr)
        finally:
            if source_path.exists() and not args.keep_source:
                source_path.unlink()

    if not args.keep_source:
        try:
            tmp_root.rmdir()
        except OSError:
            pass

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
