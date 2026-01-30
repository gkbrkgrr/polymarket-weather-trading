#!/usr/bin/env python3
"""
Download ECMWF HRES (IFS) and AIFS-single GRIB2 files using ecmwf-opendata.

Defaults:
- First 96 hours by 3-hour steps
- 2m temperature (2t)
- temperature at 1000/925/850/700 hPa
- geopotential height (gh) at 1000/925/850/700 hPa for HRES
- geopotential (z) at 1000/925/850/700 hPa for AIFS-single

Files are saved as:
  data/raster_data/<ecmwf-hres|ecmwf-aifs-single>/<yyyymmddhh>/
    <model>-<stream>_<yyyymmddhh>_<fff>.grib2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from ecmwf.opendata import Client


PRESSURE_LEVELS = (1000, 925, 850, 700)
SURFACE_PARAMS = ("2t",)
IFS_PRESSURE_PARAMS = ("t", "gh")
AIFS_PRESSURE_PARAMS = ("t", "z")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    client_model: str
    pressure_params: tuple[str, ...]


MODELS = {
    "ecmwf-hres": ModelConfig(
        name="ecmwf-hres",
        client_model="ifs",
        pressure_params=IFS_PRESSURE_PARAMS,
    ),
    "ecmwf-aifs-single": ModelConfig(
        name="ecmwf-aifs-single",
        client_model="aifs-single",
        pressure_params=AIFS_PRESSURE_PARAMS,
    ),
}


def parse_cycle(cycle_str: str) -> datetime:
    try:
        return datetime.strptime(cycle_str, "%Y%m%d%H")
    except ValueError as exc:
        raise ValueError("cycle must be in YYYYMMDDHH format") from exc


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raster_data"


def stream_for_model(model_name: str, cycle: datetime) -> str:
    if model_name == "ecmwf-aifs-single":
        return "oper"
    if model_name == "ecmwf-hres":
        if cycle.hour in (0, 12):
            return "oper"
        if cycle.hour in (6, 18):
            return "scda"
        raise ValueError("ECMWF HRES cycles must be 00, 06, 12, or 18 UTC")
    raise ValueError(f"Unknown model {model_name!r}")


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


def step_interval_for_model(model_name: str, default_interval: int) -> int:
    if model_name == "ecmwf-aifs-single":
        return 6
    return default_interval


def build_requests(
    *,
    cycle: datetime,
    step: int,
    stream: str,
    pressure_params: tuple[str, ...],
) -> list[dict[str, object]]:
    common = {
        "date": cycle.strftime("%Y%m%d"),
        "time": cycle.hour,
        "type": "fc",
        "step": step,
        "stream": stream,
    }

    requests: list[dict[str, object]] = [
        {
            **common,
            "param": list(SURFACE_PARAMS),
            "levtype": "sfc",
        },
        {
            **common,
            "param": list(pressure_params),
            "levtype": "pl",
            "levelist": list(PRESSURE_LEVELS),
        },
    ]
    return requests


def download_step(
    *,
    client: Client,
    requests: list[dict[str, object]],
    out_path: Path,
    overwrite: bool,
) -> bool:
    if out_path.exists() and not overwrite:
        print(f"Exists, skipping: {out_path}")
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_paths = [
        out_path.with_suffix(out_path.suffix + f".part{idx}") for idx in range(len(requests))
    ]
    merged_tmp = out_path.with_suffix(out_path.suffix + ".part")

    try:
        for tmp_path in tmp_paths:
            if tmp_path.exists():
                tmp_path.unlink()
        if merged_tmp.exists():
            merged_tmp.unlink()

        for req, tmp_path in zip(requests, tmp_paths):
            client.retrieve(request=req, target=str(tmp_path))

        with open(merged_tmp, "wb") as merged:
            for tmp_path in tmp_paths:
                with open(tmp_path, "rb") as src:
                    shutil.copyfileobj(src, merged)

        if merged_tmp.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty")

        merged_tmp.replace(out_path)
        print(f"Saved: {out_path}")
        return True
    except Exception as exc:
        print(f"Failed: {out_path} ({exc})", file=sys.stderr)
        return False
    finally:
        for tmp_path in tmp_paths:
            tmp_path.unlink(missing_ok=True)
        merged_tmp.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", required=True, help="UTC cycle in YYYYMMDDHH format")
    parser.add_argument(
        "--model",
        choices=("ecmwf-hres", "ecmwf-aifs-single", "all"),
        default="all",
        help="Which model to download (default: all)",
    )
    parser.add_argument("--max-step-hours", type=int, default=96)
    parser.add_argument("--step-interval-hours", type=int, default=3)
    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        help="Explicit forecast steps to download (overrides max-step-hours/interval)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Root output directory (default: data/raster_data)",
    )
    parser.add_argument(
        "--source",
        default="ecmwf",
        help="ecmwf-opendata source (ecmwf/aws/azure/google)",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)

    try:
        cycle = parse_cycle(args.cycle)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    steps = iter_steps(args.steps, args.max_step_hours, args.step_interval_hours)

    if args.model == "all":
        model_names = ("ecmwf-hres", "ecmwf-aifs-single")
    else:
        model_names = (args.model,)

    failures = 0
    for model_name in model_names:
        model_steps = steps
        if not args.steps:
            interval = step_interval_for_model(model_name, args.step_interval_hours)
            model_steps = iter_steps(None, args.max_step_hours, interval)

        config = MODELS[model_name]
        stream = stream_for_model(model_name, cycle)
        client = Client(source=args.source, model=config.client_model)
        cycle_root = args.output_root / model_name / cycle.strftime("%Y%m%d%H")
        for step in model_steps:
            filename = f"{model_name}-{stream}_{cycle.strftime('%Y%m%d%H')}_{step:03d}.grib2"
            out_path = cycle_root / filename
            reqs = build_requests(
                cycle=cycle,
                step=step,
                stream=stream,
                pressure_params=config.pressure_params,
            )
            ok = download_step(
                client=client,
                requests=reqs,
                out_path=out_path,
                overwrite=args.overwrite,
            )
            if not ok:
                failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
