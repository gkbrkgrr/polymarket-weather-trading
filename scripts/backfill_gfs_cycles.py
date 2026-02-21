#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RASTER_ROOT = REPO_ROOT / "data" / "raster_data" / "gfs_archive"
DEFAULT_POINT_ROOT = REPO_ROOT / "data" / "point_data" / "gfs" / "raw"
DEFAULT_PRED_ROOT = REPO_ROOT / "data" / "ml_predictions"
DEFAULT_CYCLE_HOURS = (0, 6, 12, 18)
DEFAULT_MODELS = ("xgb", "xgb_opt")
CYCLE_PATTERN = re.compile(r"^\d{10}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill GFS cycles: download/subset GRIBs from archive, build point_data "
            "parquets, and build ml_predictions parquets."
        )
    )
    parser.add_argument("--start-cycle", required=True, help="Inclusive start cycle YYYYMMDDHH")
    parser.add_argument("--end-cycle", required=True, help="Inclusive end cycle YYYYMMDDHH")
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for subprocess scripts.",
    )
    parser.add_argument(
        "--raster-root",
        type=Path,
        default=DEFAULT_RASTER_ROOT,
        help=f"GRIB cycle root (default: {DEFAULT_RASTER_ROOT})",
    )
    parser.add_argument(
        "--point-root",
        type=Path,
        default=DEFAULT_POINT_ROOT,
        help=f"Point parquet root (default: {DEFAULT_POINT_ROOT})",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=DEFAULT_PRED_ROOT,
        help=f"Prediction root (default: {DEFAULT_PRED_ROOT})",
    )
    parser.add_argument("--max-step-hours", type=int, default=96)
    parser.add_argument("--step-interval-hours", type=int, default=3)
    parser.add_argument(
        "--archiver-threads",
        type=int,
        default=8,
        help="Threads for gfs_forecast_archiver.py",
    )
    parser.add_argument(
        "--wgrib2-bin",
        type=Path,
        default=Path("/home/gkbrkgrr/miniconda3/envs/env_poly/bin/wgrib2"),
        help="Path to wgrib2 executable for archive downloader.",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=3,
        help="Retries per stage (download/point/predict).",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=20.0,
        help="Backoff base seconds between retries.",
    )
    parser.add_argument(
        "--sleep-between-cycles-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between cycles.",
    )
    parser.add_argument(
        "--overwrite-predictions",
        action="store_true",
        help="Always pass --overwrite to predictor script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing subprocesses.",
    )
    return parser.parse_args(argv)


def parse_cycle_token(value: str, arg_name: str) -> datetime:
    if CYCLE_PATTERN.fullmatch(value) is None:
        raise SystemExit(f"{arg_name} must be YYYYMMDDHH, got {value!r}")
    return datetime.strptime(value, "%Y%m%d%H")


def build_cycle_list(start_cycle: str, end_cycle: str) -> list[str]:
    start_dt = parse_cycle_token(start_cycle, "--start-cycle")
    end_dt = parse_cycle_token(end_cycle, "--end-cycle")
    if start_dt > end_dt:
        raise SystemExit("--start-cycle must be <= --end-cycle")
    cycles: list[str] = []
    cur = start_dt
    while cur <= end_dt:
        if cur.hour in DEFAULT_CYCLE_HOURS:
            cycles.append(cur.strftime("%Y%m%d%H"))
        cur += timedelta(hours=6)
    return cycles


def expected_step_count(max_step_hours: int, step_interval_hours: int) -> int:
    return len(range(0, max_step_hours + 1, step_interval_hours))


def run_cmd(cmd: list[str], dry_run: bool) -> subprocess.CompletedProcess | None:
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        return None
    return subprocess.run(cmd, check=True)


def grib_cycle_ready(
    raster_root: Path,
    cycle: str,
    expected_steps: int,
) -> bool:
    cycle_dir = raster_root / cycle
    if not cycle_dir.exists():
        return False
    files = sorted(cycle_dir.glob(f"gfs_{cycle}_f*.grib2"))
    return len(files) >= expected_steps


def point_cycle_ready(point_root: Path, cycle: str) -> bool:
    return bool(list(point_root.glob(f"gfs_{cycle}_*.parquet")))


def prediction_cycle_ready(pred_root: Path, model: str, cycle: str) -> bool:
    pattern = f"{model}_daily_tmax_predictions_{cycle}.parquet"
    files = list((pred_root / model).glob(f"*/{pattern}"))
    return len(files) >= 9


def run_with_retries(
    *,
    stage: str,
    cmd: list[str],
    retry_count: int,
    retry_wait_seconds: float,
    dry_run: bool,
) -> None:
    for attempt in range(1, retry_count + 1):
        try:
            run_cmd(cmd, dry_run=dry_run)
            return
        except subprocess.CalledProcessError as exc:
            if attempt >= retry_count:
                raise SystemExit(
                    f"{stage} failed after {retry_count} attempts. "
                    f"Last exit code: {exc.returncode}"
                ) from exc
            wait = retry_wait_seconds * attempt
            print(
                f"{stage} attempt {attempt}/{retry_count} failed; "
                f"retrying in {wait:.1f}s"
            )
            time.sleep(wait)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cycles = build_cycle_list(args.start_cycle, args.end_cycle)
    expected_steps = expected_step_count(args.max_step_hours, args.step_interval_hours)

    print(f"Cycles requested: {len(cycles)}")
    print(f"Range: {args.start_cycle} .. {args.end_cycle}")
    print(f"Expected GRIB steps/cycle: {expected_steps}")
    print(f"Raster root: {args.raster_root}")
    print(f"Point root: {args.point_root}")
    print(f"Prediction root: {args.pred_root}")
    print(f"Dry run: {bool(args.dry_run)}")

    scripts_dir = REPO_ROOT / "scripts"
    archive_script = REPO_ROOT / "data_gatherer" / "gfs_archiver" / "gfs_forecast_archiver.py"

    done_cycles = 0
    for idx, cycle in enumerate(cycles, start=1):
        print(f"\n[{idx}/{len(cycles)}] cycle={cycle}")

        if not grib_cycle_ready(args.raster_root, cycle, expected_steps):
            download_cmd = [
                str(args.python_bin),
                str(archive_script),
                "--cycle",
                cycle,
                "--output-root",
                str(args.raster_root),
                "--max-step-hours",
                str(args.max_step_hours),
                "--step-interval-hours",
                str(args.step_interval_hours),
                "--threads",
                str(args.archiver_threads),
                "--wgrib2-bin",
                str(args.wgrib2_bin),
            ]
            run_with_retries(
                stage=f"download[{cycle}]",
                cmd=download_cmd,
                retry_count=int(args.retry_count),
                retry_wait_seconds=float(args.retry_wait_seconds),
                dry_run=bool(args.dry_run),
            )
        else:
            print("GRIB ready: skip")

        if not point_cycle_ready(args.point_root, cycle):
            point_cmd = [
                str(args.python_bin),
                str(scripts_dir / "raw_point_extractor.py"),
                "--model",
                "gfs",
                "--cycle",
                cycle,
                "--grib-dir",
                str(args.raster_root / cycle),
                "--max-step-hours",
                str(args.max_step_hours),
                "--step-interval-hours",
                str(args.step_interval_hours),
            ]
            run_with_retries(
                stage=f"point[{cycle}]",
                cmd=point_cmd,
                retry_count=int(args.retry_count),
                retry_wait_seconds=float(args.retry_wait_seconds),
                dry_run=bool(args.dry_run),
            )
        else:
            print("Point parquet ready: skip")

        for model_name in DEFAULT_MODELS:
            if prediction_cycle_ready(args.pred_root, model_name, cycle) and not args.overwrite_predictions:
                print(f"Predictions ready ({model_name}): skip")
                continue
            pred_cmd = [
                str(args.python_bin),
                str(scripts_dir / "daily_station_tmax_predictor.py"),
                "--model",
                model_name,
                "--cycle",
                cycle,
            ]
            if args.overwrite_predictions:
                pred_cmd.append("--overwrite")
            run_with_retries(
                stage=f"predict[{model_name}][{cycle}]",
                cmd=pred_cmd,
                retry_count=int(args.retry_count),
                retry_wait_seconds=float(args.retry_wait_seconds),
                dry_run=bool(args.dry_run),
            )

        done_cycles += 1
        if args.sleep_between_cycles_seconds > 0:
            time.sleep(float(args.sleep_between_cycles_seconds))

    print(f"\nCompleted cycles: {done_cycles}/{len(cycles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
