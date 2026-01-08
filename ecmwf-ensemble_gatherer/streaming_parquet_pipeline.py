#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import re
import ssl
import threading
import sys
import traceback
from pathlib import Path
from typing import Sequence
from urllib.request import HTTPSHandler, build_opener, install_opener

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import download_pipeline as dp


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download ECMWF Open Data GRIB2 files timestep-by-timestep and process each timestep immediately into "
            "per-city parquet files (overlapping downloads with processing)."
        )
    )

    p.add_argument("--preset", default="ifs-enfo", choices=sorted(dp.PRESETS.keys()))
    p.add_argument("--date", default="utc-today", help="YYYY-MM-DD, YYYYMMDD, or 'utc-today'")
    p.add_argument("--run", default=None, help="00/06/12/18 or 00z/06z/12z/18z (default: auto)")
    p.add_argument("--steps", default=None, help="Comma-separated step hours and/or ranges, e.g. '0,24,48' or '0-240:3'")
    p.add_argument("--products", default=None, help="Comma-separated product types to include (e.g. ef,fc,cf,pf,ep)")
    p.add_argument(
        "--params",
        default="t2m,u10,v10,tp",
        help="Variables to download (subset) and write to parquet (e.g. 't2m,u10,v10,tp' or 't2m,tp,rh2m')",
    )
    p.add_argument(
        "--no-subset",
        action="store_true",
        help="Do not subset downloads by --params (still processes --params from the full GRIB files)",
    )
    p.add_argument("--name-regex", default=None, help="Regex applied to the GRIB2 filename as an additional filter")
    p.add_argument("--timeout-s", type=int, default=60, help="HTTP timeout in seconds")
    p.add_argument("--retries", type=int, default=5, help="Retries per file")
    p.add_argument("--chunk-mb", type=int, default=8, help="Download chunk size in MiB")
    p.add_argument("--ca-bundle", default=None, help="Path to a custom CA bundle (PEM) to trust for HTTPS")
    p.add_argument("--insecure", action="store_true", help="Disable HTTPS certificate verification (unsafe)")
    p.add_argument("--workers", type=int, default=4, help="Parallelize downloads across multiple files within a timestep")
    p.add_argument(
        "--connections-per-file",
        type=int,
        default=1,
        help="For full-file downloads (no effective subsetting), use N parallel HTTP Range requests per file",
    )
    p.add_argument("--keep-going", action="store_true", help="Keep downloading/processing other timesteps if one fails")
    p.add_argument("--max-total-gb", type=float, default=25.0, help="Abort if listed total size exceeds this many GiB")
    p.add_argument("--force-large", action="store_true", help="Allow large total downloads (disables --max-total-gb abort)")
    p.add_argument("--subset-merge-gap-kb", type=int, default=0, help="Merge nearby byte ranges when subsetting via .index")
    p.add_argument(
        "--dest-root",
        default=None,
        help="Override destination root for GRIB files (default: <repo>/data/raster_data/ecmwf-ensemble)",
    )

    p.add_argument(
        "--locations",
        default=str(dp._repo_root() / "locations.csv"),
        help="Path to locations.csv (name, lat_lon)",
    )
    p.add_argument(
        "--point-output-dir",
        default=str(dp._repo_root() / "data" / "point_data" / "ecmwf-ensemble"),
        help="Base output directory for per-city parquet files",
    )
    p.add_argument("--process-workers", type=int, default=1, help="Parallelize processing across timesteps (default: 1)")
    p.add_argument("--max-inflight-steps", type=int, default=2, help="Max timesteps queued for processing while downloading ahead")
    p.add_argument("--dry-run", action="store_true", help="List planned downloads/processing without doing work")
    return p


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    now_utc = dt.datetime.now(dt.timezone.utc)
    print_lock = threading.Lock()

    def _log(msg: str) -> None:
        with print_lock:
            print(msg, flush=True)

    if args.insecure and args.ca_bundle:
        raise SystemExit("Use either --insecure or --ca-bundle (not both)")
    if args.insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        install_opener(build_opener(HTTPSHandler(context=ctx)))
        _log("WARNING: HTTPS verification disabled via --insecure")
    elif args.ca_bundle:
        ctx = ssl.create_default_context(cafile=str(Path(args.ca_bundle).expanduser()))
        install_opener(build_opener(HTTPSHandler(context=ctx)))

    if args.run is None:
        date, run = dp._auto_run(now_utc)
        if args.date not in {"today", "utc-today"}:
            date = dp._parse_date(args.date)
    else:
        date = dp._parse_date(args.date)
        run = dp._parse_run(args.run)

    yyyymmdd = date.strftime("%Y%m%d")
    dest_root = Path(args.dest_root) if args.dest_root else dp._repo_root() / "data" / "raster_data" / "ecmwf-ensemble"
    dest_dir = dest_root / run / yyyymmdd

    products = dp._parse_products(args.products)
    preset = dp.PRESETS.get(args.preset, {})
    remote_template = str(preset.get("remote_dir_template", dp.DEFAULT_REMOTE_DIR_TEMPLATE))
    if products is None and "products" in preset:
        products = set(preset["products"])  # type: ignore[arg-type]

    remote_dir = remote_template.format(yyyymmdd=yyyymmdd, run=run)
    steps = dp._parse_steps(args.steps)
    name_regex = re.compile(args.name_regex) if args.name_regex else None

    # Use the download pipeline's param parsing so 'rh2m' expands to '2t'+'2d' for the subset download.
    requested_params = _split_csv(args.params)
    download_params = None if args.no_subset else dp._parse_params(args.params)  # type: ignore[arg-type]
    if not requested_params:
        raise SystemExit("--params cannot be empty")

    files = dp.list_remote_grib2(
        remote_dir,
        products=products,
        steps=steps,
        name_regex=name_regex,
        timeout_s=args.timeout_s,
    )
    if not files:
        _log(f"No GRIB2 files found at {remote_dir}")
        return 2

    total = dp._sum_sizes(files)
    _log(f"Remote: {remote_dir}")
    _log(f"Dest:   {dest_dir}")
    _log(f"Files:  {len(files)} (listed size: {dp._fmt_bytes(total)})")
    _log(f"Params: {', '.join(requested_params)}")

    if (not args.force_large) and total > int(args.max_total_gb * 1024 * 1024 * 1024):
        _log(
            f"Refusing to download {dp._fmt_bytes(total)} (exceeds --max-total-gb={args.max_total_gb}). "
            f"Use --force-large to override."
        )
        return 3

    by_step: dict[int, list[dp.RemoteGrib2]] = {}
    for f in files:
        if f.step_hours is None:
            continue
        by_step.setdefault(int(f.step_hours), []).append(f)

    if not by_step:
        _log("No files had a parseable step hour in their filenames.")
        return 2

    dest_dir.mkdir(parents=True, exist_ok=True)
    coords = None
    if not args.dry_run:
        from weather_data.grib_points import load_locations_coords, process_timestep_gribs_to_parquets

        coords = load_locations_coords(args.locations)

    def _download_one(step: int, f: dp.RemoteGrib2) -> Path | None:
        if download_params is None:
            dest_path = dest_dir / f.name
            if args.connections_per_file > 1 and f.size_bytes is not None:
                dp._download_with_parallel_ranges(
                    f.url,
                    dest_path,
                    expected_size=f.size_bytes,
                    connections=args.connections_per_file,
                    chunk_bytes=args.chunk_mb * 1024 * 1024,
                    timeout_s=args.timeout_s,
                    retries=args.retries,
                )
            else:
                dp._download_with_resume(
                    f.url,
                    dest_path,
                    expected_size=f.size_bytes,
                    chunk_bytes=args.chunk_mb * 1024 * 1024,
                    timeout_s=args.timeout_s,
                    retries=args.retries,
                )
            return dest_path

        parts, _subset_bytes = dp._read_index_parts(
            f.url,
            params=set(download_params),
            steps={step},
            timeout_s=args.timeout_s,
        )
        if not parts:
            return None
        planned = dp._estimated_download_bytes(parts, merge_gap_bytes=args.subset_merge_gap_kb * 1024)
        dest_name = dp._subset_output_name(f.name, download_params)
        dest_path = dest_dir / dest_name
        if dest_path.exists() and dest_path.stat().st_size == planned:
            return dest_path
        dp._download_ranges(
            f.url,
            dest_path,
            parts=parts,
            chunk_bytes=args.chunk_mb * 1024 * 1024,
            timeout_s=args.timeout_s,
            retries=args.retries,
            merge_gap_bytes=args.subset_merge_gap_kb * 1024,
        )
        return dest_path

    def _process_step(step: int, local_paths: list[Path]) -> list[Path]:
        if not local_paths:
            return []
        if coords is None:
            raise RuntimeError("coords not loaded")
        return process_timestep_gribs_to_parquets(
            local_paths,
            coords=coords,
            params=requested_params,
            point_output_dir=args.point_output_dir,
        )

    inflight: list[tuple[int, cf.Future[list[Path]]]] = []
    failures: list[str] = []
    abort = False

    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.process_workers))) as process_ex:
        for i, (step, step_files) in enumerate(sorted(by_step.items(), key=lambda kv: kv[0]), start=1):
            step_files = sorted(step_files, key=lambda r: r.name)
            _log(f"[{i}/{len(by_step)}] Downloading step {step}h ({len(step_files)} file(s)) {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if args.dry_run:
                for f in step_files[:10]:
                    dest_name = f.name if download_params is None else dp._subset_output_name(f.name, download_params)
                    _log(f"  - {f.name} -> {dest_name}")
                if len(step_files) > 10:
                    _log(f"  ... ({len(step_files) - 10} more)")
                continue

            local_paths: list[Path] = []
            try:
                if int(args.workers) > 1 and len(step_files) > 1:
                    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as dl_ex:
                        futs = [dl_ex.submit(_download_one, step, f) for f in step_files]
                        for fut in cf.as_completed(futs):
                            p = fut.result()
                            if p is not None:
                                local_paths.append(p)
                else:
                    for f in step_files:
                        p = _download_one(step, f)
                        if p is not None:
                            local_paths.append(p)
            except Exception as e:
                msg = f"Download failed for step {step}h: {e}"
                failures.append(msg)
                _log(msg)
                _log(traceback.format_exc())
                if not args.keep_going:
                    break
                continue

            local_paths = sorted({p.resolve() for p in local_paths})
            _log(f"[{i}/{len(by_step)}] Processing step {step}h ({len(local_paths)} local file(s)) {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            fut = process_ex.submit(_process_step, step, local_paths)
            inflight.append((step, fut))

            while len(inflight) >= max(1, int(args.max_inflight_steps)):
                step0, fut0 = inflight.pop(0)
                try:
                    written = fut0.result()
                    _log(f"Processed step {step0}h: wrote {len(written)} parquet(s) {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception as e:
                    msg = f"Processing failed for step {step0}h: {e}"
                    failures.append(msg)
                    _log(msg)
                    _log(traceback.format_exc())
                    if not args.keep_going:
                        inflight.clear()
                        abort = True
                        break
            if abort:
                break

        for step, fut in inflight:
            try:
                written = fut.result()
                _log(f"Processed step {step}h: wrote {len(written)} parquet(s)")
            except Exception as e:
                msg = f"Processing failed for step {step}h: {e}"
                failures.append(msg)
                _log(msg)
                _log(traceback.format_exc())
                if not args.keep_going:
                    abort = True
                    break

    if failures:
        _log(f"{len(failures)} failure(s).")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
