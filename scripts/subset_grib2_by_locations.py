#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from weather_data.grib_spatial_subset import bbox_from_locations_csv, subset_grib2_file, subset_grib2_inplace


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Spatially subset GRIB2 files to a lat/lon extent derived from locations.csv (+margin) to reduce disk usage."
        )
    )
    p.add_argument(
        "--locations",
        default=str(_REPO_ROOT / "locations.csv"),
        help="Path to locations.csv (must have 'lat_lon' column)",
    )
    p.add_argument("--margin-deg", type=float, default=2.0, help="Safety margin added in all directions (degrees)")
    p.add_argument(
        "--grib-root",
        default=str(_REPO_ROOT / "data" / "raster_data" / "ecmwf-ensemble"),
        help="Root directory containing GRIB2 files (searched recursively by default)",
    )
    p.add_argument("--pattern", default="**/*.grib2", help="Glob pattern relative to --grib-root")
    p.add_argument("--workers", type=int, default=1, help="Parallelize subsetting across multiple files")
    p.add_argument("--dry-run", action="store_true", help="List files that would be processed without writing output")
    p.add_argument("--force", action="store_true", help="Re-process files even if a .bbox.json marker exists")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--inplace", action="store_true", help="Replace each input file with its spatial subset")
    mode.add_argument(
        "--dest-root",
        default=None,
        help="Write outputs under this root (mirrors relative paths under --grib-root); keeps originals intact",
    )
    p.add_argument(
        "--backup-ext",
        default=None,
        help="When using --inplace, keep the original file at <file><backup-ext> (e.g. '.full')",
    )
    return p


def _iter_grib2(grib_root: Path, pattern: str) -> list[Path]:
    root = grib_root.expanduser().resolve()
    paths = sorted({p.resolve() for p in root.glob(pattern) if p.is_file() and p.suffix.lower() == ".grib2"})
    return paths


def _default_dest_for(src: Path) -> Path:
    return src.with_name(src.stem + "__bbox" + src.suffix)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    grib_root = Path(args.grib_root).expanduser().resolve()
    if not grib_root.exists():
        raise SystemExit(f"--grib-root does not exist: {str(grib_root)!r}")

    bbox = bbox_from_locations_csv(args.locations, margin_deg=float(args.margin_deg))
    print(f"bbox: north={bbox.north:.4f}, south={bbox.south:.4f}, west={bbox.west:.4f}, east={bbox.east:.4f}")

    files = _iter_grib2(grib_root, str(args.pattern))
    if not files:
        print("No .grib2 files found.")
        return 0

    def _should_skip(p: Path) -> bool:
        if args.force:
            return False
        return p.with_suffix(p.suffix + ".bbox.json").exists()

    planned = [p for p in files if not _should_skip(p)]
    print(f"Found {len(files)} GRIB2 file(s) under {str(grib_root)!r}; planned: {len(planned)}")

    if args.dry_run:
        for p in planned[:50]:
            print(f"- {str(p)}")
        if len(planned) > 50:
            print(f"... ({len(planned) - 50} more)")
        return 0

    dest_root = Path(args.dest_root).expanduser().resolve() if args.dest_root else None

    def _process_one(p: Path) -> tuple[Path, int]:
        if args.inplace:
            res = subset_grib2_inplace(p, bbox=bbox, backup_ext=args.backup_ext)
            return (p, int(res.output_bytes))

        if dest_root is None:
            dest = _default_dest_for(p)
        else:
            rel = p.relative_to(grib_root)
            dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        res = subset_grib2_file(p, dest, bbox=bbox, overwrite=True)
        return (dest, int(res.output_bytes))

    failures: list[str] = []
    if int(args.workers) <= 1:
        for i, p in enumerate(planned, start=1):
            try:
                out, size = _process_one(p)
                print(f"[{i}/{len(planned)}] {p.name} -> {out.name} ({size/1024/1024:.1f} MiB)")
            except Exception as e:
                failures.append(f"{str(p)}: {e}")
                print(f"[{i}/{len(planned)}] FAILED {p.name}: {e}")
    else:
        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            fut_map = {ex.submit(_process_one, p): p for p in planned}
            done = 0
            for fut in cf.as_completed(fut_map):
                p = fut_map[fut]
                done += 1
                try:
                    out, size = fut.result()
                    print(f"[{done}/{len(planned)}] {p.name} -> {out.name} ({size/1024/1024:.1f} MiB)")
                except Exception as e:
                    failures.append(f"{str(p)}: {e}")
                    print(f"[{done}/{len(planned)}] FAILED {p.name}: {e}")

    if failures:
        print(f"{len(failures)} failure(s).")
        for msg in failures[:20]:
            print(f"- {msg}")
        if len(failures) > 20:
            print(f"... ({len(failures) - 20} more)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
