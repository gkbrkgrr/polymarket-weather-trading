#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ECMWF_PORTAL_BASE = "https://data.ecmwf.int"
DEFAULT_REMOTE_DIR_TEMPLATE = (
    "https://data.ecmwf.int/forecasts/{yyyymmdd}/{run}/ifs/0p25/enfo/"
)

PRESETS: dict[str, dict[str, object]] = {
    "ifs-enfo": {"remote_dir_template": "https://data.ecmwf.int/forecasts/{yyyymmdd}/{run}/ifs/0p25/enfo/"},
    "ifs-enfo-ep": {
        "remote_dir_template": "https://data.ecmwf.int/forecasts/{yyyymmdd}/{run}/ifs/0p25/enfo/",
        "products": {"ep"},
    },
    "ifs-oper": {
        "remote_dir_template": "https://data.ecmwf.int/forecasts/{yyyymmdd}/{run}/ifs/0p25/oper/",
        "products": {"fc"},
    },
    "aifs-single-oper": {
        "remote_dir_template": "https://data.ecmwf.int/forecasts/{yyyymmdd}/{run}/aifs-single/0p25/oper/",
        "products": {"fc"},
    },
    "aifs-ens-cf": {
        "remote_dir_template": "https://data.ecmwf.int/forecasts/{yyyymmdd}/{run}/aifs-ens/0p25/enfo/",
        "products": {"cf"},
    },
}

PARAM_ALIASES: dict[str, str] = {
    "t2m": "2t",
    "d2m": "2d",
    "u10": "10u",
    "v10": "10v",
}


@dataclass(frozen=True)
class RemoteGrib2:
    name: str
    url: str
    size_bytes: int | None
    step_hours: int | None
    product: str | None


_GRIB2_RE = re.compile(
    r'<a href="(?P<href>[^"]+)">(?P<name>[^<]+\.grib2)</a>\s+'
    r"\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}\s+"
    r"(?P<size>\d+)\s+",
    re.IGNORECASE,
)
_NAME_META_RE = re.compile(
    r"-(?P<step>\d+)h-(?P<stream>[a-z0-9-]+)-(?P<type>[a-z0-9]+)\.grib2$",
    re.I,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_date(date_str: str) -> dt.date:
    date_str = date_str.strip()
    if date_str.lower() in {"today", "utc-today"}:
        return dt.datetime.now(dt.timezone.utc).date()
    if re.fullmatch(r"\d{8}", date_str):
        return dt.datetime.strptime(date_str, "%Y%m%d").date()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    raise ValueError(f"Unsupported date format: {date_str!r} (use YYYY-MM-DD or YYYYMMDD)")


def _parse_run(run_str: str) -> str:
    run_str = run_str.strip().lower()
    if run_str.endswith("z"):
        run_str = run_str[:-1]
    if not run_str.isdigit():
        raise ValueError(f"Unsupported run: {run_str!r} (use 00/06/12/18 or 00z/06z/12z/18z)")
    hour = int(run_str)
    if hour not in {0, 6, 12, 18}:
        raise ValueError(f"Unsupported run hour: {hour} (expected one of 0,6,12,18)")
    return f"{hour:02d}z"


def _auto_run(now_utc: dt.datetime) -> tuple[dt.date, str]:
    hour = now_utc.hour
    cycle = max(c for c in (0, 6, 12, 18) if c <= hour)
    return now_utc.date(), f"{cycle:02d}z"


def _parse_steps(steps: str | None) -> set[int] | None:
    if steps is None:
        return None
    steps = steps.strip()
    if not steps:
        return None
    out: set[int] = set()
    for part in steps.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?", part)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            stride = int(match.group(3) or 1)
            if stride <= 0:
                raise ValueError(f"Invalid step stride in {part!r}")
            if end < start:
                raise ValueError(f"Invalid step range in {part!r}")
            out.update(range(start, end + 1, stride))
            continue
        if part.isdigit():
            out.add(int(part))
            continue
        raise ValueError(
            f"Unsupported --steps token: {part!r} (use e.g. 0,24,48 or 0-240:3)"
        )
    return out or None


def _parse_products(products: str | None) -> set[str] | None:
    if products is None:
        return None
    products = products.strip()
    if not products:
        return None
    return {p.strip().lower() for p in products.split(",") if p.strip()}


def _parse_params(params: str | None) -> list[str] | None:
    if params is None:
        return None
    params = params.strip()
    if not params:
        return None
    out: list[str] = []
    for p in (x.strip().lower() for x in params.split(",")):
        if not p:
            continue
        if p == "rh2m":
            # RH at 2m is not always present as a direct param in ECMWF open-data ENFO.
            # Include the inputs needed to derive it downstream: t2m (2t) + d2m (2d).
            out.extend(["2t", "2d"])
            continue
        out.append(PARAM_ALIASES.get(p, p))
    deduped = sorted(set(out))
    return deduped or None


def _fetch_text(url: str, *, timeout_s: int) -> str:
    req = Request(url, headers={"User-Agent": "polymarket-weather-trading/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def list_remote_grib2(
    remote_dir_url: str,
    *,
    products: set[str] | None,
    steps: set[int] | None,
    name_regex: re.Pattern[str] | None,
    timeout_s: int,
) -> list[RemoteGrib2]:
    html = _fetch_text(remote_dir_url, timeout_s=timeout_s)
    out: list[RemoteGrib2] = []
    for match in _GRIB2_RE.finditer(html):
        href = match.group("href")
        name = match.group("name")
        if name_regex is not None and not name_regex.search(name):
            continue

        name_match = _NAME_META_RE.search(name)
        step_hours = int(name_match.group("step")) if name_match else None
        product = name_match.group("type").lower() if name_match else None

        if products is not None and product is not None and product not in products:
            continue
        if steps is not None and step_hours is not None and step_hours not in steps:
            continue

        url = href if href.startswith("http") else f"{ECMWF_PORTAL_BASE}{href}"
        out.append(
            RemoteGrib2(
                name=name,
                url=url,
                size_bytes=int(match.group("size")),
                step_hours=step_hours,
                product=product,
            )
        )

    out.sort(key=lambda r: (r.step_hours is None, r.step_hours or 0, r.name))
    return out


def _download_with_resume(
    url: str,
    dest: Path,
    *,
    expected_size: int | None,
    chunk_bytes: int,
    timeout_s: int,
    retries: int,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        return

    if dest.exists() and not tmp.exists():
        dest.rename(tmp)

    for attempt in range(1, retries + 1):
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "polymarket-weather-trading/1.0"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"

            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout_s) as resp:
                status = getattr(resp, "status", 200)
                if existing > 0 and status == 200:
                    existing = 0
                    tmp.unlink(missing_ok=True)
                mode = "ab" if existing > 0 else "wb"
                with tmp.open(mode) as f:
                    while True:
                        chunk = resp.read(chunk_bytes)
                        if not chunk:
                            break
                        f.write(chunk)

            if expected_size is not None:
                actual = tmp.stat().st_size
                if actual != expected_size:
                    raise IOError(f"Incomplete download for {dest.name}: {actual} != {expected_size}")

            tmp.replace(dest)
            return
        except (HTTPError, URLError, TimeoutError, ConnectionError, IOError) as e:
            if attempt >= retries:
                raise
            backoff = min(60, 2**attempt)
            print(f"Download failed (attempt {attempt}/{retries}) for {dest.name}: {e}; retrying in {backoff}s")
            time.sleep(backoff)

def _index_url(grib2_url: str) -> str:
    if not grib2_url.endswith(".grib2"):
        raise ValueError(f"Expected .grib2 url, got: {grib2_url}")
    return f"{grib2_url[:-6]}.index"


def _subset_output_name(original_name: str, params: Sequence[str]) -> str:
    if not original_name.endswith(".grib2"):
        return f"{original_name}__params-{'-'.join(params)}"
    return f"{original_name[:-6]}__params-{'-'.join(params)}.grib2"


def _read_index_parts(
    grib2_url: str,
    *,
    params: set[str],
    steps: set[int] | None,
    timeout_s: int,
) -> tuple[list[tuple[int, int]], int]:
    url = _index_url(grib2_url)
    req = Request(url, headers={"User-Agent": "polymarket-weather-trading/1.0"})
    parts: list[tuple[int, int]] = []
    with urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("param") not in params:
                continue
            if steps is not None:
                step_s = entry.get("step")
                try:
                    step_i = int(step_s)
                except (TypeError, ValueError):
                    continue
                if step_i not in steps:
                    continue
            offset = int(entry["_offset"])
            length = int(entry["_length"])
            parts.append((offset, length))

    parts.sort(key=lambda x: x[0])
    total = sum(length for _, length in parts)
    return parts, total


def _merge_ranges(parts: Sequence[tuple[int, int]], *, max_gap_bytes: int) -> list[tuple[int, int]]:
    if not parts:
        return []
    merged: list[tuple[int, int]] = []
    cur_off, cur_len = parts[0]
    cur_end = cur_off + cur_len
    for off, length in parts[1:]:
        if off < cur_off:
            raise ValueError("Ranges not sorted")
        gap = off - cur_end
        if gap <= max_gap_bytes:
            cur_end = max(cur_end, off + length)
            continue
        merged.append((cur_off, cur_end - cur_off))
        cur_off = off
        cur_end = off + length
    merged.append((cur_off, cur_end - cur_off))
    return merged


def _estimated_download_bytes(parts: Sequence[tuple[int, int]], *, merge_gap_bytes: int) -> int:
    merged = _merge_ranges(parts, max_gap_bytes=merge_gap_bytes)
    return sum(length for _, length in merged)


def _download_ranges(
    url: str,
    dest: Path,
    *,
    parts: Sequence[tuple[int, int]],
    chunk_bytes: int,
    timeout_s: int,
    retries: int,
    merge_gap_bytes: int,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    merged = _merge_ranges(parts, max_gap_bytes=merge_gap_bytes)
    expected_size = sum(length for _, length in merged)

    if dest.exists() and dest.stat().st_size == expected_size:
        return

    for attempt in range(1, retries + 1):
        try:
            tmp.unlink(missing_ok=True)
            with tmp.open("wb") as f:
                for off, length in merged:
                    start = off
                    end = off + length - 1
                    headers = {
                        "User-Agent": "polymarket-weather-trading/1.0",
                        "Range": f"bytes={start}-{end}",
                    }
                    req = Request(url, headers=headers)
                    with urlopen(req, timeout=timeout_s) as resp:
                        status = getattr(resp, "status", 200)
                        if status != 206:
                            raise IOError(f"Server did not honor Range request (status={status}) for {dest.name}")
                        while True:
                            chunk = resp.read(chunk_bytes)
                            if not chunk:
                                break
                            f.write(chunk)

            if tmp.stat().st_size != expected_size:
                raise IOError(f"Incomplete subset download for {dest.name}: {tmp.stat().st_size} != {expected_size}")
            tmp.replace(dest)
            return
        except (HTTPError, URLError, TimeoutError, ConnectionError, IOError) as e:
            if attempt >= retries:
                raise
            backoff = min(60, 2**attempt)
            print(f"Subset download failed (attempt {attempt}/{retries}) for {dest.name}: {e}; retrying in {backoff}s")
            time.sleep(backoff)


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.2f}{unit}"
        value /= 1024
    return f"{n}B"


def _sum_sizes(files: Sequence[RemoteGrib2]) -> int:
    total = 0
    for f in files:
        if f.size_bytes is not None:
            total += f.size_bytes
    return total


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download ECMWF IFS 0p25 ENFO (ensemble) GRIB2 files into data/raster_data/ecmwf-ensemble/<run>/<yyyymmdd>/",
    )
    p.add_argument(
        "--preset",
        default="ifs-enfo",
        choices=sorted(PRESETS.keys()),
        help="Preconfigured remote directory/product set (ignored if --remote-dir-template is provided)",
    )
    p.add_argument("--date", default="utc-today", help="YYYY-MM-DD, YYYYMMDD, or 'utc-today'")
    p.add_argument("--run", default=None, help="00/06/12/18 or 00z/06z/12z/18z (default: auto)")
    p.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step hours and/or ranges, e.g. '0,24,48' or '0-240:3' (default: all available)",
    )
    p.add_argument(
        "--products",
        default=None,
        help="Comma-separated product types to include (the last token in the filename, e.g. 'ef', 'ep', 'fc', 'cf'); default: all",
    )
    p.add_argument(
        "--params",
        default=None,
        help="Subset to these GRIB param shortNames using the file .index (e.g. '2t,tp,2r,10u,10v' or 't2m,tp,rh2m,u10,v10')",
    )
    p.add_argument(
        "--name-regex",
        default=None,
        help="Regex applied to the GRIB2 filename as an additional filter (optional)",
    )
    p.add_argument("--timeout-s", type=int, default=60, help="HTTP timeout in seconds")
    p.add_argument("--retries", type=int, default=5, help="Retries per file")
    p.add_argument("--chunk-mb", type=int, default=8, help="Download chunk size in MiB")
    p.add_argument(
        "--max-total-gb",
        type=float,
        default=25.0,
        help="Abort if listed total size exceeds this many GiB (use --force-large to override)",
    )
    p.add_argument(
        "--force-large",
        action="store_true",
        help="Allow large total downloads (disables --max-total-gb abort)",
    )
    p.add_argument(
        "--subset-merge-gap-kb",
        type=int,
        default=0,
        help="When using --params, merge nearby byte ranges if the gap is <= this many KiB (downloads extra messages but reduces HTTP requests)",
    )
    p.add_argument("--dry-run", action="store_true", help="List files and destination directory without downloading")
    p.add_argument(
        "--dest-root",
        default=None,
        help="Override destination root (default: <repo>/data/raster_data/ecmwf-ensemble)",
    )
    p.add_argument(
        "--remote-dir-template",
        default=DEFAULT_REMOTE_DIR_TEMPLATE,
        help="Remote directory template; variables: {yyyymmdd}, {run}",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    now_utc = dt.datetime.now(dt.timezone.utc)

    if args.run is None:
        date, run = _auto_run(now_utc)
        if args.date not in {"today", "utc-today"}:
            date = _parse_date(args.date)
    else:
        date = _parse_date(args.date)
        run = _parse_run(args.run)

    yyyymmdd = date.strftime("%Y%m%d")
    dest_root = Path(args.dest_root) if args.dest_root else _repo_root() / "data/raster_data"
    dest_dir = dest_root / run / yyyymmdd

    remote_template = args.remote_dir_template
    products = _parse_products(args.products)
    if args.remote_dir_template == DEFAULT_REMOTE_DIR_TEMPLATE:
        preset = PRESETS.get(args.preset, {})
        remote_template = str(preset.get("remote_dir_template", remote_template))
        if products is None and "products" in preset:
            products = set(preset["products"])  # type: ignore[arg-type]

    remote_dir = remote_template.format(yyyymmdd=yyyymmdd, run=run)
    steps = _parse_steps(args.steps)
    params = _parse_params(args.params)
    name_regex = re.compile(args.name_regex) if args.name_regex else None

    files = list_remote_grib2(
        remote_dir,
        products=products,
        steps=steps,
        name_regex=name_regex,
        timeout_s=args.timeout_s,
    )
    if not files:
        print(f"No GRIB2 files found at {remote_dir}")
        return 2

    total = _sum_sizes(files)
    print(f"Remote: {remote_dir}")
    print(f"Dest:   {dest_dir}")
    print(f"Files:  {len(files)} (listed size: {_fmt_bytes(total)})")

    if params is not None:
        print(f"Subset params: {', '.join(params)}")

    if args.dry_run:
        if params is None:
            for f in files[:50]:
                size_s = _fmt_bytes(f.size_bytes) if f.size_bytes is not None else "?"
                print(f"- {f.name} ({size_s})")
            if len(files) > 50:
                print(f"... ({len(files) - 50} more)")
            return 0

        subset_total = 0
        for f in files[:20]:
            parts, subset_bytes = _read_index_parts(
                f.url,
                params=set(params),
                steps=steps,
                timeout_s=args.timeout_s,
            )
            planned = _estimated_download_bytes(parts, merge_gap_bytes=args.subset_merge_gap_kb * 1024)
            subset_total += planned
            print(f"- {f.name} -> {_subset_output_name(f.name, params)} ({len(parts)} msgs, {_fmt_bytes(planned)})")
        if len(files) > 20:
            print(f"... ({len(files) - 20} more files; subset sizes not computed in dry-run output)")
        print(f"Subset (first {min(20, len(files))} files) total: {_fmt_bytes(subset_total)}")
        return 0

    if params is None and not args.force_large:
        limit_bytes = int(args.max_total_gb * 1024 * 1024 * 1024)
        if total > limit_bytes:
            print(
                f"Refusing to download {_fmt_bytes(total)} (exceeds --max-total-gb={args.max_total_gb}). "
                f"Use --force-large to override, or choose a smaller preset like --preset ifs-oper / aifs-single-oper / ifs-enfo-ep."
            )
            return 3

    subset_limit_bytes = int(args.max_total_gb * 1024 * 1024 * 1024)
    subset_running_bytes = 0

    for i, f in enumerate(files, start=1):
        if params is None:
            dest_path = dest_dir / f.name
            size_s = _fmt_bytes(f.size_bytes) if f.size_bytes is not None else "?"
            print(f"[{i}/{len(files)}] {f.name} ({size_s})")
            _download_with_resume(
                f.url,
                dest_path,
                expected_size=f.size_bytes,
                chunk_bytes=args.chunk_mb * 1024 * 1024,
                timeout_s=args.timeout_s,
                retries=args.retries,
            )
            continue

        parts, subset_bytes = _read_index_parts(
            f.url,
            params=set(params),
            steps=steps,
            timeout_s=args.timeout_s,
        )
        if not parts:
            print(f"[{i}/{len(files)}] {f.name}: no matching index entries for params={params}")
            continue
        planned = _estimated_download_bytes(parts, merge_gap_bytes=args.subset_merge_gap_kb * 1024)
        dest_name = _subset_output_name(f.name, params)
        dest_path = dest_dir / dest_name
        if dest_path.exists() and dest_path.stat().st_size == planned:
            print(f"[{i}/{len(files)}] {dest_name} already present ({_fmt_bytes(planned)})")
            continue
        subset_running_bytes += planned
        if not args.force_large and subset_running_bytes > subset_limit_bytes:
            print(
                f"Refusing to download subset {_fmt_bytes(subset_running_bytes)} (exceeds --max-total-gb={args.max_total_gb}). "
                f"Use --force-large to override, or reduce --steps / number of files."
            )
            return 3
        print(f"[{i}/{len(files)}] {f.name} -> {dest_name} ({len(parts)} msgs, {_fmt_bytes(planned)})")
        _download_ranges(
            f.url,
            dest_path,
            parts=parts,
            chunk_bytes=args.chunk_mb * 1024 * 1024,
            timeout_s=args.timeout_s,
            retries=args.retries,
            merge_gap_bytes=args.subset_merge_gap_kb * 1024,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
