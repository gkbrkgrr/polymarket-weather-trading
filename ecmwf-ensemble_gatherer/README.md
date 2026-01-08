# ECMWF ensemble gatherer

Downloads ECMWF Open Data Portal IFS ensemble (ENFO) GRIB2 files into:

`data/raster_data/ecmwf-ensemble/<run>/<yyyymmdd>/`

Example: the `2026-01-06 00Z` run is stored under `data/raster_data/ecmwf-ensemble/00z/20260106/`.

## Usage

Dry-run (lists files, creates no downloads):

```bash
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-enfo --date 2026-01-06 --run 00z --dry-run
```

Download:

```bash
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-enfo --date 2026-01-06 --run 00z --force-large
```

Faster download (parallel files + multi-connection per file via HTTP Range):

```bash
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-enfo --date 2026-01-06 --run 00z --force-large --workers 8 --connections-per-file 4
```

Note: `ifs/0p25/enfo` `ef` files are very large (multi-GB each). Prefer a smaller preset unless you truly need the full raw ensemble fields.

Smaller alternatives:

```bash
# IFS deterministic (much smaller per step)
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-oper --date 2026-01-06 --run 00z

# AIFS deterministic (even smaller per step)
python ecmwf-ensemble_gatherer/download_pipeline.py --preset aifs-single-oper --date 2026-01-06 --run 00z

# AIFS ensemble (control + perturbed members)
python ecmwf-ensemble_gatherer/download_pipeline.py --preset aifs-enfo --date 2026-01-06 --run 00z --products cf,pf

# IFS ensemble probabilities (if available for that run)
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-enfo-ep --date 2026-01-06 --run 00z
```

Subset variables before downloading (uses the `.index` file + HTTP range requests):

```bash
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-oper --date 2026-01-06 --run 00z --params t2m,tp,u10,v10
```

Notes:
- `rh2m` is commonly not present as a direct GRIB param in ENFO. If you pass `rh2m` here, the downloader includes `t2m` + `d2m` (`2t` + `2d`) so you can derive RH downstream.
- If you're bandwidth/latency constrained, `--params` + `--steps` is usually a much bigger win than parallelizing full-file downloads.
- You can still parallelize subset downloads across files with `--workers` (the script computes subset sizes first to enforce `--max-total-gb`).

## Spatial (extent) subsetting

This downloader can subset by GRIB *messages* (param/step) using the `.index` file, but it cannot do true spatial subsetting via HTTP range requests: each GRIB message typically contains the full grid, and the `.index` does not provide per-message bounding boxes.

If you want a smaller region on disk, subset after reading (e.g. crop to a bounding box and write NetCDF/Zarr). A helper exists in `weather_data/grib_reader.py`:

```python
from weather_data.grib_reader import BBox, subset_bbox

roi = subset_bbox(ds, BBox(north=50, west=-125, south=20, east=-65))  # CONUS example
```

Limit by forecast steps (hours):

```bash
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-oper --date 2026-01-06 --run 00z --steps 0,24,48
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-oper --date 2026-01-06 --run 00z --steps 0-240:3
```

Limit by product type (the last token in the filename, e.g. `ef`, `ep`, `fc`, `cf`):

```bash
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-enfo --date 2026-01-06 --run 00z --products ep
```

## Low-latency GRIB -> parquet pipeline

This repo includes a streaming pipeline that downloads timestep-by-timestep and processes each timestep into per-city parquets while the next timestep downloads (to avoid latency gaps):

```bash
python ecmwf-ensemble_gatherer/streaming_parquet_pipeline.py --preset ifs-enfo --date 2026-01-06 --run 00z --steps 0-240:3 --params t2m,u10,v10,tp --workers 8 --force-large
```

Outputs are written under:

`data/point_data/ecmwf-ensemble/<city_name>/raw/ecmwf-ensemble_indexes_<forecasted_at>_<max_datetime>.parquet`

Notes:
- By default, the streaming pipeline *subsets* downloads using the GRIB `.index` and `--params`. Use `--no-subset` to download full GRIB files.
- If you previously downloaded into `data/raster_data/<run>/<yyyymmdd>/`, pass `--dest-root data/raster_data` to keep that layout.
