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

Note: `ifs/0p25/enfo` `ef` files are very large (multi-GB each). Prefer a smaller preset unless you truly need the full raw ensemble fields.

Smaller alternatives:

```bash
# IFS deterministic (much smaller per step)
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-oper --date 2026-01-06 --run 00z

# AIFS deterministic (even smaller per step)
python ecmwf-ensemble_gatherer/download_pipeline.py --preset aifs-single-oper --date 2026-01-06 --run 00z

# IFS ensemble probabilities (if available for that run)
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-enfo-ep --date 2026-01-06 --run 00z
```

Subset variables before downloading (uses the `.index` file + HTTP range requests):

```bash
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-oper --date 2026-01-06 --run 00z --params t2m,tp,u10,v10
```

Notes:
- `rh2m` is commonly not present as a direct GRIB param in ENFO. If you pass `rh2m` here, the downloader includes `t2m` + `d2m` (`2t` + `2d`) so you can derive RH downstream.

Limit by forecast steps (hours):

```bash
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-oper --date 2026-01-06 --run 00z --steps 0,24,48
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-oper --date 2026-01-06 --run 00z --steps 0-240:3
```

Limit by product type (the last token in the filename, e.g. `ef`, `ep`, `fc`, `cf`):

```bash
python ecmwf-ensemble_gatherer/download_pipeline.py --preset ifs-enfo --date 2026-01-06 --run 00z --products ep
```
