from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import xarray as xr


class BackendUnavailableError(RuntimeError):
    pass


PARAM_ALIASES: dict[str, str] = {
    "t2m": "2t",
    "rh2m": "2r",
    "d2m": "2d",
    "u10": "10u",
    "v10": "10v",
}

_DROP_COORDS = {
    "heightAboveGround",
    "surface",
    "isobaricInhPa",
    "meanSea",
    "hybrid",
    "depthBelowLandLayer",
}


def normalize_param(param: str) -> str:
    param = param.strip().lower()
    if not param:
        raise ValueError("param cannot be empty")
    return PARAM_ALIASES.get(param, param)


def preferred_var_name(param: str) -> str:
    return param.strip()


def _default_cache_dir() -> Path:
    env = os.environ.get("POLY_WEATHER_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "data/grib_cache").resolve()


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


class _LRUCache:
    def __init__(self, max_items: int, on_evict: Optional[Callable[[Any], None]] = None):
        if max_items <= 0:
            raise ValueError("max_items must be > 0")
        self._max_items = max_items
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._on_evict = on_evict

    def get(self, key: Any) -> Any:
        value = self._data.pop(key)  # raises KeyError
        self._data[key] = value
        return value

    def put(self, key: Any, value: Any) -> None:
        if key in self._data:
            old = self._data.pop(key)
            if self._on_evict is not None:
                self._on_evict(old)
        self._data[key] = value
        while len(self._data) > self._max_items:
            _, evicted = self._data.popitem(last=False)
            if self._on_evict is not None:
                self._on_evict(evicted)

    def clear(self) -> None:
        if self._on_evict is not None:
            for v in self._data.values():
                self._on_evict(v)
        self._data.clear()


@dataclass(frozen=True)
class GribDatasetKey:
    path: str
    engine: str
    filter_by_keys_json: str
    indexpath: str


class GribXarrayReader:
    """
    Thin, cache-aware GRIB -> xarray reader.

    - Prefers `cfgrib` backend when installed (xarray engine="cfgrib").
    - Caches opened datasets to avoid repeated indexing/parsing.
    - Encourages `filter_by_keys` to avoid loading irrelevant messages.

    Notes:
    - GRIB reading requires external deps. Install one of:
      - cfgrib + eccodes (recommended): `pip install cfgrib eccodes`

    Example:
        reader = GribXarrayReader()
        ds = reader.read_ensemble_params(
            "data/raster_data/ecmwf-ensemble/00z/20260106/20260106000000-0h-enfo-ef__params-10u-10v-2r-2t-tp.grib2",
            ["t2m", "tp", "rh2m", "u10", "v10"],
        )
    """

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        dataset_cache_size: int = 8,
        datasets_cache_size: int = 4,
    ):
        self._cache_dir = (cache_dir or _default_cache_dir()).expanduser().resolve()
        self._ds_cache = _LRUCache(dataset_cache_size, on_evict=_close_xarray)
        self._dss_cache = _LRUCache(datasets_cache_size, on_evict=_close_xarray_list)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def clear_cache(self) -> None:
        self._ds_cache.clear()
        self._dss_cache.clear()

    def open_dataset(
        self,
        path: str | Path,
        *,
        filter_by_keys: dict[str, Any] | None = None,
        engine: str = "cfgrib",
        indexpath: str | Path | None = None,
        allow_empty: bool = False,
    ) -> xr.Dataset:
        path = Path(path).expanduser().resolve()
        filter_by_keys = dict(filter_by_keys or {})

        if engine != "cfgrib":
            raise BackendUnavailableError(f"Unsupported engine: {engine!r} (only 'cfgrib' is supported right now)")
        _require_cfgrib()

        indexpath_s = str(indexpath) if indexpath is not None else ""
        key = GribDatasetKey(
            path=str(path),
            engine=engine,
            filter_by_keys_json=json.dumps(filter_by_keys, sort_keys=True, separators=(",", ":"), default=str),
            indexpath=str(indexpath_s),
        )
        try:
            return self._ds_cache.get(key)
        except KeyError:
            pass

        try:
            ds = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": filter_by_keys,
                    "indexpath": indexpath_s,
                },
            )
        except Exception as e:
            if e.__class__.__name__ == "DatasetBuildError":
                raise KeyError(
                    f"cfgrib failed to build a single dataset for file {str(path)!r} with filter_by_keys={filter_by_keys!r}. "
                    "This usually means you need to add a key like 'dataType' (e.g. 'pf' vs 'cf') or otherwise narrow the filter."
                ) from e
            raise
        if not allow_empty and len(ds.data_vars) == 0:
            _close_xarray(ds)
            raise KeyError(
                "No GRIB messages matched filter_by_keys="
                f"{filter_by_keys!r} for file {str(path)!r}. "
                "If you're reading 2m/10m fields, try filter_by_keys={'typeOfLevel':'heightAboveGround','level':2 or 10}."
            )
        self._ds_cache.put(key, ds)
        return ds

    def open_datasets(
        self,
        path: str | Path,
        *,
        filter_by_keys: dict[str, Any] | None = None,
        engine: str = "cfgrib",
        indexpath: str | Path | None = None,
        allow_empty: bool = False,
    ) -> list[xr.Dataset]:
        path = Path(path).expanduser().resolve()
        filter_by_keys = dict(filter_by_keys or {})

        if engine != "cfgrib":
            raise BackendUnavailableError(f"Unsupported engine: {engine!r} (only 'cfgrib' is supported right now)")
        _require_cfgrib()

        indexpath_s = str(indexpath) if indexpath is not None else ""
        key = GribDatasetKey(
            path=str(path),
            engine=engine,
            filter_by_keys_json=json.dumps(filter_by_keys, sort_keys=True, separators=(",", ":"), default=str),
            indexpath=str(indexpath_s),
        )
        try:
            return list(self._dss_cache.get(key))
        except KeyError:
            pass

        import cfgrib  # type: ignore

        try:
            datasets = cfgrib.open_datasets(
                str(path),
                backend_kwargs={
                    "filter_by_keys": filter_by_keys,
                    "indexpath": indexpath_s,
                },
            )
        except Exception as e:
            if allow_empty:
                return []
            raise KeyError(
                f"cfgrib failed to open datasets for file {str(path)!r} with filter_by_keys={filter_by_keys!r}"
            ) from e

        if not allow_empty and not datasets:
            raise KeyError(f"No datasets returned for filter_by_keys={filter_by_keys!r} on file {str(path)!r}")

        self._dss_cache.put(key, tuple(datasets))
        return datasets

    def read_param(
        self,
        path: str | Path,
        param: str,
        *,
        filter_by_keys: dict[str, Any] | None = None,
        engine: str = "cfgrib",
        indexpath: str | Path | None = None,
        name: str | None = None,
        auto_fix_filter_by_keys: bool = True,
        data_types: tuple[str, ...] | None = None,
    ) -> xr.DataArray:
        short_name = normalize_param(param)
        merged = dict(filter_by_keys or {})

        if data_types is not None:
            da = self._read_ensemble_short_name(
                path,
                short_name,
                data_types=data_types,
                filter_by_keys=merged,
                engine=engine,
                indexpath=indexpath,
                auto_fix_filter_by_keys=auto_fix_filter_by_keys,
            )
            return da.rename(name or param)

        merged.setdefault("shortName", short_name)

        try:
            ds = self.open_dataset(
                path,
                filter_by_keys=merged,
                engine=engine,
                indexpath=indexpath,
                allow_empty=False,
            )
        except KeyError:
            if not auto_fix_filter_by_keys:
                raise
            fixed = _maybe_fix_filter_by_keys(short_name, merged)
            if fixed is None:
                raise
            ds = self.open_dataset(
                path,
                filter_by_keys=fixed,
                engine=engine,
                indexpath=indexpath,
                allow_empty=False,
            )
        da = _select_var_by_short_name(ds, short_name)
        da = _canonicalize_grib_dataarray(da)
        return da.rename(name or param)

    def read_params(
        self,
        path: str | Path,
        params: list[str],
        *,
        filter_by_keys: dict[str, Any] | None = None,
        engine: str = "cfgrib",
        indexpath: str | Path | None = None,
        auto_fix_filter_by_keys: bool = True,
    ) -> xr.Dataset:
        requested: list[tuple[str, str]] = [(preferred_var_name(p), normalize_param(p)) for p in params]
        base = dict(filter_by_keys or {})

        datasets = self.open_datasets(path, filter_by_keys=base, engine=engine, indexpath=indexpath, allow_empty=False)

        out: dict[str, xr.DataArray] = {}
        for display_name, short_name in requested:
            da = _find_by_short_name(datasets, short_name)
            if da is None and auto_fix_filter_by_keys:
                fbk = dict(base)
                fbk.setdefault("shortName", short_name)
                fixed = _maybe_fix_filter_by_keys(short_name, fbk)
                if fixed is not None:
                    datasets2 = self.open_datasets(
                        path, filter_by_keys={k: v for k, v in fixed.items() if k != "shortName"}, engine=engine, indexpath=indexpath, allow_empty=True
                    )
                    da = _find_by_short_name(datasets2, short_name)
            if da is None:
                if display_name == "rh2m":
                    # Try derive from 2m temperature + 2m dewpoint if available.
                    t2m = _find_by_short_name(datasets, "2t")
                    d2m = _find_by_short_name(datasets, "2d")
                    if t2m is None or d2m is None:
                        raise KeyError(
                            "rh2m not present as a GRIB param in this dataset. "
                            "Derivation requires `t2m` (2t) and `d2m` (2d) to be present in the file."
                        )
                    out[display_name] = derive_rh2m(_canonicalize_grib_dataarray(t2m), _canonicalize_grib_dataarray(d2m))
                    continue
                raise KeyError(f"Param shortName {short_name!r} not found in file {str(Path(path).expanduser())!r}")
            out[display_name] = _canonicalize_grib_dataarray(da)

        return xr.Dataset(out)

    def open_ensemble_dataset(
        self,
        path: str | Path,
        *,
        data_types: tuple[str, ...] = ("cf", "pf"),
        filter_by_keys: dict[str, Any] | None = None,
        engine: str = "cfgrib",
        indexpath: str | Path | None = None,
        member_dim: str = "number",
    ) -> xr.Dataset:
        base = dict(filter_by_keys or {})
        base.pop("dataType", None)
        dss: list[xr.Dataset] = []
        for dt in data_types:
            fbk = dict(base)
            fbk["dataType"] = dt
            dss.extend(self.open_datasets(path, filter_by_keys=fbk, engine=engine, indexpath=indexpath, allow_empty=True))

        if not dss:
            raise KeyError(f"No datasets found for data_types={data_types!r} with filter_by_keys={base!r}")

        combined = xr.merge([_drop_problematic_coords(ds) for ds in dss], compat="override")
        # Ensure member dim exists if possible
        if member_dim in combined.coords and member_dim not in combined.dims:
            combined = combined.expand_dims({member_dim: [int(combined.coords[member_dim].values)]})
        return combined

    def read_ensemble_params(
        self,
        path: str | Path,
        params: list[str],
        *,
        data_types: tuple[str, ...] = ("cf", "pf"),
        filter_by_keys: dict[str, Any] | None = None,
        engine: str = "cfgrib",
        indexpath: str | Path | None = None,
        member_dim: str = "number",
        auto_fix_filter_by_keys: bool = True,
    ) -> xr.Dataset:
        requested: list[tuple[str, str]] = [(preferred_var_name(p), normalize_param(p)) for p in params]
        base = dict(filter_by_keys or {})
        base.pop("dataType", None)

        by_short: dict[str, list[xr.DataArray]] = {sn: [] for _, sn in requested}

        for dt in data_types:
            fbk = dict(base)
            fbk["dataType"] = dt
            dss = self.open_datasets(path, filter_by_keys=fbk, engine=engine, indexpath=indexpath, allow_empty=True)
            for _, sn in requested:
                da = _find_by_short_name(dss, sn)
                if da is None and auto_fix_filter_by_keys:
                    fbk2 = dict(fbk)
                    fbk2.setdefault("shortName", sn)
                    fixed = _maybe_fix_filter_by_keys(sn, fbk2)
                    if fixed is not None:
                        dss2 = self.open_datasets(
                            path,
                            filter_by_keys={k: v for k, v in fixed.items() if k != "shortName"},
                            engine=engine,
                            indexpath=indexpath,
                            allow_empty=True,
                        )
                        da = _find_by_short_name(dss2, sn)
                if da is None:
                    continue
                da = _canonicalize_grib_dataarray(da)
                da = _ensure_member_dim(da, member_dim)
                by_short[sn].append(da)

        out: dict[str, xr.DataArray] = {}
        for display_name, sn in requested:
            if by_short.get(sn):
                if len(by_short[sn]) == 1:
                    out[display_name] = by_short[sn][0]
                else:
                    out[display_name] = xr.concat(by_short[sn], dim=member_dim).sortby(member_dim)
            else:
                if display_name == "rh2m":
                    t2m = self._read_ensemble_short_name(
                        path,
                        "2t",
                        data_types=data_types,
                        filter_by_keys=base,
                        engine=engine,
                        indexpath=indexpath,
                        auto_fix_filter_by_keys=auto_fix_filter_by_keys,
                        member_dim=member_dim,
                    )
                    try:
                        d2m = self._read_ensemble_short_name(
                            path,
                            "2d",
                            data_types=data_types,
                            filter_by_keys=base,
                            engine=engine,
                            indexpath=indexpath,
                            auto_fix_filter_by_keys=auto_fix_filter_by_keys,
                            member_dim=member_dim,
                        )
                    except KeyError as e:
                        raise KeyError(
                            "rh2m derivation requires `d2m` (shortName '2d') to be present in the GRIB file. "
                            "Include `d2m` when downloading/subsetting."
                        ) from e
                    out[display_name] = derive_rh2m(t2m, d2m)
                    continue
                raise KeyError(f"Param shortName {sn!r} not found for any dataType in {data_types!r}")

        return xr.Dataset(out)

    def _read_ensemble_short_name(
        self,
        path: str | Path,
        short_name: str,
        *,
        data_types: tuple[str, ...],
        filter_by_keys: dict[str, Any],
        engine: str,
        indexpath: str | Path | None,
        auto_fix_filter_by_keys: bool,
        member_dim: str = "number",
    ) -> xr.DataArray:
        base = dict(filter_by_keys)
        base.pop("dataType", None)
        arrays: list[xr.DataArray] = []

        for dt in data_types:
            fbk = dict(base)
            fbk["dataType"] = dt
            fbk["shortName"] = short_name
            if auto_fix_filter_by_keys:
                fixed = _maybe_fix_filter_by_keys(short_name, fbk)
                if fixed is not None:
                    fbk = fixed
            dss = self.open_datasets(path, filter_by_keys=fbk, engine=engine, indexpath=indexpath, allow_empty=True)
            da = _find_by_short_name(dss, short_name)
            if da is None:
                continue
            da = _canonicalize_grib_dataarray(da)
            da = _ensure_member_dim(da, member_dim)
            arrays.append(da)

        if not arrays:
            raise KeyError(f"Param shortName {short_name!r} not found for any dataType in {data_types!r}")
        if len(arrays) == 1:
            return arrays[0]
        return xr.concat(arrays, dim=member_dim).sortby(member_dim)

    def read_param_many(
        self,
        paths: list[str | Path],
        param: str,
        *,
        filter_by_keys: dict[str, Any] | None = None,
        engine: str = "cfgrib",
        indexpath: str | Path | None = None,
        name: str | None = None,
        concat_dim: str = "step",
    ) -> xr.DataArray:
        arrays: list[xr.DataArray] = []
        for i, p in enumerate(paths):
            da = self.read_param(
                p,
                param,
                filter_by_keys=filter_by_keys,
                engine=engine,
                indexpath=indexpath,
                name=name or param,
            )
            if concat_dim in da.dims:
                arrays.append(da)
            elif concat_dim in da.coords:
                arrays.append(da.expand_dims({concat_dim: da.coords[concat_dim]}))
            else:
                arrays.append(da.expand_dims({concat_dim: [i]}))
        return xr.concat(arrays, dim=concat_dim)

    def _default_indexpath(self, path: Path, filter_by_keys: dict[str, Any]) -> str:
        base = self.cache_dir / "cfgrib"
        base.mkdir(parents=True, exist_ok=True)
        st = path.stat()
        payload = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        h = _stable_hash(payload)
        return str((base / f"{path.stem}.{h}.idx").resolve())


def _close_xarray(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _close_xarray_list(obj: Any) -> None:
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _close_xarray(item)
        return
    _close_xarray(obj)


def _require_cfgrib() -> None:
    try:
        import cfgrib  # noqa: F401
    except Exception as e:
        raise BackendUnavailableError(
            "GRIB reading requires 'cfgrib' (and an eccodes install). Try: pip install cfgrib eccodes"
        ) from e


def _maybe_fix_filter_by_keys(short_name: str, filter_by_keys: dict[str, Any]) -> dict[str, Any] | None:
    recommendations: dict[str, dict[str, Any]] = {
        "2t": {"typeOfLevel": "heightAboveGround", "level": 2},
        "2r": {"typeOfLevel": "heightAboveGround", "level": 2},
        "2d": {"typeOfLevel": "heightAboveGround", "level": 2},
        "10u": {"typeOfLevel": "heightAboveGround", "level": 10},
        "10v": {"typeOfLevel": "heightAboveGround", "level": 10},
        "tp": {"typeOfLevel": "surface"},
    }
    rec = recommendations.get(short_name)
    if rec is None:
        return None

    updated = dict(filter_by_keys)
    for k in ("typeOfLevel", "level"):
        if k in rec:
            updated[k] = rec[k]
    if updated == filter_by_keys:
        return None
    return updated


def _select_var_name_and_data_by_short_name(ds: xr.Dataset, short_name: str) -> tuple[str, xr.DataArray]:
    if short_name in ds.data_vars:
        return short_name, ds[short_name]
    for name, da in ds.data_vars.items():
        if da.attrs.get("GRIB_shortName") == short_name:
            return name, da
    available = ", ".join(sorted(ds.data_vars.keys()))
    raise KeyError(f"Param shortName {short_name!r} not found in dataset; available: {available}")


def _select_var_by_short_name(ds: xr.Dataset, short_name: str) -> xr.DataArray:
    return _select_var_name_and_data_by_short_name(ds, short_name)[1]


def _find_by_short_name(datasets: list[xr.Dataset], short_name: str) -> xr.DataArray | None:
    for ds in datasets:
        try:
            _, da = _select_var_name_and_data_by_short_name(ds, short_name)
            return da
        except KeyError:
            continue
    return None


def _canonicalize_grib_dataarray(da: xr.DataArray) -> xr.DataArray:
    for coord in _DROP_COORDS:
        if coord in da.dims:
            if da.sizes.get(coord, 0) == 1:
                da = da.squeeze(coord, drop=True)
        if coord in da.coords and coord not in da.dims:
            da = da.reset_coords(coord, drop=True)
    return da


def _ensure_member_dim(da: xr.DataArray, member_dim: str) -> xr.DataArray:
    if member_dim in da.dims:
        return da
    if member_dim in da.coords:
        value = da.coords[member_dim].values
        try:
            value_i = int(value)
        except Exception:
            value_i = 0
        da = da.reset_coords(member_dim, drop=True)
        return da.expand_dims({member_dim: [value_i]})
    return da.expand_dims({member_dim: [0]})


def _drop_problematic_coords(ds: xr.Dataset) -> xr.Dataset:
    drop = [c for c in _DROP_COORDS if c in ds.coords]
    if not drop:
        return ds
    return ds.reset_coords(drop, drop=True)


def derive_rh2m(t2m_k: xr.DataArray, d2m_k: xr.DataArray) -> xr.DataArray:
    """
    Derive 2m relative humidity (%) from 2m temperature and 2m dewpoint.
    Inputs are expected in Kelvin.
    """
    t_c = t2m_k - 273.15
    td_c = d2m_k - 273.15

    # Magnus formula (over water) in hPa
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    rh = 100.0 * (e / es)
    rh = rh.clip(0.0, 100.0)
    rh.attrs = dict(t2m_k.attrs)
    rh.attrs.update(
        {
            "units": "%",
            "long_name": "2 metre relative humidity (derived from t2m and d2m)",
        }
    )
    return rh
