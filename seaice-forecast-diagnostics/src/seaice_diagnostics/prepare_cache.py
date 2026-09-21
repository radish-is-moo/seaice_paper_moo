"""Build daily caches from aligned, already regridded ERA5/PIOMAS NetCDF."""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm


def get_grid_coordinates(era5_sample_path: str) -> np.ndarray:
    ds = xr.open_dataset(era5_sample_path)
    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return np.column_stack([lat_grid.reshape(-1), lon_grid.reshape(-1)])


def _cache_date_key(date_like) -> str:
    return pd.Timestamp(date_like).strftime("%Y%m%d")


def create_or_load_grid_daily_cache(
    era5_file_list,
    piomas_sic_file_list,
    piomas_sit_file_list,
    valid_dates,
    cache_dir,
    variables=None,
    overwrite=False,
):
    variables = variables or ["t2m", "d2m", "u10", "v10", "msl", "sst", "sic", "sit"]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_paths = [cache_dir / f"{_cache_date_key(d)}.npz" for d in valid_dates]
    if all(p.exists() for p in expected_paths) and not overwrite:
        print(f"Daily grid cache already complete: {cache_dir}")
        return cache_dir

    print("Preparing daily grid cache without scale separation:", cache_dir)
    ds_era5 = xr.open_mfdataset(
        era5_file_list, combine="by_coords", engine="netcdf4", parallel=False
    )
    ds_sic = xr.open_mfdataset(
        piomas_sic_file_list, combine="by_coords", engine="netcdf4", parallel=False
    ).reindex(time=ds_era5.time, method="nearest")
    ds_sit = xr.open_mfdataset(
        piomas_sit_file_list, combine="by_coords", engine="netcdf4", parallel=False
    ).reindex(time=ds_era5.time, method="nearest")
    ds_piomas = xr.merge([ds_sic, ds_sit], compat="override")

    era5_dates = pd.DatetimeIndex(ds_era5.time.values).normalize()
    date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(era5_dates)}
    missing_dates = []

    for date_like in tqdm(valid_dates, desc="cache daily fields"):
        date_norm = pd.Timestamp(date_like).normalize()
        cache_path = cache_dir / f"{_cache_date_key(date_norm)}.npz"
        if cache_path.exists() and not overwrite:
            continue
        if date_norm not in date_to_idx:
            missing_dates.append(str(date_norm.date()))
            continue

        t_idx = date_to_idx[date_norm]
        daily_list = []
        sic_full = None
        for var in variables:
            if var in ["sic", "sit"]:
                arr_2d = ds_piomas[var].isel(time=t_idx).values.astype(np.float32)
            else:
                arr_2d = ds_era5[var].isel(time=t_idx).values.astype(np.float32)
            daily_list.append(arr_2d)
            if var == "sic":
                sic_full = arr_2d.copy()

        np.savez_compressed(
            cache_path,
            daily=np.stack(daily_list, axis=0).astype(np.float32),
            sic_full=sic_full.astype(np.float32),
        )

    ds_era5.close()
    ds_sic.close()
    ds_sit.close()
    ds_piomas.close()

    if missing_dates:
        raise ValueError(
            f"Missing dates while generating daily grid cache. Examples: {missing_dates[:5]}"
        )
    print("Daily grid cache ready:", cache_dir)
    return cache_dir
