"""Build a static PIOMAS ocean mask on the manuscript's 0.25-degree grid.

The local native PIOMAS files use a stable land convention: SIC is exactly 1
and SIT is exactly 0.  The native mask is remapped with spherical nearest
neighbour interpolation, matching the nearest-neighbour PIOMAS field remap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from .mask import (
    derive_piomas_native_ocean_mask,
    remap_boolean_mask_nearest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piomas-root", type=Path, required=True)
    parser.add_argument("--regridded-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from .config import RESULTS

    output = args.output or RESULTS / "masks" / "piomas_ocean_mask_025deg.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshots: list[np.ndarray] = []
    native_lat = native_lon = None
    validation_rows = []
    for year, indices in (
        (1979, (0, 182, 364)),
        (2000, (0, 182, 364)),
        (2025, (0, 182, 364)),
    ):
        sic_path = args.piomas_root / "SIC" / f"PIOMAS_SIC_{year}.nc"
        sit_path = args.piomas_root / "SIT" / f"PIOMAS_SIT_{year}.nc"
        with xr.open_dataset(sic_path, decode_times=False) as sic_ds, xr.open_dataset(
            sit_path, decode_times=False
        ) as sit_ds:
            if native_lat is None:
                native_lat = np.asarray(sic_ds["lat"].values, dtype=np.float64)
                native_lon = np.asarray(sic_ds["lon"].values, dtype=np.float64)
            for index in indices:
                ocean = derive_piomas_native_ocean_mask(
                    sic_ds["sic"].isel(time=index).values,
                    sit_ds["sit"].isel(time=index).values,
                )
                snapshots.append(ocean)
                validation_rows.append(
                    {
                        "year": year,
                        "time_index": index,
                        "native_ocean_cells": int(ocean.sum()),
                        "native_land_cells": int((~ocean).sum()),
                    }
                )

    reference_native = snapshots[0]
    stability_differences = [
        int(np.sum(mask != reference_native)) for mask in snapshots
    ]
    if any(stability_differences):
        raise RuntimeError(
            "PIOMAS native land convention is not static across validation snapshots: "
            f"{stability_differences}"
        )

    regridded_sic_path = (
        args.regridded_root / "SIC" / "regridded_ERA5_PIOMAS_SIC_1979.nc"
    )
    regridded_sit_path = (
        args.regridded_root / "SIT" / "regridded_ERA5_PIOMAS_SIT_1979.nc"
    )
    with xr.open_dataset(
        regridded_sic_path, decode_times=False
    ) as sic_ds, xr.open_dataset(regridded_sit_path, decode_times=False) as sit_ds:
        target_lat = np.asarray(sic_ds["latitude"].values, dtype=np.float64)
        target_lon = np.asarray(sic_ds["longitude"].values, dtype=np.float64)
        remapped_ocean = remap_boolean_mask_nearest(
            native_lat,
            native_lon,
            reference_native,
            target_lat,
            target_lon,
        )
        regridded_snapshot_ocean = derive_piomas_native_ocean_mask(
            sic_ds["sic"].isel(time=0).values,
            sit_ds["sit"].isel(time=0).values,
        )

    mismatch = remapped_ocean != regridded_snapshot_ocean
    np.savez_compressed(
        output,
        ocean_mask=remapped_ocean,
        land_mask=~remapped_ocean,
        latitude=target_lat,
        longitude=target_lon,
        native_ocean_mask=reference_native,
        native_latitude=native_lat,
        native_longitude=native_lon,
        derivation="native PIOMAS land convention: SIC == 1 and SIT == 0",
        remapping="nearest neighbour on unit-sphere coordinates",
    )

    validation = {
        "output": str(output),
        "native_shape": list(reference_native.shape),
        "target_shape": list(remapped_ocean.shape),
        "native_ocean_cells": int(reference_native.sum()),
        "native_land_cells": int((~reference_native).sum()),
        "target_ocean_cells": int(remapped_ocean.sum()),
        "target_land_cells": int((~remapped_ocean).sum()),
        "static_snapshot_max_difference_cells": max(stability_differences),
        "remapped_vs_regridded_snapshot_difference_cells": int(mismatch.sum()),
        "snapshots": validation_rows,
        "note": (
            "The regridded snapshot check is diagnostic only. The saved mask is "
            "derived from the stable native PIOMAS land convention."
        ),
    }
    validation_path = output.with_name(f"{output.stem}_validation.json")
    with validation_path.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, ensure_ascii=False)
    print(f"Saved ocean mask: {output}")
    print(f"Saved validation: {validation_path}")


if __name__ == "__main__":
    main()
