"""Portable paths; set SEAICE_CONFIG to a JSON file before importing modules."""

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path(
    os.environ.get("SEAICE_CONFIG", REPO / "configs/paths.json")
).resolve()
VALUES = (
    json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}
)


def path(key, default):
    value = Path(os.path.expandvars(VALUES.get(key, default))).expanduser()
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


RESULTS = path("results", "results")
ARCHIVES = path("archives", "data/external/archives")
CACHE = path("daily_cache", "data/external/daily_cache")
ERA5 = path("era5_sample", "data/external/era5/era5_combined_with_sst_1979.nc")
MASK = path("ocean_mask", "data/masks/piomas_ocean_mask_025deg.npz")
TRAINING = path("training", "results/training")
FIGURES = RESULTS / "figures"
OCEAN = RESULTS / "ocean"
STATES = RESULTS / "states"
INITIAL = RESULTS / "initial_conditions"
NOTEBOOKS = REPO / "notebooks"


def model_path(model, kind):
    default = (
        "checkpoints/best.pth" if kind == "checkpoint" else "channel_statistics.npz"
    )
    return path(model + "_" + kind, str(TRAINING / model / default))


def raw_path(name):
    defaults = {
        "era5_glob": "data/external/era5/era5_combined_with_sst_*.nc",
        "piomas_sic_glob": "data/external/piomas/sic/regridded_ERA5_PIOMAS_SIC_*.nc",
        "piomas_sit_glob": "data/external/piomas/sit/regridded_ERA5_PIOMAS_SIT_*.nc",
    }
    return str(path(name, defaults[name]))
