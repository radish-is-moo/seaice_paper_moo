# External inputs and file formats

No raw ERA5/PIOMAS time series, trained weights, or complete forecast archive is distributed here. Obtain the same research inputs separately under their applicable terms. The included mask and derived figure inputs are documented by SHA-256 manifests. File formats are NumPy NPZ and CSV, loaded without pickle.

## Configuration

Copy `configs/paths.example.json` to `configs/paths.json`. Paths may be absolute or relative to the repository root. Alternatively pass `--config /path/to/paths.json` before the subcommand or set `SEAICE_CONFIG` before starting a notebook kernel. Local configuration is ignored by Git.

| Key | Meaning |
| --- | --- |
| daily_cache | Directory of YYYYMMDD.npz files, 1979–2025 |
| archives | Root containing predictions/, references/, and optionally cache/ |
| era5_sample | NetCDF with latitude and longitude on the actual model grid |
| results | New output directory; defaults to results/ |
| training | Parent of cnn/, unet/, gnn/ training outputs |
| cnn_checkpoint / cnn_statistics | Optional explicit CNN weights and normalization files; similarly unet_* and gnn_* |
| era5_glob / piomas_sic_glob / piomas_sit_glob | Optional already-regridded input globs for prepare-cache |

## Daily input cache

Each YYYYMMDD.npz contains `daily` float32, shape (8,101,1440), and `sic_full` float32, shape (101,1440). Predictor order is t2m,d2m,u10,v10,msl,sst,sic,sit. Wind components are in m/s, SIC is a fraction, SIT is in m; use the original physical units and normalization for the other variables. `sic_full` retains the reference field, whereas model input preparation can replace missing predictor values as in the original training code. Do not substitute a different interpolation, mask, grid order, or unit conversion silently.

`prepare-cache` assembles aligned NetCDF data into the cache. It assumes the upstream spatial regridding has already been completed. It uses the original time-reindexing logic; therefore check source calendars/alignment before using a different dataset. This release does not implement upstream data retrieval or PIOMAS-to-ERA5 regridding.

## Forecast/reference archive

```text
archives/
  predictions/cnn/2023/2023-06-30.npz
  predictions/unet/2023/2023-06-30.npz
  predictions/gnn/2023/2023-06-30.npz
  references/2023/2023-06-30.npz
  cache/reference_calendar_day_climatology_1979_2019.npz
```

There are 846 prediction archives and 282 reference archives for the paired manuscript panel. Each contains a `prediction` or `reference` array, shape (30,101,1440), plus init_date, target_dates, lead_days. Prediction metadata includes model, notebook_sha256, checkpoint_sha256, and statistics_sha256. The published frozen SHA-256 identities are in `data/provenance/frozen_archives.json`. The climatology cache is optional: persistence can reconstruct it from daily caches for 1979–2019, which is expensive.

Full recalculation also reads initialization-day SIC/SIT/wind from the daily cache. Figure reproduction instead uses fixed small inputs in `data/figure_inputs/`; Figure 3 contains only lead 30 of the selected date. Figure 6 includes the verified persistence diagnostics, not the original multiyear daily series.

## Checkpoints and new inference

Training saves plain PyTorch state dictionaries and 8-element mean/std normalization arrays. Inference loads weights with `weights_only=True`. Training notebooks are the canonical architecture definitions; inference extracts only the named class definitions without executing training.

Use a separate archive path for newly generated predictions. The frozen evaluator is deliberately limited to the published archive hashes. Evaluating a newly trained experiment requires declaring new provenance and reviewing its scope; it is not advertised as reproducing the original numeric results.

## Rebuild the static mask

The original native-SIC/SIT mask derivation and spherical nearest-neighbour remapping are included:

```sh
python -m seaice_diagnostics.build_mask --piomas-root /data/PIOMAS --regridded-root /data/PIOMAS_REGRIDDED
```

This writes to `results/masks/` rather than replacing the bundled verified mask. Native SIC/SIT NetCDF for 1979, 2000 and 2025 and regridded 1979 SIC/SIT are required. The code checks nine snapshots for a stable land convention.
