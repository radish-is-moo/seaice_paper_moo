# Release validation — 2026-09-20

Performed locally with Python 3.10.20 on Windows; numerical/plotting package versions are recorded in `requirements-tested.txt`.

- Editable package installation succeeded.
- All analysis/plotting modules imported without the original research folders on the import path.
- All three cleaned training notebooks parsed, with outputs removed and execution counts reset.
- **14 scientific-contract tests passed**, including weighted RMSE, open-water inclusion, state boundaries, empty classes, pooled non-MIZ SSE, 282 complete windows, paired bootstrap behavior, Figure 3 case selection, real selected-case RMSE, six transition pathways, equal model metric averaging, and native-mask remapping.
- CNN, U-Net, and GNN class definitions were compared against the source notebooks and were identical. Original checkpoints loaded successfully with the reorganized inference runtime (CPU); parameter counts were 110142, 500782 and 38862 respectively.
- Figures 1–7 were generated as PNG and PDF. Plotted values and confidence-interval vertices are checked against fixed numerical inputs; Figure 5's 630 daily points agree with the previous 63 band summaries within 8.33e-17 RMSE.
- Generated PNGs for Figures 1, 2, 3, 5, 6 and 7 match the current English manuscript images pixel-for-pixel. Figure 4 uses the same source points and confidence bounds and the same dimensions; a small raster/style difference remains (27985 pixels, approximately 0.53% of the canvas). A side-by-side visual check found no scientific content or layout change.
- The bundled data files passed SHA-256 verification. Frozen archive identities are provided using repository-relative paths; full external archives are not copied into the release.

## Not performed for packaging

No full retraining, 282-initialization inference run, all-year persistence recomputation, or complete frozen-archive recalculation was repeated. Full-evaluation portability was checked through imports, source adaptation and numerical regression tests, not an end-to-end all-data run. Upstream raw-data downloading and spatial regridding are outside the included cache-assembly stage. The optional GitHub Actions workflow has not been run on GitHub because no repository was published.

Do not describe this packaging work as a new scientific experiment or a fresh validation of every result in the manuscript.
