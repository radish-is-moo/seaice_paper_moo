# Sea-ice forecast diagnostics

Code for **Regional and state-based diagnostics of AI sea-ice forecast errors near the Northern Sea Route**.

This repository diagnoses common regional and state-dependent error patterns in CNN, U-Net, and grid GraphSAGE forecasts. It is locked to the **2023–2025 ver2 experiment**, with 282 common initializations, lead days 1–30, and all target dates in July–October. It does not include the separate 2020–2025 experiment.

[한국어 안내](README.ko.md) · [Data contract](docs/data.md) · [Methods](docs/methods.md) · [Reproducibility](docs/reproducibility.md) · [Source mapping](docs/source_mapping.json)

## Quick start: reproduce the figures

From this repository root, with Python 3.10 or newer:

```sh
python -m pip install -e ".[maps,test]"
python -m seaice_diagnostics check-data
python -m pytest -q
python -m seaice_diagnostics figures
```

Figures are written to `results/figures/` as PNG and PDF. Small numerical inputs are included; full forecast archives are not needed for this command. Cartopy may download Natural Earth coastline data on first use. The supported install is **editable from a repository checkout**: notebooks, configuration, and figure data are repository resources, not wheel package data.

To generate only selected figures:

```sh
python -m seaice_diagnostics figures --only 4 5 7
```

A Conda alternative is `conda env create -f environment.yml` followed by `conda activate seaice-diagnostics`.

## Recompute the frozen experiment

Supply the external files described in [docs/data.md](docs/data.md). Copy `configs/paths.example.json` to `configs/paths.json` and edit paths. Relative paths resolve from the repository root, regardless of your working directory. Forward slashes work on Windows too.

```sh
python -m seaice_diagnostics audit
python -m seaice_diagnostics evaluate ocean
python -m seaice_diagnostics evaluate states
python -m seaice_diagnostics evaluate conditions
```

`evaluate all` runs those stages in order. The ocean stage includes regional metrics, block-bootstrap confidence intervals, endpoint state transitions, paired MIZ/non-MIZ errors, case selection, and reference persistence. The states and conditions stages recalculate **pan-Arctic ocean** results separately. Evaluation checks the frozen archive hashes; newly trained predictions do not silently replace the published experiment.

## Training and inference

Install the training extra in an environment with the appropriate PyTorch build for your CPU/GPU:

```sh
python -m pip install -e ".[training]"
python -m seaice_diagnostics prepare-cache
jupyter lab
```

Run `notebooks/train_cnn.ipynb`, `train_unet.ipynb`, or `train_gnn.ipynb`. They retain the original model definitions and training settings but omit obsolete post-training evaluation cells and saved outputs. CNN/U-Net require the daily cache first; the GNN notebook also contains the original cache preparation step.

```sh
python -m seaice_diagnostics infer --models cnn unet gnn --device cuda
```

This defaults to the 282 manuscript initializations. Use `--init-date 2024-09-24` for one case. Training outputs use `results/training/{cnn,unet,gnn}/checkpoints/best.pth` and `channel_statistics.npz`. Existing checkpoints can instead be configured with `cnn_checkpoint`, `cnn_statistics`, etc. Never point a new inference run at the frozen archive directory: use a separate configuration. New-run archives have different provenance and are intentionally rejected by the frozen evaluation command.

## Repository layout

```text
src/seaice_diagnostics/  portable numerical, inference, analysis, and plotting modules
notebooks/              three training notebooks, with outputs removed
configs/                example external-data paths
data/figure_inputs/     fixed numerical inputs for the latest Figures 1–7
data/masks/             verified PIOMAS-derived ocean mask
data/provenance/        frozen archive and bundled-input SHA-256 manifests
docs/                   data requirements, methods, code provenance, limitations
tests/                  numerical and scientific-contract regression tests
.github/workflows/      automated numerical tests on push/pull request
```

## Interpretation

Ocean masks remove land and retain open water. Target-date state classes are distinct from fixed initial-condition classes. Model-wise metrics are averaged; forecast fields are not averaged. State transitions are reference endpoint differences, not evidence of a causal mechanism. A small MIZ–non-MIZ difference does not imply small absolute errors.

## License and citation

A license has **not yet been selected**. No open-source license grant is made by this repository. Data-source terms remain applicable. Manuscript authors, DOI, repository URL, and a formal citation file should be added when finalized; they have not been invented here.

## Validation scope

See [docs/validation.md](docs/validation.md) for checks actually run for this release. Reorganization does not constitute a new model-training experiment. Full training and all-year archive recomputation were not rerun solely for packaging.
