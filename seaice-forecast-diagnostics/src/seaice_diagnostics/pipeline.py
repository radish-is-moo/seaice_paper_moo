"""Public commands and strict archive provenance checks."""

import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as c
from .protocol import paired_initializations, validate_prediction_archive


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def initialize():
    for root in (c.OCEAN, c.STATES, c.INITIAL):
        for folder in ("tables", "validation", "diagnostics", "reports", "figures"):
            (root / folder).mkdir(parents=True, exist_ok=True)
    c.FIGURES.mkdir(parents=True, exist_ok=True)


def audit():
    """Verify the published frozen archives. No checkpoint deserialization."""
    initialize()
    frozen = json.loads((c.REPO / "data/provenance/frozen_archives.json").read_text())
    mask_expected = "c73ace27347312eacd21479ecceb6297b4253ee068f199d6b04450a5ae5a11a2"
    if digest(c.MASK) != mask_expected:
        raise ValueError("Ocean-mask hash differs from the published experiment.")
    records = []
    for i, entry in enumerate(frozen, 1):
        path = c.ARCHIVES / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing external archive: {path}. Configure SEAICE_CONFIG; see docs/data.md."
            )
        actual = digest(path)
        if actual != entry["sha256"]:
            raise ValueError(f'Frozen archive changed: {entry["path"]}')
        records.append({"path": str(path), "sha256": actual})
        if i % 100 == 0:
            print(f"Verified {i}/{len(frozen)} archives", flush=True)
    (c.OCEAN / "validation/archive_manifest.json").write_text(
        json.dumps(records, indent=2)
    )
    (c.OCEAN / "validation/mask_contract.json").write_text(
        json.dumps({"sha256": mask_expected, "ocean_cells": 101206})
    )
    shutil.copy2(
        c.REPO / "data/provenance/checkpoints.json",
        c.OCEAN / "validation/archive_input_provenance.json",
    )
    print(f"PASS: {len(records)} frozen archives and ocean mask")


def evaluate_ocean():
    from .diagnostics import run_diagnostics
    from .pair_state_support import main as pair_support

    audit()
    run_diagnostics(
        output_root=c.OCEAN,
        prediction_root=c.ARCHIVES,
        bootstrap_reps=5000,
        seed=20260816,
    )
    pair_support()
    summarize_regions()


def summarize_regions():
    from .statistics import bootstrap_paired_means

    raw = pd.read_csv(
        c.OCEAN / "tables/domain_metrics_area_weighted.csv.gz",
        parse_dates=["init_date"],
    )
    raw["band"] = np.where(
        raw.lead_day <= 5, "early", np.where(raw.lead_day >= 16, "late", "middle")
    )
    wide = (
        raw[raw.band.ne("middle")]
        .groupby(["model", "init_date", "domain", "band"])[["rmse", "r"]]
        .mean()
        .unstack("band")
    )
    growth = pd.DataFrame(
        {
            "rmse_growth": wide["rmse", "late"] - wide["rmse", "early"],
            "r_loss": wide["r", "early"] - wide["r", "late"],
        }
    ).reset_index()
    growth.to_csv(
        c.OCEAN / "tables/domain_growth_by_model_initialization.csv.gz", index=False
    )
    z = growth[growth.domain.isin(["combined_nsr", "pan_arctic"])].pivot(
        index=["model", "init_date"], columns="domain", values=["rmse_growth", "r_loss"]
    )
    contrast = pd.DataFrame(
        {
            "rmse_growth_diff": z["rmse_growth", "combined_nsr"]
            - z["rmse_growth", "pan_arctic"],
            "acc_loss_diff": z["r_loss", "combined_nsr"] - z["r_loss", "pan_arctic"],
        }
    ).reset_index()
    stats = {
        str(b): bootstrap_paired_means(
            contrast, reps=5000, block_length=b, seed=20260816
        )
        for b in (7, 14, 30)
    }
    (c.OCEAN / "tables/regional_growth_bootstrap.json").write_text(
        json.dumps(stats, indent=2)
    )


def plot(numbers):
    initialize()
    import importlib
    import matplotlib.pyplot as plt

    functions = {
        1: ("plot_overview", "figure1"),
        2: ("plot_overview", "figure2"),
        3: ("plot_maps", "figure3"),
        4: ("plot_regional", "figure4"),
        5: ("plot_states", "main"),
        6: ("plot_maps", "figure6"),
        7: ("plot_transitions", "main"),
    }
    report = {}
    for n in numbers:
        print(f"Rendering Figure {n}", flush=True)
        plt.rcdefaults()
        module, function = functions[n]
        report[str(n)] = getattr(
            importlib.import_module("." + module, __package__), function
        )()
    (c.FIGURES / "render_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(c.FIGURES)


def check_data():
    """Validate bundled figure input integrity before redraw."""
    expected = json.loads((c.REPO / "data/provenance/bundled_files.json").read_text())
    for entry in expected:
        if digest(c.REPO / entry["path"]) != entry["sha256"]:
            raise ValueError("Bundled data changed: " + entry["path"])
    with np.load(c.MASK, allow_pickle=False) as z:
        assert z["ocean_mask"].sum() == 101206
    print(f"PASS: {len(expected)} bundled numerical inputs")
