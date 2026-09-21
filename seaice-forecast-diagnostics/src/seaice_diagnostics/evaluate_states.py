"""New pan-Arctic ocean target-state verification of frozen 2023-2025 ver2.

Read-only forecasts/reference; no retraining, inference, or manuscript edits.
Existing NSR state rows are imported unchanged, with explicit scope metadata.
"""

from pathlib import Path
import hashlib
import io
import json
import sys
import time
import numpy as np
import pandas as pd

from .config import STATES as OUT, ARCHIVES as SOURCE, OCEAN as PRIOR, MASK
from .diagnostics import cell_area_km2, sic_state_codes
from .protocol import paired_initializations

MODELS = {"cnn": "CNN", "unet": "U-Net", "gnn": "GNN"}
STATES = ["open_water", "miz", "compact_ice"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    for folder in ("tables", "validation", "figures", "reports"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    previous = json.loads(
        (PRIOR / "validation/archive_manifest.json").read_text(encoding="utf-8")
    )
    lookup = {str(Path(x["path"])): x for x in previous}
    audit = []
    protected_paths = []
    protected = {str(p): sha(p) for p in protected_paths}
    mask_contract = json.loads(
        (PRIOR / "validation/mask_contract.json").read_text(encoding="utf-8")
    )
    assert sha(MASK) == mask_contract["sha256"]
    with np.load(MASK, allow_pickle=False) as z:
        ocean = z["ocean_mask"].astype(bool)
        lat, lon = z["latitude"], z["longitude"]
    assert ocean.shape == (101, 1440) and ocean.sum() == 101206
    area = cell_area_km2(lat, lon)[ocean]
    assert np.isfinite(area).all() and (area > 0).all()
    weights = np.broadcast_to(area, (30, len(area))).ravel()
    lead_offset = np.arange(30, dtype=np.int16)[:, None] * 3
    cases = paired_initializations([2023, 2024, 2025], (7, 8, 9, 10))
    assert cases.groupby("target_year").size().to_dict() == {
        2023: 94,
        2024: 94,
        2025: 94,
    }
    inputs = json.loads(
        (PRIOR / "validation/archive_input_provenance.json").read_text(encoding="utf-8")
    )
    checkpoint_hashes = {
        key: next(
            x["sha256"]
            for x in inputs["inputs"]
            if x["path"].endswith(".pth")
            and ("simple_" + key if key != "gnn" else "gnn_gridonly") in x["path"]
        )
        for key in MODELS
    }

    def read_archive(path, field, date, model=None):
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        assert digest == lookup[str(path)]["sha256"], f"Archive changed: {path}"
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            assert str(z["init_date"].item()) == date
            assert np.array_equal(z["lead_days"], np.arange(1, 31))
            expected = pd.date_range(
                pd.Timestamp(date) + pd.Timedelta(days=1), periods=30
            )
            assert pd.DatetimeIndex(z["target_dates"].astype(str)).equals(expected)
            if model:
                assert str(z["model"].item()) == MODELS[model]
                assert str(z["checkpoint_sha256"].item()) == checkpoint_hashes[model]
            value = z[field][:, ocean].astype(np.float64)
        assert value.shape == (30, 101206) and np.isfinite(value).all()
        if model:
            assert value.min() >= 0 and value.max() <= 1
        audit.append(
            {
                "path": str(path),
                "bytes": len(blob),
                "sha256": digest,
                "matches_previous_verified_archive": True,
            }
        )
        return value, expected

    rows = []
    for num, case in enumerate(cases.itertuples(index=False), 1):
        date = pd.Timestamp(case.init_date).date().isoformat()
        year = int(case.target_year)
        ref, dates = read_archive(
            SOURCE / "references" / str(year) / (date + ".npz"), "reference", date
        )
        codes = sic_state_codes(ref)
        assert (codes >= 0).all()
        labels = (codes + lead_offset).ravel()
        counts = np.bincount(labels, minlength=90).reshape(30, 3)
        areas = np.bincount(labels, weights=weights, minlength=90).reshape(30, 3)
        assert np.all(counts.sum(axis=1) == 101206)
        np.testing.assert_allclose(areas.sum(axis=1), area.sum(), rtol=1e-12, atol=1e-6)
        for key, model in MODELS.items():
            pred, _ = read_archive(
                SOURCE / "predictions" / key / str(year) / (date + ".npz"),
                "prediction",
                date,
                key,
            )
            sse = np.bincount(
                labels, weights=weights * np.square(pred - ref).ravel(), minlength=90
            ).reshape(30, 3)
            rmse = np.sqrt(
                np.divide(sse, areas, out=np.full_like(sse, np.nan), where=areas > 0)
            )
            for l in range(30):
                for s, state in enumerate(STATES):
                    rows.append(
                        {
                            "experiment_id": "2023_2025_ver2",
                            "domain": "pan_arctic",
                            "state_basis": "target_date",
                            "model": model,
                            "init_date": date,
                            "target_date": dates[l].date().isoformat(),
                            "lead_day": l + 1,
                            "state": state,
                            "n_cells": int(counts[l, s]),
                            "valid_area_km2": areas[l, s],
                            "weighted_sse": sse[l, s],
                            "rmse": rmse[l, s],
                        }
                    )
        if num == 1 or num % 20 == 0 or num == len(cases):
            print(
                f"Pan-Arctic state metrics: {num}/{len(cases)} initializations; elapsed {time.monotonic()-start:.1f}s",
                flush=True,
            )
    pan = pd.DataFrame(rows)
    assert len(pan) == 76140
    nsr_path = PRIOR / "tables/figure5_condition_metrics_area_weighted.csv.gz"
    source_nsr = pd.read_csv(nsr_path)
    nsr = source_nsr.loc[source_nsr.analysis_family.eq("verification_state")].copy()
    nsr = nsr.rename(columns={"category": "state", "area_km2": "valid_area_km2"})
    nsr["experiment_id"], nsr["domain"], nsr["state_basis"] = (
        "2023_2025_ver2",
        "combined_nsr",
        "target_date",
    )
    nsr = nsr[pan.columns]
    assert len(nsr) == 76140
    both = pd.concat([pan, nsr], ignore_index=True)
    assert not both.duplicated(
        ["domain", "model", "init_date", "lead_day", "state"]
    ).any()
    both.to_csv(OUT / "tables/state_metrics.csv.gz", index=False, compression="gzip")
    protected_check = [
        {"path": p, "sha256": digest, "unchanged": sha(p) == digest}
        for p, digest in protected.items()
    ]
    assert all(x["unchanged"] for x in protected_check)
    manifest = {
        "status": "PASS",
        "experiment_id": "2023_2025_ver2",
        "state_basis": "target_date",
        "state_definitions": {
            "open_water": "reference SIC < 0.15",
            "miz": "0.15 <= reference SIC <= 0.80",
            "compact_ice": "reference SIC > 0.80",
        },
        "mask_path": str(MASK),
        "mask_sha256": sha(MASK),
        "ocean_cells": int(ocean.sum()),
        "reference_policy": "Cached raw reference retained; finite values outside [0,1] are not silently clipped. Forecasts use the existing saved [0,1] values.",
        "n_initializations": len(cases),
        "n_new_pan_arctic_rows": len(pan),
        "n_reused_nsr_rows": len(nsr),
        "n_empty_pan_rows": int(pan.rmse.isna().sum()),
        "n_empty_nsr_rows": int(nsr.rmse.isna().sum()),
        "archive_count": len(audit),
        "checkpoint_hashes": checkpoint_hashes,
        "nsr_source": {"path": str(nsr_path), "sha256": sha(nsr_path)},
        "area_formula": "R^2 * abs(sin(lat_edge_2)-sin(lat_edge_1)) * abs(delta_lon_rad); R=6371.0088km",
        "elapsed_seconds": time.monotonic() - start,
        "protected_sources": protected_check,
    }
    (OUT / "validation/run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "validation/archive_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
