"""Recalculate pan-Arctic initial-condition RMSE using frozen ver2 archives."""

from pathlib import Path
import hashlib
import io
import json
import sys
import time
import numpy as np
import pandas as pd

from .config import (
    INITIAL as OUT,
    STATES as PAN,
    OCEAN as PRIOR,
    ARCHIVES as SOURCE,
    MASK,
)
from . import diagnostics as d
from .inference import CACHE_DIR, VARIABLES
from .protocol import paired_initializations

MODELS = {"cnn": "CNN", "unet": "U-Net", "gnn": "GNN"}
LABELS = {
    "initial_gradient": ["low", "moderate", "high", "extreme"],
    "initial_sit_class": ["open_water_no_ice", "thin_ice", "medium_ice", "thick_ice"],
    "initial_sit_x_wind": [
        "thin_ice__lower_middle_wind",
        "thin_ice__high_upper_tail_wind",
        "medium_thick_ice__lower_middle_wind",
        "medium_thick_ice__high_upper_tail_wind",
    ],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    for folder in ["tables", "validation", "figures", "reports"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    prior_manifest = json.loads(
        (PAN / "validation/run_manifest.json").read_text(encoding="utf-8")
    )
    known = {
        str(Path(x["path"])): x["sha256"]
        for x in json.loads(
            (PAN / "validation/archive_manifest.json").read_text(encoding="utf-8")
        )
    }
    assert sha(MASK) == prior_manifest["mask_sha256"]
    protected = prior_manifest["protected_sources"]
    with np.load(MASK, allow_pickle=False) as z:
        ocean = z["ocean_mask"].astype(bool)
        lat = z["latitude"]
        lon = z["longitude"]
    area2 = d.cell_area_km2(lat, lon)
    area = area2[ocean]
    assert ocean.sum() == 101206 and np.isfinite(area).all() and (area > 0).all()
    nsr = d.build_domain_masks(lat, lon, ocean)["combined_nsr"]
    old = pd.read_csv(
        PRIOR / "tables/figure5_initial_thresholds_area_weighted.csv",
        parse_dates=["init_date"],
    ).set_index("init_date")
    archive_records = []
    cache_records = []
    source_checks = []
    threshold_rows = []
    rows = []
    cases = paired_initializations([2023, 2024, 2025], (7, 8, 9, 10))
    assert cases.groupby("target_year").size().to_dict() == {
        2023: 94,
        2024: 94,
        2025: 94,
    }
    previous_date = None
    previous_day_one = None

    def read_archive(path, field, date, model=None):
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        assert digest == known[str(path)], f"Archive changed: {path}"
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            assert str(z["init_date"].item()) == date
            assert np.array_equal(z["lead_days"], np.arange(1, 31))
            expected = pd.date_range(
                pd.Timestamp(date) + pd.Timedelta(days=1), periods=30
            )
            assert pd.DatetimeIndex(z["target_dates"].astype(str)).equals(expected)
            if model:
                assert str(z["model"].item()) == MODELS[model]
                assert (
                    str(z["checkpoint_sha256"].item())
                    == prior_manifest["checkpoint_hashes"][model]
                )
            values = z[field][:, ocean].astype(np.float64)
        assert values.shape == (30, 101206) and np.isfinite(values).all()
        if model:
            assert values.min() >= 0 and values.max() <= 1
        archive_records.append(
            {
                "path": str(path),
                "sha256": digest,
                "bytes": len(blob),
                "matches_frozen_ver2": True,
            }
        )
        return values, expected

    for number, case in enumerate(cases.itertuples(index=False), 1):
        init = pd.Timestamp(case.init_date)
        date = init.date().isoformat()
        year = int(case.target_year)
        cachepath = CACHE_DIR / (init.strftime("%Y%m%d") + ".npz")
        blob = cachepath.read_bytes()
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            sic = z["sic_full"].astype(np.float32)
            daily = z["daily"].astype(np.float32)
        assert daily.shape == (len(VARIABLES), 101, 1440)
        sit = daily[7]
        wind = np.hypot(daily[2].astype(float), daily[3].astype(float))
        assert all(np.isfinite(v[ocean]).all() for v in [sic, sit, wind])
        cache_records.append(
            {
                "path": str(cachepath),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "bytes": len(blob),
            }
        )
        linked = False
        if previous_date is not None and init - previous_date == pd.Timedelta(days=1):
            np.testing.assert_array_equal(sic[ocean], previous_day_one)
            linked = True
        ref, dates = read_archive(
            SOURCE / "references" / str(year) / (date + ".npz"), "reference", date
        )
        previous_date = init
        previous_day_one = ref[0].copy()
        gradient = d.physical_sic_gradient_km(sic, lat, lon, ocean)
        gradcodes, gt = d.classify_area_weighted_bins(gradient, area2, ocean)
        windcodes, wt = d.classify_area_weighted_bins(wind, area2, ocean)
        sitcodes = d.sit_class_codes(sit, sic)
        joint = d.joint_sit_wind_codes(sitcodes, windcodes)
        categories = {
            "initial_gradient": gradcodes[ocean],
            "initial_sit_class": sitcodes[ocean],
            "initial_sit_x_wind": joint[ocean],
        }
        initial_ice = ocean & (sic >= 0.15)
        gradient_valid = ocean & np.isfinite(gradient)
        threshold = {
            "experiment_id": "2023_2025_ver2",
            "domain": "pan_arctic",
            "init_date": date,
        }
        for q, v in zip([33, 67, 90], gt):
            threshold[f"gradient_q{q}_sic_per_km"] = float(v)
        for q, v in zip([33, 67, 90], wt):
            threshold[f"wind_q{q}_ms"] = float(v)
        for name, mask in [
            ("gradient_valid", gradient_valid),
            ("initial_ice", initial_ice),
            ("sit_valid", ocean),
            ("wind_valid", ocean),
        ]:
            threshold[name + "_n_cells"] = int(mask.sum())
            threshold[name + "_area_km2"] = float(area2[mask].sum())
        threshold_rows.append(threshold)
        # All initialization fields must reproduce the existing NSR thresholds.
        _, ng = d.classify_area_weighted_bins(gradient, area2, nsr)
        _, nw = d.classify_area_weighted_bins(wind, area2, nsr)
        obs = old.loc[init]
        expected_g = obs[[f"gradient_q{q}_sic_per_km" for q in [33, 67, 90]]].to_numpy(
            float
        )
        expected_w = obs[[f"wind_q{q}_ms" for q in [33, 67, 90]]].to_numpy(float)
        np.testing.assert_allclose(ng, expected_g, rtol=0, atol=1e-14)
        np.testing.assert_allclose(nw, expected_w, rtol=0, atol=1e-12)
        nsrthin = float(area2[nsr & (sitcodes == 1)].sum())
        nsrmedium = float(area2[nsr & (sitcodes >= 2)].sum())
        np.testing.assert_allclose(
            [nsrthin, nsrmedium],
            [obs.thin_area_km2, obs.medium_thick_area_km2],
            rtol=1e-12,
            atol=1e-6,
        )
        source_checks.append(
            {
                "init_date": date,
                "initial_sic_matches_prior_forecast_reference": linked,
                "nsr_gradient_threshold_max_abs_error": float(
                    np.max(np.abs(ng - expected_g))
                ),
                "nsr_wind_threshold_max_abs_error": float(
                    np.max(np.abs(nw - expected_w))
                ),
            }
        )
        prepared = {}
        for family, codes in categories.items():
            indices = np.flatnonzero(codes >= 0)
            selected = codes[indices]
            count = np.bincount(selected, minlength=4)
            areas = np.bincount(selected, weights=area[indices], minlength=4)
            labels = (
                4 * np.arange(30, dtype=np.int16)[:, None] + selected[None, :]
            ).ravel()
            weights = np.broadcast_to(area[indices], (30, len(indices))).ravel()
            prepared[family] = (indices, count, areas, labels, weights)
        assert prepared["initial_sit_class"][1].sum() == 101206
        assert prepared["initial_gradient"][1].sum() == gradient_valid.sum()
        assert prepared["initial_sit_x_wind"][1].sum() == initial_ice.sum()
        for key, model in MODELS.items():
            pred, _ = read_archive(
                SOURCE / "predictions" / key / str(year) / (date + ".npz"),
                "prediction",
                date,
                key,
            )
            error2 = np.square(pred - ref)
            for family, (indices, count, areas, labels, weights) in prepared.items():
                sse = np.bincount(
                    labels, weights=error2[:, indices].ravel() * weights, minlength=120
                ).reshape(30, 4)
                rmse = np.sqrt(
                    np.divide(
                        sse,
                        areas[None, :],
                        out=np.full_like(sse, np.nan),
                        where=areas[None, :] > 0,
                    )
                )
                for lead in range(30):
                    for j, category in enumerate(LABELS[family]):
                        rows.append(
                            {
                                "experiment_id": "2023_2025_ver2",
                                "domain": "pan_arctic",
                                "classification_basis": "initial_date_condition",
                                "model": model,
                                "init_date": date,
                                "target_date": dates[lead].date().isoformat(),
                                "lead_day": lead + 1,
                                "analysis_family": family,
                                "category": category,
                                "n_cells": int(count[j]),
                                "valid_area_km2": areas[j],
                                "weighted_sse": sse[lead, j],
                                "rmse": rmse[lead, j],
                            }
                        )
        if number == 1 or number % 20 == 0 or number == 282:
            print(
                f"Pan-Arctic initial conditions: {number}/282; elapsed {time.monotonic()-started:.1f}s",
                flush=True,
            )
    data = pd.DataFrame(rows)
    assert (
        len(data) == 304560
        and not data.duplicated(
            ["model", "init_date", "lead_day", "analysis_family", "category"]
        ).any()
    )
    data.to_csv(
        OUT / "tables/initial_condition_metrics.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(threshold_rows).to_csv(
        OUT / "tables/initial_thresholds.csv", index=False
    )
    pd.DataFrame(source_checks).to_csv(
        OUT / "validation/initial_source_crosschecks.csv", index=False
    )
    for item in protected:
        assert sha(item["path"]) == item["sha256"]
    manifest = {
        "status": "PASS",
        "experiment_id": "2023_2025_ver2",
        "domain": "pan_arctic",
        "classification_basis": "initial_date_condition",
        "quantile_domain": "pan_arctic valid ocean, not NSR",
        "n_initializations": 282,
        "new_rows": len(data),
        "empty_rows": int(data.rmse.isna().sum()),
        "ocean_cells": 101206,
        "source_root": str(SOURCE),
        "cache_root": str(CACHE_DIR),
        "cache_variables": list(VARIABLES),
        "mask_path": str(MASK),
        "mask_sha256": sha(MASK),
        "checkpoint_hashes": prior_manifest["checkpoint_hashes"],
        "forecast_reference_archive_count": len(archive_records),
        "initial_cache_count": len(cache_records),
        "initial_sic_reference_links_checked": sum(
            x["initial_sic_matches_prior_forecast_reference"] for x in source_checks
        ),
        "all_initializations_reproduce_existing_nsr_thresholds": True,
        "reference_policy": prior_manifest["reference_policy"],
        "gradient_definition": "Existing physical ocean-only gradient, central/one-sided differences with longitude periodicity; a single valid component suffices.",
        "protected_sources": protected,
        "elapsed_seconds": time.monotonic() - started,
    }
    for name, obj in [
        ("run_manifest.json", manifest),
        ("archive_manifest.json", archive_records),
        ("initial_cache_manifest.json", cache_records),
    ]:
        (OUT / "validation" / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
