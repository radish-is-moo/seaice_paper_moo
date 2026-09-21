import json
import numpy as np
import pandas as pd
from seaice_diagnostics import diagnostics as d
from seaice_diagnostics.config import REPO, MASK
from seaice_diagnostics.protocol import paired_initializations
from seaice_diagnostics.numerics import weighted_quantile, physical_sic_gradient_km
from seaice_diagnostics.statistics import (
    _bootstrap_multiseries_by_year,
    rank_figure3_candidates,
)


def test_paired_initializations_and_full_target_windows():
    cases = paired_initializations([2023, 2024, 2025], (7, 8, 9, 10))
    assert cases.groupby("target_year").size().to_dict() == {
        2023: 94,
        2024: 94,
        2025: 94,
    }
    for date in cases.init_date:
        assert set(pd.date_range(date + pd.Timedelta(days=1), periods=30).month) <= {
            7,
            8,
            9,
            10,
        }


def test_state_thresholds_and_missing_values():
    np.testing.assert_array_equal(
        d.sic_state_codes(np.array([0, 0.149, 0.15, 0.8, 0.801, 1, np.nan])),
        [0, 0, 1, 1, 2, 2, -1],
    )


def test_area_weighting_excludes_land_but_includes_open_water():
    ref = np.array([[[0.0, 0.5, 1.0]]])
    pred = np.array([[[0.2, 0.9, 0.0]]])
    result = d.batch_domain_metrics_area_weighted(
        pred, ref, np.array([[True, True, False]]), np.array([[3.0, 1.0, 99.0]])
    )
    np.testing.assert_allclose(result["rmse"], [np.sqrt((3 * 0.2**2 + 0.4**2) / 4)])


def test_non_miz_requires_pooled_sse_not_mean_rmse():
    rows = d.weighted_category_metrics(
        np.array([0.01, 0.81]), np.array([9.0, 1.0]), np.array([0, 0]), ["non_miz"]
    )
    assert np.isclose(rows[0]["rmse"], 0.3)
    assert not np.isclose(rows[0]["rmse"], (0.1 + 0.9) / 2)


def test_empty_category_is_nan_not_zero():
    rows = d.weighted_category_metrics(
        np.array([0.01]), np.array([1.0]), np.array([0]), ["present", "empty"]
    )
    assert np.isnan(rows[1]["rmse"]) and rows[1]["area_km2"] == 0


def test_ocean_mask_and_regional_union():
    with np.load(MASK, allow_pickle=False) as z:
        ocean = z["ocean_mask"]
        masks = d.build_domain_masks(z["latitude"], z["longitude"], ocean)
    assert ocean.sum() == 101206 and masks["combined_nsr"].sum() == 29595
    np.testing.assert_array_equal(masks["pan_arctic"], ocean)
    assert not (masks["combined_nsr"] & ~ocean).any()


def test_area_weighted_quantile_is_not_cell_count_quantile():
    assert weighted_quantile(np.array([1, 2, 100]), np.array([90, 5, 5]), 0.67) == 1


def test_constant_sic_has_zero_ocean_gradient():
    values = np.full((3, 4), 0.4)
    gradient = physical_sic_gradient_km(
        values,
        np.array([70, 71, 72]),
        np.array([0, 90, 180, 270]),
        np.ones((3, 4), bool),
    )
    np.testing.assert_allclose(gradient, 0, atol=1e-12)


def test_bootstrap_keeps_constant_paired_difference():
    dates = pd.DatetimeIndex(
        list(pd.date_range("2023-07-01", periods=40))
        + list(pd.date_range("2024-07-01", periods=40))
    )
    values = np.full((80, 1), 0.02)
    a = _bootstrap_multiseries_by_year(values, dates, reps=50, block_length=7, seed=42)
    b = _bootstrap_multiseries_by_year(values, dates, reps=50, block_length=7, seed=42)
    np.testing.assert_allclose(np.array(a), 0.02, atol=1e-14)
    np.testing.assert_array_equal(a, b)


def test_figure3_selected_case_reproduces_from_all_eligible_inputs():
    raw = pd.read_csv(
        REPO
        / "data/figure_inputs/ocean/tables/figure3_regime_metrics_area_weighted.csv.gz"
    )
    ranked, _ = rank_figure3_candidates(raw)
    assert str(ranked.iloc[0].init_date)[:10] == "2024-09-24"


def test_figure7_transition_paths_sum_to_exposure():
    folder = REPO / "data/figure_inputs/ocean/tables"
    exposure = (
        pd.read_csv(folder / "figure7_transition_summary.csv")
        .set_index(["domain", "lead_band"])
        .transition_fraction.sort_index()
    )
    paths = (
        pd.read_csv(folder / "figure7_transition_type_summary.csv")
        .groupby(["domain", "lead_band"])
        .area_fraction_of_valid_grid_days.sum()
        .sort_index()
    )
    np.testing.assert_allclose(exposure, paths, atol=1e-14, rtol=0)


def test_figure5_equal_model_metrics_and_daily_support():
    data = pd.read_csv(
        REPO / "data/figure_inputs/initial_conditions/tables/initial_lead_summary.csv"
    )
    keys = ["analysis_family", "category", "lead_day"]
    mean = (
        data[data.model.isin(["CNN", "U-Net", "GNN"])]
        .groupby(keys)
        .rmse_mean.mean()
        .sort_index()
    )
    shown = (
        data[data.model.eq("Three-model equal mean")]
        .set_index(keys)
        .rmse_mean.sort_index()
    )
    np.testing.assert_allclose(mean, shown, atol=1e-14, rtol=0)
    assert data.n_initializations.eq(282).all()


def test_selected_case_rmse_matches_frozen_manuscript_table():
    with np.load(MASK, allow_pickle=False) as z:
        ocean = z["ocean_mask"]
        area = d.cell_area_km2(z["latitude"], z["longitude"])
    expected = pd.read_csv(
        REPO
        / "data/figure_inputs/ocean/tables/figure3_regime_metrics_area_weighted.csv.gz"
    )
    expected = expected[expected.init_date.eq("2024-09-24")].set_index(
        ["model", "regime_bin"]
    )
    with np.load(REPO / "data/figure_inputs/figure3_case.npz", allow_pickle=False) as z:
        reference = z["reference"].astype(np.float64)
        codes = np.where(ocean, d.sic_state_codes(reference), -1)
        for key, label in [("cnn", "CNN"), ("unet", "U-Net"), ("gnn", "GNN")]:
            error2 = (z[key].astype(np.float64) - reference) ** 2
            actual = d.weighted_category_metrics(
                error2, area, codes, ["open_water", "miz", "compact_ice"]
            )
            for row in actual:
                assert np.isclose(
                    row["rmse"],
                    expected.loc[(label, row["category"]), "rmse"],
                    rtol=0,
                    atol=1e-12,
                )


def test_native_mask_remaps_to_frozen_ocean_mask():
    from seaice_diagnostics.mask import remap_boolean_mask_nearest

    with np.load(MASK, allow_pickle=False) as z:
        actual = remap_boolean_mask_nearest(
            z["native_latitude"],
            z["native_longitude"],
            z["native_ocean_mask"],
            z["latitude"],
            z["longitude"],
        )
        np.testing.assert_array_equal(actual, z["ocean_mask"])
