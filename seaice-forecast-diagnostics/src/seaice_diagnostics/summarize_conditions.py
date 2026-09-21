"""Summarize new pan-Arctic initial-condition diagnostics and paired contrasts."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

from .config import INITIAL as OUT, OCEAN
from .summarize_states import bootstrap_weights

MODELS = ["CNN", "U-Net", "GNN", "Three-model equal mean"]
GROUPS = {
    "initial_gradient": ["low", "moderate", "high", "extreme"],
    "initial_sit_class": ["open_water_no_ice", "thin_ice", "medium_ice", "thick_ice"],
    "initial_sit_x_wind": [
        "thin_ice__lower_middle_wind",
        "thin_ice__high_upper_tail_wind",
        "medium_thick_ice__lower_middle_wind",
        "medium_thick_ice__high_upper_tail_wind",
    ],
}
PAIRS = [
    ("initial_gradient", "extreme", "low"),
    ("initial_gradient", "extreme", "high"),
    ("initial_sit_class", "thin_ice", "thick_ice"),
    ("initial_sit_class", "thin_ice", "medium_ice"),
    (
        "initial_sit_x_wind",
        "thin_ice__high_upper_tail_wind",
        "thin_ice__lower_middle_wind",
    ),
    (
        "initial_sit_x_wind",
        "medium_thick_ice__high_upper_tail_wind",
        "medium_thick_ice__lower_middle_wind",
    ),
]


def main():
    t = OUT / "tables"
    raw = pd.read_csv(
        t / "initial_condition_metrics.csv.gz", parse_dates=["init_date", "target_date"]
    )
    assert len(raw) == 304560 and set(raw.domain) == {"pan_arctic"}
    keys = ["init_date", "lead_day", "analysis_family", "category"]
    eq = (
        raw.groupby(keys)
        .agg(
            rmse=("rmse", "mean"),
            valid_area_km2=("valid_area_km2", "mean"),
            model_count=("model", "nunique"),
            valid_model_count=("rmse", "count"),
        )
        .reset_index()
    )
    assert eq.model_count.eq(3).all() and eq.valid_model_count.isin([0, 3]).all()
    eq["model"] = "Three-model equal mean"
    cols = keys + ["model", "rmse", "valid_area_km2"]
    allmodels = pd.concat([raw[cols], eq[cols]], ignore_index=True)
    allmodels["lead_band"] = np.where(
        allmodels.lead_day <= 5,
        "L1-5",
        np.where(allmodels.lead_day <= 15, "L6-15", "L16-30"),
    )
    byinit = (
        allmodels.groupby(
            ["model", "init_date", "analysis_family", "category", "lead_band"]
        )
        .agg(
            rmse=("rmse", "mean"),
            valid_area_km2=("valid_area_km2", "mean"),
            n_leads=("rmse", "count"),
        )
        .reset_index()
    )
    summary = (
        byinit.groupby(["model", "analysis_family", "category", "lead_band"])
        .agg(
            rmse_mean=("rmse", "mean"),
            n_initializations=("rmse", "count"),
            mean_area_km2=("valid_area_km2", "mean"),
        )
        .reset_index()
    )
    for col, val in [
        ("experiment_id", "2023_2025_ver2"),
        ("domain", "pan_arctic"),
        ("classification_basis", "initial_date_condition"),
    ]:
        summary[col] = val
    summary.to_csv(t / "initial_condition_by_model_summary.csv", index=False)
    plot = summary[summary.model.eq("Three-model equal mean")].copy()
    assert len(plot) == 36
    plot.to_csv(t / "initial_condition_summary.csv", index=False)
    allmodels.groupby(["model", "analysis_family", "category", "lead_day"]).rmse.agg(
        rmse_mean="mean", n_initializations="count"
    ).reset_index().to_csv(t / "initial_lead_summary.csv", index=False)
    w = byinit.pivot(
        index=["model", "init_date", "analysis_family", "category"],
        columns="lead_band",
        values=["rmse", "n_leads"],
    )
    growth = pd.DataFrame(
        {
            "early_rmse": w[("rmse", "L1-5")],
            "late_rmse": w[("rmse", "L16-30")],
            "n_early": w[("n_leads", "L1-5")],
            "n_late": w[("n_leads", "L16-30")],
        }
    )
    growth["complete"] = growth.n_early.eq(5) & growth.n_late.eq(15)
    growth["growth"] = (growth.late_rmse - growth.early_rmse).where(growth.complete)
    growth = growth.reset_index()
    growth.to_csv(
        t / "initial_growth_by_initialization.csv.gz", index=False, compression="gzip"
    )
    dates = pd.DatetimeIndex(sorted(raw.init_date.unique()))
    grid = growth.pivot(
        index="init_date",
        columns=["model", "analysis_family", "category"],
        values="growth",
    ).reindex(dates)
    vectors = []
    specs = []
    for model in MODELS:
        for family, categories in GROUPS.items():
            for category in categories:
                specs.append(
                    {
                        "kind": "growth",
                        "model": model,
                        "analysis_family": family,
                        "category": category,
                    }
                )
                vectors.append(grid[(model, family, category)].to_numpy(float))
        for family, a, b in PAIRS:
            specs.append(
                {
                    "kind": "contrast",
                    "model": model,
                    "analysis_family": family,
                    "contrast": a + " minus " + b,
                }
            )
            vectors.append(
                (grid[(model, family, a)] - grid[(model, family, b)]).to_numpy(float)
            )
    matrix = np.column_stack(vectors)
    finite = np.isfinite(matrix)
    observed = np.nanmean(matrix, axis=0)
    stats = []
    for block in [7, 14, 30]:
        weights = bootstrap_weights(dates, block)
        denominator = weights @ finite.astype(float)
        boot = np.divide(
            weights @ np.nan_to_num(matrix, nan=0),
            denominator,
            out=np.full((len(weights), matrix.shape[1]), np.nan),
            where=denominator > 0,
        )
        lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
        blo, bhi = np.nanquantile(boot, [0.025 / 6, 1 - 0.025 / 6], axis=0)
        for j, spec in enumerate(specs):
            stats.append(
                {
                    **spec,
                    "mean": observed[j],
                    "ci_lower": lo[j],
                    "ci_upper": hi[j],
                    "family6_ci_lower": blo[j],
                    "family6_ci_upper": bhi[j],
                    "n_initializations": int(finite[:, j].sum()),
                    "block_days": block,
                    "bootstrap_reps": 5000,
                    "valid_bootstrap_reps": int(np.isfinite(boot[:, j]).sum()),
                }
            )
    stat = pd.DataFrame(stats)
    stat.to_csv(t / "initial_growth_statistics.csv", index=False)
    base = (
        growth[growth.complete]
        .groupby(["model", "analysis_family", "category"])
        .agg(
            early_rmse=("early_rmse", "mean"),
            late_rmse=("late_rmse", "mean"),
            growth=("growth", "mean"),
            n_initializations=("growth", "count"),
        )
        .reset_index()
    )
    base = base.merge(
        stat[stat.kind.eq("growth") & stat.block_days.eq(7)][
            ["model", "analysis_family", "category", "ci_lower", "ci_upper"]
        ],
        on=["model", "analysis_family", "category"],
        validate="one_to_one",
    )
    base.to_csv(t / "initial_growth_summary.csv", index=False)
    growth[growth.complete].assign(year=lambda x: x.init_date.dt.year).groupby(
        ["year", "model", "analysis_family", "category"]
    ).growth.agg(mean="mean", n_initializations="count").reset_index().to_csv(
        t / "initial_growth_by_year.csv", index=False
    )
    old = pd.read_csv(OCEAN / "tables/figure5_condition_summary.csv")
    old = old[old.model.eq("Three-model equal mean")]
    compare = plot.merge(
        old[
            [
                "analysis_family",
                "category",
                "lead_band",
                "rmse_mean",
                "n_initializations",
            ]
        ],
        on=["analysis_family", "category", "lead_band"],
        suffixes=("_pan_arctic", "_previous_nsr"),
        validate="one_to_one",
    )
    compare["rmse_difference_pan_minus_nsr"] = (
        compare.rmse_mean_pan_arctic - compare.rmse_mean_previous_nsr
    )
    compare.to_csv(t / "initial_pan_arctic_vs_previous_nsr.csv", index=False)
    coverage = (
        raw.assign(empty=raw.rmse.isna())
        .groupby(["model", "analysis_family", "category"])
        .agg(
            rows=("rmse", "size"),
            empty_rows=("empty", "sum"),
            minimum_cells=("n_cells", "min"),
            maximum_cells=("n_cells", "max"),
        )
        .reset_index()
    )
    coverage.to_csv(t / "coverage.csv", index=False)
    first = raw[raw.model.eq("CNN") & raw.lead_day.eq(1)][
        ["init_date", "analysis_family", "category", "n_cells", "valid_area_km2"]
    ].copy()
    first["fraction_of_family_area"] = first.valid_area_km2 / first.groupby(
        ["init_date", "analysis_family"]
    ).valid_area_km2.transform("sum")
    first.to_csv(t / "initial_class_support.csv", index=False)
    metadata = {
        "status": "PASS",
        "experiment_id": "2023_2025_ver2",
        "domain": "pan_arctic",
        "classification_basis": "initial_date_condition",
        "growth": "mean daily RMSE L16-30 minus L1-5 within initialization and model",
        "model_aggregation": "Equal mean of model-wise RMSE; forecast fields are not averaged",
        "n_initializations": 282,
        "summary_rows": len(plot),
        "bootstrap": "Hierarchical years and within-year non-circular moving blocks, shared date weights",
        "block_days": [7, 14, 30],
        "bootstrap_reps": 5000,
        "seed": 20260816,
        "contrasts": "Six exploratory growth contrasts per model; family6 intervals provide Bonferroni adjustment across those six contrasts",
        "minimum_valid_initializations": int(plot.n_initializations.min()),
        "maximum_valid_initializations": int(plot.n_initializations.max()),
    }
    (OUT / "validation/statistical_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(base[base.model.eq("Three-model equal mean")].to_string(index=False))
    print(
        stat[
            stat.kind.eq("contrast")
            & stat.model.eq("Three-model equal mean")
            & stat.block_days.eq(7)
        ][
            [
                "analysis_family",
                "contrast",
                "mean",
                "ci_lower",
                "ci_upper",
                "family6_ci_lower",
                "family6_ci_upper",
                "n_initializations",
            ]
        ].to_string(
            index=False
        )
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
