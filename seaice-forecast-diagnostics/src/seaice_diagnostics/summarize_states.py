"""Paired early-late growth, uncertainty, and plot tables for target SIC states."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

from .config import STATES as OUT
from .statistics import _bootstrap_multiseries_by_year

MODELS = ["CNN", "U-Net", "GNN", "Three-model mean"]
STATES = ["open_water", "miz", "compact_ice"]
DOMAINS = ["pan_arctic", "combined_nsr"]
CONTRASTS = [
    ("miz", "open_water"),
    ("compact_ice", "open_water"),
    ("miz", "compact_ice"),
]
SEED = 20260816
REPS = 5000


def bootstrap_weights(dates, block, seed=SEED, reps=REPS):
    """Same hierarchical year / non-circular moving-block draws as manuscript.

    Keep original daily positions when individual contrasts have missing cases.
    The identical row weights pair every model/state/domain in a replicate.
    """
    dates = pd.DatetimeIndex(dates)
    years = np.array(sorted(dates.year.unique()))
    rng = np.random.default_rng(seed)
    weights = np.zeros((reps, len(dates)), dtype=np.float64)
    for first in range(0, reps, 500):
        last = min(first + 500, reps)
        size = last - first
        chosen_years = rng.choice(years, size=(size, len(years)), replace=True)
        for slot in range(len(years)):
            for year in years:
                rows = np.flatnonzero(chosen_years[:, slot] == year) + first
                if not len(rows):
                    continue
                positions = np.flatnonzero(dates.year == year)
                n = len(positions)
                if n <= block:
                    indices = rng.integers(0, n, size=(len(rows), n))
                    np.add.at(
                        weights, (np.repeat(rows, n), positions[indices].ravel()), 1
                    )
                else:
                    nblocks = int(np.ceil(n / block))
                    starts = rng.integers(0, n - block + 1, size=(len(rows), nblocks))
                    lengths = np.full(nblocks, block)
                    lengths[-1] = n - block * (nblocks - 1)
                    for j, length in enumerate(lengths):
                        indices = starts[:, j, None] + np.arange(length)[None, :]
                        np.add.at(
                            weights,
                            (np.repeat(rows, length), positions[indices].ravel()),
                            1,
                        )
    assert np.all(weights.sum(axis=1) == len(dates))
    return weights


def main():
    table = OUT / "tables"
    raw = pd.read_csv(
        table / "state_metrics.csv.gz", parse_dates=["init_date", "target_date"]
    )
    assert len(raw) == 152280
    assert set(raw.state_basis) == {"target_date"} and set(raw.experiment_id) == {
        "2023_2025_ver2"
    }
    keys = ["domain", "model", "init_date", "state"]
    wide = raw.pivot(index=keys, columns="lead_day", values="rmse").reindex(
        columns=range(1, 31)
    )
    early, late = wide.loc[:, 1:5], wide.loc[:, 16:30]
    case = pd.DataFrame(
        {
            "early_rmse": early.mean(axis=1),
            "late_rmse": late.mean(axis=1),
            "n_early": early.count(axis=1),
            "n_late": late.count(axis=1),
        }
    )
    case["complete"] = case.n_early.eq(5) & case.n_late.eq(15)
    case["growth"] = (case.late_rmse - case.early_rmse).where(case.complete)
    case = case.reset_index()
    mean_case = (
        case.groupby(["domain", "init_date", "state"])
        .agg(
            early_rmse=("early_rmse", "mean"),
            late_rmse=("late_rmse", "mean"),
            n_early=("n_early", "min"),
            n_late=("n_late", "min"),
            valid_models=("growth", "count"),
            growth=("growth", "mean"),
        )
        .reset_index()
    )
    mean_case["complete"] = mean_case.valid_models.eq(3)
    mean_case["growth"] = mean_case.growth.where(mean_case.complete)
    mean_case["model"] = "Three-model mean"
    combined = pd.concat([case, mean_case[case.columns]], ignore_index=True)
    combined.to_csv(
        table / "growth_by_initialization.csv.gz", index=False, compression="gzip"
    )

    lead_mean_cases = (
        raw.groupby(["domain", "init_date", "lead_day", "state"])
        .rmse.agg(["mean", "count"])
        .reset_index()
    )
    lead_mean_cases["rmse"] = lead_mean_cases["mean"].where(
        lead_mean_cases["count"].eq(3)
    )
    lead_mean_cases["model"] = "Three-model mean"
    lead_all = pd.concat(
        [
            raw[["domain", "model", "init_date", "lead_day", "state", "rmse"]],
            lead_mean_cases[
                ["domain", "model", "init_date", "lead_day", "state", "rmse"]
            ],
        ]
    )
    lead = (
        lead_all.groupby(["domain", "model", "state", "lead_day"])
        .rmse.agg(rmse_mean="mean", n_cases="count")
        .reset_index()
    )
    lead.to_csv(table / "lead_summary.csv", index=False)
    dates = pd.DatetimeIndex(sorted(raw.init_date.unique()))
    assert len(dates) == 282
    base = combined.pivot(
        index="init_date", columns=["domain", "model", "state"], values="growth"
    ).reindex(dates)
    series, specs = [], []

    def add(values, **spec):
        series.append(np.asarray(values, float))
        specs.append(spec)

    for domain in DOMAINS:
        for model in MODELS:
            for state in STATES:
                add(
                    base[(domain, model, state)],
                    kind="growth",
                    domain=domain,
                    model=model,
                    state=state,
                )
            for a, b in CONTRASTS:
                add(
                    base[(domain, model, a)] - base[(domain, model, b)],
                    kind="state_contrast",
                    domain=domain,
                    model=model,
                    state_a=a,
                    state_b=b,
                    contrast=a + " minus " + b,
                )
    for model in MODELS:
        for state in STATES:
            add(
                base[("combined_nsr", model, state)]
                - base[("pan_arctic", model, state)],
                kind="domain_contrast",
                model=model,
                state=state,
                contrast="combined_nsr minus pan_arctic",
            )
        for a, b in CONTRASTS:
            add(
                (base[("combined_nsr", model, a)] - base[("combined_nsr", model, b)])
                - (base[("pan_arctic", model, a)] - base[("pan_arctic", model, b)]),
                kind="domain_state_interaction",
                model=model,
                state_a=a,
                state_b=b,
                contrast="NSR minus pan-Arctic: " + a + " minus " + b,
            )
    matrix = np.column_stack(series)
    finite = np.isfinite(matrix)
    estimates = np.nanmean(matrix, axis=0)
    stats = []
    bootstrap_match = []
    for block in (7, 14, 30):
        weights = bootstrap_weights(dates, block)
        denominator = weights @ finite.astype(float)
        assert (
            denominator > 0
        ).all(), "No valid paired sample in a bootstrap replicate"
        boot = (weights @ np.nan_to_num(matrix, nan=0.0)) / denominator
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        adjlo, adjhi = np.quantile(boot, [0.025 / 3, 1 - 0.025 / 3], axis=0)
        # Independent comparison with the established implementation for a full panel.
        fullcol = int(np.flatnonzero(finite.all(axis=0))[0])
        expected = _bootstrap_multiseries_by_year(
            matrix[:, fullcol, None], dates, REPS, block, SEED
        )
        np.testing.assert_allclose(
            [estimates[fullcol], lo[fullcol], hi[fullcol]],
            [float(v[0]) for v in expected],
            rtol=0,
            atol=1e-12,
        )
        bootstrap_match.append(
            {"block_days": block, "matches_existing_bootstrap_helper": True}
        )
        for j, spec in enumerate(specs):
            stats.append(
                {
                    **spec,
                    "mean": estimates[j],
                    "ci_lower": lo[j],
                    "ci_upper": hi[j],
                    "family3_ci_lower": adjlo[j],
                    "family3_ci_upper": adjhi[j],
                    "n_cases": int(finite[:, j].sum()),
                    "block_days": block,
                    "bootstrap_reps": REPS,
                }
            )
    stats = pd.DataFrame(stats)
    stats.to_csv(table / "growth_statistics_all_blocks.csv", index=False)
    stats.loc[stats.kind.eq("state_contrast")].to_csv(
        table / "growth_contrasts.csv", index=False
    )
    stats.loc[stats.kind.isin(["domain_contrast", "domain_state_interaction"])].to_csv(
        table / "domain_growth_contrasts.csv", index=False
    )
    summary = (
        combined.loc[combined.complete]
        .groupby(["domain", "model", "state"])
        .agg(
            early_rmse=("early_rmse", "mean"),
            late_rmse=("late_rmse", "mean"),
            growth=("growth", "mean"),
            n_cases=("growth", "count"),
        )
        .reset_index()
    )
    interval = stats.loc[
        stats.kind.eq("growth") & stats.block_days.eq(7),
        ["domain", "model", "state", "ci_lower", "ci_upper"],
    ]
    summary = summary.merge(
        interval, on=["domain", "model", "state"], validate="one_to_one"
    )
    summary.to_csv(table / "growth_summary.csv", index=False)
    annual = (
        combined.loc[combined.complete]
        .assign(year=lambda x: x.init_date.dt.year)
        .groupby(["domain", "model", "state", "year"])
        .agg(
            early_rmse=("early_rmse", "mean"),
            late_rmse=("late_rmse", "mean"),
            growth=("growth", "mean"),
            n_cases=("growth", "count"),
        )
        .reset_index()
    )
    annual.to_csv(table / "growth_by_year.csv", index=False)
    area = raw.loc[
        raw.model.eq("CNN"),
        [
            "domain",
            "init_date",
            "target_date",
            "lead_day",
            "state",
            "n_cells",
            "valid_area_km2",
        ],
    ].copy()
    total = area.groupby(["domain", "init_date", "lead_day"]).valid_area_km2.transform(
        "sum"
    )
    area["ocean_area_fraction"] = area.valid_area_km2 / total
    area.to_csv(
        table / "state_area_by_initialization_lead.csv.gz",
        index=False,
        compression="gzip",
    )
    area.groupby(["domain", "state", "lead_day"]).agg(
        mean_ocean_fraction=("ocean_area_fraction", "mean"),
        min_cells=("n_cells", "min"),
        mean_area_km2=("valid_area_km2", "mean"),
    ).reset_index().to_csv(table / "state_area_lead_summary.csv", index=False)
    missing = (
        raw.assign(empty=raw.rmse.isna())
        .groupby(["domain", "model", "state"])
        .agg(rows=("rmse", "size"), empty_rows=("empty", "sum"))
        .reset_index()
    )
    eligible = (
        case.groupby(["domain", "model", "state"])
        .complete.sum()
        .reset_index(name="complete_growth_cases")
    )
    coverage = missing.merge(eligible, on=["domain", "model", "state"])
    coverage.to_csv(table / "coverage.csv", index=False)
    manifest = {
        "status": "PASS",
        "state_basis": "target_date",
        "growth": "mean of daily RMSE at L16-30 minus mean of daily RMSE at L1-5, first calculated within model and initialization",
        "bootstrap": "Hierarchical resampling of three years and within-year non-circular daily moving blocks; same date weights for all models, domains, and states",
        "block_days": [7, 14, 30],
        "reps": REPS,
        "seed": SEED,
        "primary_ci": "95% pointwise percentile intervals, 7-day blocks",
        "state_contrast_multiplicity": "family3 intervals use percentile tails 0.025/3 and 1-0.025/3, Bonferroni for the three state contrasts within each domain/model/block setting",
        "complete_case_policy": "Require all 5 early and 15 late days; contrast uses paired nonmissing initializations; original daily positions retained in bootstrap",
        "three_model_mean": "Equal mean of three model-wise metrics within an initialization; not RMSE of an ensemble-mean forecast",
        "bootstrap_implementation_check": bootstrap_match,
        "n_empty_rows": int(raw.rmse.isna().sum()),
        "minimum_complete_cases": int(coverage.complete_growth_cases.min()),
        "maximum_complete_cases": int(coverage.complete_growth_cases.max()),
    }
    (OUT / "validation/statistical_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("Three-model mean paired state contrasts:")
    print(
        stats.loc[
            stats.kind.eq("state_contrast") & stats.model.eq("Three-model mean"),
            [
                "domain",
                "contrast",
                "mean",
                "ci_lower",
                "ci_upper",
                "family3_ci_lower",
                "family3_ci_upper",
                "block_days",
            ],
        ].to_string(index=False)
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
