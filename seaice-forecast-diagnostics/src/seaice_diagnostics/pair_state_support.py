"""Put absolute MIZ/non-MIZ means on the contrast's common support.

The original aggregate contrast is already paired. Separate component means
previously used distinct available-case sets when one SIC category was empty.
This correction aligns the explanatory components without changing the contrast.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import OCEAN as OUT


def main():
    tables = OUT / "tables"
    target = tables / "figure7_miz_non_miz_summary.csv"
    prior = pd.read_csv(target)
    backup = tables / "figure7_miz_non_miz_summary_separate_available_support.csv"
    if not backup.exists():
        prior.to_csv(backup, index=False)
    raw = pd.read_csv(tables / "figure7_miz_non_miz_model_metrics.csv.gz")
    raw["lead_band"] = np.where(
        raw.lead_day <= 5, "L1-5", np.where(raw.lead_day <= 15, "L6-15", "L16-30")
    )
    raw["paired_valid"] = np.isfinite(raw.miz_rmse) & np.isfinite(raw.non_miz_rmse)
    key = ["init_date", "lead_day", "lead_band", "domain"]
    # Shared reference categories and the complete finite forecast panel make
    # support identical across the three models; fail instead of assuming this.
    supports = raw.groupby(key).paired_valid.agg(["sum", "size"])
    assert supports["size"].eq(3).all()
    assert supports["sum"].isin([0, 3]).all()
    paired = raw[raw.paired_valid].copy()
    equal = paired.groupby(key, as_index=False).agg(
        miz_rmse=("miz_rmse", "mean"), non_miz_rmse=("non_miz_rmse", "mean")
    )
    equal["miz_minus_non_miz_rmse"] = equal.miz_rmse - equal.non_miz_rmse
    final = equal.groupby(["domain", "lead_band"], as_index=False).agg(
        miz_rmse=("miz_rmse", "mean"),
        non_miz_rmse=("non_miz_rmse", "mean"),
        miz_minus_non_miz_rmse=("miz_minus_non_miz_rmse", "mean"),
        n_initializations=("init_date", "nunique"),
        n_paired_initialization_leads=("lead_day", "size"),
    )
    counts = (
        raw.groupby(["domain", "lead_band"])
        .size()
        .div(3)
        .astype(int)
        .rename("n_total_initialization_leads")
        .reset_index()
    )
    final = final.merge(counts, on=["domain", "lead_band"])
    final["n_excluded_initialization_leads"] = (
        final.n_total_initialization_leads - final.n_paired_initialization_leads
    )
    final["summary_support"] = (
        "Same model-initialization-lead cases with nonempty MIZ and non-MIZ categories; equal mean of three model-wise RMSE values"
    )
    assert np.allclose(
        final.miz_rmse - final.non_miz_rmse,
        final.miz_minus_non_miz_rmse,
        atol=1e-14,
        rtol=0,
    )
    comparison = prior.merge(
        final, on=["domain", "lead_band"], suffixes=("_prior", "_paired")
    )
    delta = float(
        (
            comparison.miz_minus_non_miz_rmse_prior
            - comparison.miz_minus_non_miz_rmse_paired
        )
        .abs()
        .max()
    )
    assert delta < 1e-14
    final.to_csv(target, index=False)
    report = {
        "status": "PASS",
        "contrast_max_absolute_change": delta,
        "reason": "Align component means to the same paired support as the plotted contrast.",
        "rows": final.to_dict("records"),
        "before_after": comparison.to_dict("records"),
    }
    (OUT / "validation/figure7_paired_support.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
