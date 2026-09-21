from __future__ import annotations
import numpy as np
import pandas as pd

MODELS = ("CNN", "GNN", "U-Net")


def _bootstrap_multiseries_by_year(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    reps: int,
    block_length: int,
    seed: int,
    chunk_size: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized hierarchical year/moving-block bootstrap for daily series."""
    values = np.asarray(values, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if values.ndim != 2 or values.shape[0] != len(dates):
        raise ValueError("values must be a date-by-series matrix")
    if not np.isfinite(values).all():
        raise ValueError("daily series contain non-finite values")
    years = np.array(sorted(dates.year.unique()), dtype=int)
    year_arrays = {year: values[dates.year == year] for year in years}
    rng = np.random.default_rng(seed)
    boot = np.empty((reps, values.shape[1]), dtype=float)

    for chunk_start in range(0, reps, chunk_size):
        chunk_stop = min(reps, chunk_start + chunk_size)
        size = chunk_stop - chunk_start
        sampled_years = rng.choice(years, size=(size, len(years)), replace=True)
        totals = np.zeros((size, values.shape[1]), dtype=float)
        counts = np.zeros(size, dtype=int)
        for slot in range(len(years)):
            chosen = sampled_years[:, slot]
            for source_year in years:
                mask = chosen == source_year
                n_draws = int(mask.sum())
                if n_draws == 0:
                    continue
                array = year_arrays[int(source_year)]
                n_dates = array.shape[0]
                if n_dates <= block_length:
                    indices = rng.integers(0, n_dates, size=(n_draws, n_dates))
                    sampled_sums = array[indices].sum(axis=1)
                else:
                    n_blocks = int(np.ceil(n_dates / block_length))
                    starts = rng.integers(
                        0, n_dates - block_length + 1, size=(n_draws, n_blocks)
                    )
                    lengths = np.full(n_blocks, block_length, dtype=int)
                    lengths[-1] = n_dates - block_length * (n_blocks - 1)
                    cumulative = np.vstack(
                        [np.zeros((1, array.shape[1])), np.cumsum(array, axis=0)]
                    )
                    sampled_sums = np.zeros((n_draws, array.shape[1]), dtype=float)
                    for block_index, length in enumerate(lengths):
                        block_starts = starts[:, block_index]
                        sampled_sums += (
                            cumulative[block_starts + length] - cumulative[block_starts]
                        )
                totals[mask] += sampled_sums
                counts[mask] += n_dates
        boot[chunk_start:chunk_stop] = totals / counts[:, None]

    observed = values.mean(axis=0)
    lower = np.quantile(boot, 0.025, axis=0)
    upper = np.quantile(boot, 0.975, axis=0)
    return observed, lower, upper


def _zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    scale = float(values.std(ddof=0))
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - float(values.mean())) / scale


def rank_figure3_candidates(
    regime_metrics: pd.DataFrame,
    min_cells: int = 20,
    min_area_km2: float = 1000.0,
    top_k: int = 3,
    min_gap_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank lead-30 dates for localized MIZ versus compact-pack error."""
    data = regime_metrics.copy()
    data["init_date"] = pd.to_datetime(data["init_date"])
    data = data[
        data["domain"].eq("pan_arctic")
        & data["lead_day"].eq(30)
        & data["regime_family"].eq("target_sic_zone")
        & data["regime_bin"].isin(["miz", "compact_ice"])
        & data["model"].isin(MODELS)
    ].copy()
    grouped = data.groupby(["init_date", "model", "regime_bin"], as_index=False).agg(
        rmse=("rmse", "mean"),
        n_cells=("n_cells", "sum"),
        valid_area_km2=("valid_area_km2", "sum"),
    )
    wide = (
        grouped.pivot(
            index=["init_date", "model"],
            columns="regime_bin",
            values=["rmse", "n_cells", "valid_area_km2"],
        )
        .dropna()
        .reset_index()
    )
    wide.columns = [
        (
            "_".join(str(value) for value in col if str(value))
            if isinstance(col, tuple)
            else str(col)
        )
        for col in wide.columns
    ]
    wide = wide[
        wide["n_cells_miz"].ge(min_cells)
        & wide["n_cells_compact_ice"].ge(min_cells)
        & wide["valid_area_km2_miz"].ge(min_area_km2)
        & wide["valid_area_km2_compact_ice"].ge(min_area_km2)
    ].copy()
    wide["contrast"] = wide["rmse_miz"] - wide["rmse_compact_ice"]

    summary = (
        wide.groupby("init_date")
        .agg(
            n_models=("model", "nunique"),
            miz_rmse_mean=("rmse_miz", "mean"),
            compact_rmse_mean=("rmse_compact_ice", "mean"),
            contrast_mean=("contrast", "mean"),
            contrast_sd=("contrast", lambda values: float(np.std(values, ddof=0))),
            contrast_min=("contrast", "min"),
            miz_cells_min=("n_cells_miz", "min"),
            compact_cells_min=("n_cells_compact_ice", "min"),
            miz_area_km2_min=("valid_area_km2_miz", "min"),
            compact_area_km2_min=("valid_area_km2_compact_ice", "min"),
        )
        .reset_index()
    )
    summary = summary[
        summary["n_models"].eq(len(MODELS))
        & summary["contrast_min"].gt(0)
        & summary["contrast_min"].ge(0.5 * summary["contrast_mean"])
    ].copy()
    if summary.empty:
        raise ValueError(
            "No Figure 3 candidates pass support and model-agreement filters"
        )

    summary["score"] = (
        0.45 * _zscore(summary["contrast_mean"])
        + 0.25 * _zscore(summary["miz_rmse_mean"])
        - 0.20 * _zscore(summary["compact_rmse_mean"])
        - 0.10 * _zscore(summary["contrast_sd"])
    )
    ranked = summary.sort_values(
        ["score", "contrast_mean", "miz_rmse_mean"], ascending=[False, False, False]
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    selected_rows = []
    for _, row in ranked.iterrows():
        date = pd.Timestamp(row["init_date"])
        if all(
            abs((date - pd.Timestamp(existing["init_date"])).days) >= min_gap_days
            for existing in selected_rows
        ):
            selected_rows.append(row.to_dict())
        if len(selected_rows) == top_k:
            break
    selected = pd.DataFrame(selected_rows)
    if len(selected) < top_k:
        raise ValueError(
            f"Only {len(selected)} temporally separated candidates were available"
        )
    selected["selection_order"] = np.arange(1, len(selected) + 1)
    return ranked, selected


def _moving_block_dates(
    dates: pd.DatetimeIndex, block_length: int, rng: np.random.Generator
) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    if len(dates) == 0:
        return []
    if block_length < 1:
        raise ValueError("block_length must be positive")
    if len(dates) <= block_length:
        starts = np.array([0])
    else:
        count = int(np.ceil(len(dates) / block_length))
        starts = rng.integers(0, len(dates) - block_length + 1, size=count)
    sampled: list[pd.Timestamp] = []
    for start in starts:
        sampled.extend(dates[int(start) : int(start) + block_length].tolist())
    while len(sampled) < len(dates):
        start = int(rng.integers(0, max(1, len(dates) - block_length + 1)))
        sampled.extend(dates[start : start + block_length].tolist())
    return sampled[: len(dates)]


def resample_year_date_panel(
    contrast_panel: pd.DataFrame,
    block_length: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Hierarchically resample years and within-year date blocks.

    Every sampled date carries all attached model rows, preserving model and
    domain-pair dependence at the date level.
    """
    data = contrast_panel.copy()
    data["init_date"] = pd.to_datetime(data["init_date"])
    if "init_year" not in data:
        data["init_year"] = data["init_date"].dt.year
    years = np.array(sorted(data["init_year"].unique()), dtype=int)
    if len(years) == 0:
        raise ValueError("contrast_panel contains no years")
    sampled_years = rng.choice(years, size=len(years), replace=True)
    pieces = []
    draw_id = 0
    for slot, source_year in enumerate(sampled_years):
        year_panel = data[data["init_year"].eq(int(source_year))]
        sampled_dates = _moving_block_dates(
            pd.DatetimeIndex(year_panel["init_date"].unique()), block_length, rng
        )
        for sampled_date in sampled_dates:
            rows = year_panel[year_panel["init_date"].eq(sampled_date)].copy()
            rows["source_year"] = int(source_year)
            rows["source_init_date"] = pd.Timestamp(sampled_date)
            rows["sampled_year_slot"] = int(slot)
            rows["draw_id"] = int(draw_id)
            pieces.append(rows)
            draw_id += 1
    if not pieces:
        return data.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def bootstrap_paired_means(
    contrast_panel: pd.DataFrame,
    reps: int = 5000,
    block_length: int = 7,
    seed: int = 20260816,
) -> dict[str, dict[str, float | int]]:
    """Return observed means and percentile intervals for paired contrasts."""
    metrics = ("rmse_growth_diff", "acc_loss_diff")
    if reps < 1:
        raise ValueError("reps must be positive")
    data = contrast_panel.copy()
    data["init_date"] = pd.to_datetime(data["init_date"])
    if "init_year" not in data:
        data["init_year"] = data["init_date"].dt.year
    models = tuple(sorted(data["model"].unique()))
    years = np.array(sorted(data["init_year"].unique()), dtype=int)
    year_arrays: dict[int, np.ndarray] = {}
    for year in years:
        panel = data[data["init_year"].eq(int(year))]
        pivot = panel.pivot(
            index="init_date", columns="model", values=list(metrics)
        ).sort_index()
        if pivot.isna().any().any() or tuple(pivot.columns.levels[1]) != models:
            raise ValueError(
                f"Year {year} does not contain a complete date-model panel"
            )
        year_arrays[int(year)] = np.stack(
            [pivot[metric].to_numpy(dtype=float) for metric in metrics], axis=-1
        )

    def sampled_indices(n_dates: int) -> np.ndarray:
        if n_dates <= block_length:
            return rng.integers(0, n_dates, size=n_dates)
        n_blocks = int(np.ceil(n_dates / block_length))
        starts = rng.integers(0, n_dates - block_length + 1, size=n_blocks)
        offsets = np.arange(block_length)
        return (starts[:, None] + offsets[None, :]).ravel()[:n_dates]

    rng = np.random.default_rng(seed)
    distributions = {metric: np.empty(reps, dtype=float) for metric in metrics}
    for rep in range(reps):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        total = np.zeros(len(metrics), dtype=float)
        count = 0
        for source_year in sampled_years:
            values = year_arrays[int(source_year)]
            block_values = values[sampled_indices(values.shape[0])]
            total += block_values.sum(axis=(0, 1))
            count += block_values.shape[0] * block_values.shape[1]
        replicate_mean = total / count
        for metric_index, metric in enumerate(metrics):
            distributions[metric][rep] = float(replicate_mean[metric_index])

    result: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        lower, upper = np.quantile(distributions[metric], [0.025, 0.975])
        result[metric] = {
            "mean": float(data[metric].mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "n_initialization_dates": int(data["init_date"].nunique()),
            "n_model_date_cases": int(len(data)),
            "bootstrap_reps": int(reps),
            "block_days": int(block_length),
            "seed": int(seed),
        }
    return result
