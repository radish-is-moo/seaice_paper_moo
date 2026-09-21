"""Diagnostics and statistics for the ver2 2023-2025 manuscript figures."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .inference import (
    CACHE_DIR,
    ERA5_SAMPLE,
    MODEL_SPECS,
    OUTPUT_ROOT,
    DailyCache,
    prediction_relative_path,
    reference_relative_path,
)
from .protocol import paired_initializations


from .config import MASK

from .numerics import classify_area_weighted_bins, weighted_quantile  # noqa: E402
from .numerics import physical_sic_gradient_km  # noqa: E402
from .statistics import (  # noqa: E402
    _bootstrap_multiseries_by_year,
    rank_figure3_candidates,
)

MODEL_KEYS = ("cnn", "unet", "gnn")
MODEL_NAMES = tuple(MODEL_SPECS[key].display_name for key in MODEL_KEYS)
SEA_DOMAINS = ("barents", "kara", "laptev", "east_siberian", "chukchi")
DOMAIN_ORDER = ("pan_arctic", "combined_nsr", *SEA_DOMAINS)
LEAD_BANDS = {"L1-5": range(1, 6), "L6-15": range(6, 16), "L16-30": range(16, 31)}


def cell_area_km2(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    lat_mid = 0.5 * (lat[:-1] + lat[1:])
    lat_edges = np.r_[
        lat[0] + (lat[0] - lat_mid[0]), lat_mid, lat[-1] + (lat[-1] - lat_mid[-1])
    ]
    lat_edges = np.clip(lat_edges, -90.0, 90.0)
    lon_mid = 0.5 * (lon[:-1] + lon[1:])
    lon_edges = np.r_[
        lon[0] + (lon[0] - lon_mid[0]), lon_mid, lon[-1] + (lon[-1] - lon_mid[-1])
    ]
    lat_band = np.abs(
        np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1]))
    )
    lon_width = np.abs(np.deg2rad(np.diff(lon_edges)))
    return (6371.0088**2) * lat_band[:, None] * lon_width[None, :]


def _longitude_between(
    lon360: np.ndarray, minimum: float, maximum: float
) -> np.ndarray:
    minimum %= 360.0
    maximum %= 360.0
    return (
        (lon360 >= minimum) & (lon360 <= maximum)
        if minimum <= maximum
        else (lon360 >= minimum) | (lon360 <= maximum)
    )


def build_domain_masks(
    latitude: np.ndarray,
    longitude: np.ndarray,
    ocean_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    lat = np.asarray(latitude, dtype=np.float64)
    lon360 = np.mod(np.asarray(longitude, dtype=np.float64), 360.0)
    ocean = np.asarray(ocean_mask, dtype=bool)
    if ocean.shape != (lat.size, lon360.size):
        raise ValueError("ocean_mask does not match latitude-longitude grid")
    lat2d, lon2d = np.meshgrid(lat, lon360, indexing="ij")
    boxes = {
        "chukchi": (66.0, 78.0, 180.0, 205.0),
        "east_siberian": (68.0, 82.5, 140.0, 180.0),
        "laptev": (70.0, 82.5, 100.0, 140.0),
        "kara": (68.0, 82.5, 60.0, 100.0),
        "barents": (68.0, 82.5, 20.0, 60.0),
    }
    masks: dict[str, np.ndarray] = {"pan_arctic": ocean.copy()}
    for name, (lat_min, lat_max, lon_min, lon_max) in boxes.items():
        masks[name] = (
            ocean
            & (lat2d >= lat_min)
            & (lat2d <= lat_max)
            & _longitude_between(lon2d, lon_min, lon_max)
        )
    masks["combined_nsr"] = np.logical_or.reduce([masks[name] for name in SEA_DOMAINS])
    return {name: masks[name] for name in DOMAIN_ORDER}


def sic_state_codes(sic: np.ndarray) -> np.ndarray:
    values = np.asarray(sic, dtype=np.float64)
    codes = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    codes[valid & (values < 0.15)] = 0
    codes[valid & (values >= 0.15) & (values <= 0.80)] = 1
    codes[valid & (values > 0.80)] = 2
    return codes


def sit_class_codes(sit: np.ndarray, sic: np.ndarray) -> np.ndarray:
    """Classify initial SIT, retaining open-water/no-ice as its own class."""
    thickness = np.asarray(sit, dtype=np.float64)
    concentration = np.asarray(sic, dtype=np.float64)
    if thickness.shape != concentration.shape:
        raise ValueError("sit and sic must share a shape")
    codes = np.full(thickness.shape, -1, dtype=np.int8)
    valid = np.isfinite(thickness) & np.isfinite(concentration)
    ice = valid & (concentration >= 0.15)
    codes[valid & (concentration < 0.15)] = 0
    codes[ice & (thickness < 0.5)] = 1
    codes[ice & (thickness >= 0.5) & (thickness < 1.5)] = 2
    codes[ice & (thickness >= 1.5)] = 3
    return codes


def joint_sit_wind_codes(sit_codes: np.ndarray, wind_codes: np.ndarray) -> np.ndarray:
    """Return thin/medium-thick by lower-middle/high-upper-tail wind codes.

    Open-water/no-ice cells (SIT code 0) and missing cells are excluded.
    """
    sit_category = np.asarray(sit_codes, dtype=np.int8)
    wind_category = np.asarray(wind_codes, dtype=np.int8)
    if sit_category.shape != wind_category.shape:
        raise ValueError("sit_codes and wind_codes must share a shape")
    wind_group = np.where(
        (wind_category >= 0) & (wind_category <= 1),
        0,
        np.where(wind_category >= 2, 1, -1),
    )
    ice_group = np.where(sit_category == 1, 0, np.where(sit_category >= 2, 1, -1))
    return np.where(
        (ice_group >= 0) & (wind_group >= 0), ice_group * 2 + wind_group, -1
    ).astype(np.int8)


def weighted_category_metrics(
    error2: np.ndarray,
    area_km2: np.ndarray,
    codes: np.ndarray,
    labels: Iterable[str],
) -> list[dict[str, object]]:
    squared = np.asarray(error2, dtype=np.float64)
    area = np.asarray(area_km2, dtype=np.float64)
    category = np.asarray(codes, dtype=np.int64)
    labels = list(labels)
    if not (squared.shape == area.shape == category.shape):
        raise ValueError("error2, area_km2, and codes must share a shape")
    valid = (
        (category >= 0)
        & (category < len(labels))
        & np.isfinite(squared)
        & np.isfinite(area)
        & (area > 0)
    )
    selected_codes = category[valid]
    selected_area = area[valid]
    selected_error = squared[valid]
    counts = np.bincount(selected_codes, minlength=len(labels))
    area_sums = np.bincount(
        selected_codes, weights=selected_area, minlength=len(labels)
    )
    sse_sums = np.bincount(
        selected_codes,
        weights=selected_area * selected_error,
        minlength=len(labels),
    )
    rows: list[dict[str, object]] = []
    for code, label in enumerate(labels):
        area_sum = float(area_sums[code])
        sse = float(sse_sums[code])
        rows.append(
            {
                "category": label,
                "n_cells": int(counts[code]),
                "area_km2": area_sum,
                "weighted_sse": sse,
                "rmse": float(np.sqrt(sse / area_sum)) if area_sum > 0 else np.nan,
            }
        )
    return rows


def batch_domain_metrics_area_weighted(
    prediction: np.ndarray,
    reference: np.ndarray,
    domain_mask: np.ndarray,
    area_km2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized cell-area-weighted RMSE and spatial pattern correlation."""
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    domain = np.asarray(domain_mask, dtype=bool)
    area = np.asarray(area_km2, dtype=np.float64)
    if pred.ndim != 3 or pred.shape != ref.shape or pred.shape[1:] != domain.shape:
        raise ValueError(
            "prediction/reference must be lead-lat-lon and match domain_mask"
        )
    if area.shape != domain.shape:
        raise ValueError("area_km2 must match domain_mask")
    index = np.flatnonzero(domain.reshape(-1))
    selected_pred = pred.reshape(pred.shape[0], -1)[:, index]
    selected_ref = ref.reshape(ref.shape[0], -1)[:, index]
    selected_area = area.reshape(-1)[index]
    valid = (
        np.isfinite(selected_pred)
        & np.isfinite(selected_ref)
        & np.isfinite(selected_area)[None, :]
        & (selected_area[None, :] > 0)
    )
    weights = np.where(valid, selected_area[None, :], 0.0)
    weight_sum = weights.sum(axis=1)
    count = valid.sum(axis=1).astype(np.int64)
    safe_weight = np.maximum(weight_sum, 1.0)
    safe_pred = np.where(valid, selected_pred, 0.0)
    safe_ref = np.where(valid, selected_ref, 0.0)
    error = safe_pred - safe_ref
    rmse = np.sqrt(np.sum(weights * np.square(error), axis=1) / safe_weight)
    mean_pred = np.sum(weights * safe_pred, axis=1) / safe_weight
    mean_ref = np.sum(weights * safe_ref, axis=1) / safe_weight
    pred_anomaly = np.where(valid, selected_pred - mean_pred[:, None], 0.0)
    ref_anomaly = np.where(valid, selected_ref - mean_ref[:, None], 0.0)
    numerator = np.sum(weights * pred_anomaly * ref_anomaly, axis=1)
    denominator = np.sqrt(
        np.sum(weights * np.square(pred_anomaly), axis=1)
        * np.sum(weights * np.square(ref_anomaly), axis=1)
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.full(pred.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    rmse[weight_sum <= 0] = np.nan
    return {
        "rmse": rmse,
        "r": correlation,
        "n_cells": count,
        "valid_area_km2": weight_sum,
    }


def _atomic_csv(
    frame: pd.DataFrame, path: Path, *, compression: str | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(path) + ".part")
    frame.to_csv(part, index=False, compression=compression)
    os.replace(part, path)


def _load_grid_and_ocean(cache):
    with np.load(MASK, allow_pickle=False) as z:
        lat, lon, ocean = z["latitude"], z["longitude"], z["ocean_mask"].astype(bool)
    if ocean.shape != (101, 1440) or ocean.sum() != 101206:
        raise ValueError("The frozen ver2 ocean mask must contain 101206 ocean cells.")
    return lat, lon, ocean


def _load_reference(
    output_root: Path, target_year: int, init_date: object
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    path = output_root / reference_relative_path(target_year, init_date)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        reference = np.asarray(archive["reference"], dtype=np.float32)
        targets = pd.DatetimeIndex(np.asarray(archive["target_dates"]).astype(str))
    return reference, targets


def _load_prediction(
    output_root: Path, model_key: str, target_year: int, init_date: object
) -> np.ndarray:
    path = output_root / prediction_relative_path(model_key, target_year, init_date)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["prediction"], dtype=np.float32)


def _initial_categories(
    cache: DailyCache,
    init_date: object,
    lat: np.ndarray,
    lon: np.ndarray,
    area: np.ndarray,
    ocean: np.ndarray,
    combined_nsr: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    day = cache.load_day(init_date)
    daily = day["daily"]
    sic = np.asarray(day["sic_full"], dtype=np.float32)
    sit = np.asarray(daily[7], dtype=np.float32)
    wind = np.hypot(
        np.asarray(daily[2], dtype=np.float64), np.asarray(daily[3], dtype=np.float64)
    )
    gradient = physical_sic_gradient_km(sic, lat, lon, ocean)
    gradient_codes, gradient_thresholds = classify_area_weighted_bins(
        gradient, area, combined_nsr
    )
    wind_codes, wind_thresholds = classify_area_weighted_bins(wind, area, combined_nsr)
    sit_codes = sit_class_codes(sit, sic)
    state_codes = sic_state_codes(sic)
    ice = combined_nsr & np.isfinite(sic) & np.isfinite(sit) & (sic >= 0.15)
    thin_area = float(area[ice & (sit_codes == 1)].sum())
    medium_area = float(area[ice & ((sit_codes == 2) | (sit_codes == 3))].sum())
    ice_area = float(area[ice].sum())
    thresholds = {
        "gradient_q33_sic_per_km": float(gradient_thresholds[0]),
        "gradient_q67_sic_per_km": float(gradient_thresholds[1]),
        "gradient_q90_sic_per_km": float(gradient_thresholds[2]),
        "wind_q33_ms": float(wind_thresholds[0]),
        "wind_q67_ms": float(wind_thresholds[1]),
        "wind_q90_ms": float(wind_thresholds[2]),
        "thin_area_km2": thin_area,
        "medium_thick_area_km2": medium_area,
        "ice_covered_valid_area_km2": ice_area,
        "sit_partition_abs_error_km2": abs((thin_area + medium_area) - ice_area),
    }
    return {
        "sic": sic,
        "initial_state": state_codes,
        "gradient": gradient_codes,
        "wind": wind_codes,
        "sit": sit_codes,
    }, thresholds


def _append_context(
    rows: list[dict[str, object]], metrics: list[dict[str, object]], **context: object
) -> None:
    for metric in metrics:
        rows.append({**context, **metric})


def _bootstrap_lead_curves(
    domain_frame: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    keys = ["init_date", "target_date", "target_year", "lead_day", "domain"]
    aggregations: dict[str, tuple[str, str]] = {
        "rmse": ("rmse", "mean"),
        "r": ("r", "mean"),
        "n_cells": ("n_cells", "min"),
        "model_count": ("model", "nunique"),
    }
    if "valid_area_km2" in domain_frame.columns:
        aggregations["valid_area_km2"] = ("valid_area_km2", "mean")
    equal = domain_frame.groupby(keys, as_index=False).agg(**aggregations)
    if not equal["model_count"].eq(3).all():
        raise ValueError("Domain metrics are not complete across all three models")
    equal["model"] = "Three-model equal mean"
    data = pd.concat([domain_frame, equal[domain_frame.columns]], ignore_index=True)
    rows: list[dict[str, object]] = []
    for lead, group in data.groupby("lead_day", sort=True):
        pivot = group.pivot(
            index="init_date", columns=["model", "domain"], values=["rmse", "r"]
        ).sort_index()
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete paired model-domain panel at lead {lead}")
        observed, lower, upper = _bootstrap_multiseries_by_year(
            pivot.to_numpy(dtype=np.float64),
            pd.DatetimeIndex(pivot.index),
            reps=int(repetitions),
            block_length=7,
            seed=int(seed) + int(lead),
        )
        for index, (metric, model, domain) in enumerate(pivot.columns):
            rows.append(
                {
                    "model": model,
                    "domain": domain,
                    "lead_day": int(lead),
                    "metric": metric,
                    "mean": float(observed[index]),
                    "ci_lower": float(lower[index]),
                    "ci_upper": float(upper[index]),
                    "n_initializations": int(len(pivot)),
                    "bootstrap_reps": int(repetitions),
                    "block_days": 7,
                    "seed_base": int(seed),
                    "lead_seed": int(seed) + int(lead),
                }
            )
    return pd.DataFrame(rows)


def _summarize_figure5(raw: pd.DataFrame) -> pd.DataFrame:
    lead_lookup = {lead: band for band, leads in LEAD_BANDS.items() for lead in leads}
    state = raw[raw["analysis_family"].eq("verification_state")].copy()
    state["lead_band"] = state["lead_day"].map(lead_lookup)
    state_by_init = state.groupby(
        ["model", "analysis_family", "category", "lead_band", "init_date"],
        as_index=False,
    ).agg(rmse=("rmse", "mean"), area_km2=("area_km2", "mean"))
    state_summary = state_by_init.groupby(
        ["model", "analysis_family", "category", "lead_band"],
        as_index=False,
    ).agg(
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        mean_area_km2=("area_km2", "mean"),
        n_initializations=("rmse", "count"),
        n_initializations_total=("init_date", "nunique"),
    )

    initial = raw[~raw["analysis_family"].eq("verification_state")].copy()
    keys = ["init_date", "target_date", "lead_day", "analysis_family", "category"]
    equal = initial.groupby(keys, as_index=False).agg(
        rmse=("rmse", "mean"),
        area_km2=("area_km2", "mean"),
        n_cells=("n_cells", "min"),
        model_count=("model", "nunique"),
    )
    if len(equal) and not equal["model_count"].eq(3).all():
        raise ValueError("Figure 5 rows are incomplete across models")
    equal["lead_band"] = equal["lead_day"].map(lead_lookup)
    by_init = equal.groupby(
        ["analysis_family", "category", "lead_band", "init_date"], as_index=False
    ).agg(rmse=("rmse", "mean"), area_km2=("area_km2", "mean"))
    initial_summary = by_init.groupby(
        ["analysis_family", "category", "lead_band"], as_index=False
    ).agg(
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        mean_area_km2=("area_km2", "mean"),
        n_initializations=("rmse", "count"),
        n_initializations_total=("init_date", "nunique"),
    )
    initial_summary["model"] = "Three-model equal mean"
    summary = pd.concat([state_summary, initial_summary], ignore_index=True)
    growth_keys = ["model", "analysis_family", "category"]
    early = summary[summary["lead_band"].eq("L1-5")][
        growth_keys + ["rmse_mean"]
    ].rename(columns={"rmse_mean": "rmse_early"})
    late = summary[summary["lead_band"].eq("L16-30")][
        growth_keys + ["rmse_mean"]
    ].rename(columns={"rmse_mean": "rmse_late"})
    growth = early.merge(late, on=growth_keys, how="outer")
    growth["rmse_growth"] = growth["rmse_late"] - growth["rmse_early"]
    return summary.merge(growth, on=growth_keys, how="left")


def _summarize_figure7(
    transitions: pd.DataFrame, errors: pd.DataFrame, types: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lead_lookup = {lead: band for band, leads in LEAD_BANDS.items() for lead in leads}
    transitions = transitions.copy()
    transitions["lead_band"] = transitions["lead_day"].map(lead_lookup)
    transition_summary = transitions.groupby(
        ["domain", "lead_band"], as_index=False
    ).agg(
        changed_area_km2=("changed_area_km2", "sum"),
        valid_area_km2=("valid_area_km2", "sum"),
    )
    transition_summary["transition_fraction"] = (
        transition_summary["changed_area_km2"] / transition_summary["valid_area_km2"]
    )
    types = types.copy()
    types["lead_band"] = types["lead_day"].map(lead_lookup)
    type_summary = types.groupby(
        ["domain", "lead_band", "transition_type"], as_index=False
    ).agg(
        transition_area_km2=("transition_area_km2", "sum"),
        valid_area_km2=("valid_area_km2", "sum"),
    )
    type_summary["area_fraction_of_valid_grid_days"] = (
        type_summary["transition_area_km2"] / type_summary["valid_area_km2"]
    )
    errors = errors.copy()
    errors["lead_band"] = errors["lead_day"].map(lead_lookup)
    model_contrast = errors.assign(contrast=errors["miz_rmse"] - errors["non_miz_rmse"])
    equal = model_contrast.groupby(
        ["init_date", "lead_day", "lead_band", "domain"], as_index=False
    ).agg(
        miz_rmse=("miz_rmse", "mean"),
        non_miz_rmse=("non_miz_rmse", "mean"),
        miz_minus_non_miz_rmse=("contrast", "mean"),
        model_count=("model", "nunique"),
    )
    if not equal["model_count"].eq(3).all():
        raise ValueError("Figure 7 error rows are incomplete across models")
    error_summary = equal.groupby(["domain", "lead_band"], as_index=False).agg(
        miz_rmse=("miz_rmse", "mean"),
        non_miz_rmse=("non_miz_rmse", "mean"),
        miz_minus_non_miz_rmse=("miz_minus_non_miz_rmse", "mean"),
        n_initializations=("init_date", "nunique"),
    )
    return transition_summary, error_summary, type_summary


def _month_day_keys() -> tuple[list[str], dict[str, int]]:
    keys = [
        date.strftime("%m-%d")
        for date in pd.date_range("2000-01-01", "2000-12-31", freq="D")
    ]
    return keys, {key: index for index, key in enumerate(keys)}


def seasonal_reference_dates(
    years: Iterable[int],
    months: Iterable[int] = (7, 8, 9, 10),
) -> dict[int, pd.DatetimeIndex]:
    selected_months = tuple(sorted({int(month) for month in months}))
    if not selected_months or any(month < 1 or month > 12 for month in selected_months):
        raise ValueError("months must contain calendar month numbers from 1 to 12")
    windows: dict[int, pd.DatetimeIndex] = {}
    for year_value in years:
        year = int(year_value)
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        windows[year] = dates[dates.month.isin(selected_months)]
    return windows


def persistence_quality_mask(
    ocean: np.ndarray,
    pair_counts: np.ndarray,
    anomaly_std: np.ndarray,
    *,
    min_pairs: int = 60,
    min_std: float = 0.02,
) -> np.ndarray:
    ocean_mask = np.asarray(ocean, dtype=bool)
    counts = np.asarray(pair_counts)
    spread = np.asarray(anomaly_std, dtype=np.float64)
    if counts.ndim != ocean_mask.ndim + 1 or counts.shape[1:] != ocean_mask.shape:
        raise ValueError("pair_counts must be lag-lat-lon and match ocean")
    if spread.shape != ocean_mask.shape:
        raise ValueError("anomaly_std must match ocean")
    return (
        ocean_mask
        & np.isfinite(spread)
        & (spread >= float(min_std))
        & np.all(counts >= int(min_pairs), axis=0)
    )


def compute_reference_persistence(
    *,
    output_root: Path,
    cache: DailyCache,
    lat: np.ndarray,
    lon: np.ndarray,
    ocean: np.ndarray,
    domains: dict[str, np.ndarray],
    force: bool,
    climatology_root: Path | None = None,
    test_months: Iterable[int] = (7, 8, 9, 10),
    min_pairs: int = 60,
    min_std: float = 0.02,
) -> tuple[Path, Path]:
    diagnostic_dir = output_root / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    climatology_root = (
        Path(climatology_root) if climatology_root is not None else Path(output_root)
    )
    climatology_path = (
        climatology_root / "cache" / "reference_calendar_day_climatology_1979_2019.npz"
    )
    persistence_path = (
        diagnostic_dir / "figure6_reference_anomaly_persistence_jul_oct_2023_2025.npz"
    )
    summary_path = (
        output_root
        / "tables"
        / "figure6_reference_persistence_summary_jul_oct_2023_2025.csv"
    )
    keys, key_index = _month_day_keys()
    if climatology_path.exists():
        with np.load(climatology_path, allow_pickle=False) as archive:
            climatology = np.asarray(archive["climatology"], dtype=np.float32)
    else:
        sums = np.zeros((366, *ocean.shape), dtype=np.float64)
        counts = np.zeros((366, *ocean.shape), dtype=np.uint16)
        dates = pd.date_range("1979-01-01", "2019-12-31", freq="D")
        for index, date in enumerate(dates, start=1):
            values = np.asarray(cache.load_day(date)["sic_full"], dtype=np.float32)
            slot = key_index[date.strftime("%m-%d")]
            valid = ocean & np.isfinite(values)
            sums[slot][valid] += values[valid]
            counts[slot][valid] += 1
            if index % 1000 == 0:
                print(f"[Figure 6 climatology] {index}/{len(dates)}", flush=True)
        climatology = np.divide(
            sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
        ).astype(np.float32)
        from .protocol import atomic_savez_compressed

        atomic_savez_compressed(
            climatology_path,
            climatology=climatology,
            month_day=np.asarray(keys, dtype=str),
            counts=counts,
            period=np.asarray("1979-01-01/2019-12-31"),
            definition=np.asarray("calendar-day mean reference SIC; no trend removal"),
        )
    if persistence_path.exists() and summary_path.exists() and not force:
        return persistence_path, summary_path
    date_windows = seasonal_reference_dates((2023, 2024, 2025), test_months)
    anomaly_by_year: dict[int, np.ndarray] = {}
    miz_hits = np.zeros(ocean.shape, dtype=np.float64)
    miz_counts = np.zeros(ocean.shape, dtype=np.float64)
    anomaly_count = np.zeros(ocean.shape, dtype=np.uint16)
    anomaly_sum = np.zeros(ocean.shape, dtype=np.float64)
    anomaly_sum2 = np.zeros(ocean.shape, dtype=np.float64)
    for year, dates in date_windows.items():
        values = np.empty((len(dates), *ocean.shape), dtype=np.float32)
        for index, date in enumerate(dates):
            sic = np.asarray(cache.load_day(date)["sic_full"], dtype=np.float32)
            anomaly = sic - climatology[key_index[date.strftime("%m-%d")]]
            values[index] = anomaly
            valid_sic = ocean & np.isfinite(sic)
            miz_hits[valid_sic] += (
                (sic[valid_sic] >= 0.15) & (sic[valid_sic] <= 0.80)
            ).astype(float)
            miz_counts[valid_sic] += 1.0
            valid_anomaly = ocean & np.isfinite(anomaly)
            anomaly_count[valid_anomaly] += 1
            anomaly_sum[valid_anomaly] += anomaly[valid_anomaly]
            anomaly_sum2[valid_anomaly] += np.square(
                anomaly[valid_anomaly], dtype=np.float64
            )
        anomaly_by_year[year] = values
    correlations = np.full((30, *ocean.shape), np.nan, dtype=np.float32)
    pair_counts = np.zeros((30, *ocean.shape), dtype=np.uint16)
    for lag in range(1, 31):
        count = np.zeros(ocean.shape, dtype=np.uint32)
        sx = np.zeros(ocean.shape, dtype=np.float64)
        sy = np.zeros(ocean.shape, dtype=np.float64)
        sxx = np.zeros(ocean.shape, dtype=np.float64)
        syy = np.zeros(ocean.shape, dtype=np.float64)
        sxy = np.zeros(ocean.shape, dtype=np.float64)
        for values in anomaly_by_year.values():
            for start in range(0, values.shape[0] - lag, 31):
                x = values[start : min(values.shape[0] - lag, start + 31)]
                y = values[start + lag : min(values.shape[0], start + lag + x.shape[0])]
                valid = np.isfinite(x) & np.isfinite(y)
                xv = np.where(valid, x, 0.0).astype(np.float64)
                yv = np.where(valid, y, 0.0).astype(np.float64)
                count += valid.sum(axis=0, dtype=np.uint32)
                sx += xv.sum(axis=0)
                sy += yv.sum(axis=0)
                sxx += np.square(xv).sum(axis=0)
                syy += np.square(yv).sum(axis=0)
                sxy += (xv * yv).sum(axis=0)
        pair_counts[lag - 1] = count.astype(np.uint16)
        safe = np.maximum(count.astype(np.float64), 1.0)
        numerator = sxy - sx * sy / safe
        varx = np.maximum(sxx - sx * sx / safe, 0.0)
        vary = np.maximum(syy - sy * sy / safe, 0.0)
        denominator = np.sqrt(varx * vary)
        correlations[lag - 1] = np.divide(
            numerator,
            denominator,
            out=np.full(ocean.shape, np.nan),
            where=(denominator > 0) & (count >= 2) & ocean,
        ).astype(np.float32)
        print(f"[Figure 6 persistence] lag {lag}/30", flush=True)
    anomaly_safe_count = np.maximum(anomaly_count.astype(np.float64), 1.0)
    anomaly_variance = np.maximum(
        anomaly_sum2 / anomaly_safe_count - np.square(anomaly_sum / anomaly_safe_count),
        0.0,
    )
    anomaly_std = np.sqrt(anomaly_variance).astype(np.float32)
    valid_display = persistence_quality_mask(
        ocean,
        pair_counts,
        anomaly_std,
        min_pairs=min_pairs,
        min_std=min_std,
    )
    persistence = np.full(ocean.shape, 31.0, dtype=np.float32)
    previous = np.ones(ocean.shape, dtype=np.float32)
    unresolved = ocean.copy()
    for lag in range(1, 31):
        current = correlations[lag - 1]
        crossing = (
            unresolved & np.isfinite(previous) & np.isfinite(current) & (current <= 0.5)
        )
        denominator = previous[crossing] - current[crossing]
        fraction = np.divide(
            previous[crossing] - 0.5,
            denominator,
            out=np.ones_like(denominator),
            where=denominator != 0,
        )
        persistence[crossing] = (lag - 1 + fraction).astype(np.float32)
        unresolved[crossing] = False
        previous = current
    persistence[~valid_display] = np.nan
    miz_frequency = np.divide(
        miz_hits, miz_counts, out=np.full(ocean.shape, np.nan), where=miz_counts > 0
    ).astype(np.float32)
    from .protocol import atomic_savez_compressed

    atomic_savez_compressed(
        persistence_path,
        lat=np.asarray(lat, dtype=np.float32),
        lon=np.asarray(lon, dtype=np.float32),
        ocean_mask=ocean.astype(np.uint8),
        persistence_days=persistence,
        r_by_lag=correlations,
        n_pairs_by_lag=pair_counts,
        anomaly_std=anomaly_std,
        valid_display_mask=valid_display.astype(np.uint8),
        lags=np.arange(1, 31, dtype=np.int16),
        miz_frequency=miz_frequency,
        climatology_period=np.asarray("1979-01-01/2019-12-31"),
        test_period=np.asarray(
            "2023-07-01/2023-10-31;2024-07-01/2024-10-31;2025-07-01/2025-10-31"
        ),
        test_months=np.asarray(
            sorted({int(value) for value in test_months}), dtype=np.int8
        ),
        min_pair_count=np.asarray(int(min_pairs), dtype=np.int16),
        min_anomaly_std=np.asarray(float(min_std), dtype=np.float32),
        definition=np.asarray(
            "same-year July-October Pearson r of reference SIC anomalies; require every lag count>=60 and seasonal anomaly std>=0.02; linear interpolation at first r<=0.5; 31 means >30 days"
        ),
    )
    rows = []
    area_grid = cell_area_km2(lat, lon)
    for domain, mask in domains.items():
        valid = (
            mask
            & valid_display
            & np.isfinite(persistence)
            & np.isfinite(miz_frequency)
            & np.isfinite(area_grid)
        )
        weights = area_grid[valid]
        rows.append(
            {
                "domain": domain,
                "median_persistence_days_area_weighted": weighted_quantile(
                    persistence[valid], weights, 0.5
                ),
                "area_fraction_persistence_le_14_days": float(
                    np.sum(weights * (persistence[valid] <= 14)) / np.sum(weights)
                ),
                "mean_miz_frequency_area_weighted": float(
                    np.sum(weights * miz_frequency[valid]) / np.sum(weights)
                ),
                "n_cells": int(valid.sum()),
                "minimum_pair_count_across_displayed_cells_and_lags": (
                    int(pair_counts[:, valid].min()) if valid.any() else 0
                ),
                "minimum_anomaly_std_across_displayed_cells": (
                    float(anomaly_std[valid].min()) if valid.any() else np.nan
                ),
            }
        )
    _atomic_csv(pd.DataFrame(rows), summary_path)
    return persistence_path, summary_path


def run_diagnostics(
    *,
    target_months: Iterable[int] = (7, 8, 9, 10),
    bootstrap_reps: int = 5000,
    seed: int = 20260816,
    output_root: Path = OUTPUT_ROOT,
    prediction_root: Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    output_root = Path(output_root)
    prediction_root = (
        Path(prediction_root) if prediction_root is not None else output_root
    )
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    cases = paired_initializations([2023, 2024, 2025], target_months)
    if cases.groupby("target_year").size().to_dict() != {2023: 94, 2024: 94, 2025: 94}:
        raise ValueError(
            "Paired target-month filtering did not yield 94 initializations per year"
        )
    cache = DailyCache()
    lat, lon, ocean = _load_grid_and_ocean(cache)
    area = cell_area_km2(lat, lon)
    domains = build_domain_masks(lat, lon, ocean)
    domain_indices = {
        name: np.flatnonzero(mask.reshape(-1)) for name, mask in domains.items()
    }
    combined_index = domain_indices["combined_nsr"]
    area_flat = area.reshape(-1)
    domain_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    figure5_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    transition_type_rows: list[dict[str, object]] = []
    figure7_error_rows: list[dict[str, object]] = []
    transition_labels = {
        (0, 1): "open_water_to_miz",
        (0, 2): "open_water_to_compact_ice",
        (1, 0): "miz_to_open_water",
        (1, 2): "miz_to_compact_ice",
        (2, 0): "compact_ice_to_open_water",
        (2, 1): "compact_ice_to_miz",
    }
    for case_number, case in enumerate(cases.itertuples(index=False), start=1):
        reference, target_dates = _load_reference(
            prediction_root, case.target_year, case.init_date
        )
        predictions = {
            model_key: _load_prediction(
                prediction_root, model_key, case.target_year, case.init_date
            )
            for model_key in MODEL_KEYS
        }
        categories, thresholds = _initial_categories(
            cache, case.init_date, lat, lon, area, ocean, domains["combined_nsr"]
        )
        threshold_rows.append(
            {"init_date": case.init_date, "target_year": case.target_year, **thresholds}
        )
        init_state = categories["initial_state"]
        for model_key in MODEL_KEYS:
            model_name = MODEL_SPECS[model_key].display_name
            for domain_name, domain_mask in domains.items():
                metrics = batch_domain_metrics_area_weighted(
                    predictions[model_key], reference, domain_mask, area
                )
                for lead_index, target_date in enumerate(target_dates):
                    domain_rows.append(
                        {
                            "model": model_name,
                            "init_date": case.init_date,
                            "target_date": target_date,
                            "target_year": case.target_year,
                            "lead_day": lead_index + 1,
                            "domain": domain_name,
                            "rmse": float(metrics["rmse"][lead_index]),
                            "r": float(metrics["r"][lead_index]),
                            "n_cells": int(metrics["n_cells"][lead_index]),
                            "valid_area_km2": float(
                                metrics["valid_area_km2"][lead_index]
                            ),
                        }
                    )
        for lead_index, target_date in enumerate(target_dates):
            lead_day = lead_index + 1
            target_state = sic_state_codes(reference[lead_index])
            for domain_name in SEA_DOMAINS:
                index = domain_indices[domain_name]
                initial_selected = init_state.reshape(-1)[index]
                target_selected = target_state.reshape(-1)[index]
                area_selected = area_flat[index]
                valid = (
                    (initial_selected >= 0)
                    & (target_selected >= 0)
                    & np.isfinite(area_selected)
                    & (area_selected > 0)
                )
                changed = valid & (initial_selected != target_selected)
                transition_rows.append(
                    {
                        "init_date": case.init_date,
                        "target_date": target_date,
                        "target_year": case.target_year,
                        "lead_day": lead_day,
                        "domain": domain_name,
                        "changed_area_km2": float(area_selected[changed].sum()),
                        "valid_area_km2": float(area_selected[valid].sum()),
                        "transition_fraction": (
                            float(
                                area_selected[changed].sum()
                                / area_selected[valid].sum()
                            )
                            if valid.any()
                            else np.nan
                        ),
                    }
                )
                for (source_code, target_code), label in transition_labels.items():
                    selected = (
                        valid
                        & (initial_selected == source_code)
                        & (target_selected == target_code)
                    )
                    transition_type_rows.append(
                        {
                            "init_date": case.init_date,
                            "target_date": target_date,
                            "target_year": case.target_year,
                            "lead_day": lead_day,
                            "domain": domain_name,
                            "transition_type": label,
                            "transition_area_km2": float(area_selected[selected].sum()),
                            "valid_area_km2": float(area_selected[valid].sum()),
                        }
                    )
            for model_key in MODEL_KEYS:
                model_name = MODEL_SPECS[model_key].display_name
                prediction = predictions[model_key][lead_index]
                ref = reference[lead_index]
                error2 = np.square(
                    prediction.astype(np.float64) - ref.astype(np.float64)
                )
                error_flat = error2.reshape(-1)
                target_flat = target_state.reshape(-1)
                for domain_name in SEA_DOMAINS:
                    index = domain_indices[domain_name]
                    domain_error = error_flat[index]
                    domain_area = area_flat[index]
                    domain_state = target_flat[index]
                    miz_codes = np.where(domain_state == 1, 0, -1)
                    non_codes = np.where(
                        (domain_state == 0) | (domain_state == 2), 0, -1
                    )
                    miz_metric = weighted_category_metrics(
                        domain_error, domain_area, miz_codes, ["miz"]
                    )[0]
                    non_metric = weighted_category_metrics(
                        domain_error, domain_area, non_codes, ["non_miz"]
                    )[0]
                    figure7_error_rows.append(
                        {
                            "model": model_name,
                            "init_date": case.init_date,
                            "target_date": target_date,
                            "target_year": case.target_year,
                            "lead_day": lead_day,
                            "domain": domain_name,
                            "miz_rmse": miz_metric["rmse"],
                            "miz_area_km2": miz_metric["area_km2"],
                            "miz_weighted_sse": miz_metric["weighted_sse"],
                            "non_miz_rmse": non_metric["rmse"],
                            "non_miz_area_km2": non_metric["area_km2"],
                            "non_miz_weighted_sse": non_metric["weighted_sse"],
                        }
                    )
                if lead_day == 30:
                    pan_codes = np.where(domains["pan_arctic"], target_state, -1)
                    state_metrics = weighted_category_metrics(
                        error2, area, pan_codes, ["open_water", "miz", "compact_ice"]
                    )
                    for item in state_metrics:
                        regime_rows.append(
                            {
                                "model": model_name,
                                "init_date": case.init_date,
                                "target_date": target_date,
                                "target_year": case.target_year,
                                "lead_day": 30,
                                "domain": "pan_arctic",
                                "regime_family": "target_sic_zone",
                                "regime_bin": item["category"],
                                "rmse": item["rmse"],
                                "n_cells": item["n_cells"],
                                "valid_area_km2": item["area_km2"],
                            }
                        )
                combined_error = error_flat[combined_index]
                combined_area = area_flat[combined_index]
                combined_target_state = target_flat[combined_index]
                _append_context(
                    figure5_rows,
                    weighted_category_metrics(
                        combined_error,
                        combined_area,
                        combined_target_state,
                        ["open_water", "miz", "compact_ice"],
                    ),
                    model=model_name,
                    init_date=case.init_date,
                    target_date=target_date,
                    target_year=case.target_year,
                    lead_day=lead_day,
                    analysis_family="verification_state",
                )
                gradient_codes = categories["gradient"].reshape(-1)[combined_index]
                _append_context(
                    figure5_rows,
                    weighted_category_metrics(
                        combined_error,
                        combined_area,
                        gradient_codes,
                        ["low", "moderate", "high", "extreme"],
                    ),
                    model=model_name,
                    init_date=case.init_date,
                    target_date=target_date,
                    target_year=case.target_year,
                    lead_day=lead_day,
                    analysis_family="initial_gradient",
                )
                sit_codes = categories["sit"].reshape(-1)[combined_index]
                _append_context(
                    figure5_rows,
                    weighted_category_metrics(
                        combined_error,
                        combined_area,
                        sit_codes,
                        ["open_water_no_ice", "thin_ice", "medium_ice", "thick_ice"],
                    ),
                    model=model_name,
                    init_date=case.init_date,
                    target_date=target_date,
                    target_year=case.target_year,
                    lead_day=lead_day,
                    analysis_family="initial_sit_class",
                )
                wind_codes = categories["wind"].reshape(-1)[combined_index]
                joint = joint_sit_wind_codes(sit_codes, wind_codes)
                _append_context(
                    figure5_rows,
                    weighted_category_metrics(
                        combined_error,
                        combined_area,
                        joint,
                        [
                            "thin_ice__lower_middle_wind",
                            "thin_ice__high_upper_tail_wind",
                            "medium_thick_ice__lower_middle_wind",
                            "medium_thick_ice__high_upper_tail_wind",
                        ],
                    ),
                    model=model_name,
                    init_date=case.init_date,
                    target_date=target_date,
                    target_year=case.target_year,
                    lead_day=lead_day,
                    analysis_family="initial_sit_x_wind",
                )
        print(
            f"[diagnostics] {case_number}/{len(cases)} init={case.init_date.date()}",
            flush=True,
        )
    domain_frame = pd.DataFrame(domain_rows)
    regime_frame = pd.DataFrame(regime_rows)
    figure5_frame = pd.DataFrame(figure5_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    transition_frame = pd.DataFrame(transition_rows)
    transition_type_frame = pd.DataFrame(transition_type_rows)
    figure7_error_frame = pd.DataFrame(figure7_error_rows)
    _atomic_csv(
        domain_frame,
        table_dir / "domain_metrics_area_weighted.csv.gz",
        compression="gzip",
    )
    _atomic_csv(
        regime_frame,
        table_dir / "figure3_regime_metrics_area_weighted.csv.gz",
        compression="gzip",
    )
    ranked, selected = rank_figure3_candidates(
        regime_frame.rename(columns={"combined_nsr": "nsr_corridor"})
    )
    _atomic_csv(ranked, table_dir / "figure3_candidate_ranking.csv")
    _atomic_csv(selected, table_dir / "figure3_selected_candidates.csv")
    bootstrap = _bootstrap_lead_curves(
        domain_frame, repetitions=bootstrap_reps, seed=seed
    )
    _atomic_csv(
        bootstrap,
        table_dir / "figure4_lead_time_metrics_area_weighted_bootstrap5000.csv.gz",
        compression="gzip",
    )
    _atomic_csv(
        figure5_frame,
        table_dir / "figure5_condition_metrics_area_weighted.csv.gz",
        compression="gzip",
    )
    _atomic_csv(
        threshold_frame, table_dir / "figure5_initial_thresholds_area_weighted.csv"
    )
    figure5_summary = _summarize_figure5(figure5_frame)
    _atomic_csv(figure5_summary, table_dir / "figure5_condition_summary.csv")
    _atomic_csv(
        transition_frame,
        table_dir / "figure7_reference_transition_cases.csv.gz",
        compression="gzip",
    )
    _atomic_csv(
        transition_type_frame,
        table_dir / "figure7_reference_transition_types.csv.gz",
        compression="gzip",
    )
    _atomic_csv(
        figure7_error_frame,
        table_dir / "figure7_miz_non_miz_model_metrics.csv.gz",
        compression="gzip",
    )
    transition_summary, error_summary, type_summary = _summarize_figure7(
        transition_frame, figure7_error_frame, transition_type_frame
    )
    _atomic_csv(transition_summary, table_dir / "figure7_transition_summary.csv")
    _atomic_csv(error_summary, table_dir / "figure7_miz_non_miz_summary.csv")
    _atomic_csv(type_summary, table_dir / "figure7_transition_type_summary.csv")
    persistence_path, persistence_summary = compute_reference_persistence(
        output_root=output_root,
        cache=cache,
        lat=lat,
        lon=lon,
        ocean=ocean,
        domains=domains,
        force=force,
        climatology_root=prediction_root,
        test_months=target_months,
    )
    old_case = ranked[
        pd.to_datetime(ranked["init_date"]).eq(pd.Timestamp("2024-09-24"))
    ]
    report = {
        "paired_initializations": int(cases["init_date"].nunique()),
        "paired_by_year": {
            str(key): int(value)
            for key, value in cases.groupby("target_year").size().items()
        },
        "target_months": sorted({int(value) for value in target_months}),
        "domain_metric_weighting": "grid-cell area weighted over jointly valid ocean cells",
        "figure5_weighting": "grid-cell area weighted",
        "figure3_top_init_date": pd.Timestamp(ranked.iloc[0]["init_date"])
        .date()
        .isoformat(),
        "figure3_2024_09_24_rank": (
            int(old_case.iloc[0]["rank"]) if not old_case.empty else None
        ),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(seed),
        "persistence_path": str(persistence_path),
        "persistence_summary": str(persistence_summary),
    }
    report_path = output_root / "diagnostics" / "diagnostic_run_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(report_path) + ".part")
    part.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(part, report_path)
    return report
