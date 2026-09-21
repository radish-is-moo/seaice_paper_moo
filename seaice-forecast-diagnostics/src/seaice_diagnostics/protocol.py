"""Pure protocol and numerical helpers for the ver2 manuscript regeneration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

GRID_SHAPE = (101, 1440)
HORIZON_DAYS = 30


def complete_initializations(years: Iterable[int]) -> pd.DataFrame:
    """List initializations whose 30 target days stay inside the requested test span.

    ``target_year`` is the year containing the first target date.  The final
    requested year is clipped at 31 December, matching the notebook's complete
    target-window split.
    """
    normalized_years = sorted({int(year) for year in years})
    if not normalized_years:
        raise ValueError("years must not be empty")
    test_start = pd.Timestamp(year=min(normalized_years), month=1, day=1)
    test_end = pd.Timestamp(year=max(normalized_years), month=12, day=31)
    first_init = test_start - pd.Timedelta(days=1)
    last_init = test_end - pd.Timedelta(days=HORIZON_DAYS)
    rows: list[dict[str, object]] = []
    for init_date in pd.date_range(first_init, last_init, freq="D"):
        target_dates = pd.date_range(
            init_date + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D"
        )
        target_year = int(target_dates[0].year)
        if target_year not in normalized_years:
            continue
        rows.append(
            {
                "init_date": init_date.normalize(),
                "target_start": target_dates[0].normalize(),
                "target_end": target_dates[-1].normalize(),
                "target_year": target_year,
                "target_dates": tuple(target_dates),
            }
        )
    return pd.DataFrame(rows)


def paired_initializations(
    years: Iterable[int], target_months: Iterable[int]
) -> pd.DataFrame:
    """Return cases for which every lead target month is in ``target_months``."""
    months = {int(month) for month in target_months}
    if not months or not months.issubset(set(range(1, 13))):
        raise ValueError("target_months must be integers in 1..12")
    frame = complete_initializations(years)
    keep = frame["target_dates"].map(
        lambda dates: all(int(date.month) in months for date in dates)
    )
    return frame.loc[keep].reset_index(drop=True)


def pooled_weighted_rmse(weighted_sse: np.ndarray, area: np.ndarray) -> float:
    """Pool category SSE and area before taking the square root."""
    sse = np.asarray(weighted_sse, dtype=np.float64)
    weights = np.asarray(area, dtype=np.float64)
    valid = np.isfinite(sse) & np.isfinite(weights) & (weights > 0)
    total_area = float(weights[valid].sum())
    if total_area <= 0:
        return np.nan
    return float(np.sqrt(float(sse[valid].sum()) / total_area))


def atomic_savez_compressed(path: str | Path, **arrays: object) -> Path:
    """Write an NPZ through ``.part`` and atomically replace the destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(destination) + ".part")
    if part.exists():
        part.unlink()
    safe_arrays = {
        name: (
            np.asarray(value, dtype=str)
            if np.asarray(value).dtype.kind == "O"
            else value
        )
        for name, value in arrays.items()
    }
    try:
        with part.open("wb") as handle:
            np.savez_compressed(handle, **safe_arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(part, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError("temporary NPZ archive is empty")
        os.replace(part, destination)
    finally:
        if part.exists():
            part.unlink()
    return destination


def validate_prediction_archive(
    path: str | Path,
    *,
    expected_model: str | None = None,
    expected_init_date: object | None = None,
) -> dict[str, object]:
    """Validate the restart contract for one model-initialization prediction."""
    archive_path = Path(path)
    result: dict[str, object] = {"path": str(archive_path), "valid": False}
    if not archive_path.exists():
        result["reason"] = "missing"
        return result
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            required = {
                "prediction",
                "model",
                "init_date",
                "target_dates",
                "lead_days",
                "shape",
                "dtype",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"missing arrays: {sorted(missing)}")
            prediction = np.asarray(archive["prediction"])
            model = str(np.asarray(archive["model"]).item())
            init_date = str(np.asarray(archive["init_date"]).item())
            lead_days = np.asarray(archive["lead_days"], dtype=int)
            target_dates = np.asarray(archive["target_dates"]).astype(str)
            stored_shape = tuple(np.asarray(archive["shape"], dtype=int).tolist())
            stored_dtype = str(np.asarray(archive["dtype"]).item())
        if prediction.shape != (HORIZON_DAYS, *GRID_SHAPE):
            raise ValueError(f"unexpected prediction shape: {prediction.shape}")
        if stored_shape != prediction.shape:
            raise ValueError("stored shape metadata differs from the prediction")
        if (
            prediction.dtype.itemsize < np.dtype(np.float32).itemsize
            or prediction.dtype.kind != "f"
        ):
            raise ValueError(f"prediction dtype is below float32: {prediction.dtype}")
        if stored_dtype != str(prediction.dtype):
            raise ValueError("stored dtype metadata differs from the prediction")
        if not np.array_equal(lead_days, np.arange(1, HORIZON_DAYS + 1)):
            raise ValueError("lead_days are not complete 1..30")
        if (
            len(target_dates) != HORIZON_DAYS
            or len(set(target_dates.tolist())) != HORIZON_DAYS
        ):
            raise ValueError("target_dates are incomplete or duplicated")
        finite = np.isfinite(prediction)
        if finite.any() and (
            float(prediction[finite].min()) < 0.0
            or float(prediction[finite].max()) > 1.0
        ):
            raise ValueError("finite predicted SIC is outside 0..1")
        if expected_model is not None and model != str(expected_model):
            raise ValueError(f"model mismatch: {model}")
        if expected_init_date is not None:
            wanted = pd.Timestamp(expected_init_date).date().isoformat()
            if init_date != wanted:
                raise ValueError(f"init_date mismatch: {init_date}")
        result.update(
            {
                "valid": True,
                "model": model,
                "init_date": init_date,
                "shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
                "finite_min": float(prediction[finite].min()) if finite.any() else None,
                "finite_max": float(prediction[finite].max()) if finite.any() else None,
            }
        )
    except Exception as exc:
        result["reason"] = str(exc)
    return result
