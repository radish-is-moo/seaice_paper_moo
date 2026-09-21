from __future__ import annotations
from collections.abc import Iterable, Sequence
import numpy as np

EARTH_RADIUS_KM = 6371.0088


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    data = np.asarray(values, dtype=np.float64)
    area = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(data) & np.isfinite(area) & (area > 0)
    if not np.any(valid):
        return np.nan
    order = np.argsort(data[valid])
    sorted_values = data[valid][order]
    sorted_weights = area[valid][order]
    cutoff = float(quantile) * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def classify_area_weighted_bins(
    values: np.ndarray,
    area: np.ndarray,
    domain_mask: np.ndarray,
    quantiles: Sequence[float] = (0.33, 0.67, 0.90),
) -> tuple[np.ndarray, np.ndarray]:
    """Classify values using area-weighted thresholds in an ocean-only domain."""
    data = np.asarray(values, dtype=np.float64)
    weights = np.asarray(area, dtype=np.float64)
    domain = np.asarray(domain_mask, dtype=bool)
    if not (data.shape == weights.shape == domain.shape):
        raise ValueError("Values, area, and domain mask must share a shape")
    valid = domain & np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    codes = np.full(data.shape, -1, dtype=np.int8)
    thresholds = np.asarray(
        [weighted_quantile(data[valid], weights[valid], q) for q in quantiles],
        dtype=np.float64,
    )
    if np.any(valid):
        codes[valid] = np.searchsorted(
            thresholds,
            data[valid],
            side="left",
        ).astype(np.int8)
    return codes, thresholds


def _validate_grid_inputs(
    sic: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    ocean_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(sic, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    ocean = np.asarray(ocean_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("sic must be a two-dimensional latitude-longitude field")
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("latitude and longitude must be one-dimensional")
    if values.shape != (lat.size, lon.size) or ocean.shape != values.shape:
        raise ValueError("sic, coordinates, and ocean_mask must share one grid")
    if lat.size < 2:
        raise ValueError("physical gradients require at least two latitudes")
    if lon.size < 3:
        raise ValueError(
            "physical gradients require at least three periodic longitudes"
        )
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("latitude and longitude must be finite")
    if np.any(np.diff(lat) == 0.0):
        raise ValueError("adjacent latitude coordinates must be distinct")
    longitude_steps = np.diff(lon)
    if not np.all(longitude_steps > 0.0):
        raise ValueError("longitude coordinates must be strictly increasing")
    longitude_step = longitude_steps[0]
    if not np.allclose(longitude_steps, longitude_step, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("longitude coordinates must be regularly spaced")
    if not np.isclose(longitude_step * lon.size, 360.0, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError(
            "longitude coordinates must form a regularly spaced full-period grid"
        )
    return values, lat, lon, ocean


def physical_sic_gradient_km(
    sic: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
) -> np.ndarray:
    """Return an ocean-only SIC-gradient magnitude in SIC per kilometre.

    Each directional component uses a central difference only when both
    neighboring ocean cells are finite; otherwise it uses the available
    one-sided difference. Longitude wraps periodically.
    """
    values, latitude, longitude, ocean = _validate_grid_inputs(
        sic, lat, lon, ocean_mask
    )
    valid = ocean & np.isfinite(values)
    n_lat, n_lon = values.shape
    meridional = np.full(values.shape, np.nan, dtype=np.float64)
    zonal = np.full(values.shape, np.nan, dtype=np.float64)
    y = np.deg2rad(latitude) * EARTH_RADIUS_KM

    for row in range(n_lat):
        for column in range(n_lon):
            if not valid[row, column]:
                continue
            north_valid = row + 1 < n_lat and valid[row + 1, column]
            south_valid = row > 0 and valid[row - 1, column]
            if north_valid and south_valid:
                meridional[row, column] = (
                    values[row + 1, column] - values[row - 1, column]
                ) / (y[row + 1] - y[row - 1])
            elif north_valid:
                meridional[row, column] = (
                    values[row + 1, column] - values[row, column]
                ) / (y[row + 1] - y[row])
            elif south_valid:
                meridional[row, column] = (
                    values[row, column] - values[row - 1, column]
                ) / (y[row] - y[row - 1])

    longitude_rad = np.deg2rad(longitude)
    east_steps = np.mod(
        np.diff(np.r_[longitude_rad, longitude_rad[0] + 2.0 * np.pi]),
        2.0 * np.pi,
    )
    cosine_latitude = np.cos(np.deg2rad(latitude))
    for row in range(n_lat):
        if np.isclose(cosine_latitude[row], 0.0, atol=1.0e-12):
            continue
        km_per_radian = EARTH_RADIUS_KM * abs(cosine_latitude[row])
        for column in range(n_lon):
            if not valid[row, column]:
                continue
            east = (column + 1) % n_lon
            west = (column - 1) % n_lon
            east_valid = valid[row, east]
            west_valid = valid[row, west]
            east_distance = km_per_radian * east_steps[column]
            west_distance = km_per_radian * east_steps[west]
            if east_valid and west_valid:
                zonal[row, column] = (values[row, east] - values[row, west]) / (
                    east_distance + west_distance
                )
            elif east_valid:
                zonal[row, column] = (
                    values[row, east] - values[row, column]
                ) / east_distance
            elif west_valid:
                zonal[row, column] = (
                    values[row, column] - values[row, west]
                ) / west_distance

    has_component = np.isfinite(meridional) | np.isfinite(zonal)
    magnitude = np.hypot(
        np.nan_to_num(meridional, nan=0.0), np.nan_to_num(zonal, nan=0.0)
    )
    magnitude[~valid | ~has_component] = np.nan
    return magnitude
