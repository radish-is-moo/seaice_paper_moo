from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def derive_piomas_native_ocean_mask(
    sic: np.ndarray,
    sit: np.ndarray,
    *,
    sic_atol: float = 1.0e-7,
    sit_atol: float = 1.0e-12,
) -> np.ndarray:
    """Return the PIOMAS ocean mask from its static land-value convention.

    In the local native PIOMAS files, land is represented by exactly unit SIC
    and zero SIT. Missing values are not treated as ocean.
    """

    concentration = np.asarray(sic, dtype=np.float64)
    thickness = np.asarray(sit, dtype=np.float64)
    if concentration.shape != thickness.shape:
        raise ValueError("sic and sit must have equal shapes")
    finite = np.isfinite(concentration) & np.isfinite(thickness)
    land = (
        finite
        & np.isclose(concentration, 1.0, rtol=0.0, atol=float(sic_atol))
        & np.isclose(thickness, 0.0, rtol=0.0, atol=float(sit_atol))
    )
    return finite & ~land


def _unit_sphere_xyz(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(latitude, dtype=np.float64).ravel())
    lon = np.deg2rad(np.asarray(longitude, dtype=np.float64).ravel())
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def remap_boolean_mask_nearest(
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
    source_mask: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour remap of a static mask on the unit sphere."""

    source_lat = np.asarray(source_latitude, dtype=np.float64)
    source_lon = np.asarray(source_longitude, dtype=np.float64)
    mask = np.asarray(source_mask, dtype=bool)
    if not (source_lat.shape == source_lon.shape == mask.shape):
        raise ValueError("source latitude, longitude, and mask must match")

    target_lat = np.asarray(target_latitude, dtype=np.float64)
    target_lon = np.asarray(target_longitude, dtype=np.float64)
    if target_lat.ndim == 1 and target_lon.ndim == 1:
        target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
    elif target_lat.shape == target_lon.shape:
        target_lat_2d, target_lon_2d = target_lat, target_lon
    else:
        raise ValueError("target coordinates must be matching 2-D arrays or 1-D axes")

    tree = cKDTree(_unit_sphere_xyz(source_lat, source_lon))
    _, nearest = tree.query(
        _unit_sphere_xyz(target_lat_2d, target_lon_2d),
        k=1,
    )
    return mask.ravel()[nearest].reshape(target_lat_2d.shape)
