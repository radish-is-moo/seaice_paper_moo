import numpy as np

REGION_LABELS = {
    "barents": "Barents",
    "kara": "Kara",
    "laptev": "Laptev",
    "east_siberian": "East Siberian",
    "chukchi": "Chukchi",
}

NSR_REGION_BOXES = [
    ("barents", 68.0, 82.5, 20.0, 60.0),
    ("kara", 68.0, 82.5, 60.0, 100.0),
    ("laptev", 70.0, 82.5, 100.0, 140.0),
    ("east_siberian", 68.0, 82.5, 140.0, 180.0),
    ("chukchi", 66.0, 78.0, 180.0, 205.0),
]

REGION_COLORS = {
    "barents": "#6FC7BF",
    "kara": "#E7C45F",
    "laptev": "#F2AA62",
    "east_siberian": "#E98B78",
    "chukchi": "#9B88C2",
}

LABEL_POSITIONS = {
    "barents": (40.0, 74.2),
    "kara": (82.0, 73.5),
    "laptev": (121.0, 75.7),
    "east_siberian": (160.0, 76.8),
    "chukchi": (192.0, 72.0),
}


def sector_polygon(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a densely sampled polar-sector polygon in lon/lat coordinates."""
    top_lon = np.linspace(lon_min, lon_max, 80)
    right_lat = np.linspace(lat_max, lat_min, 32)
    bottom_lon = np.linspace(lon_max, lon_min, 80)
    left_lat = np.linspace(lat_min, lat_max, 32)

    lon = np.concatenate(
        [
            top_lon,
            np.full_like(right_lat, lon_max),
            bottom_lon,
            np.full_like(left_lat, lon_min),
            np.asarray([lon_min]),
        ]
    )
    lat = np.concatenate(
        [
            np.full_like(top_lon, lat_max),
            right_lat,
            np.full_like(bottom_lon, lat_min),
            left_lat,
            np.asarray([lat_max]),
        ]
    )
    return lon360_to_plot(lon), lat


def lon360_to_plot(lon: np.ndarray) -> np.ndarray:
    """Keep 0-360 longitudes so polygons crossing 180E do not wrap globally."""
    return np.asarray(lon)
