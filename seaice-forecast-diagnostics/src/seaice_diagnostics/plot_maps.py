from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.text import Text
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from PIL import Image

from .plot_config import FIGURES as OUT, OCEAN as OLD, INPUT
from .config import MASK
from .diagnostics import build_domain_masks

DPI = 450
CHECKS = []
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source(path):
    path = Path(path)
    INPUTS[str(path)] = sha(path)
    return path


def check(name, condition, **details):
    item = {"name": name, "passed": bool(condition), **details}
    CHECKS.append(item)
    assert condition, item


def digest(array):
    # Canonical float64/NaN representation makes the audit independent of storage dtype.
    a = np.ascontiguousarray(array, dtype=np.float64)
    a[np.isnan(a)] = np.nan
    return hashlib.sha256(a.tobytes()).hexdigest()


def plot_arrays(lat, lon, field):
    lon_pm = ((np.asarray(lon, dtype=float) + 180) % 360) - 180
    order = np.argsort(lon_pm)
    values = np.asarray(field, dtype=float)[:, order]
    values, lon_plot = add_cyclic_point(values, coord=lon_pm[order], axis=1)
    lon2d, lat2d = np.meshgrid(lon_plot, lat)
    result = lon2d, lat2d, np.ma.masked_invalid(values)
    return result


def polar_axis(axis, nsr=False, persistence=False):
    axis.set_extent(
        [10, 210, 64, 86] if nsr else [-180, 180, 50, 90], ccrs.PlateCarree()
    )
    axis.add_feature(cfeature.LAND, facecolor="#DDDCD7", edgecolor="none", zorder=4)
    axis.coastlines(
        linewidth=0.4 if persistence else 0.30,
        color="#555555" if persistence else "#888888",
        zorder=5,
    )
    # Unlabelled sparse guide lines preserve geographic context without tiny text.
    axis.gridlines(
        xlocs=np.arange(-180, 181, 60),
        ylocs=[60, 70, 80],
        linewidth=0.3,
        color="#777777" if persistence else "#999999",
        alpha=0.30 if persistence else 0.20,
        linestyle="--",
        draw_labels=False,
    )


def mask_union(lat, lon, ocean):
    lat2d, lon2d = np.meshgrid(lat, np.mod(lon, 360), indexing="ij")
    combined = np.zeros(ocean.shape, dtype=bool)
    for south, north, west, east in [
        (66, 78, 180, 205),
        (68, 82.5, 140, 180),
        (70, 82.5, 100, 140),
        (68, 82.5, 60, 100),
        (68, 82.5, 20, 60),
    ]:
        combined |= (
            (lat2d >= south) & (lat2d <= north) & (lon2d >= west) & (lon2d <= east)
        )
    combined &= ocean
    expected = build_domain_masks(lat, lon, ocean)["combined_nsr"]
    check(
        "combined_nsr_matches_original",
        np.array_equal(combined, expected),
        cells=int(combined.sum()),
    )
    return combined


def save(fig, name):
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    minimum = min(t.get_fontsize() for t in texts)
    check(
        name + "_minimum_font",
        minimum >= (9 if name == "figure3" else 9.5),
        points=minimum,
    )
    files = {}
    for ext in ("png", "pdf"):
        target = OUT / f"{name}.{ext}"
        fig.savefig(
            target, dpi=DPI
        )  # No tight bbox: exact physical dimensions are intentional.
        files[ext] = str(target)
    with Image.open(files["png"]) as im:
        pixel_size = list(im.size)
    expected = [round(v * DPI) for v in fig.get_size_inches()]
    check(
        name + "_png_dimensions",
        pixel_size == expected and pixel_size[0] == round(6.5 * DPI),
        inches=list(fig.get_size_inches()),
        pixels=pixel_size,
    )
    plt.close(fig)
    return {
        "files": files,
        "inches": list(np.array(expected) / DPI),
        "pixels": pixel_size,
        "minimum_font_points": minimum,
    }


def figure3():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.5,
            "savefig.facecolor": "white",
        }
    )
    ranking = pd.read_csv(source(OLD / "tables/figure3_candidate_ranking.csv"))
    date = str(ranking.sort_values("rank").iloc[0]["init_date"])[:10]
    year = date[:4]
    with np.load(source(INPUT / "figure3_case.npz"), allow_pickle=False) as z:
        raw_reference = z["reference"]
        target_date = str(z["target_date"].item())
    with np.load(source(MASK), allow_pickle=False) as z:
        ocean, lat, lon = z["ocean_mask"].astype(bool), z["latitude"], z["longitude"]
    reference = np.where(ocean, raw_reference, np.nan)
    ref_coords = plot_arrays(lat, lon, reference)
    fig = plt.figure(figsize=(6.5, 5.4), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        left=0.066,
        right=0.992,
        top=0.918,
        bottom=0.170,
        width_ratios=[1, 1, 1, 1.62],
        hspace=0.105,
        wspace=0.09,
    )
    fields_report = {"reference": digest(reference)}
    titles = ["Reference SIC", "Predicted SIC", "SIC error", "NSR zoom"]
    models = [
        ("CNN", "cnn", "#0072B2"),
        ("U-Net", "unet", "#D55E00"),
        ("GNN", "gnn", "#009E73"),
    ]
    sic_mesh = error_mesh = None
    for row, (model, key, color) in enumerate(models):
        with np.load(source(INPUT / "figure3_case.npz"), allow_pickle=False) as z:
            raw_prediction = np.asarray(z[key], dtype=np.float32)
        prediction = np.where(ocean, raw_prediction, np.nan)
        error = (
            prediction - reference
        )  # Preserve the original float32 subtraction exactly.
        check(
            f"figure3_{key}_prediction_same",
            np.array_equal(
                prediction, np.where(ocean, raw_prediction, np.nan), equal_nan=True
            ),
        )
        check(
            f"figure3_{key}_error_same",
            np.array_equal(
                error,
                np.where(ocean, raw_prediction, np.nan)
                - np.where(ocean, raw_reference, np.nan),
                equal_nan=True,
            ),
            dtype=str(error.dtype),
        )
        fields_report[model] = {
            "prediction": digest(prediction),
            "error": digest(error),
            "error_finite_cells": int(np.isfinite(error).sum()),
        }
        pred_coords = plot_arrays(lat, lon, prediction)
        for col, field in enumerate([reference, prediction, error, error]):
            axis = fig.add_subplot(
                grid[row, col],
                projection=ccrs.NorthPolarStereo(
                    central_longitude=110 if col == 3 else 0
                ),
            )
            polar_axis(axis, nsr=col == 3)
            coords = plot_arrays(lat, lon, field)
            kwargs = (
                {"cmap": "Blues", "vmin": 0, "vmax": 1}
                if col < 2
                else {
                    "cmap": "RdBu_r",
                    "norm": TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5),
                }
            )
            mesh = axis.pcolormesh(
                *coords,
                transform=ccrs.PlateCarree(),
                shading="auto",
                rasterized=True,
                **kwargs,
            )
            if col < 2:
                sic_mesh = mesh
            else:
                error_mesh = mesh
            if col == 3:
                lw = 0.90 if col < 3 else 1.05
                # Both halos underneath both colored contours, so one halo cannot erase the other line.
                for coords, style in [(ref_coords, "-"), (pred_coords, "--")]:
                    axis.contour(
                        *coords,
                        levels=[0.15],
                        colors="white",
                        linewidths=lw + 0.65,
                        linestyles=style,
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                    )
                axis.contour(
                    *ref_coords,
                    levels=[0.15],
                    colors="black",
                    linewidths=lw,
                    transform=ccrs.PlateCarree(),
                    zorder=7,
                )
                axis.contour(
                    *pred_coords,
                    levels=[0.15],
                    colors="#C000C0",
                    linewidths=lw,
                    linestyles="--",
                    transform=ccrs.PlateCarree(),
                    zorder=8,
                )
            axis.text(
                0.025,
                0.975,
                f"({chr(97 + row * 4 + col)})",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                zorder=10,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": 0.3,
                    "alpha": 0.85,
                },
            )
            if row == 0:
                axis.set_title(titles[col], fontsize=9.5, fontweight="bold", pad=5)
            if col == 0:
                axis.text(
                    -0.07,
                    0.5,
                    model,
                    transform=axis.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontweight="bold",
                    fontsize=10,
                    color=color,
                )
    sic_cb = fig.colorbar(
        sic_mesh,
        cax=fig.add_axes([0.105, 0.105, 0.32, 0.018]),
        orientation="horizontal",
        ticks=[0, 0.25, 0.5, 0.75, 1],
    )
    sic_cb.set_label("SIC", labelpad=2)
    err_cb = fig.colorbar(
        error_mesh,
        cax=fig.add_axes([0.54, 0.105, 0.40, 0.018]),
        orientation="horizontal",
        ticks=[-0.5, -0.25, 0, 0.25, 0.5],
    )
    err_cb.set_label("SIC error (forecast − reference)", labelpad=2)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], color="black", lw=0.95, label="Reference ice edge"),
            Line2D(
                [0], [0], color="#C000C0", lw=0.95, ls="--", label="Predicted ice edge"
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.54, 0.146),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=2.8,
        columnspacing=2.2,
        borderaxespad=0,
    )
    result = save(fig, "figure3")
    return {
        **result,
        "init_date": date,
        "target_date": target_date,
        "lead_day": 30,
        "eligible_cases": len(ranking),
        "row_order": [m[0] for m in models],
        "field_sha256": fields_report,
        "all_northern_extent": [-180, 180, 50, 90],
        "nsr_zoom_extent": [10, 210, 64, 86],
        "nsr_zoom_semantics": "Regional enlargement of the full ocean-masked error field, as in the source figure; no combined-NSR mask applied.",
    }


def figure6():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.5,
            "savefig.facecolor": "white",
        }
    )
    cached = source(
        OLD / "diagnostics/figure6_reference_anomaly_persistence_jul_oct_2023_2025.npz"
    )
    with np.load(cached, allow_pickle=False) as z:
        lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
        ocean, valid = z["ocean_mask"].astype(bool), z["valid_display_mask"].astype(
            bool
        )
        persistence, miz = np.asarray(z["persistence_days"], dtype=float), np.asarray(
            z["miz_frequency"], dtype=float
        )
    combined = mask_union(lat, lon, ocean)
    a = np.where(combined & valid, persistence, np.nan)
    b = np.where(combined, miz, np.nan)
    check(
        "figure6_persistence_exact_cached_values",
        np.array_equal(
            a[combined & valid], persistence[combined & valid], equal_nan=True
        ),
    )
    check(
        "figure6_miz_exact_cached_values",
        np.array_equal(b[combined], miz[combined], equal_nan=True),
    )
    check(
        "figure6_persistence_mask_unchanged",
        np.array_equal(np.isfinite(a), combined & valid & np.isfinite(persistence)),
    )
    vmax = max(0.6, float(np.nanmax(miz[combined])))
    fig = plt.figure(figsize=(6.5, 3.4), facecolor="white")
    meshes = []
    for i, field, cmap, upper, title in [
        (0, a, "viridis_r", 31, "(a) SIC anomaly persistence"),
        (1, b, "Blues", vmax, "(b) MIZ frequency"),
    ]:
        axis = fig.add_axes(
            [0.019 + i * 0.50, 0.300, 0.462, 0.605],
            projection=ccrs.NorthPolarStereo(central_longitude=110),
        )
        polar_axis(axis, nsr=True, persistence=True)
        mesh = axis.pcolormesh(
            *plot_arrays(lat, lon, field),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=0,
            vmax=upper,
            shading="auto",
            rasterized=True,
        )
        meshes.append(mesh)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=6)
    cb0 = fig.colorbar(
        meshes[0],
        cax=fig.add_axes([0.052, 0.218, 0.396, 0.028]),
        orientation="horizontal",
        ticks=[0, 7, 14, 21, 31],
    )
    cb0.ax.set_xticklabels(["0", "7", "14", "21", ">30"])
    cb0.set_label("Persistence (days)", labelpad=3)
    cb1 = fig.colorbar(
        meshes[1],
        cax=fig.add_axes([0.552, 0.218, 0.396, 0.028]),
        orientation="horizontal",
        ticks=[0, 0.2, 0.4, 0.6, 0.8],
    )
    cb1.set_label("Fraction of reference days", labelpad=3)
    check(
        "figure6_terminal_label",
        cb0.ax.get_xticklabels()[-1].get_text() == ">30" and cb0.get_ticks()[-1] == 31,
        final_tick_position=31,
        final_tick_label=">30",
    )
    result = save(fig, "figure6")
    return {
        **result,
        "combined_nsr_cells": int(combined.sum()),
        "panel_a_finite_cells": int(np.isfinite(a).sum()),
        "panel_b_finite_cells": int(np.isfinite(b).sum()),
        "persistence_gt30_cells": int(np.sum(a == 31)),
        "persistence_scale": [0, 31],
        "miz_scale": [0, vmax],
        "panel_a_sha256": digest(a),
        "panel_b_sha256": digest(b),
        "nsr_extent": [10, 210, 64, 86],
        "numeric_data_31_retained": True,
    }
