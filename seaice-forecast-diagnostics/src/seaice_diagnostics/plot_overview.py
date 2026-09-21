from pathlib import Path
import hashlib
import importlib.util
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from .plot_config import FIGURES as FIG

MANIFEST = []
MODELS = ["CNN", "U-Net", "GNN"]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#151515",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "savefig.bbox": None,
        }
    )


def save(fig, key, source=None, numerical=None):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()
    ]
    text_outside = []
    fw, fh = fig.bbox.width, fig.bbox.height
    for t in texts:
        b = t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fw + 1 or b.y1 > fh + 1:
            text_outside.append(t.get_text())
    assert not text_outside, (key, text_outside)
    assert min(t.get_fontsize() for t in texts) >= 9
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figure{key}.{ext}", dpi=350, facecolor="white")
    MANIFEST.append(
        {
            "figure": key,
            "canvas_inches": fig.get_size_inches().tolist(),
            "minimum_font_pt": min(t.get_fontsize() for t in texts),
            "source": str(source) if source else "Verified source workflow diagram",
            "source_sha256": (
                hashlib.sha256(source.read_bytes()).hexdigest() if source else None
            ),
            "text_outside_canvas": text_outside,
            "numerical_validation": numerical,
        }
    )
    plt.close(fig)


def clean_axes(a):
    a.grid(axis="y", color="#DEE2E7", lw=0.6)
    a.set_axisbelow(True)
    a.tick_params(length=3, width=0.75, pad=2.5)


def figure1():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from . import geography as m

    expected = [
        ("barents", 68.0, 82.5, 20.0, 60.0),
        ("kara", 68.0, 82.5, 60.0, 100.0),
        ("laptev", 70.0, 82.5, 100.0, 140.0),
        ("east_siberian", 68.0, 82.5, 140.0, 180.0),
        ("chukchi", 66.0, 78.0, 180.0, 205.0),
    ]
    assert m.NSR_REGION_BOXES == expected
    style()
    fig = plt.figure(figsize=(4.7, 4.9))
    ax = fig.add_axes(
        [0.015, 0.065, 0.97, 0.915],
        projection=ccrs.NorthPolarStereo(central_longitude=95),
    )
    ax.set_extent([-180, 180, 54, 90], crs=ccrs.PlateCarree())
    ax.set_facecolor("#EAF3FA")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#EAF3FA", zorder=0)
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="#E5DED3",
        edgecolor="#7A8491",
        linewidth=0.4,
        zorder=2,
    )
    ax.coastlines(resolution="50m", linewidth=0.45, color="#64748B", zorder=5)
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.45,
        color="#8AA0B5",
        alpha=0.4,
        linestyle=":",
    )
    polygons = []
    for name, lat0, lat1, lon0, lon1 in expected:
        lon, lat = m.sector_polygon(lat0, lat1, lon0, lon1)
        ax.fill(
            lon,
            lat,
            transform=ccrs.PlateCarree(),
            facecolor=m.REGION_COLORS[name],
            alpha=0.55,
            edgecolor="#26313B",
            linewidth=1.0,
            zorder=4,
        )
        x, y = m.LABEL_POSITIONS[name]
        label = m.REGION_LABELS[name].replace("East Siberian", "East\nSiberian")
        ax.text(
            x,
            y,
            label,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=9.5,
            weight="bold",
            bbox=dict(
                boxstyle="square,pad=.13",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
            zorder=7,
        )
        polygons.append(
            {
                "name": name,
                "coordinate_count": len(lon),
                "lon_sha256": hashlib.sha256(lon.tobytes()).hexdigest(),
                "lat_sha256": hashlib.sha256(lat.tobytes()).hexdigest(),
                "label_location": [x, y],
            }
        )
    lon = np.linspace(-180, 180, 721)
    ax.plot(
        lon,
        np.full_like(lon, 65),
        transform=ccrs.PlateCarree(),
        color="#334155",
        lw=1.2,
        ls=(0, (5, 3)),
        zorder=6,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#334155",
                ls=(0, (5, 3)),
                lw=1.3,
                label="Pan-Arctic domain boundary (65°N)",
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        fontsize=9.5,
        handlelength=2.8,
    )
    save(
        fig,
        "1",
        Path(m.__file__),
        {
            "region_boxes": expected,
            "polygons": polygons,
            "projection_central_longitude": 95,
            "map_extent": [-180, 180, 54, 90],
            "pan_arctic_boundary_latitude": 65,
        },
    )


def figure2():
    style()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.015, top=0.985)

    def box(x, y, w, h, title, detail="", fill="#F0F3F5"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.008,rounding_size=.014",
                facecolor=fill,
                edgecolor="#78858F",
                lw=0.85,
            )
        )
        if detail:
            ax.text(
                x + w / 2,
                y + h * 0.75,
                title,
                ha="center",
                va="center",
                weight="bold",
                fontsize=10.5,
            )
            ax.text(
                x + w / 2,
                y + h * 0.32,
                detail,
                ha="center",
                va="center",
                fontsize=9.5,
                linespacing=1.2,
            )
        else:
            ax.text(
                x + w / 2,
                y + h / 2,
                title,
                ha="center",
                va="center",
                weight="bold",
                fontsize=10.5,
            )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.1,
                color="#57636D",
            )
        )

    box(
        0.035,
        0.812,
        0.93,
        0.166,
        "Daily inputs",
        "PIOMAS SIC and SIT; six ERA5 variables\n15 days × 8 variables = 120 input features",
    )
    for c in (0.185, 0.5, 0.815):
        arrow(0.5, 0.8, c, 0.739)
    for c, name in zip((0.185, 0.5, 0.815), MODELS):
        box(c - 0.135, 0.635, 0.27, 0.094, name, fill="#EDF3F6")
        arrow(c, 0.623, c, 0.561)
    ax.text(
        0.5,
        0.531,
        "Each model predicts 30 days of SIC change",
        ha="center",
        va="center",
        fontsize=9.8,
    )
    arrow(0.5, 0.506, 0.5, 0.453)
    box(
        0.035,
        0.316,
        0.93,
        0.125,
        "SIC forecasts at leads 1–30",
        "Initial SIC + predicted change, clipped to [0, 1]",
    )
    arrow(0.5, 0.304, 0.5, 0.239)
    box(
        0.035,
        0.04,
        0.93,
        0.185,
        "Ocean verification for each model",
        "Static ocean mask and area-weighted metrics\nPan-Arctic ocean → combined NSR → five seas",
        fill="#F4F3ED",
    )
    save(
        fig,
        "2",
        numerical={
            "models": MODELS,
            "input_days": 15,
            "input_variables": 8,
            "input_features": 120,
            "lead_days": [1, 30],
            "forecast_clip": [0, 1],
            "same_text_and_workflow_as_source": True,
        },
    )
