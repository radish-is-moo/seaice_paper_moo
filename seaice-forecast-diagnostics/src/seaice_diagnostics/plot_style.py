import numpy as np
import matplotlib.pyplot as plt

MODELS = ["CNN", "U-Net", "GNN"]

DOMAINS = [
    "pan_arctic",
    "combined_nsr",
    "barents",
    "kara",
    "laptev",
    "east_siberian",
    "chukchi",
]

SEAS = DOMAINS[2:]

BANDS = ["L1-5", "L6-15", "L16-30"]

LABELS = dict(
    zip(
        DOMAINS,
        [
            "Pan-Arctic",
            "Combined NSR",
            "Barents",
            "Kara",
            "Laptev",
            "East Siberian",
            "Chukchi",
        ],
    )
)

COLORS = dict(
    zip(
        DOMAINS,
        ["#202020", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
    )
)

STYLES = dict(zip(DOMAINS, ["-", "-", "--", "-.", ":", "--", "-."]))


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10.5,
            "axes.edgecolor": "#3d3d3d",
            "axes.linewidth": 0.8,
            "text.color": "#111111",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def shared_limits(values, lower_bound=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low, high = float(values.min()), float(values.max())
    span = high - low if high > low else max(abs(high), 1.0)
    return (
        low - span * 0.05 if lower_bound is None else float(lower_bound),
        high + span * 0.05,
    )
