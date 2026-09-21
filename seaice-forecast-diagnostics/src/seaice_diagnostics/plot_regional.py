from pathlib import Path
import hashlib
import importlib.util
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from . import plot_style as original
from .plot_style import MODELS, DOMAINS, SEAS, BANDS, LABELS, COLORS, STYLES
from .plot_config import FIGURES, OCEAN

SOURCE = OCEAN / "tables"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    original.style()
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
        }
    )


def finish(fig, number):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        bounds = item.get_window_extent(renderer)
        assert item.get_fontsize() >= 9.5, (
            number,
            item.get_text(),
            item.get_fontsize(),
        )
        assert (
            bounds.x0 >= -1
            and bounds.y0 >= -1
            and bounds.x1 <= fig.bbox.x1 + 1
            and bounds.y1 <= fig.bbox.y1 + 1
        ), (number, item.get_text(), list(bounds.extents), list(fig.bbox.extents))
        texts.append(
            {
                "text": item.get_text(),
                "font_pt": item.get_fontsize(),
                "bounds_px": bounds.extents.tolist(),
            }
        )
    assert fig.get_figwidth() == 6.5
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"figure{number}.{ext}", dpi=400, facecolor="white")
    record = {
        "width_inches": fig.get_figwidth(),
        "height_inches": fig.get_figheight(),
        "minimum_visible_text_pt": min(t["font_pt"] for t in texts),
        "visible_text_bounds_checked": True,
        "texts": texts,
        "output_sha256": {
            ext: sha(FIGURES / f"figure{number}.{ext}") for ext in ["png", "pdf"]
        },
    }
    plt.close(fig)
    return record


def assert_interval_polygon(collection, x, lower, upper):
    vertices = collection.get_paths()[0].vertices
    # Both exact source bounds at every lead must occur in the rendered polygon.
    for xx, lo, hi in zip(x, lower, upper):
        assert np.any(np.all(vertices == [xx, lo], axis=1))
        assert np.any(np.all(vertices == [xx, hi], axis=1))


def figure4():
    style()
    source = SOURCE / "figure4_lead_time_metrics_area_weighted_bootstrap5000.csv.gz"
    data = pd.read_csv(source)
    selected = data[data.model.isin(MODELS)].copy()
    assert len(selected) == 1260
    assert not selected.duplicated(["model", "domain", "metric", "lead_day"]).any()
    assert selected.n_initializations.eq(282).all()
    assert selected.bootstrap_reps.eq(5000).all() and selected.block_days.eq(7).all()
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 5.1), sharex=True)
    fig.subplots_adjust(
        left=0.113, right=0.983, bottom=0.105, top=0.813, hspace=0.32, wspace=0.15
    )
    rmse_limits = original.shared_limits(
        data[data.metric.eq("rmse")][["ci_lower", "ci_upper"]], 0
    )
    r_limits = original.shared_limits(
        data[data.metric.eq("r")][["ci_lower", "ci_upper"]]
    )
    r_limits = (max(-1, r_limits[0]), min(1, r_limits[1]))
    plotted, bounds = [], []
    for col, model in enumerate(MODELS):
        for row, metric in enumerate(["rmse", "r"]):
            ax = axes[row, col]
            for domain in DOMAINS:
                frame = selected[
                    (selected.model == model)
                    & (selected.metric == metric)
                    & (selected.domain == domain)
                ].sort_values("lead_day")
                assert frame.lead_day.tolist() == list(range(1, 31))
                (line,) = ax.plot(
                    frame.lead_day,
                    frame["mean"],
                    label=LABELS[domain],
                    color=COLORS[domain],
                    linestyle=STYLES[domain],
                    linewidth=1.65 if domain in DOMAINS[:2] else 1.15,
                    alpha=1,
                    zorder=4 if domain in DOMAINS[:2] else 3,
                )
                np.testing.assert_array_equal(
                    line.get_xdata(), frame.lead_day.to_numpy()
                )
                np.testing.assert_array_equal(
                    line.get_ydata(), frame["mean"].to_numpy()
                )
                plotted.extend(frame.to_dict("records"))
                if domain in DOMAINS[:2]:
                    band = ax.fill_between(
                        frame.lead_day,
                        frame.ci_lower,
                        frame.ci_upper,
                        color=COLORS[domain],
                        alpha=0.13,
                        linewidth=0,
                        zorder=1,
                    )
                    assert_interval_polygon(
                        band, frame.lead_day, frame.ci_lower, frame.ci_upper
                    )
                    bounds.extend(frame.to_dict("records"))
            ax.set_xlim(1, 30)
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
            ax.set_ylim(*(rmse_limits if metric == "rmse" else r_limits))
            ax.set_yticks(
                [0, 0.1, 0.2, 0.3] if metric == "rmse" else [0.4, 0.6, 0.8, 1]
            )
            ax.grid(color="#e3e3e3", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", pad=3, length=3)
            ax.tick_params(axis="y", pad=3, length=3)
            ax.set_title(
                (
                    f"({chr(97 + 3*row + col)}) {model}"
                    if row == 0
                    else f"({chr(100 + col)})"
                ),
                color="#111111",
                fontweight="bold",
                loc="left",
                pad=5,
            )
            if row == 1:
                ax.set_xlabel("Lead time (days)", labelpad=5)
            if col == 0:
                ax.set_ylabel(
                    (
                        "RMSE (SIC fraction)"
                        if metric == "rmse"
                        else "Spatial pattern correlation"
                    ),
                    labelpad=5,
                )
            else:
                ax.tick_params(labelleft=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[:4],
        labels[:4],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.987),
        frameon=False,
        handlelength=2.1,
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0,
    )
    fig.legend(
        handles[4:],
        labels[4:],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.934),
        frameon=False,
        handlelength=2.1,
        columnspacing=1.4,
        handletextpad=0.5,
        borderaxespad=0,
    )
    preserved = pd.DataFrame(plotted)
    previous = pd.read_csv(OCEAN / "figure4_plotted_values.csv")
    keys = ["model", "domain", "metric", "lead_day"]
    pd.testing.assert_frame_equal(
        preserved.sort_values(keys).reset_index(drop=True),
        previous.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    preserved.to_csv(FIGURES / "figure4_portrait_plotted_values.csv", index=False)
    record = finish(fig, 4)
    record.update(
        {
            "source": str(source),
            "source_sha256": sha(source),
            "plotted_mean_values": len(plotted),
            "source_line_x_y_exactly_preserved": True,
            "previous_plotted_csv_exactly_preserved": True,
            "confidence_interval_cases": len(bounds),
            "rendered_interval_vertices_exactly_match_source": True,
            "interval_scope": "Unchanged pointwise 95% bounds for pan-Arctic and combined NSR only",
            "model_order": MODELS,
            "domain_order": DOMAINS,
            "rmse_limits": rmse_limits,
            "r_limits": r_limits,
            "layout": "2 rows by 3 columns; original panel and model order preserved",
        }
    )
    return record
