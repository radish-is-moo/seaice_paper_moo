"""Daily Figure 5 preview; redraw existing 2023-2025 ver2 summaries only."""

from pathlib import Path
import hashlib
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .plot_config import FIGURES as OUT, STATES, INITIAL

SOURCES = {
    "target_daily": STATES / "tables/lead_summary.csv",
    "initial_daily": INITIAL / "tables/initial_lead_summary.csv",
    "previous_bands": INITIAL / "figure5_band_crosscheck.csv",
}
PROTECTED = []


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    protected_before = {str(p): sha(p) for p in PROTECTED}
    data = {k: pd.read_csv(p) for k, p in SOURCES.items()}
    target = (
        data["target_daily"]
        .query("domain == 'pan_arctic' and model in ['CNN','U-Net','GNN']")
        .copy()
    )
    initial = data["initial_daily"].query("model == 'Three-model equal mean'").copy()
    assert len(target) == 270 and target.n_cases.eq(282).all()
    assert len(initial) == 360 and initial.n_initializations.eq(282).all()
    assert not target.duplicated(["model", "state", "lead_day"]).any()
    assert not initial.duplicated(["analysis_family", "category", "lead_day"]).any()
    parts = (
        data["initial_daily"]
        .query("model in ['CNN','U-Net','GNN']")
        .groupby(["analysis_family", "category", "lead_day"])
        .rmse_mean.mean()
    )
    np.testing.assert_allclose(
        initial.set_index(
            ["analysis_family", "category", "lead_day"]
        ).rmse_mean.sort_index(),
        parts.sort_index(),
        atol=1e-14,
        rtol=0,
    )
    old = data["previous_bands"]
    assert set(old.domain) == {"pan_arctic"}
    assert set(old.experiment_id) == {"2023_2025_ver2"}
    checks = []
    for _, row in old.iterrows():
        lo, hi = map(int, row.lead_band[1:].split("-"))
        if row.model in ["CNN", "U-Net", "GNN"]:
            z = target[(target.model == row.model) & (target.state == row.category)]
        else:
            z = initial[
                (initial.analysis_family == row.analysis_family)
                & (initial.category == row.category)
            ]
        v = float(z[z.lead_day.between(lo, hi)].rmse_mean.mean())
        checks.append(abs(v - row.rmse_mean))
    assert len(checks) == 63 and max(checks) < 1e-14
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "text.color": "#202020",
            "axes.edgecolor": "#444444",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 6.0), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.105, right=0.978, bottom=0.267, top=0.873, hspace=0.66, wspace=0.18
    )
    fig.text(0.105, 0.983, "Target-date states", fontsize=10.5, weight="bold", va="top")
    fig.text(
        0.105,
        0.545,
        "Initial conditions: three-model mean",
        fontsize=10.5,
        weight="bold",
        va="bottom",
    )
    specs = [
        ("open_water", "Open water", "#009E73", "--", "^"),
        ("miz", "MIZ", "#D55E00", "-", "s"),
        ("compact_ice", "Compact ice", "#0072B2", ":", "o"),
    ]
    rows = []

    def plot(ax, z, **kw):
        z = z.sort_values("lead_day")
        assert z.lead_day.tolist() == list(range(1, 31))
        (line,) = ax.plot(
            z.lead_day,
            z.rmse_mean,
            linewidth=1.6,
            markersize=3.3,
            markevery=[0, 9, 19, 29],
            **kw,
        )
        np.testing.assert_array_equal(line.get_xdata(), z.lead_day)
        np.testing.assert_array_equal(line.get_ydata(), z.rmse_mean)
        return z

    for c, model in enumerate(["CNN", "U-Net", "GNN"]):
        a = axs[0, c]
        for cat, label, color, ls, marker in specs:
            z = plot(
                a,
                target[(target.model == model) & (target.state == cat)],
                label=label,
                color=color,
                linestyle=ls,
                marker=marker,
            )
            rows.extend(
                dict(
                    panel=chr(97 + c),
                    domain="pan_arctic",
                    model=model,
                    classification="target-date SIC",
                    category=cat,
                    lead_day=int(r.lead_day),
                    rmse_mean=float(r.rmse_mean),
                    n_initializations=282,
                )
                for _, r in z.iterrows()
            )
        a.set_title(f"({chr(97+c)}) {model}", loc="left", weight="bold", pad=7)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.545, 0.957),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.35,
        handletextpad=0.45,
    )
    families = [
        (
            "initial_gradient",
            "Initial SIC gradient",
            ["low", "moderate", "high", "extreme"],
            ["Low", "Moderate", "High", "Extreme"],
            ["#76A5AF", "#4C78A8", "#F58518", "#C44E52"],
        ),
        (
            "initial_sit_class",
            "Initial SIT",
            ["open_water_no_ice", "thin_ice", "medium_ice", "thick_ice"],
            ["Open water", "Thin", "Medium", "Thick"],
            ["#009E73", "#D55E00", "#4C78A8", "#6B6B6B"],
        ),
    ]
    for c, (family, title, cats, labels, colors) in enumerate(families):
        a = axs[1, c]
        for j, (cat, label, color) in enumerate(zip(cats, labels, colors)):
            z = plot(
                a,
                initial[
                    (initial.analysis_family == family) & (initial.category == cat)
                ],
                label=label,
                color=color,
                linestyle=["-", "--", "-.", ":"][j],
                marker=["o", "s", "^", "D"][j],
            )
            rows.extend(
                dict(
                    panel=chr(100 + c),
                    domain="pan_arctic",
                    model="Three-model equal mean",
                    classification=family,
                    category=cat,
                    lead_day=int(r.lead_day),
                    rmse_mean=float(r.rmse_mean),
                    n_initializations=282,
                )
                for _, r in z.iterrows()
            )
        a.set_title(
            f"({chr(100+c)}) {title}", loc="left", weight="bold", pad=7, fontsize=9.5
        )
        h, l = a.get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="upper left",
            bbox_to_anchor=(a.get_position().x0, 0.177),
            frameon=False,
            borderaxespad=0,
            handlelength=1.9,
            labelspacing=0.4,
            handletextpad=0.45,
        )
    a = axs[1, 2]
    for cat, color, ls in [
        ("thin_ice__lower_middle_wind", "#D55E00", "-"),
        ("thin_ice__high_upper_tail_wind", "#D55E00", "--"),
        ("medium_thick_ice__lower_middle_wind", "#0072B2", "-"),
        ("medium_thick_ice__high_upper_tail_wind", "#0072B2", "--"),
    ]:
        z = plot(
            a,
            initial[
                (initial.analysis_family == "initial_sit_x_wind")
                & (initial.category == cat)
            ],
            color=color,
            linestyle=ls,
        )
        rows.extend(
            dict(
                panel="f",
                domain="pan_arctic",
                model="Three-model equal mean",
                classification="initial_sit_x_wind",
                category=cat,
                lead_day=int(r.lead_day),
                rmse_mean=float(r.rmse_mean),
                n_initializations=282,
            )
            for _, r in z.iterrows()
        )
    a.set_title(
        "(f) Initial SIT and wind", loc="left", weight="bold", pad=7, fontsize=9.5
    )
    factor_handles = [
        Patch(facecolor="#D55E00", label="Thin"),
        Patch(facecolor="#0072B2", label="Medium/thick"),
        Line2D([0], [0], color="#333333", lw=1.6, label="Lower wind"),
        Line2D([0], [0], color="#333333", lw=1.6, ls="--", label="Higher wind"),
    ]
    fig.legend(
        handles=factor_handles,
        loc="upper left",
        bbox_to_anchor=(a.get_position().x0, 0.177),
        frameon=False,
        borderaxespad=0,
        handlelength=1.9,
        labelspacing=0.4,
        handletextpad=0.45,
    )
    ymax = max(target.rmse_mean.max(), initial.rmse_mean.max())
    assert ymax < 0.30, ymax
    for a in axs.flat:
        a.axvspan(1, 5, color="#E6E9ED", alpha=0.55, zorder=-2, linewidth=0)
        a.axvspan(16, 30, color="#E6E9ED", alpha=0.55, zorder=-2, linewidth=0)
        a.set_xlim(1, 30)
        a.set_ylim(0, 0.30)
        a.set_xticks([1, 5, 10, 15, 20, 25, 30])
        a.set_yticks([0, 0.1, 0.2, 0.3])
        a.tick_params(length=3, pad=3, labelbottom=True)
        a.grid(axis="y", color="#DDE1E5", linewidth=0.55)
        a.set_axisbelow(True)
    for a in axs[:, 0]:
        a.set_ylabel("RMSE (SIC fraction)", labelpad=5)
    for a in axs[1, :]:
        a.set_xlabel("Lead time (days)", labelpad=5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()
    ]
    outside = []
    for t in texts:
        bb = t.get_window_extent(renderer)
        if (
            bb.x0 < -1
            or bb.y0 < -1
            or bb.x1 > fig.bbox.width + 1
            or bb.y1 > fig.bbox.height + 1
        ):
            outside.append(t.get_text())
    assert not outside, outside
    assert min(t.get_fontsize() for t in texts) >= 9.5
    # Tick intervals and adjacent titles must remain distinct at manuscript size.
    for a in axs.flat:
        bbs = [
            t.get_window_extent(renderer)
            for t in a.get_xticklabels()
            if t.get_visible()
        ]
        assert all(bbs[i].x1 < bbs[i + 1].x0 for i in range(len(bbs) - 1))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figure5.{ext}", dpi=450, facecolor="white")
    fig.savefig(OUT / "figure5_manuscript_scale.png", dpi=150, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "figure5_plotted_daily_values.csv", index=False)
    assert len(rows) == 630
    assert protected_before == {str(p): sha(p) for p in PROTECTED}
    report = {
        "status": "PASS",
        "figure": "5",
        "experiment": "2023-2025 ver2",
        "domain": "pan_arctic ocean including open water",
        "operation": "Existing daily summaries redrawn; no new forecast or raw-field metric computation",
        "sources": {k: {"path": str(p), "sha256": sha(p)} for k, p in SOURCES.items()},
        "plotted_values": len(rows),
        "cases_per_category": 282,
        "previous_63_band_values_max_difference": max(checks),
        "initial_equal_model_mean_reproduced": True,
        "daily_ci_available": False,
        "figure_inches": [6.5, 6.0],
        "font_minimum_pt": 9.5,
        "text_outside_canvas": outside,
        "x_tick_overlap": False,
        "lead_ticks": [1, 5, 10, 15, 20, 25, 30],
        "all_panel_y_limits": [0, 0.3],
        "max_displayed_rmse": float(ymax),
        "shade": "Lead days1-5 and16-30, not uncertainty",
        "protected_files_unchanged": protected_before,
        "visual_qa": "Pending independent image inspection",
    }
    (OUT / "plot_states_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "plot_states_caption_ko.txt").write_text(
        "그림 5. 2023–2025년 ver2 실험에서 범북극 해양의 목표일 상태 및 초기조건별 일별 SIC RMSE. (a–c) 목표일 기준 SIC로 구분한 개방수역, MIZ, 고밀도 해빙의 모델별 결과. (d–f) 초기 SIC 구배, SIT, SIT–풍속 범주별로 각각 계산한 세 모델 RMSE의 동일 가중 평균. 초기조건 범주의 격자 구성은 선행시간에 걸쳐 고정한다. (f)의 색은 두께군, 실선과 파선은 각각 낮은 풍속군과 높은 풍속군을 나타낸다. 풍속 구분은 각 초기일의 범북극 해양 면적가중 67백분위수를 기준으로 한다. 회색 배경은 본문의 증가량 비교에 사용하는 초기 1–5일과 후기 16–30일 구간이며 신뢰구간이 아니다. 모든 곡선은 동일한 초기일 사례 282개와 개방수역을 포함한 해양 마스크를 사용한다. 시안에는 일별 신뢰구간을 추가하지 않았다. 세부 범주 정의는 기존 Methods와 동일하다.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(rows),
                "max_rmse": ymax,
                "max_band_difference": max(checks),
                "outputs": str(OUT),
            }
        )
    )


if __name__ == "__main__":
    main()
