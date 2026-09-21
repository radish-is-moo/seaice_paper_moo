"""Preview only: regional transition exposure and absolute state errors.

No forecast arrays or statistics are recomputed. The three verified summary
CSVs are the sole numerical inputs. The current manuscript is not modified.
"""

from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


from .plot_config import FIGURES as OUT, OCEAN

SOURCE = OCEAN / "tables"
SEAS = ["barents", "kara", "laptev", "east_siberian", "chukchi"]
SEA_LABELS = ["Barents", "Kara", "Laptev", "East Siberian", "Chukchi"]
BANDS = ["L1-5", "L6-15", "L16-30"]
BAND_LABELS = ["1–5", "6–15", "16–30"]
PATHS = [
    "open_water_to_miz",
    "open_water_to_compact_ice",
    "miz_to_open_water",
    "miz_to_compact_ice",
    "compact_ice_to_open_water",
    "compact_ice_to_miz",
]
PATH_LABELS = ["OW → MIZ", "OW → CI", "MIZ → OW", "MIZ → CI", "CI → OW", "CI → MIZ"]
PATH_COLORS = ["#70B7C8", "#2E6F9E", "#D4941E", "#B9603A", "#43566F", "#8F6CAB"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(frame, value):
    assert not frame.duplicated(["domain", "lead_band"]).any()
    result = (
        frame.set_index(["domain", "lead_band"])[value]
        .reindex(pd.MultiIndex.from_product([SEAS, BANDS]))
        .to_numpy()
        .reshape(5, 3)
    )
    assert np.isfinite(result).all()
    return result


def annotation_color(rgba):
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
    return (
        "white"
        if (1.05 / (luminance + 0.05)) > ((luminance + 0.05) / 0.05)
        else "#111111"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "exposure": SOURCE / "figure7_transition_summary.csv",
        "errors": SOURCE / "figure7_miz_non_miz_summary.csv",
        "pathways": SOURCE / "figure7_transition_type_summary.csv",
    }
    protected = []
    before = {str(p): sha(p) for p in protected}
    transitions, errors, pathways = [
        pd.read_csv(sources[k]) for k in ["exposure", "errors", "pathways"]
    ]
    assert len(transitions) == 15 and len(errors) == 15 and len(pathways) == 90
    assert errors.n_initializations.eq(282).all()
    assert errors.summary_support.str.contains("Same model-initialization-lead").all()
    np.testing.assert_allclose(
        transitions.changed_area_km2 / transitions.valid_area_km2,
        transitions.transition_fraction,
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        pathways.transition_area_km2 / pathways.valid_area_km2,
        pathways.area_fraction_of_valid_grid_days,
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        errors.miz_rmse - errors.non_miz_rmse,
        errors.miz_minus_non_miz_rmse,
        rtol=0,
        atol=1e-14,
    )
    exposure = matrix(transitions, "transition_fraction") * 100
    miz = matrix(errors, "miz_rmse")
    non_miz = matrix(errors, "non_miz_rmse")
    assert max(miz.max(), non_miz.max()) < 0.25
    assert exposure.max() < 35

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
            "text.color": "#111111",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(6.5, 6.2), dpi=150)
    top_y, top_h, panel_w = 0.602, 0.295, 0.235
    heat_axes = [
        fig.add_axes([x, top_y, panel_w, top_h]) for x in [0.184, 0.452, 0.720]
    ]
    exposure_cax = fig.add_axes([0.184, 0.499, 0.235, 0.017])
    rmse_cax = fig.add_axes([0.452, 0.499, 0.503, 0.017])
    bottom_ax = fig.add_axes([0.184, 0.176, 0.771, 0.244])
    exposure_norm, rmse_norm = Normalize(0, 35), Normalize(0, 0.25)
    exposure_cmap, rmse_cmap = plt.get_cmap("Blues"), plt.get_cmap("YlOrRd")
    plotted = []
    images = []
    configs = [
        (
            "a",
            exposure,
            "transition_fraction",
            "exposure",
            "(a) Transition\nexposure (%)",
            ".1f",
            exposure_cmap,
            exposure_norm,
            100,
            "%",
        ),
        (
            "b",
            miz,
            "miz_rmse",
            "errors",
            "(b) MIZ\nRMSE",
            ".3f",
            rmse_cmap,
            rmse_norm,
            1,
            "SIC fraction",
        ),
        (
            "c",
            non_miz,
            "non_miz_rmse",
            "errors",
            "(c) Non-MIZ\nRMSE",
            ".3f",
            rmse_cmap,
            rmse_norm,
            1,
            "SIC fraction",
        ),
    ]
    for ax, (
        panel,
        values,
        metric,
        source_key,
        title,
        fmt,
        cmap,
        norm,
        multiplier,
        unit,
    ) in zip(heat_axes, configs):
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        np.testing.assert_array_equal(im.get_array().data, values)
        images.append(im)
        ax.set_title(title, loc="left", fontweight="bold", pad=8, linespacing=1.05)
        ax.set_xticks(range(3), BAND_LABELS)
        ax.set_yticks(range(5), SEA_LABELS)
        ax.tick_params(length=0, pad=5, labelleft=(panel == "a"))
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row, sea in enumerate(SEAS):
            for col, band in enumerate(BANDS):
                val = float(values[row, col])
                ax.text(
                    col,
                    row,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=annotation_color(cmap(norm(val))),
                )
                source_frame = transitions if source_key == "exposure" else errors
                source_value = float(
                    source_frame.loc[
                        (source_frame.domain == sea) & (source_frame.lead_band == band),
                        metric,
                    ].iloc[0]
                )
                assert val == source_value * multiplier
                plotted.append(
                    {
                        "panel": panel,
                        "domain": sea,
                        "lead_band": band,
                        "metric": metric,
                        "transition_type": "",
                        "source_value": source_value,
                        "display_value": val,
                        "display_unit": unit,
                        "source_file": sources[source_key].name,
                    }
                )
    fig.text(0.570, 0.543, "Lead time (days)", ha="center", va="center", fontsize=9.5)
    fig.colorbar(
        images[0], cax=exposure_cax, orientation="horizontal", ticks=[0, 10, 20, 30]
    )
    fig.colorbar(
        images[1],
        cax=rmse_cax,
        orientation="horizontal",
        ticks=[0, 0.05, 0.10, 0.15, 0.20, 0.25],
        format="%.2f",
    )
    for ax in [exposure_cax, rmse_cax]:
        ax.tick_params(length=2.5, pad=3)
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
    assert images[1].norm is images[2].norm

    late = pathways[pathways.lead_band.eq("L16-30")].set_index(
        ["domain", "transition_type"]
    )
    left = np.zeros(len(SEAS))
    handles = []
    for path, label, color in zip(PATHS, PATH_LABELS, PATH_COLORS):
        values = np.array(
            [
                late.loc[(sea, path), "area_fraction_of_valid_grid_days"] * 100
                for sea in SEAS
            ]
        )
        bars = bottom_ax.barh(
            range(5),
            values,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            height=0.70,
        )
        np.testing.assert_allclose(
            [b.get_width() for b in bars], values, rtol=0, atol=1e-14
        )
        np.testing.assert_allclose([b.get_x() for b in bars], left, rtol=0, atol=1e-14)
        for sea, value in zip(SEAS, values):
            plotted.append(
                {
                    "panel": "d",
                    "domain": sea,
                    "lead_band": "L16-30",
                    "metric": "area_fraction_of_valid_grid_days",
                    "transition_type": path,
                    "source_value": float(
                        late.loc[(sea, path), "area_fraction_of_valid_grid_days"]
                    ),
                    "display_value": float(value),
                    "display_unit": "%",
                    "source_file": sources["pathways"].name,
                }
            )
        left += values
        handles.append(Patch(facecolor=color, label=label))
    np.testing.assert_allclose(left, exposure[:, 2], rtol=0, atol=1e-12)
    bottom_ax.set_yticks(range(5), SEA_LABELS)
    bottom_ax.invert_yaxis()
    bottom_ax.set_xlim(0, 35)
    bottom_ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    bottom_ax.set_xlabel("Area-weighted exposure (%)", labelpad=5)
    bottom_ax.set_title(
        "(d) Transition pathways: lead days 16–30", loc="left", fontweight="bold", pad=8
    )
    bottom_ax.tick_params(axis="y", length=0, pad=5)
    bottom_ax.grid(axis="x", color="#dedede", linewidth=0.6)
    bottom_ax.set_axisbelow(True)
    bottom_ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles[:3],
        PATH_LABELS[:3],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.078),
        frameon=False,
        borderaxespad=0,
        columnspacing=1.5,
        handlelength=1.5,
        handletextpad=0.5,
    )
    fig.legend(
        handles[3:],
        PATH_LABELS[3:],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.041),
        frameon=False,
        borderaxespad=0,
        columnspacing=1.5,
        handlelength=1.5,
        handletextpad=0.5,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_records = []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        bounds = item.get_window_extent(renderer)
        assert item.get_fontsize() >= 9.5, (item.get_text(), item.get_fontsize())
        assert (
            bounds.x0 >= -1
            and bounds.y0 >= -1
            and bounds.x1 <= fig.bbox.x1 + 1
            and bounds.y1 <= fig.bbox.y1 + 1
        ), (item.get_text(), bounds.extents)
        text_records.append(
            {
                "text": item.get_text(),
                "font_pt": item.get_fontsize(),
                "bounds_px": bounds.extents.tolist(),
            }
        )
    # Each annotation stays inside its own heatmap cell; shared axes have identical row alignment.
    for ax in heat_axes:
        for txt in ax.texts:
            bb = txt.get_window_extent(renderer)
            x, y = txt.get_position()
            corners = ax.transData.transform([[x - 0.5, y - 0.5], [x + 0.5, y + 0.5]])
            assert bb.x0 >= corners[:, 0].min() and bb.x1 <= corners[:, 0].max()
            assert bb.y0 >= corners[:, 1].min() and bb.y1 <= corners[:, 1].max()
    assert (
        len({(ax.get_position().y0, ax.get_position().height) for ax in heat_axes}) == 1
    )
    plotted_frame = pd.DataFrame(plotted)
    assert len(plotted_frame) == 75
    plotted_frame.to_csv(
        OUT / "figure7_plotted_values.csv", index=False, float_format="%.17g"
    )
    for filename, dpi in [
        ("figure7.png", 450),
        ("figure7.pdf", 450),
        ("figure7_manuscript_scale.png", 150),
    ]:
        fig.savefig(OUT / filename, dpi=dpi, facecolor="white")
    plt.close(fig)
    caption = """Figure 7 preview — proposed caption

Reference-state transitions and absolute SIC forecast errors in five seas near the Northern Sea Route, July–October 2023–2025. (a) Area-weighted fraction of valid ocean grid–date pairs whose reference state differs between initialization and verification, summarized over three lead-time windows. (b, c) Target-date MIZ and non-MIZ RMSE on a common color scale. MIZ and non-MIZ errors are summarized over the same model–initialization–lead combinations with both categories present. Non-MIZ combines open-water and compact-ice squared errors and valid areas before taking the square root; it is not the mean of their separate RMSE values. Each model is evaluated independently, and model-wise RMSE values receive equal weight. (d) Six directional reference-state changes during lead days 16–30. Bar segments are shares of all valid ocean grid–date area, rather than shares of transitioned area; their sum equals the last column of (a). The reference comparison records endpoint state differences, without identifying intermediate transitions or their physical causes. All panels include open water and exclude land. OW denotes open water, MIZ the marginal ice zone, and CI compact ice. RMSE is expressed as a SIC fraction. Panels show point estimates, not confidence intervals.

한글 설명

범북극 평균이나 상태 간 오차 차이만으로 숨겨질 수 있는 해역별 절대오차를 확인하기 위한 미리보기이다. 상단 왼쪽은 기준장 상태 전이 노출도이며, 가운데와 오른쪽은 같은 색 범위를 사용하는 목표일 MIZ·비MIZ 절대 RMSE이다. 하단 전이 경로의 합계는 상단 왼쪽 후기 값과 일치한다. 2023–2025년 ver2 실험의 기존 검증된 요약값만 다시 배치했으며, 새 지표·신뢰구간·인과관계를 계산하지 않았다. 세 모델의 예측장을 평균한 것이 아니라 모델별 RMSE에 같은 가중치를 부여했다. 각 RMSE 구간은 282개 초기일을 포함하며, 상태가 없는 일부 초기일–선행시간 조합은 MIZ·비MIZ 공통 지지집합에서 함께 제외된다.
"""
    (OUT / "figure7_caption.txt").write_text(caption, encoding="utf-8")
    assert before == {str(p): sha(p) for p in protected}
    report = {
        "status": "PASS",
        "operation": "Preview only; no manuscript edits or new metric calculations",
        "scope": "2023–2025 ver2; 282 initializations; ocean mask including open water",
        "sources": {k: {"path": str(p), "sha256": sha(p)} for k, p in sources.items()},
        "protected_files_unchanged": before,
        "size_inches": [6.5, 6.2],
        "minimum_visible_font_pt": 9.5,
        "text_bounds_and_cell_annotations_verified": True,
        "aligned_heatmap_rows_verified": True,
        "heatmap_values_exactly_match_source": 45,
        "pathway_values_preserved": 30,
        "exposure_scale_percent": [0, 35],
        "shared_absolute_rmse_scale": [0, 0.25],
        "absolute_rmse_shared_norm_identity_verified": True,
        "max_pathway_sum_minus_exposure_percent": float(
            np.max(np.abs(left - exposure[:, 2]))
        ),
        "domain_order": SEAS,
        "lead_band_order": BANDS,
        "transition_order": PATHS,
        "transition_colors": PATH_COLORS,
        "aggregation": "RMSE: matched MIZ/non-MIZ support, equal mean of three model-wise metrics; reference exposure: area-weighted grid-date pairs",
        "drawn_values": plotted,
        "visible_text": text_records,
        "output_sha256": {
            name: sha(OUT / name)
            for name in [
                "figure7.png",
                "figure7.pdf",
                "figure7_manuscript_scale.png",
                "figure7_plotted_values.csv",
                "figure7_caption.txt",
            ]
        },
        "visual_qa": "Pending direct image inspection",
    }
    (OUT / "figure7_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(OUT),
                "plotted_values": len(plotted),
                "size_inches": report["size_inches"],
                "max_pathway_sum_error": report[
                    "max_pathway_sum_minus_exposure_percent"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
