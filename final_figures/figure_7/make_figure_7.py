"""Build the submission-ready frozen-core synthesis Figure 7."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from final_figures.figure_7.figure7_data import (
    CONDITION_LABELS,
    EQUIVALENCE_BOUND,
    METRIC_LABELS,
    TASK_LABELS,
    build_figure7_data,
)
from final_figures.style import (
    COLORS,
    clean_axis,
    export_figure,
    manuscript_style,
    mm,
    panel_label,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_STEM = OUTPUT_DIR / "figure_7_causal_synthesis"

LESION_COLORS = (
    COLORS["terracotta"],
    COLORS["decoder"],
    COLORS["teal"],
    COLORS["rep15"],
)
LESION_SHORT = (
    "- depression",
    "- recurrent learning",
    "uniform inhibition",
    "single-rate readout",
)

ABLATION_CMAP = LinearSegmentedColormap.from_list(
    "ablation_modern_diverging",
    [
        COLORS["rep15"],
        "#A9C3DE",
        COLORS["white"],
        "#E9C7C7",
        COLORS["terracotta"],
        COLORS["rep1"],
    ],
    N=256,
)
COUPLING_CMAP = LinearSegmentedColormap.from_list(
    "coupling_linen_violet",
    [
        COLORS["linen"],
        "#D9D2E6",
        "#A99AC5",
        COLORS["decoder"],
        "#3D304C",
    ],
    N=256,
)


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _heading(ax, title: str, subtitle: str, *, y: float = 1.14) -> None:
    ax.text(
        0,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="semibold",
        color=COLORS["charcoal"],
        clip_on=False,
    )
    ax.text(
        0,
        y - 0.105,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=COLORS["ash"],
        clip_on=False,
    )


def _status_glyph(status: str) -> str:
    return {"changed": "Δ", "equivalent": "≈", "inconclusive": "?"}.get(
        status, ""
    )


def _panel_matrix(fig, spec, data) -> None:
    sub = spec.subgridspec(1, 2, width_ratios=(1.0, 0.035), wspace=0.16)
    ax = fig.add_subplot(sub[0, 0])
    cax = fig.add_subplot(sub[0, 1])
    values = np.asarray(data["ablation_effect"], dtype=float)
    status = np.asarray(data["status"]).astype(str)

    norm = TwoSlopeNorm(vmin=-0.36, vcenter=0.0, vmax=1.05)
    image = ax.imshow(values, cmap=ABLATION_CMAP, norm=norm, aspect="auto")

    row_labels = (
        "Roving novelty\nsuppression index",
        "Directional prediction\nunexpected - predicted",
        "Coherent dependency\nW_FF - W_GG",
        "Sequence composition\nwords held in order",
    )
    column_labels = (
        "- depression",
        "- recurrent\nlearning",
        "uniform\ninhibition",
        "single-rate\nreadout",
    )
    ax.set_yticks(range(4), row_labels)
    ax.set_xticks(range(4), column_labels)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False,
                   length=0, pad=3.2)
    ax.tick_params(axis="y", length=0, pad=5.5)
    for tick in ax.get_yticklabels():
        tick.set_linespacing(1.35)
    for tick in ax.get_xticklabels():
        tick.set_linespacing(1.25)

    # Hairline cell separators keep the map legible in grayscale without
    # making it look like a spreadsheet.
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color=COLORS["white"], linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(4):
        for column in range(4):
            value = values[row, column]
            if not np.isfinite(value):
                ax.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=COLORS["mist"],
                        edgecolor=COLORS["ash"],
                        hatch="////",
                        linewidth=0.45,
                        zorder=4,
                    )
                )
                ax.text(
                    column,
                    row,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize=6.1,
                    color=COLORS["ash"],
                    zorder=6,
                )
                continue
            text_color = COLORS["white"] if value > 0.64 else COLORS["charcoal"]
            glyph = _status_glyph(status[row, column])
            ax.text(
                column,
                row - 0.07,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="semibold",
                color=text_color,
                zorder=6,
            )
            ax.text(
                column,
                row + 0.24,
                glyph,
                ha="center",
                va="center",
                fontsize=5.8,
                color=text_color,
                zorder=6,
            )

    colorbar = fig.colorbar(image, cax=cax, orientation="vertical")
    colorbar.set_ticks([-0.3, 0, 0.5, 1.0])
    colorbar.set_ticklabels(["-0.3", "0", "0.5", "1"])
    colorbar.set_label("Ablation effect\n1 - effect retained", labelpad=3.5)
    cax.yaxis.set_label_position("left")
    colorbar.outline.set_linewidth(0.45)
    cax.tick_params(length=1.8, pad=1.5)

    _heading(
        ax,
        "One frozen core reveals a sparse causal sensitivity map",
        (
            "effect = 1 - condition/intact; 1, abolished; 0, intact-like; "
            "<0, enhanced | Δ, changed; ≈, equivalent within ±20%"
        ),
        y=1.30,
    )
    panel_label(ax, "A", x=-0.17, y=1.26)


def _native_intact_label(task_index: int, values: np.ndarray) -> str:
    mean = float(np.mean(values))
    if task_index == 0:
        return f"intact SI = {mean:.3f}"
    if task_index == 1:
        return f"intact Delta rate = {mean:.3f}"
    if task_index == 2:
        return f"intact Delta W = {mean:.4f}"
    return f"intact = {mean:.2f} of vocabulary"


def _panel_effects(fig, spec, data) -> None:
    grid = spec.subgridspec(2, 2, wspace=0.34, hspace=0.67)
    retention = np.asarray(data["retention"], dtype=float)
    mean = np.asarray(data["retention_mean"], dtype=float)
    ci = np.asarray(data["retention_ci"], dtype=float)
    status = np.asarray(data["status"]).astype(str)
    raw = np.asarray(data["raw_effect"], dtype=float)
    rng = np.random.default_rng(77)

    axes = []
    for task_index in range(4):
        ax = fig.add_subplot(grid[task_index // 2, task_index % 2])
        axes.append(ax)
        applicable = np.flatnonzero(np.isfinite(mean[task_index]))
        y_positions = np.arange(applicable.size)[::-1]

        ax.axvspan(
            1 - EQUIVALENCE_BOUND,
            1 + EQUIVALENCE_BOUND,
            color=COLORS["mist"],
            alpha=0.82,
            linewidth=0,
            zorder=0,
        )
        ax.axvline(1.0, color=COLORS["charcoal"], lw=0.65, zorder=1)
        ax.axvline(0.0, color=COLORS["ash"], lw=0.45, ls=(0, (2.2, 2.2)), zorder=1)

        for y, lesion_index in zip(y_positions, applicable):
            values = retention[task_index, lesion_index]
            jitter = np.linspace(-0.14, 0.14, values.size)
            rng.shuffle(jitter)
            for value, dy in zip(values, jitter):
                ax.plot(
                    [1.0, value],
                    [y + dy, y + dy],
                    color=LESION_COLORS[lesion_index],
                    alpha=0.13,
                    lw=0.45,
                    zorder=2,
                )
            ax.scatter(
                values,
                y + jitter,
                s=7.5,
                facecolor=COLORS["white"],
                edgecolor=LESION_COLORS[lesion_index],
                linewidth=0.55,
                alpha=0.85,
                zorder=3,
            )
            low, high = ci[task_index, lesion_index]
            centre = mean[task_index, lesion_index]
            ax.errorbar(
                centre,
                y,
                xerr=np.array([[centre - low], [high - centre]]),
                fmt="D",
                ms=3.7,
                mfc=LESION_COLORS[lesion_index],
                mec=COLORS["white"],
                mew=0.45,
                ecolor=COLORS["charcoal"],
                elinewidth=0.9,
                capsize=2.0,
                capthick=0.8,
                zorder=5,
            )
            ax.text(
                1.43,
                y,
                _status_glyph(status[task_index, lesion_index]),
                ha="right",
                va="center",
                fontsize=6.4,
                fontweight="semibold",
                color=LESION_COLORS[lesion_index],
                clip_on=False,
            )

        ax.set_xlim(-0.13, 1.47)
        ax.set_ylim(-0.55, max(y_positions) + 0.55)
        ax.set_xticks([0, 0.5, 1.0, 1.4])
        if task_index < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Primary effect retained  (intact = 1)")
        ax.set_yticks(
            y_positions,
            [LESION_SHORT[index] for index in applicable],
        )
        ax.tick_params(axis="y", length=0, pad=3.5)
        clean_axis(ax)
        ax.set_title(TASK_LABELS[task_index], loc="left", pad=8.0)
        ax.text(
            1.0,
            1.055,
            _native_intact_label(task_index, raw[task_index, 0]),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.6,
            color=COLORS["ash"],
            clip_on=False,
        )

    _heading(
        axes[0],
        "Paired sessions separate statistical change from practical equivalence",
        (
            "n = 12 per task; points are paired sessions, diamonds are means, "
            "bars are paired-bootstrap 95% CIs; gray band is the prespecified +/-20% SESOI"
        ),
        y=1.56,
    )
    panel_label(axes[0], "B", x=-0.35, y=1.76)


def _panel_aba(fig, spec, data) -> None:
    grid = spec.subgridspec(1, 2, width_ratios=(1.25, 0.72), wspace=0.34)
    ax = fig.add_subplot(grid[0, 0])
    audit = fig.add_subplot(grid[0, 1])

    coupling = np.asarray(data["aba_coupling"], dtype=float)
    rates = np.asarray(data["aba_rates_hz"], dtype=float)
    separations = np.asarray(data["aba_separations"], dtype=int)
    image = ax.imshow(
        coupling,
        origin="lower",
        aspect="auto",
        cmap=COUPLING_CMAP,
        norm=Normalize(vmin=0, vmax=0.50),
        extent=(0.5, separations[-1] + 0.5, rates[0] - 0.5, rates[-1] + 0.5),
        interpolation="nearest",
    )
    ax.set_xticks(separations)
    ax.set_yticks(rates[::2])
    ax.set_xlabel("Nominal A-B separation  (channels)")
    ax.set_ylabel("Tone-slot rate  (Hz)")
    clean_axis(ax)
    colorbar = fig.colorbar(image, ax=ax, orientation="vertical", fraction=0.045, pad=0.025)
    colorbar.set_label("Learned A-B / self coupling", labelpad=3)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_ticks([0, 0.25, 0.5])

    spread = float(np.max(np.ptp(coupling, axis=1)))
    ax.text(
        0.02,
        0.03,
        f"max spread across separation = {spread:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=COLORS["white"],
        bbox=dict(facecolor=COLORS["charcoal"], edgecolor="none", pad=1.8, alpha=0.88),
    )

    _heading(
        ax,
        "The no-retuning ABA-silence test has rate sensitivity but no Δf axis",
        (
            "80 ABA-silence cycles per cell; one-hot tones, frozen core, zero initial W; "
            "the map reports model coupling, not a perceptual boundary"
        ),
        y=1.25,
    )
    panel_label(ax, "C", x=-0.19, y=1.42)

    audit.set_axis_off()
    audit.set_xlim(0, 1)
    audit.set_ylim(0, 1)
    for y, x_b, delta in ((0.72, 0.47, "Δch = 1"), (0.37, 0.82, "Δch = 7")):
        audit.add_patch(
            plt.Circle((0.16, y), 0.055, facecolor=COLORS["white"],
                       edgecolor=COLORS["decoder"], lw=1.1)
        )
        audit.add_patch(
            plt.Circle((x_b, y), 0.055, facecolor=COLORS["white"],
                       edgecolor=COLORS["decoder"], lw=1.1)
        )
        audit.text(0.16, y, "A", ha="center", va="center", fontsize=6.4,
                   fontweight="semibold")
        audit.text(x_b, y, "B", ha="center", va="center", fontsize=6.4,
                   fontweight="semibold")
        audit.add_patch(
            FancyArrowPatch(
                (0.22, y + 0.018),
                (x_b - 0.06, y + 0.018),
                arrowstyle="-|>",
                mutation_scale=6,
                color=COLORS["teal"],
                lw=0.9,
            )
        )
        audit.add_patch(
            FancyArrowPatch(
                (x_b - 0.06, y - 0.018),
                (0.22, y - 0.018),
                arrowstyle="-|>",
                mutation_scale=6,
                color=COLORS["teal"],
                lw=0.9,
            )
        )
        audit.text((0.16 + x_b) / 2, y + 0.075, "same w_lat", ha="center",
                   va="bottom", fontsize=5.6, color=COLORS["teal"])
        audit.text((0.16 + x_b) / 2, y - 0.085, delta, ha="center", va="top",
                   fontsize=5.5, color=COLORS["ash"])

    audit.text(
        0.02,
        0.98,
        "Permutation audit",
        ha="left",
        va="top",
        fontsize=7.3,
        fontweight="semibold",
        color=COLORS["charcoal"],
    )
    audit.text(
        0.02,
        0.88,
        r"$w_{ij}=w_{lat}$ for every $i\ne j$",
        ha="left",
        va="top",
        fontsize=6.0,
        color=COLORS["ash"],
    )
    audit.text(
        0.02,
        0.20,
        "A classic van Noorden boundary cannot emerge until the fixed front end "
        "or connectivity contains spectral distance.",
        ha="left",
        va="top",
        fontsize=5.9,
        color=COLORS["charcoal"],
        linespacing=1.45,
        wrap=True,
    )


def _panel_synthesis(fig, spec) -> None:
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    stages = (
        ("Novelty gate", "multiscale depression", "Roving · SFG", COLORS["terracotta"]),
        ("Dependency map", "recurrent learning", "AB/BA · SFG", COLORS["decoder"]),
        ("Competitive expression", "selective inhibition", "AB/BA", COLORS["teal"]),
        ("Sequence composition", "multiscale readout", "Saffran", COLORS["rep15"]),
    )
    x_positions = np.linspace(0.03, 0.77, 4)
    width, height = 0.19, 0.54
    for index, (title, mechanism, evidence, color) in enumerate(stages):
        x = x_positions[index]
        box = FancyBboxPatch(
            (x, 0.29),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=COLORS["linen"] if index % 2 == 0 else COLORS["white"],
            edgecolor=color,
            linewidth=1.15,
        )
        ax.add_patch(box)
        ax.text(x + 0.018, 0.68, title, ha="left", va="center", fontsize=6.35,
                fontweight="semibold", color=COLORS["charcoal"])
        ax.text(x + 0.018, 0.51, mechanism, ha="left", va="center", fontsize=5.8,
                color=COLORS["ash"])
        ax.plot([x + 0.018, x + width - 0.018], [0.43, 0.43], color=color,
                lw=1.0, alpha=0.8)
        ax.text(x + 0.018, 0.35, evidence, ha="left", va="bottom", fontsize=5.6,
                color=color)
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.008, 0.56),
                    (x_positions[index + 1] - 0.012, 0.56),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    color=COLORS["ash"],
                    lw=0.8,
                )
            )

    ax.text(
        0.0,
        0.04,
        "A coordinated toolkit is supported within one frozen core; spectral-distance structure remains the explicit missing computation.",
        ha="left",
        va="bottom",
        fontsize=6.4,
        fontweight="semibold",
        color=COLORS["charcoal"],
    )
    ax.text(
        0,
        1.04,
        "Mechanistic synthesis",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="semibold",
        color=COLORS["charcoal"],
        clip_on=False,
    )
    panel_label(ax, "D", x=-0.045, y=1.17)


def _write_figure_provenance(paths: dict[str, Path]) -> None:
    data_provenance = DATA_DIR / "figure_7_data_provenance.json"
    value = {
        "figure": "Figure 7 - frozen-core causal synthesis",
        "final_size_mm": [183, 245],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claim": (
            "A coordinated set of mechanisms supports distinct computations "
            "within one frozen core, while the present circuit lacks a spectral-distance axis."
        ),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "data_provenance": {
            "path": str(data_provenance.resolve()),
            "sha256": _sha256(data_provenance),
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "layout_checks": {
            "fixed_bounds": True,
            "pdf_fonttype": 42,
            "png_dpi": 600,
            "minimum_nominal_font_pt": 5.5,
        },
    }
    path = DATA_DIR / "figure_7_provenance.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def make_figure(*, force_data: bool = False) -> dict[str, Path]:
    data = build_figure7_data(force=force_data, data_dir=DATA_DIR)
    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(245)))
        grid = fig.add_gridspec(
            4,
            1,
            left=0.145,
            right=0.955,
            bottom=0.045,
            top=0.935,
            height_ratios=(1.0, 1.45, 1.08, 0.60),
            hspace=0.57,
        )
        _panel_matrix(fig, grid[0], data)
        _panel_effects(fig, grid[1], data)
        _panel_aba(fig, grid[2], data)
        _panel_synthesis(fig, grid[3])
        paths = export_figure(fig, OUTPUT_STEM, fixed_bounds=True)
        plt.close(fig)
    _write_figure_provenance(paths)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-data", action="store_true")
    arguments = parser.parse_args()
    paths = make_figure(force_data=arguments.force_data)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
