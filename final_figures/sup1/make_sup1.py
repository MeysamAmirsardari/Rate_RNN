"""Build Supplementary Figure 1: roving ECoG global field power."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from final_figures.figure_2.ecog_data import build_ecog_cache
from final_figures.figure_2.inference import load_or_build_inference
from final_figures.figure_2.model_data import load_or_build_model_data
from final_figures.style import COLORS, clean_axis, export_figure, manuscript_style, mm


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "figure_2" / "data"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_STEM = OUTPUT_DIR / "sup1_ecog_gfp"


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(stop - 1)) for start, stop in edges.reshape(-1, 2)]


def _rail(ax, time: np.ndarray, mask: np.ndarray, y: float, color: str) -> None:
    for start, stop in _segments(mask):
        ax.plot(
            [time[start], time[stop]],
            [y, y],
            color=COLORS["white"],
            lw=3.1,
            solid_capstyle="butt",
            zorder=8,
        )
        ax.plot(
            [time[start], time[stop]],
            [y, y],
            color=color,
            lw=1.8,
            solid_capstyle="butt",
            zorder=9,
        )


def build_sup1() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ecog = build_ecog_cache(data_dir=DATA_DIR)
    model = load_or_build_model_data(data_dir=DATA_DIR)
    inference = load_or_build_inference(ecog, model, data_dir=DATA_DIR)

    values = np.concatenate(
        [
            ecog[f"pos{position}_gfp_rep{repetition}"]
            for position in (1, 2, 3)
            for repetition in (1, 15)
        ]
    )
    y_max = float(np.ceil(np.nanmax(values) * 5.0) / 5.0)
    rail_y = 1.08 * y_max

    with manuscript_style():
        fig = plt.figure(figsize=(mm(89), mm(82)))
        grid = fig.add_gridspec(
            3,
            1,
            left=0.17,
            right=0.97,
            bottom=0.14,
            top=0.80,
            hspace=0.12,
        )
        axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
        for index, (ax, position) in enumerate(zip(axes, (1, 2, 3))):
            time = np.asarray(ecog[f"pos{position}_time_ms"], dtype=float)
            rep1 = np.asarray(ecog[f"pos{position}_gfp_rep1"], dtype=float)
            rep15 = np.asarray(ecog[f"pos{position}_gfp_rep15"], dtype=float)
            difference = np.asarray(
                inference[f"erp_pos{position}_difference"], dtype=float
            )
            significant = np.asarray(
                inference[f"erp_pos{position}_significant"], dtype=bool
            )
            ax.axvspan(0, 180, color=COLORS["peach"], alpha=0.50, lw=0)
            ax.axvline(180, color=COLORS["ash"], lw=0.48, alpha=0.75)
            ax.plot(time, rep1, color=COLORS["rep1"], lw=1.15)
            ax.plot(time, rep15, color=COLORS["rep15"], lw=1.15)
            _rail(
                ax,
                time,
                significant & (difference > 0),
                rail_y,
                COLORS["rep1"],
            )
            _rail(
                ax,
                time,
                significant & (difference < 0),
                rail_y,
                COLORS["rep15"],
            )
            ax.text(
                0.015,
                0.88,
                f"position {position}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.2,
                fontweight="semibold",
            )
            ax.set_xlim(0, 360)
            ax.set_ylim(0, 1.16 * y_max)
            ax.set_xticks([0, 180, 360])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            if index < 2:
                ax.set_xticklabels([])
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", length=0)
            else:
                ax.set_xlabel("Time from variable-tone onset (ms)")
            clean_axis(ax, bottom=index == 2)

        axes[1].set_ylabel("Baseline-normalized GFP (a.u.)", labelpad=6.0)
        fig.text(
            0.055,
            0.955,
            "S1",
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
        )
        fig.text(
            0.17,
            0.955,
            "ECoG global field power",
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="semibold",
        )
        fig.legend(
            handles=[
                Line2D([0], [0], color=COLORS["rep1"], lw=1.35, label="Rep 1"),
                Line2D([0], [0], color=COLORS["rep15"], lw=1.35, label="Rep 15"),
                Line2D(
                    [0],
                    [0],
                    color=COLORS["rep1"],
                    lw=1.8,
                    label="cluster-corrected P < 0.05",
                ),
            ],
            loc="upper right",
            bbox_to_anchor=(0.97, 0.895),
            ncol=3,
            fontsize=5.8,
            handlelength=1.2,
            columnspacing=0.75,
            borderaxespad=0,
        )
        paths = export_figure(fig, OUTPUT_STEM, fixed_bounds=True)
        plt.close(fig)

    provenance = {
        "figure": "Supplementary Figure 1 - ECoG global field power",
        "measure": (
            "Spatial standard deviation across all 32 baseline-SD-normalized "
            "channel ERPs after block averaging, shown for actual Rep 1 and "
            "Rep 15."
        ),
        "inference": (
            "Two-sided whole-block Rep-1/Rep-15 sign-flip cluster-mass test; "
            "4,999 randomizations; maximum cluster jointly over all latencies "
            "and the three recordings."
        ),
        "inputs": {
            "ecog": str((DATA_DIR / "ecog_figure2_data.npz").resolve()),
            "inference": str((DATA_DIR / "figure_2_inference.npz").resolve()),
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    provenance_path = HERE / "sup1_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    paths["provenance"] = provenance_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for name, path in build_sup1().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
