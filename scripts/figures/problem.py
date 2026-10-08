"""Render the accepted problem-geometry schematic and no other figure.

Run this module directly from the project root.  It evaluates only the exact
wall geometry and does not invoke FEM, PINN, validation, or sweep code.
"""

from __future__ import annotations

import argparse

from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = SCRIPT_DIR.parent
SRC = PROJECT_ROOT / "src"
for search_path in (SCRIPT_DIR, SCRIPTS_DIR, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from _common import save_figure_pair  # noqa: E402
from _result_root import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    add_result_root_arguments,
    parse_result_root,
)
from corrugated_stokes.config import PHYSICAL  # noqa: E402
from corrugated_stokes.geometry import lower_wall, upper_wall  # noqa: E402


def output_stem(results_root: Path) -> Path:
    """Return the geometry figure stem below one selected result root."""

    return Path(results_root) / "figures" / "problem" / "geometry"


OUTPUT_STEM = output_stem(DEFAULT_RESULT_ROOT)


def geometry_plot_data(samples: int = 801) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact FEM-coordinate wall data on one cell ``x in [0, L]``."""
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("geometry samples must be an integer of at least two")
    x = np.linspace(0.0, PHYSICAL.L, samples)
    return x, upper_wall(x), lower_wall(x)


def make_geometry_figure(output_stem: Path = OUTPUT_STEM) -> tuple[Path, Path]:
    """Create the accepted engineering schematic of the target channel."""
    import matplotlib.pyplot as plt

    x, upper, lower = geometry_plot_data()
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    neutral = "#303030"
    dimension_width = 1.1
    extension_width = 0.75
    annotation_fontsize = 14

    # Exact channel walls remain the visual focus; dimensions stay outside.
    ax.fill_between(x, lower, upper, color="#f1f1f1")
    ax.plot(x, upper, color="black", linewidth=1.5)
    ax.plot(x, lower, color="black", linewidth=1.5)

    # Cell period below the full channel, with extension lines and end caps.
    period_y = -0.31
    cap_half_height = 0.009
    ax.plot([0.0, PHYSICAL.L], [period_y, period_y], color=neutral, linewidth=dimension_width)
    ax.plot(
        [0.0, 0.0],
        [lower[0], period_y - cap_half_height],
        color=neutral,
        linewidth=extension_width,
    )
    ax.plot(
        [PHYSICAL.L, PHYSICAL.L],
        [lower[-1], period_y - cap_half_height],
        color=neutral,
        linewidth=extension_width,
    )
    for xpos in (0.0, PHYSICAL.L):
        ax.plot(
            [xpos, xpos],
            [period_y - cap_half_height, period_y + cap_half_height],
            color=neutral,
            linewidth=dimension_width,
        )
    ax.text(
        0.5 * PHYSICAL.L,
        period_y - 0.022,
        "$L$",
        ha="center",
        va="top",
        fontsize=annotation_fontsize,
    )
    ax.text(
        0.02 * PHYSICAL.L,
        -0.272,
        "$x=0$",
        ha="left",
        va="center",
        fontsize=annotation_fontsize,
    )
    ax.text(
        0.98 * PHYSICAL.L,
        -0.272,
        "$x=L$",
        ha="right",
        va="center",
        fontsize=annotation_fontsize,
    )

    # Minimum width at x=0, dimensioned completely outside the left boundary.
    minimum_x = -0.075
    minimum_half = 0.5 * PHYSICAL.width_min
    cap_half_width = 0.010
    for ypos in (-minimum_half, minimum_half):
        ax.plot([minimum_x, 0.0], [ypos, ypos], color=neutral, linewidth=extension_width)
        ax.plot(
            [minimum_x - cap_half_width, minimum_x + cap_half_width],
            [ypos, ypos],
            color=neutral,
            linewidth=dimension_width,
        )
    ax.plot(
        [minimum_x, minimum_x],
        [-minimum_half, minimum_half],
        color=neutral,
        linewidth=dimension_width,
    )
    ax.text(
        minimum_x - 0.018,
        0.0,
        "$\\Delta\\omega$",
        ha="right",
        va="center",
        fontsize=annotation_fontsize,
    )

    # Maximum-width wall points are at x=L/2; their extension lines remain
    # outside the fluid as they run toward the right-hand dimension line.
    maximum_x = 1.075 * PHYSICAL.L
    maximum_half = 0.5 * PHYSICAL.width_max
    for ypos in (-maximum_half, maximum_half):
        ax.plot(
            [0.5 * PHYSICAL.L, maximum_x],
            [ypos, ypos],
            color=neutral,
            linewidth=extension_width,
        )
        ax.plot(
            [maximum_x - cap_half_width, maximum_x + cap_half_width],
            [ypos, ypos],
            color=neutral,
            linewidth=dimension_width,
        )
    ax.plot(
        [maximum_x, maximum_x],
        [-maximum_half, maximum_half],
        color=neutral,
        linewidth=dimension_width,
    )
    ax.text(
        maximum_x + 0.018,
        0.0,
        "$\\Delta\\Omega$",
        ha="left",
        va="center",
        fontsize=annotation_fontsize,
    )

    # Compact boundary-pressure labels sit near the openings; the secondary
    # inlet/outlet labels sit just inside the shaded end regions.
    pressure_fontsize = 13
    secondary_fontsize = 11
    ax.text(
        -0.035 * PHYSICAL.L,
        0.10,
        "$p(0)=p_{\\mathrm{in}}$",
        ha="left",
        va="bottom",
        color=neutral,
        fontsize=pressure_fontsize,
    )
    ax.text(
        1.038 * PHYSICAL.L,
        0.10,
        "$p(L)=p_{\\mathrm{out}}$",
        ha="right",
        va="bottom",
        color=neutral,
        fontsize=pressure_fontsize,
    )
    ax.text(
        0.035 * PHYSICAL.L,
        0.0,
        "inlet",
        ha="left",
        va="center",
        color="#555555",
        fontsize=secondary_fontsize,
    )
    ax.text(
        0.965 * PHYSICAL.L,
        0.0,
        "outlet",
        ha="right",
        va="center",
        color="#555555",
        fontsize=secondary_fontsize,
    )
    ax.text(
        0.5 * PHYSICAL.L,
        0.285,
        "$p_{\\mathrm{in}}>p_{\\mathrm{out}}$",
        ha="center",
        va="bottom",
        color="#444444",
        fontsize=pressure_fontsize,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.15 * PHYSICAL.L, 1.15 * PHYSICAL.L)
    ax.set_ylim(-0.36, 0.34)
    ax.set_axis_off()
    return save_figure_pair(fig, output_stem)


def main(argv: list[str] | None = None) -> int:
    """Generate exactly the final geometry PNG/PDF pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_result_root_arguments(parser)
    args = parser.parse_args(argv)
    png_path, pdf_path = make_geometry_figure(
        output_stem(parse_result_root(parser, args))
    )
    print("Created geometry outputs:")
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
