"""Render the final stored analytical/FEM relative-flux-error figure.

The renderer reads only the selected-root analytical/FEM sweep.  It does
not run the sweep, solve FEM equations, or evaluate a PINN.
"""

from __future__ import annotations

import argparse

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = SCRIPT_DIR.parent
for search_path in (SCRIPT_DIR, SCRIPTS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from _common import save_figure_pair  # noqa: E402
from _result_root import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    add_result_root_arguments,
    parse_result_root,
)


def source_path(results_root: Path) -> Path:
    """Return the analytical sweep below one selected result root."""

    return Path(results_root) / "analytical_fem" / "sweep.json"


def output_stem(results_root: Path) -> Path:
    """Return the analytical figure stem below one selected result root."""

    return Path(results_root) / "figures" / "analytical" / "relative_flux_error"


SOURCE_PATH = source_path(DEFAULT_RESULT_ROOT)
OUTPUT_STEM = output_stem(DEFAULT_RESULT_ROOT)
EXPECTED_DELTAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
EXPECTED_FEM_SETTINGS = {
    "nx": 256,
    "ny": 128,
    "integration_order": 6,
    "flux_quadrature_order": 64,
}


def load_payload(path: Path = SOURCE_PATH) -> dict[str, Any]:
    """Load the stored sweep without importing or running sweep code."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("analytical/FEM sweep payload must be a JSON object")
    return payload


def validated_plot_data(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Validate the scientific study definition and return increasing-x data."""
    points = payload.get("points")
    if not isinstance(points, list) or len(points) != len(EXPECTED_DELTAS):
        raise ValueError("sweep.json must contain exactly ten points")

    study = payload.get("study")
    if not isinstance(study, dict):
        raise ValueError("sweep.json is missing its study definition")
    if study.get("delta_values") != list(EXPECTED_DELTAS):
        raise ValueError("study delta values do not match the defined sweep")
    if study.get("fem_settings") != EXPECTED_FEM_SETTINGS:
        raise ValueError("study FEM settings do not match the defined sweep")

    deltas: list[float] = []
    percent_errors: list[float] = []
    for expected_delta, point in zip(EXPECTED_DELTAS, points):
        if not isinstance(point, dict):
            raise ValueError("every sweep point must be a JSON object")
        try:
            delta = float(point["delta"])
            analytical_flux = float(point["analytical_flux"])
            fem_flux = float(point["fem_flux"])
            stored_error = float(point["relative_flux_error"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("sweep point is missing a finite plotting value") from error
        values = (delta, analytical_flux, fem_flux, stored_error)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("sweep plotting values must be finite")
        if delta != expected_delta:
            raise ValueError("point delta values do not match the defined ordered sweep")
        if stored_error < 0.0:
            raise ValueError("relative_flux_error must be nonnegative")
        if fem_flux == 0.0:
            raise ValueError("relative flux error is undefined for zero FEM flux")
        recomputed_error = abs(analytical_flux - fem_flux) / abs(fem_flux)
        if not math.isclose(stored_error, recomputed_error, rel_tol=1.0e-13, abs_tol=1.0e-15):
            raise ValueError("stored relative_flux_error disagrees with its definition")
        deltas.append(delta)
        percent_errors.append(100.0 * stored_error)

    # The result is stored from 1.0 down to 0.1; reverse only for the
    # conventional left-to-right increasing x-axis, without interpolation.
    return (
        np.asarray(deltas[::-1], dtype=np.float64),
        np.asarray(percent_errors[::-1], dtype=np.float64),
    )


def make_figure(
    deltas: np.ndarray,
    percent_errors: np.ndarray,
    output_stem: Path = OUTPUT_STEM,
) -> tuple[Path, Path]:
    """Plot the ten exact stored samples with a restrained connecting line."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax.plot(
        deltas,
        percent_errors,
        color="#204a87",
        linewidth=1.6,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    ax.set_xlabel("Constriction ratio $\\delta=\\Delta\\omega/\\Delta\\Omega$")
    ax.set_ylabel("Relative flux error [%]")
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.set_xlim(0.075, 1.025)
    ax.set_ylim(bottom=0.0)
    ax.grid(color="#d0d0d0", linewidth=0.7, alpha=0.65)
    return save_figure_pair(fig, output_stem)


def main(argv: list[str] | None = None) -> int:
    """Validate stored data and generate exactly one PNG/PDF figure pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_result_root_arguments(parser)
    args = parser.parse_args(argv)
    results_root = parse_result_root(parser, args)
    deltas, percent_errors = validated_plot_data(load_payload(source_path(results_root)))
    png_path, pdf_path = make_figure(
        deltas, percent_errors, output_stem(results_root)
    )
    print("Created analytical outputs:")
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
