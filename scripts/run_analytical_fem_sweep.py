"""Run the ten-point lubrication/FEM constriction-ratio study.

The script varies only ``width_min / width_max``, reuses the analytical
and FEM implementations, and writes scalar results below the selected result
root. It never imports or runs PINN code.
"""

from __future__ import annotations

import os

# Set conservative single-threaded native-library defaults before numerical imports.
try:
    from scripts._native_environment import configure_native_library_environment
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _native_environment import configure_native_library_environment

configure_native_library_environment(os.name, os.environ)

import argparse
import json
from pathlib import Path
from typing import Any

from corrugated_stokes.analytical import lubrication_flux, poiseuille_flux
from corrugated_stokes.fem import solve_stokes

try:
    from scripts._analytical_fem_study import (
        DELTA_VALUES,
        FINE_SETTINGS,
        FIXED_PARAMETERS,
        RELATIVE_FLUX_ERROR_DEFINITION,
        make_result_record,
        physical_for_delta,
        relative_difference,
        require_finite,
        same_number,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _analytical_fem_study import (
        DELTA_VALUES,
        FINE_SETTINGS,
        FIXED_PARAMETERS,
        RELATIVE_FLUX_ERROR_DEFINITION,
        make_result_record,
        physical_for_delta,
        relative_difference,
        require_finite,
        same_number,
    )

try:
    from scripts._result_root import add_result_root_arguments, parse_result_root
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _result_root import add_result_root_arguments, parse_result_root


FEM_SWEEP = FINE_SETTINGS
OUTPUT_RELATIVE_PATH = Path("analytical_fem/sweep.json")


def output_path(results_root: Path) -> Path:
    """Return the sweep output below the selected top-level result root."""

    return Path(results_root) / OUTPUT_RELATIVE_PATH


def validate_sweep_payload(payload: dict[str, Any]) -> None:
    """Validate scalar consistency, protocol metadata, and straight-channel checks."""

    require_finite(payload, "sweep")
    study = payload.get("study", {})
    if tuple(study.get("delta_values", ())) != DELTA_VALUES:
        raise ValueError("study metadata has incorrect delta values")
    if study.get("fem_settings") != {
        "nx": FEM_SWEEP.nx,
        "ny": FEM_SWEEP.ny,
        "integration_order": FEM_SWEEP.integration_order,
        "flux_quadrature_order": FEM_SWEEP.flux_quadrature_order,
    }:
        raise ValueError("study metadata has incorrect FEM settings")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) != len(DELTA_VALUES):
        raise ValueError("sweep must contain exactly ten points")
    observed_deltas = tuple(point.get("delta") for point in points)
    if observed_deltas != DELTA_VALUES:
        raise ValueError("sweep delta values or ordering are incorrect")
    for point in points:
        delta = point["delta"]
        width_max = point["width_max"]
        width_min = point["width_min"]
        mean_width = 0.5 * (width_max + width_min)
        expected = {
            "width_min": delta * width_max,
            "epsilon": (width_max - width_min) / FIXED_PARAMETERS["L"],
            "mean_width": mean_width,
            "relative_amplitude": 0.5 * (width_max - width_min) / mean_width,
            "relative_flux_error": relative_difference(
                point["analytical_flux"], point["fem_flux"]
            ),
        }
        for name, value in expected.items():
            if not same_number(point[name], value):
                raise ValueError(f"inconsistent {name} at delta={delta}")

    checks = payload.get("checks", {})
    if checks.get("straight_channel_analytical_vs_exact_relative_error") > 2.0e-14:
        raise ValueError("straight-channel lubrication result does not match Poiseuille")
    if checks.get("straight_channel_fem_vs_exact_relative_error") > 2.0e-11:
        raise ValueError("straight-channel FEM result does not match Poiseuille")


def run_study() -> dict[str, Any]:
    """Execute exactly the final ten FEM solves and return their scalar study."""

    points: list[dict[str, float]] = []
    for index, delta in enumerate(DELTA_VALUES, 1):
        physical = physical_for_delta(delta)
        analytical = lubrication_flux(physical)
        fem_model = solve_stokes(physical=physical, settings=FEM_SWEEP)
        diagnostics = fem_model["diagnostics"]
        point = make_result_record(
            physical,
            analytical,
            float(diagnostics["mean_flux"]),
            float(diagnostics["relative_flux_variation"]),
        )
        points.append(point)
        print(f"FEM sweep {index}/{len(DELTA_VALUES)} delta={delta:.1f}", flush=True)

    straight = points[0]
    exact_straight_flux = poiseuille_flux(
        straight["width_max"], FIXED_PARAMETERS["mu"],
        -FIXED_PARAMETERS["delta_p"] / FIXED_PARAMETERS["L"],
    )
    checks = {
        "straight_channel_analytical_vs_exact_relative_error": relative_difference(
            straight["analytical_flux"], exact_straight_flux
        ),
        "straight_channel_fem_vs_exact_relative_error": relative_difference(
            straight["fem_flux"], exact_straight_flux
        ),
    }
    payload = {
        "study": {
            "fixed_parameters": FIXED_PARAMETERS,
            "delta_values": list(DELTA_VALUES),
            "fem_settings": {
                "nx": FEM_SWEEP.nx,
                "ny": FEM_SWEEP.ny,
                "integration_order": FEM_SWEEP.integration_order,
                "flux_quadrature_order": FEM_SWEEP.flux_quadrature_order,
            },
            "analytical_flux": "corrugated_stokes.analytical.lubrication_flux",
            "relative_flux_error_definition": (
                RELATIVE_FLUX_ERROR_DEFINITION
            ),
            "swept_parameter": "delta = width_min / width_max",
        },
        "points": points,
        "checks": checks,
    }
    validate_sweep_payload(payload)
    return payload


def write_study(output: Path, *, force: bool = False) -> Path:
    """Run the study and atomically write stable, timestamp-free JSON."""

    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"analytical/FEM sweep exists; refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    payload = run_study()
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def main(argv: list[str] | None = None) -> int:
    """Select a result root, then run the fixed study."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_result_root_arguments(parser)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing selected-root sweep"
    )
    args = parser.parse_args(argv)
    results_root = parse_result_root(parser, args)
    output = write_study(output_path(results_root), force=args.force)
    print(f"ANALYTICAL_FEM_SWEEP_WRITTEN {output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
