"""Recompute the coarse analytical/FEM sweep and compare it with the fine sweep.

The resolution check reads the selected-root ten-point 256 x 128 sweep,
independently recomputes all ten 128 x 64 FEM points, and stores only the
coarse records plus compact pointwise comparisons. It never evaluates a PINN.
"""

from __future__ import annotations

import os

try:
    from scripts._native_environment import configure_native_library_environment
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _native_environment import configure_native_library_environment

configure_native_library_environment(os.name, os.environ)

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

from corrugated_stokes.analytical import lubrication_flux, poiseuille_flux
from corrugated_stokes.config import FEMConfig
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


COARSE_SETTINGS = FEMConfig(
    nx=128, ny=64, integration_order=6, flux_quadrature_order=64
)
FINE_SWEEP_RELATIVE_PATH = Path("analytical_fem/sweep.json")
OUTPUT_RELATIVE_PATH = Path("analytical_fem/resolution_check.json")
RELATIVE_MESH_DIFFERENCE_DEFINITION = "abs(Q_fine - Q_coarse) / abs(Q_coarse)"


def fine_sweep_path(results_root: Path) -> Path:
    """Return the input sweep below the selected top-level result root."""

    return Path(results_root) / FINE_SWEEP_RELATIVE_PATH


def output_path(results_root: Path) -> Path:
    """Return the resolution output below the selected top-level result root."""

    return Path(results_root) / OUTPUT_RELATIVE_PATH


def solve_one(delta: float, settings: FEMConfig) -> dict[str, float]:
    """Run one FEM point from scratch and retain scalar diagnostics only."""
    physical = physical_for_delta(delta)
    model = solve_stokes(physical=physical, settings=settings)
    diagnostics = model["diagnostics"]
    return make_result_record(
        physical,
        lubrication_flux(physical),
        diagnostics["mean_flux"],
        diagnostics["relative_flux_variation"],
    )


def validate_result_record(record: dict[str, Any], expected_delta: float) -> None:
    """Validate geometry identities and the stored analytical/FEM error formula."""
    required = {
        "delta", "epsilon", "width_max", "width_min", "mean_width",
        "relative_amplitude", "analytical_flux", "fem_flux",
        "fem_relative_flux_variation", "relative_flux_error",
    }
    if set(record) != required:
        raise ValueError("resolution record has an unexpected field set")
    physical = physical_for_delta(expected_delta)
    expected = {
        "delta": expected_delta,
        "width_max": physical.width_max,
        "width_min": physical.width_min,
        "epsilon": physical.epsilon,
        "mean_width": physical.mean_width,
        "relative_amplitude": physical.relative_amplitude,
        "relative_flux_error": relative_difference(
            record["analytical_flux"], record["fem_flux"]
        ),
    }
    if any(not same_number(record[name], value) for name, value in expected.items()):
        raise ValueError(f"resolution record is inconsistent at delta={expected_delta}")
    require_finite(record, f"record[{expected_delta}]")


def validate_fine_sweep(payload: dict[str, Any]) -> list[dict[str, float]]:
    """Strictly validate the selected-root ten-point 256 x 128 fine sweep."""
    if set(payload) != {"study", "points", "checks"}:
        raise ValueError("fine sweep has an unexpected top-level schema")
    study = payload["study"]
    if tuple(study.get("delta_values", ())) != DELTA_VALUES:
        raise ValueError("fine sweep has incorrect delta values")
    if study.get("fem_settings") != asdict(FINE_SETTINGS):
        raise ValueError("fine sweep has incorrect FEM settings")
    fixed = study.get("fixed_parameters", {})
    if set(fixed) != set(FIXED_PARAMETERS) or any(
        not same_number(fixed[name], value) for name, value in FIXED_PARAMETERS.items()
    ):
        raise ValueError("fine sweep has incorrect physical parameters")
    if study.get("relative_flux_error_definition") != RELATIVE_FLUX_ERROR_DEFINITION:
        raise ValueError("fine sweep has incorrect error definition")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) != len(DELTA_VALUES):
        raise ValueError("fine sweep must contain exactly ten points")
    for delta, record in zip(DELTA_VALUES, points):
        validate_result_record(record, delta)
    require_finite(payload, "fine_sweep")
    return [dict(record) for record in points]


def compare_resolutions(
    coarse_results: list[dict[str, float]],
    fine_results: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Calculate prescribed pointwise coarse-fine diagnostics."""
    if len(coarse_results) != 10 or len(fine_results) != 10:
        raise ValueError("both resolutions must contain exactly ten points")
    comparisons = []
    for delta, coarse, fine in zip(DELTA_VALUES, coarse_results, fine_results):
        validate_result_record(coarse, delta)
        validate_result_record(fine, delta)
        if not same_number(coarse["analytical_flux"], fine["analytical_flux"]):
            raise ValueError("analytical flux must be resolution independent")
        comparisons.append({
            "delta": delta,
            "relative_coarse_fine_flux_difference": relative_difference(
                fine["fem_flux"], coarse["fem_flux"]
            ),
            "absolute_flux_error_change": (
                coarse["relative_flux_error"] - fine["relative_flux_error"]
            ),
        })
    return comparisons


def _maximum_with_delta(records: list[dict[str, float]], key: str) -> dict[str, float]:
    """Return one deterministic maximum and its corresponding delta."""
    record = max(records, key=lambda item: item[key])
    return {"value": float(record[key]), "delta": float(record["delta"])}


def build_payload(
    coarse_results: list[dict[str, float]],
    fine_results: list[dict[str, float]],
    fine_source: Path = FINE_SWEEP_RELATIVE_PATH,
) -> dict[str, Any]:
    """Build the compact result without duplicating fine records."""
    comparisons = compare_resolutions(coarse_results, fine_results)
    exact = poiseuille_flux(
        width=FIXED_PARAMETERS["width_max"],
        mu=FIXED_PARAMETERS["mu"],
        G=-FIXED_PARAMETERS["delta_p"] / FIXED_PARAMETERS["L"],
    )
    coarse_by_delta = {record["delta"]: record for record in coarse_results}
    fine_by_delta = {record["delta"]: record for record in fine_results}
    payload = {
        "protocol": {
            "fixed_parameters": dict(FIXED_PARAMETERS),
            "delta_values": list(DELTA_VALUES),
            "coarse_fem_settings": asdict(COARSE_SETTINGS),
            "fine_fem_settings": asdict(FINE_SETTINGS),
            "fine_sweep_source": Path(fine_source).as_posix(),
            "coarse_fem_solves": len(DELTA_VALUES),
            "relative_flux_error_definition": RELATIVE_FLUX_ERROR_DEFINITION,
            "relative_coarse_fine_flux_difference_definition": (
                RELATIVE_MESH_DIFFERENCE_DEFINITION
            ),
            "absolute_flux_error_change_definition": (
                "coarse_relative_flux_error - fine_relative_flux_error"
            ),
        },
        "coarse_results": coarse_results,
        "comparisons": comparisons,
        "straight_channel_checks": {
            "exact_poiseuille_flux": exact,
            "analytical_vs_exact_relative_difference": relative_difference(
                coarse_by_delta[1.0]["analytical_flux"], exact
            ),
            "coarse_fem_vs_exact_relative_difference": relative_difference(
                coarse_by_delta[1.0]["fem_flux"], exact
            ),
            "fine_fem_vs_exact_relative_difference": relative_difference(
                fine_by_delta[1.0]["fem_flux"], exact
            ),
        },
        "summary_diagnostics": {
            "maximum_relative_coarse_fine_flux_difference": _maximum_with_delta(
                comparisons, "relative_coarse_fine_flux_difference"
            ),
            "maximum_coarse_relative_flux_error": _maximum_with_delta(
                coarse_results, "relative_flux_error"
            ),
            "maximum_fine_relative_flux_error": _maximum_with_delta(
                fine_results, "relative_flux_error"
            ),
            "largest_analytical_fem_discrepancy_delta_matches": bool(
                max(coarse_results, key=lambda item: item["relative_flux_error"])["delta"]
                == max(fine_results, key=lambda item: item["relative_flux_error"])["delta"]
                == 0.3
            ),
        },
    }
    validate_payload(payload, fine_results, fine_source)
    return payload


def validate_payload(
    payload: dict[str, Any],
    fine_results: list[dict[str, float]],
    fine_source: Path = FINE_SWEEP_RELATIVE_PATH,
) -> None:
    """Validate the compact payload against the separately loaded fine records."""
    require_finite(payload)
    if "fine_results" in payload:
        raise ValueError("resolution payload must not duplicate fine-sweep results")
    protocol = payload.get("protocol", {})
    if tuple(protocol.get("delta_values", ())) != DELTA_VALUES:
        raise ValueError("payload has incorrect delta values")
    if protocol.get("coarse_fem_settings") != asdict(COARSE_SETTINGS):
        raise ValueError("payload coarse FEM settings mismatch")
    if protocol.get("fine_fem_settings") != asdict(FINE_SETTINGS):
        raise ValueError("payload fine FEM settings mismatch")
    if protocol.get("fine_sweep_source") != Path(fine_source).as_posix():
        raise ValueError("payload fine sweep source mismatch")
    if protocol.get("coarse_fem_solves") != 10:
        raise ValueError("payload must record ten independent coarse solves")
    if protocol.get("relative_coarse_fine_flux_difference_definition") != (
        RELATIVE_MESH_DIFFERENCE_DEFINITION
    ):
        raise ValueError("payload mesh-difference definition mismatch")
    coarse = payload.get("coarse_results")
    comparisons = payload.get("comparisons")
    if not isinstance(coarse, list) or not isinstance(comparisons, list):
        raise ValueError("payload result collections must be lists")
    if len(coarse) != 10 or len(comparisons) != 10 or len(fine_results) != 10:
        raise ValueError("payload must compare ten records at each resolution")
    expected_comparisons = compare_resolutions(coarse, fine_results)
    for actual, expected in zip(comparisons, expected_comparisons):
        if set(actual) != set(expected) or any(
            not same_number(actual[name], value) for name, value in expected.items()
        ):
            raise ValueError("coarse-fine comparison formula mismatch")
    if payload["straight_channel_checks"].get(
        "analytical_vs_exact_relative_difference", math.inf
    ) > 2.0e-14:
        raise ValueError("straight-channel lubrication flux does not equal Poiseuille")


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON atomically without timestamps or NaN values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_study(results_root: Path) -> dict[str, Any]:
    """Read the selected-root fine sweep and independently solve all coarse points."""
    fine_path = fine_sweep_path(results_root)
    if not fine_path.is_file():
        raise FileNotFoundError(f"selected-root fine sweep is missing: {fine_path}")
    fine_payload = json.loads(fine_path.read_text(encoding="utf-8"))
    fine_results = validate_fine_sweep(fine_payload)
    coarse_results = []
    for index, delta in enumerate(DELTA_VALUES, start=1):
        print(f"Resolution check: coarse {index}/10 delta={delta:.1f}", flush=True)
        coarse_results.append(solve_one(delta, COARSE_SETTINGS))
    return build_payload(coarse_results, fine_results, fine_path)


def print_summary(payload: dict[str, Any]) -> None:
    """Print the compact ten-row coarse-fine comparison table."""
    print("delta  coarse_fem_flux   relative_coarse_fine_difference")
    for coarse, comparison in zip(payload["coarse_results"], payload["comparisons"]):
        print(
            f"{coarse['delta']:>4.1f}  {coarse['fem_flux']:.12e}  "
            f"{comparison['relative_coarse_fine_flux_difference']:.12e}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the resolution check and write deterministic JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_result_root_arguments(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing selected-root resolution check",
    )
    args = parser.parse_args(argv)
    results_root = parse_result_root(parser, args)
    output = output_path(results_root)
    if output.exists() and not args.force:
        raise FileExistsError(
            f"analytical/FEM resolution check exists; refusing overwrite: {output}"
        )
    payload = run_study(results_root)
    atomic_json_write(output, payload)
    print_summary(payload)
    print(f"ANALYTICAL_FEM_RESOLUTION_CHECK_WRITTEN {output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
