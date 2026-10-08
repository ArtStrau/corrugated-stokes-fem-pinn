"""Compute the five-level FEM convergence study on explicit request.

Importing this module does not start the calculation. ``run_convergence()``
performs the five mapped Taylor--Hood solves, and ``main()`` writes their
validated results.
"""

from __future__ import annotations

import argparse
import gc
import os

try:
    from scripts._native_environment import configure_native_library_environment
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _native_environment import configure_native_library_environment

configure_native_library_environment(os.name, os.environ)

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts._result_root import add_result_root_arguments, parse_result_root
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _result_root import add_result_root_arguments, parse_result_root

from corrugated_stokes.config import FEMConfig, PHYSICAL
from corrugated_stokes.fem import (
    make_field_evaluators,
    solve_stokes,
    solve_with_retained_side,
)
from corrugated_stokes.geometry import discrete_wall_bounds, maximum_wall_geometry_error


REFINEMENT_LEVELS = ((16, 8), (32, 16), (64, 32), (128, 64), (256, 128))
INTEGRATION_ORDER = 6
FLUX_QUADRATURE_ORDER = 64
COMPARISON_X_POINTS = 129
COMPARISON_ETA_ORDER = 64
OUTPUT_RELATIVE_PATH = Path("fem/convergence.json")
COMPARISON_MODEL_KEYS = frozenset(
    {"settings", "geometry", "diagnostics", "velocity_at", "pressure_at"}
)
RETAINED_SIDE_MODEL_KEYS = frozenset(
    {
        "K",
        "rhs",
        "periodic_data",
        "periodic_pairs",
        "N_u",
        "N_full",
        "wall_velocity_dofs",
        "basis_u",
        "basis_p",
        "geometry",
        "full_solution",
        "velocity_at",
        "pressure_at",
    }
)


def output_path(results_root: Path) -> Path:
    """Return the convergence output below the selected top-level root."""

    return Path(results_root) / OUTPUT_RELATIVE_PATH


def comparison_grid(model_a, model_b) -> dict[str, np.ndarray]:
    """Build one physical-area-weighted grid inside both discrete domains.

    Each periodic x-section has equal width ``L/nx``.  Gauss--Legendre nodes
    span the intersection of the two discrete cross-sections, so both FEM
    evaluators are sampled at identical physical ``(x,y)`` locations.
    """
    x = np.linspace(0.0, PHYSICAL.L, COMPARISON_X_POINTS, endpoint=False)
    eta, eta_weights = np.polynomial.legendre.leggauss(COMPARISON_ETA_ORDER)
    lo_a, hi_a = discrete_wall_bounds(x, model_a["geometry"])
    lo_b, hi_b = discrete_wall_bounds(x, model_b["geometry"])
    lower = np.maximum(lo_a, lo_b)
    upper = np.minimum(hi_a, hi_b)
    if np.any(upper <= lower):
        raise ValueError("adjacent FEM domains have an empty common cross-section")
    xx, ee = np.meshgrid(x, eta)
    half_width = 0.5 * (upper - lower)
    center = 0.5 * (upper + lower)
    yy = center[None, :] + half_width[None, :] * ee
    weights = eta_weights[:, None] * half_width[None, :] * (PHYSICAL.L / x.size)
    return {"x": xx.ravel(), "y": yy.ravel(), "weights": weights.ravel()}


def weighted_zero_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the physical-area-weighted zero-mean pressure representative."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or np.any(weights <= 0.0):
        raise ValueError("values and positive physical-area weights must match")
    return values - np.sum(weights * values) / np.sum(weights)


def relative_field_changes(model_coarse, model_fine) -> tuple[float, float]:
    """Compare adjacent velocity and canonical pressure on one physical grid."""
    grid = comparison_grid(model_coarse, model_fine)
    coarse_u = model_coarse["velocity_at"](grid["x"], grid["y"])
    fine_u = model_fine["velocity_at"](grid["x"], grid["y"])
    coarse_p = weighted_zero_mean(
        model_coarse["pressure_at"](grid["x"], grid["y"]), grid["weights"]
    )
    fine_p = weighted_zero_mean(
        model_fine["pressure_at"](grid["x"], grid["y"]), grid["weights"]
    )
    weights = grid["weights"]
    velocity_difference = np.sum(weights * np.sum((coarse_u - fine_u) ** 2, axis=0))
    velocity_scale = np.sum(weights * np.sum(fine_u**2, axis=0))
    pressure_difference = np.sum(weights * (coarse_p - fine_p) ** 2)
    pressure_scale = np.sum(weights * fine_p**2)
    tiny = np.finfo(np.float64).tiny
    return (
        float(np.sqrt(velocity_difference / max(velocity_scale, tiny))),
        float(np.sqrt(pressure_difference / max(pressure_scale, tiny))),
    )


def divergence_l2_norm(model) -> float:
    """Integrate the FEM velocity divergence over the physical mapped mesh."""
    field = model["basis_u"].interpolate(model["velocity"])
    divergence = field.grad[0, 0] + field.grad[1, 1]
    return float(np.sqrt(np.sum(divergence**2 * model["basis_u"].dx)))


def level_record(model) -> dict[str, float | int]:
    """Collect required scalar diagnostics for one completed mesh solve."""
    diagnostics = model["diagnostics"]
    return {
        "nx": int(model["settings"].nx),
        "ny": int(model["settings"].ny),
        "mean_flux": float(diagnostics["mean_flux"]),
        "relative_flux_variation": float(diagnostics["relative_flux_variation"]),
        "divergence_l2_norm": divergence_l2_norm(model),
        "maximum_wall_geometry_error": maximum_wall_geometry_error(model["geometry"]),
    }


def successive_record(coarse, fine) -> dict[str, float | int]:
    """Collect adjacent-mesh relative velocity, pressure, and flux changes."""
    velocity_change, pressure_change = relative_field_changes(coarse, fine)
    coarse_flux = float(coarse["diagnostics"]["mean_flux"])
    fine_flux = float(fine["diagnostics"]["mean_flux"])
    return {
        "coarse_nx": int(coarse["settings"].nx),
        "coarse_ny": int(coarse["settings"].ny),
        "fine_nx": int(fine["settings"].nx),
        "fine_ny": int(fine["settings"].ny),
        "relative_velocity_l2_change": velocity_change,
        "relative_zero_mean_pressure_l2_change": pressure_change,
        "relative_mean_flux_change": abs(coarse_flux - fine_flux) / abs(fine_flux),
    }


def lightweight_comparison_model(model) -> dict[str, Any]:
    """Keep only data required for one adjacent-level field comparison."""
    comparison = {
        "settings": model["settings"],
        "geometry": model["geometry"],
        "diagnostics": {"mean_flux": float(model["diagnostics"]["mean_flux"])},
        "velocity_at": model["velocity_at"],
        "pressure_at": model["pressure_at"],
    }
    if set(comparison) != COMPARISON_MODEL_KEYS:
        raise RuntimeError("unexpected lightweight comparison-model contents")
    return comparison


def retained_side_model(model) -> dict[str, Any]:
    """Keep only data required for the finest opposite-side solve and comparison."""
    retained = {name: model[name] for name in RETAINED_SIDE_MODEL_KEYS}
    if set(retained) != RETAINED_SIDE_MODEL_KEYS:
        raise RuntimeError("unexpected retained-side model contents")
    return retained


def retained_side_record(model) -> dict[str, float]:
    """Compare finest-grid left/right periodic reductions after gauge removal."""
    alternate = solve_with_retained_side(model, "right")
    alternate_solution = alternate["full_solution"]
    del alternate
    gc.collect()
    velocity_right, pressure_right = np.split(alternate_solution, [model["N_u"]])
    velocity_at_right, pressure_at_right = make_field_evaluators(
        model["basis_u"], model["basis_p"], velocity_right, pressure_right
    )
    right_model = {
        "geometry": model["geometry"],
        "velocity_at": velocity_at_right,
        "pressure_at": pressure_at_right,
    }
    grid = comparison_grid(model, right_model)
    velocity_left = model["velocity_at"](grid["x"], grid["y"])
    velocity_right_values = velocity_at_right(grid["x"], grid["y"])
    pressure_left = weighted_zero_mean(
        model["pressure_at"](grid["x"], grid["y"]), grid["weights"]
    )
    pressure_right_values = weighted_zero_mean(
        pressure_at_right(grid["x"], grid["y"]), grid["weights"]
    )
    pairs = model["periodic_pairs"]
    left_mismatch = np.max(
        np.abs(model["full_solution"][pairs[:, 0]] - model["full_solution"][pairs[:, 1]])
    )
    right_mismatch = np.max(
        np.abs(
            alternate_solution[pairs[:, 0]]
            - alternate_solution[pairs[:, 1]]
        )
    )
    return {
        "maximum_velocity_difference": float(
            np.max(np.linalg.norm(velocity_left - velocity_right_values, axis=0))
        ),
        "maximum_zero_mean_pressure_difference": float(
            np.max(np.abs(pressure_left - pressure_right_values))
        ),
        "maximum_periodic_dof_mismatch": float(max(left_mismatch, right_mismatch)),
    }


def protocol_record() -> dict[str, Any]:
    """Return the deterministic convergence-study definition."""
    return {
        "physical": asdict(PHYSICAL),
        "refinement_levels": [list(level) for level in REFINEMENT_LEVELS],
        "integration_order": INTEGRATION_ORDER,
        "flux_quadrature_order": FLUX_QUADRATURE_ORDER,
        "comparison_grid": {
            "x_points": COMPARISON_X_POINTS,
            "x_rule": "uniform endpoint-excluded points on [0,L)",
            "eta_quadrature_order": COMPARISON_ETA_ORDER,
            "physical_domain": "intersection of adjacent discrete cross-sections",
            "weighting": "Gauss-Legendre transverse weights times physical half-width times L/x_points",
            "pressure_gauge": "separate physical-area-weighted zero mean on the common grid",
        },
    }


def build_payload(levels, successive_changes, finest_retained_side) -> dict[str, Any]:
    """Build the timestamp-free JSON payload from completed scalar results."""
    return {
        "study": protocol_record(),
        "levels": list(levels),
        "successive_changes": list(successive_changes),
        "finest_retained_side": dict(finest_retained_side),
    }


def _require_finite(value: Any, location: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite(item, f"{location}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite value at {location}")


def validate_payload(payload: dict[str, Any]) -> None:
    """Reject incomplete, non-finite, or protocol-inconsistent output."""
    if payload.get("study") != protocol_record():
        raise ValueError("convergence payload protocol mismatch")
    levels = payload.get("levels")
    changes = payload.get("successive_changes")
    if not isinstance(levels, list) or len(levels) != len(REFINEMENT_LEVELS):
        raise ValueError("convergence payload must contain five levels")
    if [(row.get("nx"), row.get("ny")) for row in levels] != list(REFINEMENT_LEVELS):
        raise ValueError("convergence level order mismatch")
    required_level = {
        "mean_flux", "relative_flux_variation", "divergence_l2_norm",
        "maximum_wall_geometry_error",
    }
    if any(not required_level.issubset(row) for row in levels):
        raise ValueError("convergence level diagnostics are incomplete")
    if not isinstance(changes, list) or len(changes) != len(REFINEMENT_LEVELS) - 1:
        raise ValueError("convergence payload must contain four successive comparisons")
    required_change = {
        "relative_velocity_l2_change", "relative_zero_mean_pressure_l2_change",
        "relative_mean_flux_change",
    }
    if any(not required_change.issubset(row) for row in changes):
        raise ValueError("successive comparison diagnostics are incomplete")
    expected_pairs = [(*coarse, *fine) for coarse, fine in zip(REFINEMENT_LEVELS[:-1], REFINEMENT_LEVELS[1:])]
    actual_pairs = [
        (row.get("coarse_nx"), row.get("coarse_ny"), row.get("fine_nx"), row.get("fine_ny"))
        for row in changes
    ]
    if actual_pairs != expected_pairs:
        raise ValueError("successive comparison order mismatch")
    retained = payload.get("finest_retained_side")
    required_retained = {
        "maximum_velocity_difference", "maximum_zero_mean_pressure_difference",
        "maximum_periodic_dof_mismatch",
    }
    if not isinstance(retained, dict) or not required_retained.issubset(retained):
        raise ValueError("finest retained-side diagnostics are incomplete")
    _require_finite(payload)
    scalar_diagnostics = [
        *(value for row in levels for key, value in row.items() if key not in {"nx", "ny"}),
        *(value for row in changes for key, value in row.items() if key not in {"coarse_nx", "coarse_ny", "fine_nx", "fine_ny"}),
        *retained.values(),
    ]
    if any(float(value) < 0.0 for value in scalar_diagnostics):
        raise ValueError("convergence diagnostics must be nonnegative")


def run_convergence() -> dict[str, Any]:
    """Execute the five mesh solves and build their result."""
    levels = []
    changes = []
    previous_comparison = None
    finest_model = None
    final_index = len(REFINEMENT_LEVELS) - 1
    for index, (nx, ny) in enumerate(REFINEMENT_LEVELS):
        settings = FEMConfig(
            nx=nx,
            ny=ny,
            integration_order=INTEGRATION_ORDER,
            flux_quadrature_order=FLUX_QUADRATURE_ORDER,
        )
        current_model = solve_stokes(
            physical=PHYSICAL, settings=settings, retain="left"
        )
        levels.append(level_record(current_model))
        if previous_comparison is not None:
            changes.append(successive_record(previous_comparison, current_model))
            del previous_comparison
            gc.collect()
        if index < final_index:
            previous_comparison = lightweight_comparison_model(current_model)
            del current_model
            gc.collect()
        else:
            finest_model = retained_side_model(current_model)
            del current_model
            gc.collect()

    payload = build_payload(
        levels, changes, retained_side_record(finest_model)
    )
    validate_payload(payload)
    return payload


def main(argv=None) -> int:
    """Validate an existing result or explicitly compute and write one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute and overwrite an existing validated or invalid result",
    )
    add_result_root_arguments(parser)
    args = parser.parse_args(argv)
    target = output_path(parse_result_root(parser, args))

    if target.exists() and not args.force:
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            validate_payload(existing)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                f"existing convergence result is invalid; refusing to overwrite: {target}"
            ) from error
        print(f"{target.as_posix()} already exists and is valid; nothing to do.")
        return 0

    payload = run_convergence()
    validate_payload(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
