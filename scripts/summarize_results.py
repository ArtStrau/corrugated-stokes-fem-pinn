"""Consolidate validated three-seed PINN and common FEM JSON results.

This script reads completed results only and writes the compact
``summary.json`` below the selected result root. It does
not import solver code, construct models, or recompute any scientific result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any

try:
    from scripts._result_root import add_result_root_arguments, parse_result_root
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _result_root import add_result_root_arguments, parse_result_root


SEEDS = (0, 1, 2)
FEM_FLUX_RELATIVE_TOLERANCE = 1.0e-12
FEM_FLUX_ABSOLUTE_TOLERANCE = 1.0e-15
AGGREGATE_FIELDS = (
    "relative_physical_l2_velocity_error",
    "relative_zero_mean_pressure_error",
    "relative_mean_flux_error",
)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object with a path-specific error."""

    if not path.is_file():
        raise FileNotFoundError(f"required result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required result is not a JSON object: {path}")
    return value


def _finite_number(value: Any, label: str) -> int | float:
    """Return a finite JSON number while rejecting booleans and non-numbers."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return value


def _require_exact_seed_directories(pinn_root: Path) -> None:
    """Require exactly the completed seed directories 0, 1, and 2."""

    discovered: set[int] = set()
    if pinn_root.is_dir():
        for path in pinn_root.iterdir():
            if path.is_dir() and path.name.startswith("seed_"):
                suffix = path.name.removeprefix("seed_")
                if suffix.isdigit():
                    discovered.add(int(suffix))
    if discovered != set(SEEDS):
        raise ValueError(
            f"expected exactly seed directories {list(SEEDS)}, found {sorted(discovered)}"
        )


def _seed_summary(results_root: Path, seed: int, fem_mean_flux: float) -> dict[str, Any]:
    """Extract and validate the requested compact record for one seed."""

    seed_root = results_root / "pinn" / f"seed_{seed}"
    training = _load_json(seed_root / "training.json")
    validation = _load_json(seed_root / "validation.json")
    if training.get("seed") != seed or validation.get("seed") != seed:
        raise ValueError(f"training/validation seed mismatch for seed {seed}")
    if training.get("adam_steps") != 16500:
        raise ValueError(f"seed {seed} Adam step count is not 16500")
    if training.get("lbfgs_closure_evaluations") != 500:
        raise ValueError(f"seed {seed} L-BFGS closure count is not 500")

    fem = validation.get("fem", {})
    fem_metrics = fem.get("metrics", {})
    validation_fem_flux = _finite_number(
        fem_metrics.get("fem_mean_flux"), f"seed {seed} FEM mean flux",
    )
    if not math.isclose(
        float(validation_fem_flux), fem_mean_flux,
        rel_tol=FEM_FLUX_RELATIVE_TOLERANCE,
        abs_tol=FEM_FLUX_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"seed {seed} FEM mean flux disagrees with reference.json")

    physics = validation.get("physics", {})
    flux_metrics = physics.get("flux", {}).get("metrics", {})
    boundary_metrics = physics.get("boundary", {}).get("metrics", {})
    unseen_metrics = (
        physics.get("interior_sets", {}).get("unseen", {}).get("metrics", {})
    )
    extracted = {
        "seed": seed,
        "adam_steps": training["adam_steps"],
        "lbfgs_closure_evaluations": training["lbfgs_closure_evaluations"],
        "final_adam_loss": training.get("final_adam_total"),
        "final_lbfgs_loss": training.get("final_total"),
        "training_seconds": training.get("training_seconds"),
        "relative_physical_l2_velocity_error": fem_metrics.get(
            "relative_physical_l2_velocity_error"
        ),
        "relative_zero_mean_pressure_error": fem_metrics.get(
            "relative_zero_mean_pressure_error"
        ),
        "relative_mean_flux_error": fem_metrics.get("relative_mean_flux_error"),
        "relative_flux_variation": flux_metrics.get("relative_flux_variation"),
        "periodic_u_rms": boundary_metrics.get("periodic_u_rms"),
        "periodic_v_rms": boundary_metrics.get("periodic_v_rms"),
        "periodic_pi_rms": boundary_metrics.get("periodic_pi_rms"),
        "wall_speed_rms_over_U0": boundary_metrics.get("wall_speed_rms_over_U0"),
        "wall_speed_maximum_over_U0": boundary_metrics.get(
            "wall_speed_maximum_over_U0"
        ),
        "unseen_horizontal_rms": unseen_metrics.get("horizontal_rms"),
        "unseen_vertical_rms": unseen_metrics.get("vertical_rms"),
    }
    for name, value in extracted.items():
        _finite_number(value, f"seed {seed} {name}")
    return extracted


def build_summary(results_root: Path) -> dict[str, Any]:
    """Build a validated compact summary from existing JSON results only."""

    results_root = Path(results_root)
    _require_exact_seed_directories(results_root / "pinn")
    fem_record = _load_json(results_root / "fem" / "reference.json")
    physical = fem_record.get("physical", {})
    fem_controls = fem_record.get("fem", {})
    diagnostics = fem_record.get("diagnostics", {})
    fem_summary = {
        "L": physical.get("L"),
        "Re": physical.get("Re"),
        "mu": physical.get("mu"),
        "delta_p": physical.get("delta_p"),
        "width_max": physical.get("width_max"),
        "width_min": physical.get("width_min"),
        "nx": fem_controls.get("nx"),
        "ny": fem_controls.get("ny"),
        "integration_order": fem_controls.get("integration_order"),
        "flux_quadrature_order": fem_controls.get("flux_quadrature_order"),
        "mean_flux": diagnostics.get("mean_flux"),
        "relative_flux_variation": diagnostics.get("relative_flux_variation"),
    }
    for name, value in fem_summary.items():
        _finite_number(value, f"FEM {name}")
    fem_mean_flux = float(fem_summary["mean_flux"])
    seeds = [_seed_summary(results_root, seed, fem_mean_flux) for seed in SEEDS]

    aggregate: dict[str, Any] = {
        "standard_deviation_convention": (
            "population standard deviation; seeds 0, 1, and 2 are the complete reported ensemble"
        )
    }
    for name in AGGREGATE_FIELDS:
        values = [float(record[name]) for record in seeds]
        aggregate[name] = {
            "mean": statistics.fmean(values),
            "standard_deviation": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {"aggregate": aggregate, "fem_reference": fem_summary, "seeds": seeds}


def write_summary(results_root: Path) -> Path:
    """Write deterministic summary JSON atomically and return its path."""

    results_root = Path(results_root)
    output = results_root / "summary.json"
    temporary = output.with_suffix(".json.tmp")
    payload = build_summary(results_root)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def main(argv: list[str] | None = None) -> int:
    """Select one result root and rebuild its derived summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_result_root_arguments(parser)
    args = parser.parse_args(argv)
    output = write_summary(parse_result_root(parser, args))
    print(f"RESULT_SUMMARY_WRITTEN {output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
