"""Shared scalar definitions for the analytical/FEM constriction study."""

from __future__ import annotations

import math
from typing import Any

from corrugated_stokes.config import FEMConfig, PhysicalConfig


DELTA_VALUES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
FIXED_PARAMETERS = {
    "L": 1.0,
    "width_max": 0.5,
    "mu": 1.0,
    "delta_p": -1.0,
    "Re": 0.0,
}
FINE_SETTINGS = FEMConfig(
    nx=256,
    ny=128,
    integration_order=6,
    flux_quadrature_order=64,
)
RELATIVE_FLUX_ERROR_DEFINITION = (
    "abs(analytical_flux - fem_flux) / abs(fem_flux)"
)


def physical_for_delta(delta: float) -> PhysicalConfig:
    """Return the study geometry for one prescribed constriction ratio."""

    value = float(delta)
    if not math.isfinite(value) or value not in DELTA_VALUES:
        raise ValueError(f"delta must be one of {DELTA_VALUES}")
    return PhysicalConfig(
        L=FIXED_PARAMETERS["L"],
        width_max=FIXED_PARAMETERS["width_max"],
        width_min=value * FIXED_PARAMETERS["width_max"],
        mu=FIXED_PARAMETERS["mu"],
        delta_p=FIXED_PARAMETERS["delta_p"],
        Re=FIXED_PARAMETERS["Re"],
    )


def relative_difference(value: float, reference: float) -> float:
    """Return ``abs(value-reference)/abs(reference)`` for finite inputs."""

    value, reference = float(value), float(reference)
    if not math.isfinite(value) or not math.isfinite(reference) or reference == 0.0:
        raise ValueError("relative-difference inputs must be finite and reference nonzero")
    return abs(value - reference) / abs(reference)


def make_result_record(
    physical: PhysicalConfig,
    analytical_flux: float,
    fem_flux: float,
    fem_relative_flux_variation: float,
) -> dict[str, float]:
    """Return one finite scalar study record without FEM field arrays."""

    analytical_flux = float(analytical_flux)
    fem_flux = float(fem_flux)
    fem_relative_flux_variation = float(fem_relative_flux_variation)
    if not all(
        math.isfinite(value)
        for value in (analytical_flux, fem_flux, fem_relative_flux_variation)
    ) or fem_flux == 0.0:
        raise ValueError("flux results must be finite and FEM flux must be nonzero")
    return {
        "delta": float(physical.constriction_ratio),
        "epsilon": float(physical.epsilon),
        "width_max": float(physical.width_max),
        "width_min": float(physical.width_min),
        "mean_width": float(physical.mean_width),
        "relative_amplitude": float(physical.relative_amplitude),
        "analytical_flux": analytical_flux,
        "fem_flux": fem_flux,
        "relative_flux_error": relative_difference(analytical_flux, fem_flux),
        "fem_relative_flux_variation": fem_relative_flux_variation,
    }


def require_finite(value: Any, location: str = "result") -> None:
    """Reject non-finite numbers in a nested study result."""

    if isinstance(value, dict):
        for name, nested in value.items():
            require_finite(nested, f"{location}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            require_finite(nested, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")


def same_number(actual: float, expected: float) -> bool:
    """Compare deterministic scalar metadata at float64 roundoff."""

    return math.isclose(
        float(actual), float(expected), rel_tol=2.0e-15, abs_tol=2.0e-15
    )
