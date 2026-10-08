"""Supported exact and lubrication-level analytical references."""

from __future__ import annotations

import numpy as np

from .config import PHYSICAL, PhysicalConfig, closed_form_shape_factor


def poiseuille_velocity(y, width: float, mu: float, G: float):
    """Exact axial velocity in a straight symmetric channel of full width."""
    y = np.asarray(y, dtype=float)
    return G * (0.25 * width**2 - y**2) / (2.0 * mu)


def poiseuille_flux(width: float, mu: float, G: float) -> float:
    """Exact two-dimensional straight-channel volume flux."""
    return G * width**3 / (12.0 * mu)


def lubrication_shape_factor(relative_amplitude: float) -> float:
    """Closed-form inverse-cube shape factor for the sinusoidal width."""
    return closed_form_shape_factor(relative_amplitude)


def lubrication_flux(physical: PhysicalConfig = PHYSICAL) -> float:
    """Leading-order lubrication flux for the corrugated channel."""
    F = lubrication_shape_factor(physical.relative_amplitude)
    return physical.G * physical.mean_width**3 / (12.0 * physical.mu * F)
