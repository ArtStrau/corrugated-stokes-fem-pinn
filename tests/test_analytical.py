"""Tests for exact straight-channel and supported lubrication references."""

import numpy as np

from corrugated_stokes.analytical import lubrication_flux, lubrication_shape_factor, poiseuille_flux, poiseuille_velocity
from corrugated_stokes.config import PHYSICAL, PINN_SCALES


def test_poiseuille_profile_and_flux():
    width, mu, G = 0.5, 2.0, 1.5
    y = np.array([-width / 2, 0, width / 2])
    u = poiseuille_velocity(y, width, mu, G)
    np.testing.assert_allclose(u, [0, G * width**2 / (8 * mu), 0], rtol=0, atol=1e-16)
    assert np.isclose(poiseuille_flux(width, mu, G), G * width**3 / (12 * mu))


def test_lubrication_flux_reproduces_target_scale():
    assert np.isclose(lubrication_shape_factor(PHYSICAL.relative_amplitude), PINN_SCALES.F)
    assert np.isclose(lubrication_flux(), PINN_SCALES.Q0)


def test_straight_limit_lubrication_equals_poiseuille():
    from corrugated_stokes.config import PhysicalConfig
    straight = PhysicalConfig(width_max=0.4, width_min=0.4)
    assert np.isclose(lubrication_flux(straight), poiseuille_flux(0.4, straight.mu, straight.G))
