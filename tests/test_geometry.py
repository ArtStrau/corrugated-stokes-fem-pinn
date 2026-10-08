"""Inexpensive tests for the paper-aligned geometry."""

import numpy as np

from corrugated_stokes.config import PHYSICAL, PINN_SCALES
from corrugated_stokes.geometry import (
    centered_channel_width, channel_width, fem_x_from_centered,
    lower_wall, make_mapped_mesh, maximum_wall_geometry_error, upper_wall,
)


def test_notation_identities_and_target_values():
    assert PHYSICAL.width_max == 0.5
    assert PHYSICAL.width_min == 0.1
    assert PHYSICAL.constriction_ratio == 0.2
    assert PHYSICAL.epsilon == 0.4
    assert PHYSICAL.mean_width == 0.3
    assert PHYSICAL.amplitude == 0.2
    assert np.isclose(PHYSICAL.relative_amplitude, 2 / 3)
    assert PINN_SCALES.relative_amplitude == PHYSICAL.relative_amplitude


def test_wall_extrema_positive_width_and_symmetry():
    x = np.linspace(0, PHYSICAL.L, 2001)
    width = channel_width(x)
    assert np.isclose(width.min(), PHYSICAL.width_min)
    assert np.isclose(width.max(), PHYSICAL.width_max)
    assert np.all(width > 0)
    assert np.array_equal(lower_wall(x), -upper_wall(x))


def test_centered_to_fem_translation_is_exact_geometry_identity():
    x = np.linspace(-0.5 * PHYSICAL.L, 0.5 * PHYSICAL.L, 1001)
    np.testing.assert_allclose(centered_channel_width(x), channel_width(fem_x_from_centered(x)), rtol=0, atol=2e-15)


def test_mapped_wall_error_decreases():
    coarse = maximum_wall_geometry_error(make_mapped_mesh(nx=4, ny=2))
    fine = maximum_wall_geometry_error(make_mapped_mesh(nx=8, ny=2))
    assert fine < coarse
