"""Reduced-cost tests for mapped periodic Q2--Q1 FEM behavior."""

import numpy as np

from corrugated_stokes.analytical import poiseuille_flux, poiseuille_velocity
from corrugated_stokes.config import FEMConfig, PhysicalConfig
from corrugated_stokes.fem import (
    build_periodic_prolongation,
    cross_sectional_flux,
    make_field_evaluators,
    solve_stokes,
    solve_with_retained_side,
)


def test_periodic_prolongation_pairs_both_sides():
    pairs = np.array([[0, 3], [1, 4]])
    left, _, _ = build_periodic_prolongation(6, pairs, "left")
    right, _, _ = build_periodic_prolongation(6, pairs, "right")
    z = np.arange(left.shape[1], dtype=float)
    assert (left @ z)[0] == (left @ z)[3]
    assert (right @ z)[1] == (right @ z)[4]


def test_small_straight_channel_matches_quadratic_and_flux():
    physical = PhysicalConfig(width_max=0.5, width_min=0.5)
    model = solve_stokes(physical, FEMConfig(nx=4, ny=2, integration_order=6, flux_quadrature_order=16))
    x = np.full(9, 0.37)
    y = np.linspace(-0.25, 0.25, 9)
    velocity = model["velocity_at"](x, y)
    np.testing.assert_allclose(velocity[0], poiseuille_velocity(y, 0.5, 1, 1), rtol=0, atol=2e-13)
    np.testing.assert_allclose(velocity[1], 0, rtol=0, atol=2e-13)
    flux = cross_sectional_flux(model, [0.13, 0.71], 20)
    np.testing.assert_allclose(flux, poiseuille_flux(0.5, 1, 1), rtol=0, atol=2e-13)


def test_chunked_field_evaluation_matches_direct_interpolation():
    physical = PhysicalConfig(width_max=0.5, width_min=0.5)
    model = solve_stokes(
        physical,
        FEMConfig(nx=4, ny=2, integration_order=6, flux_quadrature_order=16),
    )
    x = np.linspace(0.05, 0.95, 11)
    y = np.linspace(-0.2, 0.2, 11)
    points = np.vstack((x, y))
    velocity_at, pressure_at = make_field_evaluators(
        model["basis_u"],
        model["basis_p"],
        model["velocity"],
        model["pressure"],
        chunk_size=3,
    )
    expected_velocity = np.asarray(
        model["basis_u"].interpolator(model["velocity"])(points)
    )
    expected_pressure = np.asarray(
        model["basis_p"].interpolator(model["pressure"])(points)
    )
    actual_velocity = velocity_at(x, y)
    actual_pressure = pressure_at(x, y)
    assert actual_velocity.shape == expected_velocity.shape == (2, 11)
    assert actual_pressure.shape == expected_pressure.shape == (11,)
    np.testing.assert_allclose(actual_velocity, expected_velocity, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_pressure, expected_pressure, rtol=0.0, atol=0.0)


def test_small_corrugated_solve_is_finite_no_slip_periodic_and_retain_independent():
    model = solve_stokes(settings=FEMConfig(nx=6, ny=3, integration_order=6, flux_quadrature_order=12))
    assert np.isfinite(model["full_solution"]).all()
    assert model["diagnostics"]["wall_dof_max"] == 0.0
    assert model["periodic_data"]["max_coordinate_mismatch"] < 1e-14
    alternate = solve_with_retained_side(model, "right")
    np.testing.assert_allclose(model["full_solution"][:model["N_u"]], alternate["full_solution"][:model["N_u"]], rtol=1e-10, atol=1e-12)
    assert np.isfinite(model["diagnostics"]["mean_flux"])
