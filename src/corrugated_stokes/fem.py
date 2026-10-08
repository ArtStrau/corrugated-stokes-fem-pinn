"""Mapped Q2--Q1 Taylor--Hood FEM for periodic corrugated Stokes flow."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from skfem import Basis, ElementQuad1, ElementQuad2, ElementVector, LinearForm, asm, bmat, condense, solve
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace

from .config import FEM_REFERENCE, PHYSICAL, FEMConfig, PhysicalConfig
from .geometry import discrete_wall_bounds, make_mapped_mesh


FIELD_EVALUATION_CHUNK_SIZE = 1024


def make_taylor_hood_spaces(mesh, integration_order: int):
    """Create vector biquadratic velocity and bilinear pressure spaces."""
    return (Basis(mesh, ElementVector(ElementQuad2()), intorder=integration_order),
            Basis(mesh, ElementQuad1(), intorder=integration_order))


def assemble_stokes_system(basis_u, basis_p, mu: float, G: float):
    """Assemble the full mixed Stokes matrix and forcing vector."""
    A = mu * asm(vector_laplace, basis_u)
    B = -asm(divergence, basis_u, basis_p)
    K = bmat([[A, B.T], [B, None]], "csr")

    @LinearForm
    def force(test_velocity, _):
        return G * test_velocity[0]

    rhs = np.concatenate((asm(force, basis_u), basis_p.zeros()))
    return K, rhs, A, B


def pair_boundary_dofs_by_y(left_dofs, right_dofs, dof_locations, label="field", tolerance=1e-12):
    """Pair periodic scalar degrees of freedom by physical y-coordinate."""
    left, right = np.asarray(left_dofs, int), np.asarray(right_dofs, int)
    left, right = left[np.argsort(dof_locations[1, left])], right[np.argsort(dof_locations[1, right])]
    if left.size != right.size:
        raise ValueError(f"{label}: unequal left/right DOF counts")
    mismatch = 0.0 if left.size == 0 else float(np.max(np.abs(dof_locations[1, left] - dof_locations[1, right])))
    if mismatch > tolerance:
        raise ValueError(f"{label}: coordinate mismatch {mismatch:.3e}")
    return left, right, mismatch


def collect_periodic_pairs(basis_u, basis_p, tolerance=1e-12):
    """Collect velocity and pressure pairs in global mixed numbering."""
    lu, ru, eu = pair_boundary_dofs_by_y(basis_u.get_dofs("left").all("u^1"), basis_u.get_dofs("right").all("u^1"), basis_u.doflocs, "u", tolerance)
    lv, rv, ev = pair_boundary_dofs_by_y(basis_u.get_dofs("left").all("u^2"), basis_u.get_dofs("right").all("u^2"), basis_u.doflocs, "v", tolerance)
    lp, rp, ep = pair_boundary_dofs_by_y(basis_p.get_dofs("left").all(), basis_p.get_dofs("right").all(), basis_p.doflocs, "p_tilde", tolerance)
    offset = basis_u.N
    pairs = np.vstack((np.column_stack((lu, ru)), np.column_stack((lv, rv)), np.column_stack((offset + lp, offset + rp)))).astype(int)
    return {"pairs": pairs, "left_u": lu, "right_u": ru, "left_v": lv, "right_v": rv,
            "left_p": lp, "right_p": rp, "max_coordinate_mismatch": max(eu, ev, ep)}


def build_periodic_prolongation(n_full: int, pairs_left_right, retain="left"):
    """Build sparse ``P`` such that the reduced system is ``P.T K P``."""
    if retain not in {"left", "right"}:
        raise ValueError("retain must be 'left' or 'right'")
    pairs = np.asarray(pairs_left_right, dtype=int)
    masters, slaves = (pairs[:, 0], pairs[:, 1]) if retain == "left" else (pairs[:, 1], pairs[:, 0])
    if np.unique(slaves).size != slaves.size or np.intersect1d(masters, slaves).size:
        raise ValueError("invalid overlapping periodic pairs")
    representative = np.arange(n_full, dtype=int)
    representative[slaves] = masters
    retained = np.unique(representative)
    index = {dof: column for column, dof in enumerate(retained)}
    global_to_reduced = np.fromiter((index[dof] for dof in representative), dtype=int, count=n_full)
    P = csr_matrix((np.ones(n_full), (np.arange(n_full), global_to_reduced)), shape=(n_full, retained.size))
    return P, global_to_reduced, retained


def choose_pressure_gauge(periodic_data, velocity_dof_count: int, retain: str):
    """Choose a pressure gauge on the retained periodic side."""
    if retain not in {"left", "right"}:
        raise ValueError("retain must be 'left' or 'right'")
    return velocity_dof_count + periodic_data[f"{retain}_p"][0]


def solve_reduced_periodic_system(K, rhs, P, global_to_reduced, wall_velocity_dofs, pressure_gauge_full):
    """Solve the constrained reduced system and reconstruct the full vector."""
    Kr, rr = (P.T @ K @ P).tocsr(), np.asarray(P.T @ rhs).ravel()
    constrained = np.unique(np.concatenate((np.unique(global_to_reduced[wall_velocity_dofs]), [global_to_reduced[pressure_gauge_full]])))
    reduced = solve(*condense(Kr, rr, x=np.zeros(P.shape[1]), D=constrained))
    return np.asarray(P @ reduced).ravel(), Kr, constrained


def make_field_evaluators(
    basis_u,
    basis_p,
    velocity,
    pressure,
    *,
    chunk_size: int = FIELD_EVALUATION_CHUNK_SIZE,
):
    """Return memory-bounded velocity and pressure interpolation callables."""
    if chunk_size <= 0:
        raise ValueError("field-evaluation chunk size must be positive")
    vi, pi = basis_u.interpolator(velocity), basis_p.interpolator(pressure)

    def evaluate(interpolator, x, y):
        x_values = np.asarray(x, dtype=float)
        y_values = np.asarray(y, dtype=float)
        if x_values.shape != y_values.shape:
            raise ValueError("FEM evaluation coordinates must have matching shapes")
        x_flat, y_flat = x_values.ravel(), y_values.ravel()
        if x_flat.size == 0:
            return np.asarray(interpolator(np.empty((2, 0), dtype=float)))
        pieces = [
            np.asarray(interpolator(np.vstack((x_flat[start:stop], y_flat[start:stop]))))
            for start in range(0, x_flat.size, chunk_size)
            for stop in (min(start + chunk_size, x_flat.size),)
        ]
        return np.concatenate(pieces, axis=-1)

    def velocity_at(x, y):
        return evaluate(vi, x, y)

    def pressure_at(x, y):
        return evaluate(pi, x, y)

    return velocity_at, pressure_at


def cross_sectional_flux(model, x, order: int = 64):
    """Integrate axial velocity across one or more physical cross-sections."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    flux = []
    for section in x:
        lo, hi = discrete_wall_bounds(section, model["geometry"])
        half_width, center = 0.5 * (hi - lo), 0.5 * (hi + lo)
        y = center + half_width * nodes
        u = model["velocity_at"](np.full_like(y, section), y)[0]
        flux.append(half_width * np.dot(weights, u))
    return np.asarray(flux)


def solve_stokes(physical: PhysicalConfig = PHYSICAL, settings: FEMConfig = FEM_REFERENCE, retain="left"):
    """Construct, reduce, solve, and expose one complete corrugated FEM model."""
    geometry = make_mapped_mesh(physical, settings.nx, settings.ny)
    basis_u, basis_p = make_taylor_hood_spaces(geometry["mesh"], settings.integration_order)
    K, rhs, A, B = assemble_stokes_system(basis_u, basis_p, physical.mu, physical.G)
    periodic = collect_periodic_pairs(basis_u, basis_p)
    n_u, n_p = basis_u.N, basis_p.N
    P, mapping, retained = build_periodic_prolongation(n_u + n_p, periodic["pairs"], retain)
    wall = basis_u.get_dofs({"top", "bottom"}).all()
    gauge = choose_pressure_gauge(periodic, n_u, retain)
    solution, Kr, constrained = solve_reduced_periodic_system(K, rhs, P, mapping, wall, gauge)
    velocity, pressure = np.split(solution, [n_u])
    velocity_at, pressure_at = make_field_evaluators(basis_u, basis_p, velocity, pressure)
    model = {"physical": physical, "settings": settings, "geometry": geometry, "basis_u": basis_u,
             "basis_p": basis_p, "K": K, "rhs": rhs, "A": A, "B": B, "N_u": n_u, "N_p": n_p,
             "N_full": n_u + n_p, "periodic_data": periodic, "periodic_pairs": periodic["pairs"],
             "P": P, "global_to_reduced": mapping, "retained_global": retained,
             "wall_velocity_dofs": wall, "pressure_gauge": gauge, "full_solution": solution,
             "velocity": velocity, "pressure": pressure, "velocity_at": velocity_at,
             "pressure_at": pressure_at, "K_reduced": Kr, "constrained_reduced": constrained}
    xflux = np.linspace(0, physical.L, 33)
    flux = cross_sectional_flux(model, xflux, settings.flux_quadrature_order)
    model["diagnostics"] = {"flux_x": xflux, "flux": flux, "mean_flux": float(np.mean(flux)),
                            "relative_flux_variation": float((np.max(flux) - np.min(flux)) / abs(np.mean(flux))),
                            "periodic_coordinate_mismatch": periodic["max_coordinate_mismatch"],
                            "wall_dof_max": float(np.max(np.abs(velocity[wall])))}
    return model


def solve_with_retained_side(model, retain: str):
    """Resolve an assembled model while retaining the opposite periodic side."""
    P, mapping, retained = build_periodic_prolongation(model["N_full"], model["periodic_pairs"], retain)
    gauge = choose_pressure_gauge(model["periodic_data"], model["N_u"], retain)
    solution, Kr, constrained = solve_reduced_periodic_system(model["K"], model["rhs"], P, mapping, model["wall_velocity_dofs"], gauge)
    return {"full_solution": solution, "P": P, "global_to_reduced": mapping,
            "retained_global": retained, "K_reduced": Kr, "constrained": constrained}
