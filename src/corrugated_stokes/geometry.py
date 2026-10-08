"""Physical and scaled sinusoidal channel geometry and coordinate mappings."""

from __future__ import annotations

import numpy as np

from .config import PHYSICAL, PhysicalConfig


def channel_width(x, physical: PhysicalConfig = PHYSICAL):
    """Full width on the FEM cell ``x in [0,L]``, narrow at ``x=0``."""
    return upper_wall(x, physical) - lower_wall(x, physical)


def upper_wall(x, physical: PhysicalConfig = PHYSICAL):
    """Upper physical wall on the FEM cell."""
    mean_half_width = (physical.width_max + physical.width_min) / 4.0
    modulation_amplitude = (physical.width_max - physical.width_min) / 4.0
    return mean_half_width - modulation_amplitude * np.cos(2.0 * np.pi * np.asarray(x) / physical.L)


def lower_wall(x, physical: PhysicalConfig = PHYSICAL):
    """Lower physical wall on the FEM cell."""
    return -upper_wall(x, physical)


def centered_channel_width(x, physical: PhysicalConfig = PHYSICAL):
    """Full width on the centred PINN physical cell ``[-L/2,L/2]``."""
    x = np.asarray(x)
    return physical.mean_width + physical.amplitude * np.cos(2.0 * np.pi * x / physical.L)


def fem_x_from_centered(x, physical: PhysicalConfig = PHYSICAL):
    """Translate centred PINN coordinates to the FEM cell."""
    return np.asarray(x) + 0.5 * physical.L


def make_mapped_mesh(physical: PhysicalConfig = PHYSICAL, nx: int = 16, ny: int = 8):
    """Create the verified structured mapped quadrilateral topology."""
    from skfem import MeshQuad

    if not isinstance(nx, int) or not isinstance(ny, int) or nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers")
    xv = np.linspace(0.0, physical.L, nx + 1)
    ev = np.linspace(-1.0, 1.0, ny + 1)
    reference = MeshQuad.init_tensor(xv, ev).with_defaults()
    xp, eta = reference.p[0], reference.p[1]
    lo, hi = lower_wall(xp, physical), upper_wall(xp, physical)
    yp = 0.5 * (1.0 - eta) * lo + 0.5 * (1.0 + eta) * hi
    mesh = MeshQuad(np.vstack((xp.copy(), yp)), reference.t.copy()).with_boundaries(reference.boundaries)
    return {"mesh": mesh, "reference_mesh": reference, "x_vertices": xv, "eta_vertices": ev,
            "lower_vertices": lower_wall(xv, physical), "upper_vertices": upper_wall(xv, physical),
            "nx": nx, "ny": ny}


def discrete_wall_bounds(x, mesh_data):
    """Piecewise-linear wall bounds represented by an FEM mesh."""
    x = np.asarray(x, dtype=float)
    return (np.interp(x, mesh_data["x_vertices"], mesh_data["lower_vertices"]),
            np.interp(x, mesh_data["x_vertices"], mesh_data["upper_vertices"]))


def maximum_wall_geometry_error(mesh_data, physical: PhysicalConfig = PHYSICAL, samples_per_cell: int = 21):
    """Maximum exact-versus-discrete wall error."""
    x = np.linspace(0, physical.L, mesh_data["nx"] * samples_per_cell + 1)
    lo, hi = discrete_wall_bounds(x, mesh_data)
    return float(max(np.max(np.abs(lo - lower_wall(x, physical))), np.max(np.abs(hi - upper_wall(x, physical)))))
