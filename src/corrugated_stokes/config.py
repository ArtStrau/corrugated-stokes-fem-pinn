"""Validated configuration for the corrugated Stokes FEM/PINN study.

The code/paper notation is intentionally explicit: ``width_max`` is
``Delta Omega``, ``width_min`` is ``Delta omega``, ``constriction_ratio`` is
``delta``, and ``epsilon`` is the PRL parameter ``(Delta Omega-Delta omega)/L``.
The scaled PINN uses the distinct ``relative_amplitude = A/H0``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhysicalConfig:
    """Physical inputs for one periodic, symmetric corrugated channel."""

    L: float = 1.0
    width_max: float = 0.5
    width_min: float = 0.1
    mu: float = 1.0
    delta_p: float = -1.0
    Re: float = 0.0

    def __post_init__(self) -> None:
        values = (self.L, self.width_max, self.width_min, self.mu, self.delta_p, self.Re)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all physical parameters must be finite")
        if self.L <= 0 or self.width_min <= 0 or self.width_max < self.width_min:
            raise ValueError("require L > 0 and 0 < width_min <= width_max")
        if self.mu <= 0:
            raise ValueError("mu must be positive")
        if self.G <= 0:
            raise ValueError("this pressure-driven study requires G=-delta_p/L > 0")
        if self.Re != 0.0:
            raise ValueError("corrugated_stokes implements the Re=0 Stokes problem")

    @property
    def G(self) -> float:
        """Positive pressure-gradient magnitude ``-delta_p/L``."""
        return -self.delta_p / self.L

    @property
    def constriction_ratio(self) -> float:
        """Paper constriction ratio ``delta = width_min/width_max``."""
        return self.width_min / self.width_max

    @property
    def epsilon(self) -> float:
        """PRL geometry parameter ``epsilon=(width_max-width_min)/L``."""
        return (self.width_max - self.width_min) / self.L

    @property
    def mean_width(self) -> float:
        """Mean full width ``H0``."""
        return 0.5 * (self.width_max + self.width_min)

    @property
    def amplitude(self) -> float:
        """Full-width sinusoidal amplitude ``A``."""
        return 0.5 * (self.width_max - self.width_min)

    @property
    def relative_amplitude(self) -> float:
        """Scaled-PINN amplitude ``a=A/H0`` (not the PRL epsilon)."""
        return self.amplitude / self.mean_width


def closed_form_shape_factor(relative_amplitude: float) -> float:
    """Return ``H0**3 * mean(h**-3)`` for a sinusoidal full width."""
    a = float(relative_amplitude)
    if not math.isfinite(a) or not 0.0 <= a < 1.0:
        raise ValueError("relative_amplitude must be finite and in [0, 1)")
    return (2.0 + a * a) / (2.0 * (1.0 - a * a) ** 2.5)


@dataclass(frozen=True)
class PINNScales:
    """Dimensionless coordinates and physical reconstruction scales."""

    mean_width: float
    relative_amplitude: float
    ell: float
    F: float
    Q0: float
    U0: float
    P0: float
    alpha: float

    @classmethod
    def from_physical(cls, physical: PhysicalConfig) -> "PINNScales":
        """Derive the scaled formulation without fitted data."""
        H0 = physical.mean_width
        a = physical.relative_amplitude
        ell = physical.L / H0
        F = closed_form_shape_factor(a)
        Q0 = physical.G * H0**3 / (12.0 * physical.mu * F)
        U0 = Q0 / H0
        P0 = physical.G * H0
        alpha = physical.mu * Q0 / (physical.G * H0**3)
        result = cls(H0, a, ell, F, Q0, U0, P0, alpha)
        if not all(math.isfinite(v) and v > 0 for v in (H0, ell, F, Q0, U0, P0, alpha)):
            raise ValueError("derived PINN scales must be finite and positive")
        return result


@dataclass(frozen=True)
class NetworkConfig:
    """Fixed network capacity."""

    hidden_layers: int = 4
    hidden_width: int = 100


@dataclass(frozen=True)
class PointConfig:
    """Deterministic collocation counts and near-wall allocation."""

    interior_points: int
    wall_points_total: int
    periodic_pairs: int
    near_wall_fraction: float = 0.25
    near_wall_band_fraction: float = 0.10

    def __post_init__(self) -> None:
        if min(self.interior_points, self.wall_points_total, self.periodic_pairs) <= 0:
            raise ValueError("all point counts must be positive")
        if not 0 < self.near_wall_fraction < 1 or not 0 < self.near_wall_band_fraction < 0.5:
            raise ValueError("invalid near-wall fractions")


@dataclass(frozen=True)
class LBFGSConfig:
    """Fixed scientific L-BFGS controls."""

    learning_rate: float = 1.0
    maximum_closure_evaluations: int = 500
    history_size: int = 100
    line_search: str = "strong_wolfe"
    tolerance_grad: float = 1.0e-12
    tolerance_change: float = 1.0e-14


@dataclass(frozen=True)
class ValidationConfig:
    """Deterministic read-only validation grids."""

    structured_nx: int = 129
    structured_eta_order: int = 64
    finer_nx: int = 257
    finer_eta_order: int = 96
    unseen_interior: int = 8192
    wall_points_total: int = 4096
    periodic_pairs: int = 2049
    flux_sections: int = 513
    flux_gauss_order: int = 64
    fem_comparison_nx: int = 257
    fem_comparison_eta_order: int = 96
    chunk_size: int = 512


@dataclass(frozen=True)
class FEMConfig:
    """FEM mesh and quadrature controls."""

    nx: int = 256
    ny: int = 128
    integration_order: int = 6
    flux_quadrature_order: int = 64


PHYSICAL = PhysicalConfig()
PINN_SCALES = PINNScales.from_physical(PHYSICAL)
NETWORK = NetworkConfig()
ADAM_POINTS = PointConfig(1024, 256, 128)
REFINED_POINTS = PointConfig(4096, 1024, 512)
LBFGS = LBFGSConfig()
VALIDATION = ValidationConfig()
FEM_REFERENCE = FEMConfig()

ADAM_SCHEDULE = (
    (1, 3000, 5.0e-4),
    (3001, 9000, 1.0e-4),
    (9001, 16000, 1.0e-5),
    (16001, 16500, 1.0e-6),
)
RECOVERY_INTERVAL = 1500
