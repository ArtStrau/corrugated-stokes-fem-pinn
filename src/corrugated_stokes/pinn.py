"""Scaled-coordinate streamfunction PINN and fixed training protocol.

This is the seed-parameterized implementation used by
``scripts/train_pinn.py`` and the read-only validator.  It contains no FEM or
reference-solution supervision; the eight training terms are raw unit-weight
mean-square residuals.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import (
    ADAM_POINTS, ADAM_SCHEDULE, LBFGS, NETWORK, PHYSICAL, PINN_SCALES,
    REFINED_POINTS,
    NetworkConfig, PINNScales, PointConfig,
)

DTYPE = torch.float64
DEVICE = torch.device("cpu")
PROFILE = "corrugated_stokes_scaled_v1"
POINT_HASH_SCHEME = "sha256-canonical-float64-round14-v1"
POINT_HASH_DECIMALS = 14
LOSS_COMPONENT_ORDER = (
    "horizontal", "vertical", "continuity", "wall_u", "wall_v",
    "periodic_u", "periodic_v", "periodic_pi",
)


def dimensionless_height(X: torch.Tensor, scales: PINNScales = PINN_SCALES) -> torch.Tensor:
    """Return the scaled height ``1+a*cos(2*pi*X/ell)``."""
    return 1.0 + scales.relative_amplitude * torch.cos(2.0 * math.pi * X / scales.ell)


def dimensionless_upper_wall(X: torch.Tensor, scales: PINNScales = PINN_SCALES) -> torch.Tensor:
    """Upper scaled wall."""
    return 0.5 * dimensionless_height(X, scales)


def dimensionless_lower_wall(X: torch.Tensor, scales: PINNScales = PINN_SCALES) -> torch.Tensor:
    """Lower scaled wall."""
    return -dimensionless_upper_wall(X, scales)


def equidistant_subset(candidates: np.ndarray, count: int) -> np.ndarray:
    """Select exactly ``floor(linspace(0,N,count,endpoint=False))`` rows."""
    candidates = np.asarray(candidates)
    if count <= 0 or candidates.shape[0] < count:
        raise ValueError("candidate array must contain at least count rows")
    indices = np.floor(np.linspace(0, candidates.shape[0], count, endpoint=False)).astype(int)
    return candidates[indices]


def generate_interior_points(controls: PointConfig, scales: PINNScales = PINN_SCALES) -> dict[str, torch.Tensor]:
    """Generate the fixed deterministic interior and near-wall set."""
    near_count = int(round(controls.interior_points * controls.near_wall_fraction))
    ordinary_count = controls.interior_points - near_count
    ordinary_n = max(32, int(math.ceil(math.sqrt(ordinary_count * 6.0))))
    refined_n = max(64, 2 * ordinary_n)
    global_half_height = 0.5 * (1.0 + scales.relative_amplitude)

    def candidates(n: int) -> tuple[np.ndarray, np.ndarray]:
        X_values = -0.5 * scales.ell + np.arange(n, dtype=np.float64) * scales.ell / n
        Y_values = -global_half_height + (np.arange(n, dtype=np.float64) + 1.0) * (2.0 * global_half_height) / (n + 1.0)
        X, Y = np.meshgrid(X_values, Y_values)
        height = 1.0 + scales.relative_amplitude * np.cos(2.0 * np.pi * X / scales.ell)
        eta = 2.0 * Y / height
        inside = np.abs(eta) < 1.0
        return np.column_stack((X[inside], Y[inside])), np.abs(eta[inside])

    ordinary_candidates, ordinary_eta = candidates(ordinary_n)
    refined_candidates, refined_eta = candidates(refined_n)
    band_start = 1.0 - controls.near_wall_band_fraction
    ordinary = equidistant_subset(ordinary_candidates[ordinary_eta <= band_start], ordinary_count)
    near = equidistant_subset(refined_candidates[refined_eta > band_start], near_count)
    points = np.vstack((ordinary, near))
    region = np.concatenate((np.zeros(ordinary_count, dtype=np.int64), np.ones(near_count, dtype=np.int64)))
    return {
        "x": torch.tensor(points[:, :1], dtype=DTYPE, device=DEVICE),
        "y": torch.tensor(points[:, 1:], dtype=DTYPE, device=DEVICE),
        "region": torch.tensor(region.reshape(-1, 1), dtype=torch.int64),
    }


def generate_wall_points(total_count: int, scales: PINNScales = PINN_SCALES) -> dict[str, torch.Tensor]:
    """Generate endpoint-excluded points on both exact scaled walls."""
    if total_count < 2:
        raise ValueError("at least two wall points are required")
    bottom_count, top_count = total_count // 2, total_count - total_count // 2
    Xb = -0.5 * scales.ell + torch.arange(bottom_count, dtype=DTYPE).reshape(-1, 1) * scales.ell / bottom_count
    Xt = -0.5 * scales.ell + torch.arange(top_count, dtype=DTYPE).reshape(-1, 1) * scales.ell / top_count
    Yb, Yt = dimensionless_lower_wall(Xb, scales), dimensionless_upper_wall(Xt, scales)
    return {"x_bottom": Xb, "y_bottom": Yb, "x_top": Xt, "y_top": Yt,
            "x": torch.cat((Xb, Xt)), "y": torch.cat((Yb, Yt))}


def generate_periodic_pairs(pair_count: int, scales: PINNScales = PINN_SCALES) -> dict[str, torch.Tensor]:
    """Generate matching pairs at the two scaled periodic endpoints."""
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    half_height = 0.5 * (1.0 - scales.relative_amplitude)
    Y = torch.linspace(-half_height, half_height, pair_count + 2, dtype=DTYPE)[1:-1].reshape(-1, 1)
    return {"x_left": torch.full_like(Y, -0.5 * scales.ell),
            "x_right": torch.full_like(Y, 0.5 * scales.ell),
            "y_left": Y.clone(), "y_right": Y.clone()}


def generate_point_bundle(controls: PointConfig, scales: PINNScales = PINN_SCALES):
    """Return interior, wall, and periodic point dictionaries."""
    return (generate_interior_points(controls, scales),
            generate_wall_points(controls.wall_points_total, scales),
            generate_periodic_pairs(controls.periodic_pairs, scales))


def point_bundle_hash(points) -> str:
    """Hash point structure and coordinates after portable hash-only rounding."""
    digest = hashlib.sha256()

    def update(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    update(POINT_HASH_SCHEME.encode("ascii"))
    for group_name, group in zip(("interior", "walls", "periodic"), points):
        for name in sorted(group):
            array = group[name].detach().cpu().numpy()
            metadata = json.dumps(
                [group_name, name, array.dtype.name, list(array.shape)],
                separators=(",", ":"),
            ).encode("ascii")
            update(metadata)
            if np.issubdtype(array.dtype, np.floating):
                if not np.isfinite(array).all():
                    raise ValueError("point bundle contains non-finite coordinates")
                canonical = np.round(array.astype(np.float64), POINT_HASH_DECIMALS)
                canonical[canonical == 0.0] = 0.0
                canonical = np.ascontiguousarray(canonical, dtype="<f8")
            elif np.issubdtype(array.dtype, np.integer):
                dtype = array.dtype.newbyteorder("<")
                canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
            else:
                canonical = np.ascontiguousarray(array)
            update(canonical.tobytes(order="C"))
    return digest.hexdigest().upper()


class ScaledStokesNetwork(nn.Module):
    """Four-by-100 tanh map from raw scaled ``(X,Y)`` to ``(Psi,Pi)``."""

    def __init__(self, controls: NetworkConfig = NETWORK) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        sizes = [2] + [controls.hidden_width] * controls.hidden_layers + [2]
        for index, (input_size, output_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(input_size, output_size, dtype=DTYPE, device=DEVICE))
            if index < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.layers(coordinates)


def configure_determinism(seed: int) -> None:
    """Seed Python, NumPy, and Torch for deterministic CPU execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_network(seed: int, controls: NetworkConfig = NETWORK) -> ScaledStokesNetwork:
    """Build a fresh Xavier-normal, zero-bias network for an explicit seed."""
    configure_determinism(seed)
    model = ScaledStokesNetwork(controls)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def derivative(values: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    """Differentiate batched scalar values while retaining higher derivatives."""
    return torch.autograd.grad(values, coordinate, grad_outputs=torch.ones_like(values), create_graph=True, retain_graph=True)[0]


def kinematics(Psi: torch.Tensor, Pi: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Construct structurally divergence-free scaled velocity."""
    U, V = derivative(Psi, Y), -derivative(Psi, X)
    return {"Psi": Psi, "Pi": Pi, "U": U, "V": V}


def residuals(Psi: torch.Tensor, Pi: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
              scales: PINNScales = PINN_SCALES) -> dict[str, torch.Tensor]:
    """Evaluate the dimensionless Stokes equations."""
    field = kinematics(Psi, Pi, X, Y)
    U, V = field["U"], field["V"]
    U_X, U_Y = derivative(U, X), derivative(U, Y)
    V_X, V_Y = derivative(V, X), derivative(V, Y)
    U_XX, U_YY = derivative(U_X, X), derivative(U_Y, Y)
    V_XX, V_YY = derivative(V_X, X), derivative(V_Y, Y)
    Pi_X, Pi_Y = derivative(Pi, X), derivative(Pi, Y)
    return {**field, "U_X": U_X, "U_Y": U_Y, "V_X": V_X, "V_Y": V_Y,
            "U_XX": U_XX, "U_YY": U_YY, "V_XX": V_XX, "V_YY": V_YY,
            "Pi_X": Pi_X, "Pi_Y": Pi_Y,
            "R_x": 1.0 - Pi_X + scales.alpha * (U_XX + U_YY),
            "R_y": -Pi_Y + scales.alpha * (V_XX + V_YY),
            "R_c": U_X + V_Y}


def network_fields(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                   scales: PINNScales = PINN_SCALES) -> dict[str, torch.Tensor]:
    """Evaluate outputs, velocity, derivatives, and residuals."""
    outputs = model(torch.cat((X, Y), dim=1))
    return residuals(outputs[:, :1], outputs[:, 1:], X, Y, scales)


def network_kinematics(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Evaluate outputs and velocity without second derivatives."""
    outputs = model(torch.cat((X, Y), dim=1))
    return kinematics(outputs[:, :1], outputs[:, 1:], X, Y)


def loss_components(model: nn.Module, interior, walls, periodic,
                    scales: PINNScales = PINN_SCALES) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the exact unweighted sum of the eight MSE terms."""
    def leaves(a, b):
        return a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True)
    Xi, Yi = leaves(interior["x"], interior["y"])
    Xw, Yw = leaves(walls["x"], walls["y"])
    Xl, Yl = leaves(periodic["x_left"], periodic["y_left"])
    Xr, Yr = leaves(periodic["x_right"], periodic["y_right"])
    field, wall = network_fields(model, Xi, Yi, scales), network_kinematics(model, Xw, Yw)
    left, right = network_kinematics(model, Xl, Yl), network_kinematics(model, Xr, Yr)
    components = {
        "horizontal": torch.mean(field["R_x"] ** 2),
        "vertical": torch.mean(field["R_y"] ** 2),
        "continuity": torch.mean(field["R_c"] ** 2),
        "wall_u": torch.mean(wall["U"] ** 2),
        "wall_v": torch.mean(wall["V"] ** 2),
        "periodic_u": torch.mean((left["U"] - right["U"]) ** 2),
        "periodic_v": torch.mean((left["V"] - right["V"]) ** 2),
        "periodic_pi": torch.mean((left["Pi"] - right["Pi"]) ** 2),
    }
    return sum(components.values()), components


def canonical_pressure(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Select the weighted zero-mean physical pressure representative."""
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or np.any(weights <= 0):
        raise ValueError("pressure and positive weights must have identical shapes")
    mean = float(np.sum(weights * values) / np.sum(weights))
    return values - mean, mean


def component_values(components) -> dict[str, float]:
    """Detach scalar components for metadata histories."""
    return {name: float(value.detach()) for name, value in components.items()}


def model_state_hash(model: nn.Module) -> str:
    """Hash parameter names, types, shapes, and values."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest().upper()


def adam_learning_rate(step: int) -> float:
    """Return the fixed Adam learning rate for one-based step ``step``."""
    for start, end, rate in ADAM_SCHEDULE:
        if start <= step <= end:
            return rate
    raise ValueError("Adam step must be in [1,16500]")


def protocol_record(seed: int) -> dict[str, Any]:
    """Return the complete seed-parameterized scientific protocol."""
    return {"profile": PROFILE, "physical_inputs": asdict(PHYSICAL),
            "derived_scales": asdict(PINN_SCALES), "seed": int(seed),
            "dtype": "torch.float64", "device": "cpu", "network": asdict(NETWORK),
            "adam_points": asdict(ADAM_POINTS), "refined_points": asdict(REFINED_POINTS),
            "adam_schedule": [{"start": s, "end": e, "learning_rate": lr} for s, e, lr in ADAM_SCHEDULE],
            "lbfgs": asdict(LBFGS), "loss_components": list(LOSS_COMPONENT_ORDER),
            "loss_weights": {name: 1.0 for name in LOSS_COMPONENT_ORDER},
            "training_reference_inputs": [], "initialization_checkpoint": None}


def protocol_hash(seed: int) -> str:
    """Hash the canonical portable protocol record."""
    raw = json.dumps(protocol_record(seed), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest().upper()
