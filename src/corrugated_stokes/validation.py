"""Read-only PINN diagnostics and independent FEM comparison."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import (
    ADAM_POINTS, LBFGS, PHYSICAL, PINN_SCALES, REFINED_POINTS, VALIDATION,
    ValidationConfig,
)
from .geometry import (
    centered_channel_width, channel_width, discrete_wall_bounds,
    fem_x_from_centered,
)
from .pinn import (
    DTYPE, POINT_HASH_SCHEME, build_network, canonical_pressure,
    generate_periodic_pairs, generate_point_bundle, generate_wall_points,
    model_state_hash, network_fields, network_kinematics, point_bundle_hash,
    protocol_hash, protocol_record,
)


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Write finite JSON through a sibling temporary file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def structured_scaled_points(nx: int, eta_order: int) -> dict[str, np.ndarray]:
    """Return an endpoint-excluded X/Gauss-eta grid and area weights."""
    eta, eta_weights = np.polynomial.legendre.leggauss(eta_order)
    x_values = -0.5 * PINN_SCALES.ell + np.arange(nx) * PINN_SCALES.ell / nx
    X, ETA = np.meshgrid(x_values, eta)
    height = 1.0 + PINN_SCALES.relative_amplitude * np.cos(
        2.0 * np.pi * X / PINN_SCALES.ell
    )
    return {
        "X": X.ravel(), "Y": (0.5 * height * ETA).ravel(),
        "weights": (0.5 * height * eta_weights[:, None]).ravel(),
        "shape": np.asarray([eta_order, nx], dtype=np.int64),
        "X_values": x_values,
    }


def deterministic_unseen_points(count: int) -> dict[str, np.ndarray]:
    """Return deterministic irrational-rotation points with width weights."""
    index = np.arange(1, count + 1, dtype=np.float64)
    x_fraction = np.mod(index * ((math.sqrt(5.0) - 1.0) / 2.0), 1.0)
    eta_fraction = np.mod(index * (math.sqrt(2.0) - 1.0), 1.0)
    X = -0.5 * PINN_SCALES.ell + PINN_SCALES.ell * x_fraction
    eta = 2.0 * eta_fraction - 1.0
    height = 1.0 + PINN_SCALES.relative_amplitude * np.cos(
        2.0 * np.pi * X / PINN_SCALES.ell
    )
    return {
        "X": X, "Y": 0.5 * height * eta, "weights": height,
        "shape": np.asarray([count], dtype=np.int64),
    }


def evaluate_fields_chunked(model, X, Y, *, chunk_size: int, full: bool):
    """Evaluate residual or kinematic fields in bounded independent graphs."""
    X, Y = np.asarray(X), np.asarray(Y)
    if X.shape != Y.shape or chunk_size <= 0:
        raise ValueError("coordinate shapes must agree and chunk_size must be positive")
    evaluator = network_fields if full else network_kinematics
    collected: dict[str, list[np.ndarray]] = {}
    for start in range(0, X.size, chunk_size):
        stop = min(start + chunk_size, X.size)
        Xt = torch.tensor(X[start:stop], dtype=DTYPE).reshape(-1, 1).requires_grad_(True)
        Yt = torch.tensor(Y[start:stop], dtype=DTYPE).reshape(-1, 1).requires_grad_(True)
        for name, value in evaluator(model, Xt, Yt).items():
            collected.setdefault(name, []).append(value.detach().numpy().ravel())
    result = {name: np.concatenate(parts) for name, parts in collected.items()}
    if not all(np.isfinite(values).all() for values in result.values()):
        raise FloatingPointError("non-finite validation field")
    return result


def weighted_rms(values, weights) -> float:
    """Return the physical-area-weighted RMS of one diagnostic field."""
    values, weights = np.asarray(values), np.asarray(weights)
    if values.shape != weights.shape or np.any(weights <= 0):
        raise ValueError("values and positive weights must have identical shapes")
    return math.sqrt(float(np.sum(weights * values**2) / np.sum(weights)))


def _require_finite(metrics: dict[str, Any], label: str) -> None:
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError(f"non-finite {label} diagnostic")


def interior_metrics(model, points, *, chunk_size: int = 512) -> dict[str, Any]:
    """Evaluate weighted PDE and pressure metrics on one interior set."""
    field = evaluate_fields_chunked(
        model, points["X"], points["Y"], chunk_size=chunk_size, full=True
    )
    weights = points["weights"]
    pressure, raw_mean = canonical_pressure(field["Pi"], weights)
    metrics = {
        "horizontal_rms": weighted_rms(field["R_x"], weights),
        "vertical_rms": weighted_rms(field["R_y"], weights),
        "continuity_rms": weighted_rms(field["R_c"], weights),
        "continuity_maximum": float(np.max(np.abs(field["R_c"]))),
        "raw_pressure_physical_domain_mean": raw_mean,
        "canonical_pressure_physical_domain_mean": float(
            np.sum(weights * pressure) / np.sum(weights)
        ),
        "all_finite": bool(all(np.isfinite(value).all() for value in field.values())),
    }
    _require_finite(metrics, "interior")
    return {"metrics": metrics}


def boundary_metrics(model, controls: ValidationConfig = VALIDATION):
    """Evaluate wall and paired-periodic diagnostics."""
    walls = generate_wall_points(controls.wall_points_total)
    periodic = generate_periodic_pairs(controls.periodic_pairs)
    wall = evaluate_fields_chunked(
        model, walls["x"].numpy().ravel(), walls["y"].numpy().ravel(),
        chunk_size=controls.chunk_size, full=False,
    )
    left = evaluate_fields_chunked(
        model, periodic["x_left"].numpy().ravel(), periodic["y_left"].numpy().ravel(),
        chunk_size=controls.chunk_size, full=False,
    )
    right = evaluate_fields_chunked(
        model, periodic["x_right"].numpy().ravel(), periodic["y_right"].numpy().ravel(),
        chunk_size=controls.chunk_size, full=False,
    )
    speed = np.hypot(wall["U"], wall["V"])
    metrics = {
        "wall_speed_rms_over_U0": float(np.sqrt(np.mean(speed**2))),
        "wall_speed_maximum_over_U0": float(np.max(speed)),
        "periodic_u_rms": float(np.sqrt(np.mean((left["U"] - right["U"]) ** 2))),
        "periodic_v_rms": float(np.sqrt(np.mean((left["V"] - right["V"]) ** 2))),
        "periodic_pi_rms": float(np.sqrt(np.mean((left["Pi"] - right["Pi"]) ** 2))),
        "all_finite": bool(all(
            np.isfinite(value).all() for field in (wall, left, right)
            for value in field.values()
        )),
    }
    _require_finite(metrics, "boundary")
    return {"metrics": metrics}


def flux_metrics(model, controls: ValidationConfig = VALIDATION):
    """Evaluate the Gauss-64 flux diagnostic."""
    eta, eta_weights = np.polynomial.legendre.leggauss(controls.flux_gauss_order)
    x_values = (
        -0.5 * PINN_SCALES.ell
        + np.arange(controls.flux_sections) * PINN_SCALES.ell / controls.flux_sections
    )
    X, ETA = np.meshgrid(x_values, eta)
    height = 1.0 + PINN_SCALES.relative_amplitude * np.cos(
        2.0 * np.pi * X / PINN_SCALES.ell
    )
    field = evaluate_fields_chunked(
        model, X.ravel(), (0.5 * height * ETA).ravel(),
        chunk_size=controls.chunk_size, full=False,
    )
    U = field["U"].reshape(controls.flux_gauss_order, controls.flux_sections)
    flux = np.sum(eta_weights[:, None] * U * 0.5 * height, axis=0)
    mean_flux = float(np.mean(flux))
    metrics = {
        "mean_dimensionless_flux": mean_flux,
        "mean_physical_flux": PINN_SCALES.Q0 * mean_flux,
        "relative_flux_variation": float(
            np.ptp(flux) / max(abs(mean_flux), np.finfo(float).eps)
        ),
        "all_finite": bool(np.isfinite(flux).all()),
    }
    _require_finite(metrics, "flux")
    return {"metrics": metrics}


def verify_coordinate_translation(sample_count: int = 1025) -> dict[str, Any]:
    """Verify the centred PINN and FEM channel-height conventions."""
    x_pinn = np.linspace(-0.5 * PHYSICAL.L, 0.5 * PHYSICAL.L, sample_count)
    error = float(np.max(np.abs(
        centered_channel_width(x_pinn) - channel_width(fem_x_from_centered(x_pinn))
    )))
    return {"maximum_height_error": error,
            "float64_identity": bool(error <= 8 * np.finfo(float).eps)}


def evaluate_pinn_diagnostics(model, controls: ValidationConfig = VALIDATION):
    """Measure PINN residual, wall, periodicity, and flux diagnostics."""
    before = model_state_hash(model)
    sets = {
        "structured": interior_metrics(
            model, structured_scaled_points(controls.structured_nx, controls.structured_eta_order),
            chunk_size=controls.chunk_size,
        ),
        "finer_structured": interior_metrics(
            model, structured_scaled_points(controls.finer_nx, controls.finer_eta_order),
            chunk_size=controls.chunk_size,
        ),
        "unseen": interior_metrics(
            model, deterministic_unseen_points(controls.unseen_interior),
            chunk_size=controls.chunk_size,
        ),
    }
    result = {
        "interior_sets": sets,
        "boundary": boundary_metrics(model, controls),
        "flux": flux_metrics(model, controls),
        "controls": asdict(controls),
        "model_state_hash_before": before,
        "model_state_hash_after": model_state_hash(model),
    }
    if result["model_state_hash_after"] != before:
        raise RuntimeError("PINN diagnostics changed the model state")
    return result


def compare_with_fem(model, fem_model, controls: ValidationConfig = VALIDATION):
    """Compare PINN fields with an independent mapped FEM solution."""
    alignment = verify_coordinate_translation()
    if not alignment["float64_identity"]:
        raise ValueError("PINN/FEM coordinate translation is not valid in float64")
    before = model_state_hash(model)
    eta, eta_weights = np.polynomial.legendre.leggauss(controls.fem_comparison_eta_order)
    x_values = np.linspace(0, PHYSICAL.L, controls.fem_comparison_nx, endpoint=False)
    X_fem, ETA = np.meshgrid(x_values, eta)
    lower, upper = discrete_wall_bounds(X_fem.ravel(), fem_model["geometry"])
    width = upper - lower
    y = lower + 0.5 * (ETA.ravel() + 1.0) * width
    fields = evaluate_fields_chunked(
        model, (X_fem.ravel() - 0.5 * PHYSICAL.L) / PINN_SCALES.mean_width,
        y / PINN_SCALES.mean_width, chunk_size=controls.chunk_size, full=False,
    )
    pinn_u, pinn_v = PINN_SCALES.U0 * fields["U"], PINN_SCALES.U0 * fields["V"]
    pinn_pressure = PINN_SCALES.P0 * fields["Pi"]
    fem_velocity = fem_model["velocity_at"](X_fem.ravel(), y)
    fem_pressure = fem_model["pressure_at"](X_fem.ravel(), y).ravel()
    weights = (
        np.repeat(eta_weights, controls.fem_comparison_nx) * 0.5 * width
        * PHYSICAL.L / controls.fem_comparison_nx
    )
    velocity_error = math.sqrt(float(np.sum(
        weights * ((pinn_u - fem_velocity[0]) ** 2 + (pinn_v - fem_velocity[1]) ** 2)
    )))
    velocity_scale = math.sqrt(float(np.sum(
        weights * (fem_velocity[0] ** 2 + fem_velocity[1] ** 2)
    )))
    pinn_zero, pinn_mean = canonical_pressure(pinn_pressure, weights)
    fem_zero, fem_mean = canonical_pressure(fem_pressure, weights)
    pressure_error = math.sqrt(float(np.sum(weights * (pinn_zero - fem_zero) ** 2)))
    pressure_scale = math.sqrt(float(np.sum(weights * fem_zero**2)))
    flux_controls = ValidationConfig(
        flux_sections=controls.fem_comparison_nx,
        flux_gauss_order=controls.flux_gauss_order,
        chunk_size=controls.chunk_size,
    )
    pinn_flux = PINN_SCALES.Q0 * flux_metrics(model, flux_controls)["metrics"][
        "mean_dimensionless_flux"
    ]
    fem_flux = float(fem_model["diagnostics"]["mean_flux"])
    metrics = {
        "relative_physical_l2_velocity_error": velocity_error / max(velocity_scale, np.finfo(float).eps),
        "relative_zero_mean_pressure_error": pressure_error / max(pressure_scale, np.finfo(float).eps),
        "relative_mean_flux_error": abs(pinn_flux - fem_flux) / max(abs(fem_flux), np.finfo(float).eps),
        "pinn_mean_flux": pinn_flux, "fem_mean_flux": fem_flux,
        "pinn_pressure_weighted_mean_before_gauge": pinn_mean,
        "fem_pressure_weighted_mean_before_gauge": fem_mean,
        "minimum_discrete_width": float(np.min(width)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite FEM/PINN comparison metric")
    if metrics["minimum_discrete_width"] <= 0:
        raise ValueError("FEM comparison geometry has non-positive width")
    after = model_state_hash(model)
    if after != before:
        raise RuntimeError("FEM/PINN comparison changed the model state")
    return {
        "metrics": metrics, "coordinate_alignment": alignment,
        "controls": asdict(controls), "model_state_hash_before": before,
        "model_state_hash_after": after,
    }


def _point_sets_match(training: dict[str, Any]) -> bool:
    """Regenerate and verify both portable collocation-set hashes."""
    recorded = training.get("adam_point_hash"), training.get("refined_point_hash")
    if training.get("point_hash_scheme") != POINT_HASH_SCHEME:
        return False
    expected = (
        point_bundle_hash(generate_point_bundle(ADAM_POINTS)),
        point_bundle_hash(generate_point_bundle(REFINED_POINTS)),
    )
    return recorded == expected


def verify_training_results(model_path: Path, training_path: Path, seed: int):
    """Load a completed model and verify its scientific training record."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    training = json.loads(Path(training_path).read_text(encoding="utf-8"))
    expected, expected_hash = protocol_record(seed), protocol_hash(seed)
    adam_history, lbfgs_history = training.get("adam_history"), training.get("lbfgs_history")
    closures = training.get("lbfgs_closure_evaluations")
    checks = {
        "seed": checkpoint.get("seed") == training.get("seed") == seed,
        "completed": checkpoint.get("state") == training.get("state") == "TRAINING_COMPLETE",
        "protocol": (
            checkpoint.get("protocol") == expected
            and training.get("protocol") == expected
            and checkpoint.get("protocol_hash") == training.get("protocol_hash") == expected_hash
        ),
        "optimizer_counts": (
            isinstance(adam_history, list)
            and training.get("adam_steps") == len(adam_history) == 16500
            and isinstance(lbfgs_history, list)
            and isinstance(closures, int)
            and 0 < closures <= LBFGS.maximum_closure_evaluations
            and len(lbfgs_history) == closures
        ),
        "point_sets": _point_sets_match(training),
    }
    if not all(checks.values()):
        raise ValueError(f"training result consistency failed: {checks}")
    model = build_network(seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("trained model contains non-finite parameters")
    current_hash = model_state_hash(model)
    recorded = checkpoint.get("model_state_hash"), training.get("model_state_hash")
    if recorded != (current_hash, current_hash):
        raise ValueError("trained model state hash mismatch")
    checks["model_state"] = True
    return model, training, {"checks": checks, "model_state_hash": current_hash}
