"""Inexpensive tests for read-only PINN diagnostics and FEM comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from corrugated_stokes.config import (
    ADAM_POINTS, REFINED_POINTS, VALIDATION, ValidationConfig,
)
from corrugated_stokes.pinn import (
    POINT_HASH_SCHEME, build_network, generate_point_bundle, model_state_hash,
    network_kinematics, point_bundle_hash,
)
from corrugated_stokes import validation


ROOT = Path(__file__).resolve().parents[1]


def test_validation_configuration_is_fixed():
    assert VALIDATION.chunk_size == 512


def test_weighted_rms_uses_physical_weights():
    values = np.array([1.0, 3.0])
    weights = np.array([1.0, 4.0])
    expected = np.sqrt((1.0 + 4.0 * 9.0) / 5.0)
    assert validation.weighted_rms(values, weights) == pytest.approx(expected)


def test_chunked_evaluator_matches_direct_kinematics():
    model = build_network(1)
    X = np.linspace(-0.3, 0.3, 7)
    Y = np.linspace(-0.1, 0.1, 7)
    chunked = validation.evaluate_fields_chunked(
        model, X, Y, chunk_size=3, full=False
    )
    Xt = torch.tensor(X, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    Yt = torch.tensor(Y, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    direct = network_kinematics(model, Xt, Yt)
    for name, values in chunked.items():
        np.testing.assert_allclose(
            values, direct[name].detach().numpy().ravel(), rtol=0, atol=2e-15
        )


def test_validation_module_is_read_only_and_does_not_construct_optimizers():
    source = (ROOT / "src/corrugated_stokes/validation.py").read_text(encoding="utf-8")
    assert "torch.optim" not in source
    model = build_network(1)
    before = model_state_hash(model)
    controls = ValidationConfig(
        wall_points_total=16,
        periodic_pairs=7,
        flux_sections=9,
        flux_gauss_order=4,
        chunk_size=8,
    )
    validation.boundary_metrics(model, controls)
    validation.flux_metrics(model, controls)
    assert model_state_hash(model) == before


def test_structured_and_unseen_point_sets():
    structured = validation.structured_scaled_points(7, 6)
    assert structured["X"].shape == structured["Y"].shape == (42,)
    assert structured["weights"].shape == (42,)
    assert np.all(structured["weights"] > 0)
    unseen = validation.deterministic_unseen_points(19)
    assert unseen["X"].shape == unseen["Y"].shape == (19,)
    assert np.all(unseen["weights"] > 0)


def test_coordinate_translation_payload_is_json_serializable():
    result = validation.verify_coordinate_translation()
    assert type(result["float64_identity"]) is bool
    assert result["float64_identity"] is True
    json.dumps(result, allow_nan=False)


def test_fem_comparison_uses_coordinate_audit_and_preserves_model(monkeypatch):
    audit = {"maximum_height_error": 0.0, "float64_identity": True}
    monkeypatch.setattr(validation, "verify_coordinate_translation", lambda: audit)
    monkeypatch.setattr(validation, "model_state_hash", lambda model: "UNCHANGED")
    monkeypatch.setattr(
        validation,
        "discrete_wall_bounds",
        lambda x, geometry: (np.zeros_like(x), np.ones_like(x)),
    )
    monkeypatch.setattr(
        validation,
        "evaluate_fields_chunked",
        lambda model, X, Y, **kwargs: {
            "U": np.zeros_like(np.asarray(X)),
            "V": np.zeros_like(np.asarray(X)),
            "Pi": np.zeros_like(np.asarray(X)),
        },
    )
    monkeypatch.setattr(
        validation,
        "flux_metrics",
        lambda *args, **kwargs: {"metrics": {"mean_dimensionless_flux": 0.0}},
    )
    fem_model = {
        "geometry": object(),
        "velocity_at": lambda x, y: np.zeros((2, np.asarray(x).size)),
        "pressure_at": lambda x, y: np.zeros(np.asarray(x).size),
        "diagnostics": {"mean_flux": 1.0},
    }
    controls = ValidationConfig(
        fem_comparison_nx=2,
        fem_comparison_eta_order=2,
        flux_gauss_order=2,
        chunk_size=2,
    )
    result = validation.compare_with_fem(object(), fem_model, controls)
    assert result["coordinate_alignment"] == audit
    assert result["model_state_hash_before"] == result["model_state_hash_after"]


def test_existing_completed_model_loads_without_current_source_gate():
    seed_root = ROOT / "results/pinn/seed_1"
    model, training, consistency = validation.verify_training_results(
        seed_root / "model.pt", seed_root / "training.json", 1
    )
    assert training["seed"] == 1
    assert consistency["checks"]["model_state"] is True
    assert model_state_hash(model) == training["model_state_hash"]


def test_existing_completed_model_rejects_wrong_seed():
    seed_root = ROOT / "results/pinn/seed_1"
    with pytest.raises(ValueError, match="training result consistency"):
        validation.verify_training_results(
            seed_root / "model.pt", seed_root / "training.json", 0
        )


def test_new_point_hash_scheme_rejects_an_incorrect_hash(tmp_path):
    seed_root = ROOT / "results/pinn/seed_1"
    training = json.loads((seed_root / "training.json").read_text(encoding="utf-8"))
    training["point_hash_scheme"] = POINT_HASH_SCHEME
    training["adam_point_hash"] = point_bundle_hash(generate_point_bundle(ADAM_POINTS))
    training["refined_point_hash"] = point_bundle_hash(
        generate_point_bundle(REFINED_POINTS)
    )
    candidate = tmp_path / "training.json"
    candidate.write_text(json.dumps(training), encoding="utf-8")
    _, _, consistency = validation.verify_training_results(
        seed_root / "model.pt", candidate, 1
    )
    assert consistency["checks"]["point_sets"] is True

    training["adam_point_hash"] = "0" * 64
    candidate.write_text(json.dumps(training), encoding="utf-8")
    with pytest.raises(ValueError, match="training result consistency"):
        validation.verify_training_results(seed_root / "model.pt", candidate, 1)


def test_completed_training_requires_current_exact_format(tmp_path):
    seed_root = ROOT / "results/pinn/seed_1"
    original = json.loads((seed_root / "training.json").read_text(encoding="utf-8"))
    candidate = tmp_path / "training.json"

    missing_scheme = dict(original)
    missing_scheme.pop("point_hash_scheme")
    candidate.write_text(json.dumps(missing_scheme), encoding="utf-8")
    with pytest.raises(ValueError, match="training result consistency"):
        validation.verify_training_results(seed_root / "model.pt", candidate, 1)

    extra_protocol = json.loads(json.dumps(original))
    extra_protocol["protocol"]["obsolete_metadata"] = True
    candidate.write_text(json.dumps(extra_protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="training result consistency"):
        validation.verify_training_results(seed_root / "model.pt", candidate, 1)

    missing_training_hash = dict(original)
    missing_training_hash.pop("model_state_hash")
    candidate.write_text(json.dumps(missing_training_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="model state hash mismatch"):
        validation.verify_training_results(seed_root / "model.pt", candidate, 1)


def test_completed_checkpoint_requires_model_state_hash(tmp_path):
    seed_root = ROOT / "results/pinn/seed_1"
    checkpoint = torch.load(seed_root / "model.pt", map_location="cpu", weights_only=False)
    checkpoint.pop("model_state_hash")
    candidate = tmp_path / "model.pt"
    torch.save(checkpoint, candidate)
    with pytest.raises(ValueError, match="model state hash mismatch"):
        validation.verify_training_results(candidate, seed_root / "training.json", 1)


def test_atomic_validation_json_write_leaves_no_partial_file(tmp_path):
    output = tmp_path / "validation.json"
    validation.atomic_json_write(output, {"seed": 1, "training_performed": False})
    assert json.loads(output.read_text(encoding="utf-8"))["seed"] == 1
    assert not output.with_suffix(".json.tmp").exists()
