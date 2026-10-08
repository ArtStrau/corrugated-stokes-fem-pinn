"""Inexpensive tests for the FEM convergence study."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from corrugated_stokes.config import FEMConfig, PhysicalConfig
from corrugated_stokes.fem import solve_stokes


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_fem_convergence.py"
SPEC = importlib.util.spec_from_file_location("fem_convergence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
convergence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = convergence
SPEC.loader.exec_module(convergence)


def synthetic_payload():
    levels = [
        {
            "nx": nx,
            "ny": ny,
            "mean_flux": 1.0 / nx,
            "relative_flux_variation": 1.0 / nx,
            "divergence_l2_norm": 1.0 / nx,
            "maximum_wall_geometry_error": 1.0 / nx,
        }
        for nx, ny in convergence.REFINEMENT_LEVELS
    ]
    changes = [
        {
            "coarse_nx": coarse[0],
            "coarse_ny": coarse[1],
            "fine_nx": fine[0],
            "fine_ny": fine[1],
            "relative_velocity_l2_change": 0.1,
            "relative_zero_mean_pressure_l2_change": 0.1,
            "relative_mean_flux_change": 0.1,
        }
        for coarse, fine in zip(convergence.REFINEMENT_LEVELS[:-1], convergence.REFINEMENT_LEVELS[1:])
    ]
    return convergence.build_payload(
        levels,
        changes,
        {
            "maximum_velocity_difference": 1.0e-12,
            "maximum_zero_mean_pressure_difference": 1.0e-12,
            "maximum_periodic_dof_mismatch": 0.0,
        },
    )


class MemoryPath:
    """Minimal in-memory Path double for output-admission tests."""

    def __init__(self, content=None):
        self.content = content
        self.parent = self

    def exists(self):
        return self.content is not None

    def read_text(self, encoding=None):
        return self.content

    def write_text(self, content, encoding=None):
        self.content = content

    def mkdir(self, parents=False, exist_ok=False):
        return None

    def as_posix(self):
        return "reproduction/fem/convergence.json"

    def __str__(self):
        return self.as_posix()


def test_exact_protocol_and_side_effect_free_import():
    assert convergence.REFINEMENT_LEVELS == (
        (16, 8), (32, 16), (64, 32), (128, 64), (256, 128)
    )
    assert convergence.INTEGRATION_ORDER == 6
    assert convergence.FLUX_QUADRATURE_ORDER == 64
    assert convergence.OUTPUT_RELATIVE_PATH.as_posix() == "fem/convergence.json"
    assert convergence.output_path(Path("reproduction")).as_posix() == (
        "reproduction/fem/convergence.json"
    )
    assert "schema_" + "version" not in convergence.build_payload([], [], {})
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "corrugated_stokes.pinn" not in source


def test_synthetic_payload_validation_and_rejection():
    payload = synthetic_payload()
    convergence.validate_payload(payload)
    broken = copy.deepcopy(payload)
    broken["levels"][2]["mean_flux"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        convergence.validate_payload(broken)
    wrong_protocol = copy.deepcopy(payload)
    wrong_protocol["study"]["flux_quadrature_order"] = 48
    with pytest.raises(ValueError, match="protocol mismatch"):
        convergence.validate_payload(wrong_protocol)


def test_common_grid_weighting_and_pressure_gauge_are_physical():
    geometry = {
        "x_vertices": np.array([0.0, 1.0]),
        "lower_vertices": np.array([-0.25, -0.25]),
        "upper_vertices": np.array([0.25, 0.25]),
    }

    def model(velocity_scale, pressure_offset):
        return {
            "geometry": geometry,
            "velocity_at": lambda x, y: np.vstack((velocity_scale * np.ones_like(x), np.zeros_like(x))),
            "pressure_at": lambda x, y: 2.0 * y + pressure_offset,
        }

    velocity, pressure = convergence.relative_field_changes(model(0.9, 5.0), model(1.0, -7.0))
    assert velocity == pytest.approx(0.1, rel=1.0e-14)
    assert pressure == pytest.approx(0.0, abs=1.0e-13)
    grid = convergence.comparison_grid(model(1.0, 0.0), model(1.0, 0.0))
    assert np.sum(grid["weights"]) == pytest.approx(0.5, rel=1.0e-14)


def test_divergence_diagnostic_on_inexpensive_straight_fixture():
    physical = PhysicalConfig(width_max=0.5, width_min=0.5)
    model = solve_stokes(
        physical,
        FEMConfig(nx=4, ny=2, integration_order=6, flux_quadrature_order=16),
    )
    assert convergence.divergence_l2_norm(model) < 1.0e-11


def test_convergence_retains_only_lightweight_data_across_solves(monkeypatch):
    heavy_keys = {
        "K", "rhs", "A", "B", "K_reduced", "P", "basis_u", "basis_p",
        "full_solution", "global_to_reduced", "retained_global",
    }
    comparisons = []
    finest_keys = []

    def fake_solve(*, physical, settings, retain):
        assert physical == convergence.PHYSICAL
        assert retain == "left"
        model = {
            "settings": settings,
            "geometry": object(),
            "diagnostics": {"mean_flux": 1.0},
            "velocity_at": object(),
            "pressure_at": object(),
            "periodic_data": object(),
            "periodic_pairs": object(),
            "N_u": 1,
            "N_full": 2,
            "wall_velocity_dofs": object(),
        }
        model.update({name: object() for name in heavy_keys})
        return model

    def fake_level(model):
        return {
            "nx": model["settings"].nx,
            "ny": model["settings"].ny,
            "mean_flux": 1.0,
            "relative_flux_variation": 0.0,
            "divergence_l2_norm": 0.0,
            "maximum_wall_geometry_error": 0.0,
        }

    def fake_change(coarse, fine):
        comparisons.append((set(coarse), set(fine)))
        assert set(coarse) == convergence.COMPARISON_MODEL_KEYS
        assert heavy_keys.isdisjoint(coarse)
        assert heavy_keys.issubset(fine)
        return {
            "coarse_nx": coarse["settings"].nx,
            "coarse_ny": coarse["settings"].ny,
            "fine_nx": fine["settings"].nx,
            "fine_ny": fine["settings"].ny,
            "relative_velocity_l2_change": 0.0,
            "relative_zero_mean_pressure_l2_change": 0.0,
            "relative_mean_flux_change": 0.0,
        }

    monkeypatch.setattr(convergence, "solve_stokes", fake_solve)
    monkeypatch.setattr(convergence, "level_record", fake_level)
    monkeypatch.setattr(convergence, "successive_record", fake_change)
    monkeypatch.setattr(
        convergence,
        "retained_side_record",
        lambda model: (
            finest_keys.append(set(model))
            or {
                "maximum_velocity_difference": 0.0,
                "maximum_zero_mean_pressure_difference": 0.0,
                "maximum_periodic_dof_mismatch": 0.0,
            }
        ),
    )

    payload = convergence.run_convergence()
    assert len(payload["levels"]) == 5
    assert len(payload["successive_changes"]) == 4
    assert len(comparisons) == 4
    assert finest_keys == [set(convergence.RETAINED_SIDE_MODEL_KEYS)]
    assert {"A", "B", "K_reduced", "P", "global_to_reduced"}.isdisjoint(
        finest_keys[0]
    )


def test_valid_existing_result_exits_without_scientific_solve(monkeypatch, capsys):
    output = MemoryPath(json.dumps(synthetic_payload()))
    monkeypatch.setattr(convergence, "output_path", lambda root: output)

    def forbidden_solve():
        raise AssertionError("scientific solve must not run for a valid existing result")

    monkeypatch.setattr(convergence, "run_convergence", forbidden_solve)
    assert convergence.main([]) == 0
    assert "already exists and is valid; nothing to do." in capsys.readouterr().out


def test_invalid_existing_result_fails_without_overwrite(monkeypatch):
    original = "{not valid json"
    output = MemoryPath(original)
    monkeypatch.setattr(convergence, "output_path", lambda root: output)

    def forbidden_solve():
        raise AssertionError("scientific solve must not run for an invalid existing result")

    monkeypatch.setattr(convergence, "run_convergence", forbidden_solve)
    with pytest.raises(ValueError, match="invalid; refusing to overwrite"):
        convergence.main([])
    assert output.content == original


def test_force_recomputes_and_overwrites_with_stubbed_payload(monkeypatch):
    output = MemoryPath("old content")
    payload = synthetic_payload()
    calls = []
    monkeypatch.setattr(convergence, "output_path", lambda root: output)
    monkeypatch.setattr(convergence, "run_convergence", lambda: calls.append("run") or payload)

    assert convergence.main(["--force"]) == 0
    assert calls == ["run"]
    assert json.loads(output.content) == payload
