"""Scalar tests for the analytical/FEM resolution study."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_analytical_fem_resolution.py"
SPEC = importlib.util.spec_from_file_location("check_analytical_fem_resolution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resolution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolution)


def artificial_results(fine_offset: float = 0.001):
    """Build deterministic scalar fixtures without running FEM."""
    coarse, fine = [], []
    for delta in resolution.DELTA_VALUES:
        physical = resolution.physical_for_delta(delta)
        analytical = resolution.lubrication_flux(physical)
        coarse_flux = analytical * (1.0 - 0.01 * (1.0 - delta))
        fine_flux = coarse_flux * (1.0 + fine_offset)
        coarse.append(
            resolution.make_result_record(physical, analytical, coarse_flux, 1.0e-5)
        )
        fine.append(
            resolution.make_result_record(physical, analytical, fine_flux, 5.0e-6)
        )
    return coarse, fine


def fine_payload(fine):
    """Wrap scalar fixtures in the fine-sweep format."""
    return {
        "study": {
            "fixed_parameters": dict(resolution.FIXED_PARAMETERS),
            "delta_values": list(resolution.DELTA_VALUES),
            "fem_settings": asdict(resolution.FINE_SETTINGS),
            "relative_flux_error_definition": resolution.RELATIVE_FLUX_ERROR_DEFINITION,
        },
        "points": fine,
        "checks": {},
    }


def test_protocol_uses_exact_ten_points_and_declared_meshes():
    assert resolution.DELTA_VALUES == (
        1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
    )
    assert asdict(resolution.COARSE_SETTINGS) == {
        "nx": 128, "ny": 64, "integration_order": 6, "flux_quadrature_order": 64,
    }
    assert asdict(resolution.FINE_SETTINGS) == {
        "nx": 256, "ny": 128, "integration_order": 6, "flux_quadrature_order": 64,
    }
    assert resolution.FINE_SWEEP_RELATIVE_PATH.as_posix() == (
        "analytical_fem/sweep.json"
    )
    assert resolution.OUTPUT_RELATIVE_PATH.as_posix() == (
        "analytical_fem/resolution_check.json"
    )
    assert resolution.fine_sweep_path(Path("reproduction")).as_posix() == (
        "reproduction/analytical_fem/sweep.json"
    )
    assert resolution.output_path(Path("reproduction")).as_posix() == (
        "reproduction/analytical_fem/resolution_check.json"
    )


def test_fine_sweep_format_and_comparison_formula_are_exact():
    coarse, fine = artificial_results()
    assert resolution.validate_fine_sweep(fine_payload(fine)) == fine
    comparisons = resolution.compare_resolutions(coarse, fine)
    for delta, coarse_record, fine_record, comparison in zip(
        resolution.DELTA_VALUES, coarse, fine, comparisons
    ):
        assert coarse_record["width_min"] == pytest.approx(delta * 0.5)
        expected = abs(fine_record["fem_flux"] - coarse_record["fem_flux"]) / abs(
            coarse_record["fem_flux"]
        )
        assert comparison["relative_coarse_fine_flux_difference"] == pytest.approx(
            expected, rel=1.0e-15
        )
    assert resolution.RELATIVE_MESH_DIFFERENCE_DEFINITION == (
        "abs(Q_fine - Q_coarse) / abs(Q_coarse)"
    )


def test_compact_payload_has_all_coarse_points_and_no_fine_copy():
    coarse, fine = artificial_results()
    fine_source = Path("reproduction/analytical_fem/sweep.json")
    payload = resolution.build_payload(coarse, fine, fine_source)
    resolution.validate_payload(payload, fine, fine_source)
    assert payload["protocol"]["coarse_fem_solves"] == 10
    assert payload["protocol"]["fine_sweep_source"] == fine_source.as_posix()
    assert len(payload["coarse_results"]) == 10
    assert len(payload["comparisons"]) == 10
    assert "fine_results" not in payload


def test_stored_mesh_change_uses_128_by_64_coarse_denominator():
    payload = json.loads(
        (ROOT / "results" / "analytical_fem" / "resolution_check.json").read_text(
            encoding="utf-8"
        )
    )
    fine_payload_data = json.loads(
        (ROOT / "results" / "analytical_fem" / "sweep.json").read_text(
            encoding="utf-8"
        )
    )
    fine = resolution.validate_fine_sweep(fine_payload_data)
    resolution.validate_payload(
        payload,
        fine,
        Path("results/analytical_fem/sweep.json"),
    )
    for coarse_record, fine_record, comparison in zip(
        payload["coarse_results"], fine, payload["comparisons"]
    ):
        expected = abs(fine_record["fem_flux"] - coarse_record["fem_flux"]) / abs(
            coarse_record["fem_flux"]
        )
        assert comparison["relative_coarse_fine_flux_difference"] == pytest.approx(
            expected, rel=2.0e-15, abs=2.0e-15
        )


def test_run_study_reads_selected_root_fine_sweep_and_solves_all_ten_coarse_points(monkeypatch):
    coarse, fine = artificial_results()
    import json

    serialized = json.dumps(fine_payload(fine))
    original_is_file = Path.is_file
    original_read_text = Path.read_text

    def fake_is_file(path):
        if path.as_posix().endswith(resolution.FINE_SWEEP_RELATIVE_PATH.as_posix()):
            return True
        return original_is_file(path)

    def fake_read_text(path, *args, **kwargs):
        if path.as_posix().endswith(resolution.FINE_SWEEP_RELATIVE_PATH.as_posix()):
            return serialized
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    calls = []

    def fake_solve(delta, settings):
        calls.append((delta, settings))
        return coarse[resolution.DELTA_VALUES.index(delta)]

    monkeypatch.setattr(resolution, "solve_one", fake_solve)
    payload = resolution.run_study(ROOT)
    assert [delta for delta, _ in calls] == list(resolution.DELTA_VALUES)
    assert all(settings == resolution.COARSE_SETTINGS for _, settings in calls)
    assert len(payload["coarse_results"]) == 10


def test_payload_validation_rejects_nonfinite_or_bad_formula():
    coarse, fine = artificial_results()
    payload = resolution.build_payload(coarse, fine)
    payload["coarse_results"][0]["fem_flux"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        resolution.validate_payload(payload, fine)

    coarse, fine = artificial_results()
    payload = resolution.build_payload(coarse, fine)
    payload["comparisons"][0]["relative_coarse_fine_flux_difference"] += 0.1
    with pytest.raises(ValueError, match="comparison formula"):
        resolution.validate_payload(payload, fine)


def test_script_uses_current_result_format_without_pinn_dependency():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "sweep_" + "v1.json" not in source
    assert "sweep_" + "v2.json" not in source
    assert "resolution_check_" + "v1.json" not in source
    forbidden = "pi" + "nn"
    assert f"corrugated_stokes.{forbidden}" not in source
    assert f"results/{forbidden}" not in source
