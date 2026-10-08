"""Small scalar tests for the analytical/FEM sweep."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import shutil
import sys

import pytest

from corrugated_stokes.analytical import lubrication_flux, poiseuille_flux


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analytical_fem_sweep", ROOT / "scripts/run_analytical_fem_sweep.py"
)
SWEEP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def _valid_payload():
    points = []
    for delta in SWEEP.DELTA_VALUES:
        physical = SWEEP.physical_for_delta(delta)
        analytical = lubrication_flux(physical)
        points.append(SWEEP.make_result_record(
            physical, analytical, analytical * 1.01, 1.0e-5,
        ))
    # Make the two endpoint consistency checks explicit fixture facts; no FEM
    # solve is performed in routine tests.
    exact = poiseuille_flux(0.5, 1.0, 1.0)
    points[0] = SWEEP.make_result_record(
        SWEEP.physical_for_delta(1.0), exact, exact, 1.0e-5,
    )
    return {
        "study": {
            "delta_values": list(SWEEP.DELTA_VALUES),
            "fem_settings": {
                "nx": 256,
                "ny": 128,
                "integration_order": 6,
                "flux_quadrature_order": 64,
            },
        },
        "points": points,
        "checks": {
            "straight_channel_analytical_vs_exact_relative_error": 0.0,
            "straight_channel_fem_vs_exact_relative_error": 0.0,
        },
    }


def test_prescribed_deltas_and_width_relationships():
    assert SWEEP.DELTA_VALUES == (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
    for delta in SWEEP.DELTA_VALUES:
        physical = SWEEP.physical_for_delta(delta)
        assert physical.width_min == pytest.approx(delta * physical.width_max)
        assert physical.constriction_ratio == pytest.approx(delta)
        assert physical.epsilon == pytest.approx(
            (physical.width_max - physical.width_min) / physical.L
        )
        assert physical.relative_amplitude == pytest.approx(
            (physical.width_max - physical.width_min)
            / (physical.width_max + physical.width_min)
        )


def test_final_fem_settings_and_output_name():
    assert (SWEEP.FEM_SWEEP.nx, SWEEP.FEM_SWEEP.ny) == (256, 128)
    assert SWEEP.FEM_SWEEP.integration_order == 6
    assert SWEEP.FEM_SWEEP.flux_quadrature_order == 64
    assert SWEEP.OUTPUT_RELATIVE_PATH.as_posix() == "analytical_fem/sweep.json"
    assert SWEEP.output_path(Path("reproduction")).as_posix() == (
        "reproduction/analytical_fem/sweep.json"
    )
    assert "FEM_REFERENCE" not in (ROOT / "scripts/run_analytical_fem_sweep.py").read_text(
        encoding="utf-8"
    )
    assert "results/fem/reference.json" not in (
        ROOT / "scripts/run_analytical_fem_sweep.py"
    ).read_text(encoding="utf-8")


def test_relative_flux_error_formula():
    physical = SWEEP.physical_for_delta(0.5)
    point = SWEEP.make_result_record(physical, 0.8, 1.0, 0.01)
    assert point["relative_flux_error"] == pytest.approx(0.2)


def test_straight_lubrication_equals_exact_poiseuille():
    physical = SWEEP.physical_for_delta(1.0)
    assert lubrication_flux(physical) == pytest.approx(
        poiseuille_flux(physical.width_max, physical.mu, physical.G),
        rel=0.0, abs=1.0e-16,
    )


def test_payload_validation_accepts_consistent_scalar_fixture():
    SWEEP.validate_sweep_payload(_valid_payload())


def test_payload_validation_rejects_nonfinite_and_inconsistent_values():
    nonfinite = _valid_payload()
    nonfinite["points"][4]["fem_flux"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        SWEEP.validate_sweep_payload(nonfinite)

    inconsistent = copy.deepcopy(_valid_payload())
    inconsistent["points"][3]["width_min"] += 0.01
    with pytest.raises(ValueError, match="inconsistent width_min"):
        SWEEP.validate_sweep_payload(inconsistent)


def test_existing_sweep_requires_force_before_scientific_work(monkeypatch):
    root = ROOT / "tests/.sweep_overwrite"
    shutil.rmtree(root, ignore_errors=True)
    output = root / "analytical_fem" / "sweep.json"
    try:
        output.parent.mkdir(parents=True)
        output.write_text("existing", encoding="utf-8")
        calls = []
        monkeypatch.setattr(
            SWEEP, "run_study", lambda: calls.append("run") or _valid_payload()
        )
        with pytest.raises(FileExistsError, match="refusing overwrite"):
            SWEEP.write_study(output)
        assert calls == []
        assert SWEEP.write_study(output, force=True) == output
        assert calls == ["run"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
