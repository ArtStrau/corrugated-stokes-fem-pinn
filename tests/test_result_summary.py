"""Fast fixture-based tests for the compact result summarizer."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "result_summarizer", ROOT / "scripts/summarize_results.py"
)
SUMMARIZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SUMMARIZER
SPEC.loader.exec_module(SUMMARIZER)


@pytest.fixture
def compact_results():
    """Create and remove small JSON fixtures inside the writable test tree."""

    directory = ROOT / "tests/.result_summary_fixture"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir()
    try:
        fem_root = directory / "results/fem"
        fem_root.mkdir(parents=True)
        fem = {
            "physical": {"L": 1.0, "Re": 0.0, "mu": 1.0, "delta_p": -1.0,
                         "width_max": 0.5, "width_min": 0.1},
            "fem": {"nx": 128, "ny": 64, "integration_order": 6,
                    "flux_quadrature_order": 64},
            "diagnostics": {"mean_flux": 0.25, "relative_flux_variation": 1.0e-5},
        }
        (fem_root / "reference.json").write_text(json.dumps(fem), encoding="utf-8")
        for seed in (0, 1, 2):
            seed_root = directory / "results/pinn" / f"seed_{seed}"
            seed_root.mkdir(parents=True)
            training = {
                "seed": seed, "adam_steps": 16500,
                "lbfgs_closure_evaluations": 500,
                "final_adam_total": 0.1 + seed,
                "final_total": 0.01 + seed,
                "training_seconds": 10.0 + seed,
                "adam_history": ["must not be copied"],
                "lbfgs_history": ["must not be copied"],
            }
            validation = {
                "seed": seed,
                "fem": {"metrics": {
                    "fem_mean_flux": 0.25,
                    "relative_physical_l2_velocity_error": float(seed + 1),
                    "relative_zero_mean_pressure_error": float(2 * (seed + 1)),
                    "relative_mean_flux_error": float(3 * (seed + 1)),
                }},
                "physics": {
                    "flux": {"metrics": {"relative_flux_variation": 0.001 + seed}},
                    "boundary": {"metrics": {
                        "periodic_u_rms": 0.01, "periodic_v_rms": 0.02,
                        "periodic_pi_rms": 0.03, "wall_speed_rms_over_U0": 0.04,
                        "wall_speed_maximum_over_U0": 0.05,
                    }},
                    "interior_sets": {"unseen": {"metrics": {
                        "horizontal_rms": 0.06 + seed,
                        "vertical_rms": 0.07 + seed,
                    }}},
                },
            }
            (seed_root / "training.json").write_text(
                json.dumps(training), encoding="utf-8"
            )
            (seed_root / "validation.json").write_text(
                json.dumps(validation), encoding="utf-8"
            )
        yield directory / "results"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_three_validated_seeds_produce_compact_structure_without_histories(compact_results):
    summary = SUMMARIZER.build_summary(compact_results)
    assert [record["seed"] for record in summary["seeds"]] == [0, 1, 2]
    assert all("classification" not in record for record in summary["seeds"])
    assert [record["unseen_horizontal_rms"] for record in summary["seeds"]] == [0.06, 1.06, 2.06]
    assert [record["unseen_vertical_rms"] for record in summary["seeds"]] == [0.07, 1.07, 2.07]
    assert summary["fem_reference"]["mean_flux"] == 0.25
    serialized = json.dumps(summary, allow_nan=False)
    assert "adam_history" not in serialized
    assert "lbfgs_history" not in serialized
    assert "must not be copied" not in serialized


def test_population_aggregate_values_are_correct(compact_results):
    aggregate = SUMMARIZER.build_summary(compact_results)["aggregate"]
    velocity = aggregate["relative_physical_l2_velocity_error"]
    assert velocity["mean"] == 2.0
    assert velocity["minimum"] == 1.0
    assert velocity["maximum"] == 3.0
    assert velocity["standard_deviation"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert "population standard deviation" in aggregate["standard_deviation_convention"]


def test_missing_required_unseen_metric_is_rejected(compact_results):
    path = compact_results / "pinn/seed_1/validation.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["physics"]["interior_sets"]["unseen"]["metrics"]["horizontal_rms"]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="unseen_horizontal_rms must be a number"):
        SUMMARIZER.build_summary(compact_results)


def test_mismatching_fem_mean_flux_is_rejected(compact_results):
    path = compact_results / "pinn/seed_2/validation.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["fem"]["metrics"]["fem_mean_flux"] = 0.3
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with reference.json"):
        SUMMARIZER.build_summary(compact_results)


def test_missing_or_extra_seed_directory_is_rejected(compact_results):
    shutil.rmtree(compact_results / "pinn/seed_2")
    with pytest.raises(ValueError, match="expected exactly seed directories"):
        SUMMARIZER.build_summary(compact_results)


def test_summary_write_stays_inside_selected_root(compact_results):
    output = SUMMARIZER.write_summary(compact_results)
    assert output == compact_results / "summary.json"
    assert json.loads(output.read_text(encoding="utf-8")) == (
        SUMMARIZER.build_summary(compact_results)
    )


def test_summary_cli_routes_default_results_and_custom_roots(monkeypatch):
    calls = []

    def fake_write(root):
        calls.append(root)
        return root / "summary.json"

    monkeypatch.setattr(SUMMARIZER, "write_summary", fake_write)
    assert SUMMARIZER.main([]) == 0
    assert SUMMARIZER.main(["--canonical"]) == 0
    assert SUMMARIZER.main(["--results-root", "custom/root"]) == 0
    assert calls == [Path("reproduction"), Path("results"), Path("custom/root")]
