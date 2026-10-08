"""Inexpensive tests for shared scientific result-root selection."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

from scripts import _result_root as roots


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    roots.add_result_root_arguments(parser)
    return parser


def test_shared_root_selection_and_mutual_exclusion():
    parser = _parser()
    assert roots.resolve_result_root(parser.parse_args([])) == Path("reproduction")
    assert roots.resolve_result_root(parser.parse_args(["--canonical"])) == Path("results")
    custom = Path("custom/output")
    assert roots.resolve_result_root(
        parser.parse_args(["--results-root", str(custom)])
    ) == custom
    with pytest.raises(SystemExit):
        parser.parse_args(["--canonical", "--results-root", "custom"])


def test_custom_root_cannot_bypass_explicit_results_selector():
    parser = _parser()
    for value in ("results", "results/nested", str(ROOT / "results" / "nested")):
        with pytest.raises(ValueError, match="use --canonical"):
            roots.resolve_result_root(parser.parse_args(["--results-root", value]))
    outside = ROOT / "tests/.external_result_root/results"
    assert roots.resolve_result_root(
        parser.parse_args(["--results-root", str(outside)])
    ) == outside


def test_all_six_scripts_append_their_stage_subdirectory():
    fem_reference = _load("root_fem_reference", "scripts/generate_fem_reference.py")
    fem_convergence = _load("root_fem_convergence", "scripts/run_fem_convergence.py")
    sweep = _load("root_analytical_sweep", "scripts/run_analytical_fem_sweep.py")
    resolution = _load("root_analytical_resolution", "scripts/check_analytical_fem_resolution.py")
    training = _load("root_pinn_training", "scripts/train_pinn.py")
    validation = _load("root_pinn_validation", "scripts/validate_pinn.py")
    root = Path("reproduction")
    assert fem_reference.output_dir(root) == root / "fem"
    assert fem_convergence.output_path(root) == root / "fem/convergence.json"
    assert sweep.output_path(root) == root / "analytical_fem/sweep.json"
    assert resolution.fine_sweep_path(root) == root / "analytical_fem/sweep.json"
    assert resolution.output_path(root) == root / "analytical_fem/resolution_check.json"
    assert training.result_paths(root, 1)["root"] == root / "pinn/seed_1"
    assert validation.result_paths(root, 1)["root"] == root / "pinn/seed_1"


def test_fem_reference_main_forwards_root_and_force_without_solving(monkeypatch):
    module = _load("root_fem_reference_main", "scripts/generate_fem_reference.py")
    calls = []

    def fake_generate(directory, *, force=False):
        calls.append((directory, force))
        return directory / "reference.npz", directory / "reference.json"

    monkeypatch.setattr(module, "generate", fake_generate)
    assert module.main([]) == 0
    assert module.main(["--canonical", "--force"]) == 0
    assert calls == [
        (Path("reproduction/fem"), False),
        (Path("results/fem"), True),
    ]


def test_resolution_refuses_existing_output_before_any_solve(monkeypatch):
    module = _load("root_resolution_admission", "scripts/check_analytical_fem_resolution.py")
    root = ROOT / "tests/.resolution_root_safety"
    shutil.rmtree(root, ignore_errors=True)
    output = root / "analytical_fem" / "resolution_check.json"
    try:
        output.parent.mkdir(parents=True)
        output.write_text("existing", encoding="utf-8")
        monkeypatch.setattr(
            module,
            "run_study",
            lambda selected: (_ for _ in ()).throw(
                AssertionError("scientific solve ran")
            ),
        )
        with pytest.raises(FileExistsError, match="refusing overwrite"):
            module.main(["--results-root", str(root)])
    finally:
        shutil.rmtree(root, ignore_errors=True)
