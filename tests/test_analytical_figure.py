"""Focused scalar tests for the stored-data analytical figure renderer."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "figures" / "analytical.py"
SPEC = importlib.util.spec_from_file_location("analytical_figure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
figure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = figure
SPEC.loader.exec_module(figure)


def test_stored_sweep_loads_as_exact_increasing_axis_data():
    result_source = figure.source_path(Path("results"))
    delta, percent_error = figure.validated_plot_data(
        figure.load_payload(result_source)
    )
    np.testing.assert_array_equal(
        delta, np.asarray(figure.EXPECTED_DELTAS[::-1], dtype=np.float64)
    )
    assert percent_error.shape == (10,)
    assert np.isfinite(percent_error).all()
    assert np.all(percent_error >= 0.0)
    assert figure.SOURCE_PATH == Path("reproduction/analytical_fem/sweep.json")
    assert figure.OUTPUT_STEM == Path(
        "reproduction/figures/analytical/relative_flux_error"
    )
    assert figure.output_stem(Path("results")) == Path(
        "results/figures/analytical/relative_flux_error"
    )


def test_validation_rejects_changed_format_or_error_formula():
    payload = figure.load_payload(figure.source_path(Path("results")))
    bad_settings = copy.deepcopy(payload)
    bad_settings["study"]["fem_settings"]["nx"] = 128
    with pytest.raises(ValueError, match="FEM settings"):
        figure.validated_plot_data(bad_settings)

    bad_formula = copy.deepcopy(payload)
    bad_formula["points"][3]["relative_flux_error"] += 0.01
    with pytest.raises(ValueError, match="disagrees"):
        figure.validated_plot_data(bad_formula)


def test_renderer_has_no_solver_or_alternate_sweep_dependency():
    source = SCRIPT.read_text(encoding="utf-8")
    assert figure.source_path(Path("custom")) == Path(
        "custom/analytical_fem/sweep.json"
    )
    assert "sweep_" + "v1" not in source
    assert "resolution_check_" + "v1" not in source
    assert "corrugated_stokes.fem" not in source
    assert "corrugated_stokes.pinn" not in source


def test_cli_uses_one_selected_root_for_input_and_output(monkeypatch):
    calls = []
    monkeypatch.setattr(
        figure, "load_payload", lambda path: calls.append(("input", path)) or {}
    )
    monkeypatch.setattr(
        figure,
        "validated_plot_data",
        lambda payload: (np.asarray([0.1]), np.asarray([1.0])),
    )
    monkeypatch.setattr(
        figure,
        "make_figure",
        lambda x, y, stem: calls.append(("output", stem))
        or (stem.with_suffix(".png"), stem.with_suffix(".pdf")),
    )
    assert figure.main(["--results-root", "custom/root"]) == 0
    assert calls == [
        ("input", Path("custom/root/analytical_fem/sweep.json")),
        ("output", Path("custom/root/figures/analytical/relative_flux_error")),
    ]
