"""Inexpensive checks for the deterministic PINN collocation renderer."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

from corrugated_stokes.config import ADAM_POINTS, PHYSICAL


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "figures" / "pinn.py"
SPEC = importlib.util.spec_from_file_location("final_pinn_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
figures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = figures
SPEC.loader.exec_module(figures)


def test_windows_native_defaults_are_added_without_unsafe_workaround():
    environment = {}
    figures.configure_native_library_environment("nt", environment)
    assert environment == {
        "MKL_THREADING_LAYER": "SEQUENTIAL",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    assert "KMP_DUPLICATE_LIB_OK" not in environment


def test_windows_native_defaults_preserve_existing_user_values():
    environment = {
        "MKL_THREADING_LAYER": "USER_MKL",
        "OMP_NUM_THREADS": "7",
        "MKL_NUM_THREADS": "9",
    }
    figures.configure_native_library_environment("nt", environment)
    assert environment == {
        "MKL_THREADING_LAYER": "USER_MKL",
        "OMP_NUM_THREADS": "7",
        "MKL_NUM_THREADS": "9",
    }


def test_non_windows_adds_no_native_library_defaults():
    environment = {"EXISTING": "value"}
    figures.configure_native_library_environment("posix", environment)
    assert environment == {"EXISTING": "value"}


def test_native_environment_is_configured_before_numerical_imports():
    source = SCRIPT.read_text(encoding="utf-8")
    call = source.index("configure_native_library_environment(os.name, os.environ)")
    assert call < source.index("import numpy as np")
    assert call < source.index("import torch")
    assert call < source.index("from corrugated_stokes.pinn import")


def test_collocation_output_paths_are_exact():
    assert figures.EXPECTED_FILENAMES == (
        "collocation_adam.png",
        "collocation_adam.pdf",
        "adam_total.png",
        "adam_total.pdf",
        "lbfgs_total.png",
        "lbfgs_total.pdf",
    )
    assert figures.COLLOCATION_STEM == (
        Path("reproduction/figures/pinn/collocation_adam")
    )


def test_history_output_stems_and_training_sources_are_exact():
    assert figures.ADAM_TOTAL_STEM == figures.OUTPUT_DIR / "adam_total"
    assert figures.LBFGS_TOTAL_STEM == figures.OUTPUT_DIR / "lbfgs_total"
    assert figures.TRAINING_PATHS == {
        seed: Path("reproduction") / "pinn" / f"seed_{seed}" / "training.json"
        for seed in (0, 1, 2)
    }
    assert figures.training_paths(Path("results"))[1] == Path(
        "results/pinn/seed_1/training.json"
    )
    assert figures.output_stems(Path("custom"))["adam_total"] == Path(
        "custom/figures/pinn/adam_total"
    )


def test_no_argument_main_generates_only_three_output_pairs(monkeypatch):
    expected = [Path(f"figure_{index}.{suffix}") for index in range(3) for suffix in ("png", "pdf")]
    calls = []

    def fake_all(root):
        calls.append(("all", root))
        return expected

    monkeypatch.setattr(figures, "generate_all_figures", fake_all)
    assert figures.main([]) == 0
    assert calls == [("all", Path("reproduction"))]


def test_only_flag_without_value_fails_normally():
    with pytest.raises(SystemExit):
        figures.parse_args(["--only"])


def test_only_collocation_generates_only_collocation(monkeypatch):
    calls = []
    outputs = (Path("collocation.png"), Path("collocation.pdf"))
    monkeypatch.setattr(
        figures,
        "make_collocation_figure",
        lambda stem: calls.append(("collocation", stem)) or outputs,
    )
    monkeypatch.setattr(
        figures,
        "generate_history_figures",
        lambda: (_ for _ in ()).throw(AssertionError("histories must not run")),
    )
    assert figures.main(["--only", "collocation"]) == 0
    assert calls == [
        ("collocation", Path("reproduction/figures/pinn/collocation_adam"))
    ]


def test_explicit_results_collocation_routes_output_without_histories(monkeypatch):
    calls = []
    outputs = (Path("collocation.png"), Path("collocation.pdf"))
    monkeypatch.setattr(
        figures,
        "make_collocation_figure",
        lambda stem: calls.append(stem) or outputs,
    )
    assert figures.main(["--canonical", "--only", "collocation"]) == 0
    assert calls == [Path("results/figures/pinn/collocation_adam")]


def test_only_histories_generates_exactly_two_history_pairs(monkeypatch):
    outputs = [Path(f"history_{index}.{suffix}") for index in range(2) for suffix in ("png", "pdf")]
    calls = []
    monkeypatch.setattr(
        figures,
        "generate_history_figures",
        lambda root: calls.append(("histories", root)) or outputs,
    )
    monkeypatch.setattr(
        figures,
        "make_collocation_figure",
        lambda: (_ for _ in ()).throw(AssertionError("collocation must not run")),
    )
    assert figures.main(["--only", "histories"]) == 0
    assert calls == [("histories", Path("reproduction"))]


def test_individual_history_selectors_remain_independent(monkeypatch):
    calls = []
    monkeypatch.setattr(figures, "load_training_records", lambda paths: {})
    functions = {
        "adam-total": "make_adam_total_figure",
        "lbfgs-total": "make_lbfgs_total_figure",
    }
    for selector, function_name in functions.items():
        monkeypatch.setattr(
            figures,
            function_name,
            lambda stem, records, name=selector: calls.append(name) or (
                Path(f"{name}.png"),
                Path(f"{name}.pdf"),
            ),
        )
        assert figures.main(["--only", selector]) == 0
    assert calls == list(functions)


def test_total_histories_use_fixed_s2_style_without_markers():
    assert figures.SEED_STYLES == {
        0: {"color": "#67B7E1", "linestyle": "-", "marker": None},
        1: {"color": "#2D3138", "linestyle": "-", "marker": None},
        2: {"color": "#6FA85B", "linestyle": "-", "marker": None},
    }
    assert figures.HISTORY_LINEWIDTH == 1.05
    source = inspect.getsource(figures._make_total_history_figure)
    assert "**SEED_STYLES[seed]" in source
    assert "linewidth=HISTORY_LINEWIDTH" in source


def test_docs_exclude_component_figure_and_keep_eight_loss_terms():
    text = (ROOT / "docs" / "04_pinn.md").read_text(encoding="utf-8")
    assert "adam_components_seed1" not in text
    for term in (
        r"\mathcal L_x", r"\mathcal L_y", r"\mathcal L_c",
        r"\mathcal L_{\mathrm{wall},U}", r"\mathcal L_{\mathrm{wall},V}",
        r"\mathcal L_{\mathrm{per},U}", r"\mathcal L_{\mathrm{per},V}",
        r"\mathcal L_{\mathrm{per},\Pi}",
    ):
        assert term in text


def test_stored_histories_use_true_indices_totals_and_exact_lengths():
    records = figures.load_training_records(figures.training_paths(Path("results")))
    assert tuple(records) == (0, 1, 2)
    for seed in (0, 1, 2):
        adam_x, adam_y = figures.extract_training_history(records[seed], "adam")
        lbfgs_x, lbfgs_y = figures.extract_training_history(records[seed], "lbfgs")
        assert np.array_equal(adam_x, np.arange(1, 16501))
        assert np.array_equal(lbfgs_x, np.arange(1, 501))
        assert np.array_equal(
            adam_y,
            np.asarray([entry["total"] for entry in records[seed]["adam_history"]]),
        )
        assert np.array_equal(
            lbfgs_y,
            np.asarray([entry["total"] for entry in records[seed]["lbfgs_history"]]),
        )

def test_history_renderer_reuses_unsmoothed_stored_history_semantics():
    total_source = inspect.getsource(figures._make_total_history_figure)
    assert "extract_training_history" in total_source
    assert "semilogy" in total_source
    source = SCRIPT.read_text(encoding="utf-8").lower()
    for forbidden in (
        "running_minimum",
        "minimum.accumulate",
        "moving_average",
        "rolling(",
        "resample(",
        "polyfit(",
        "savgol",
    ):
        assert forbidden not in source


def test_adam_bundle_counts_and_region_classification_are_exact():
    interior, walls, periodic = figures.load_adam_collocation()
    region = interior["region"].detach().cpu().numpy().ravel()
    assert interior["x"].shape[0] == ADAM_POINTS.interior_points == 1024
    assert walls["x"].shape[0] == ADAM_POINTS.wall_points_total == 256
    assert periodic["x_left"].shape[0] == ADAM_POINTS.periodic_pairs == 128
    assert periodic["x_right"].shape[0] == ADAM_POINTS.periodic_pairs == 128
    assert np.count_nonzero(region == 0) == 768
    assert np.count_nonzero(region == 1) == 256
    assert figures.PHYSICAL is PHYSICAL


def test_load_calls_figure_generator_with_fixed_adam_controls(monkeypatch):
    calls = []
    sentinel = (object(), object(), object())

    def fake_generate_point_bundle(controls):
        calls.append(controls)
        return sentinel

    monkeypatch.setattr(figures, "generate_point_bundle", fake_generate_point_bundle)
    assert figures.load_adam_collocation() is sentinel
    assert calls == [figures.ADAM_POINTS]


def test_import_is_side_effect_free():
    before = {
        name: (figures.OUTPUT_DIR / name).stat().st_mtime_ns
        if (figures.OUTPUT_DIR / name).exists() else None
        for name in figures.EXPECTED_FILENAMES
    }
    second_spec = importlib.util.spec_from_file_location(
        "final_pinn_figures_second", SCRIPT
    )
    assert second_spec is not None and second_spec.loader is not None
    second = importlib.util.module_from_spec(second_spec)
    second_spec.loader.exec_module(second)
    after = {
        name: (figures.OUTPUT_DIR / name).stat().st_mtime_ns
        if (figures.OUTPUT_DIR / name).exists() else None
        for name in figures.EXPECTED_FILENAMES
    }
    assert after == before


def test_module_has_no_model_optimizer_or_scientific_workflow_dependency():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "generate_point_bundle(ADAM_POINTS)" in source
    for forbidden in (
        "model.pt",
        "torch.load",
        "build_network",
        "ScaledStokesNetwork",
        "torch.optim",
        "scripts.train_pinn",
        "scripts.validate_pinn",
        "corrugated_stokes.fem",
        "corrugated_stokes.analytical",
    ):
        assert forbidden not in source
