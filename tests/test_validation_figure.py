"""Inexpensive checks for the read-only validation-panel renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from corrugated_stokes.config import VALIDATION


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "figures" / "validation.py"
SPEC = importlib.util.spec_from_file_location("final_validation_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
figures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = figures
SPEC.loader.exec_module(figures)


def test_representative_seed_paths_and_controls_are_exact():
    assert figures.REPRESENTATIVE_SEED == 1
    assert figures.COMPARISON_SHAPE == (
        VALIDATION.fem_comparison_eta_order,
        VALIDATION.fem_comparison_nx,
    ) == (96, 257)
    assert tuple(figures.OUTPUT_STEMS) == figures.PANEL_NAMES
    assert all(
        path == Path("reproduction") / "figures" / "validation" / name
        for name, path in figures.OUTPUT_STEMS.items()
    )
    assert len(figures.EXPECTED_FILENAMES) == 24
    inputs = figures.input_paths(Path("custom"))
    assert inputs == {
        "reference_npz": Path("custom/fem/reference.npz"),
        "reference_json": Path("custom/fem/reference.json"),
        "model": Path("custom/pinn/seed_1/model.pt"),
        "training": Path("custom/pinn/seed_1/training.json"),
    }
    assert all(
        path.parent == Path("custom/figures/validation")
        for path in figures.output_stems(Path("custom")).values()
    )


def test_derived_error_fields_are_exact():
    uf = np.asarray([[1.0, -2.0], [0.5, 3.0]])
    vf = np.asarray([[0.0, 0.5], [-1.0, 2.0]])
    pf = np.asarray([[2.0, -1.0], [0.5, -0.25]])
    up = uf + np.asarray([[0.1, -0.2], [0.3, 0.4]])
    vp = vf + np.asarray([[0.2, 0.1], [-0.3, 0.5]])
    pp = pf + np.asarray([[0.4, -0.5], [0.6, -0.7]])
    fields = figures.derive_plot_fields(uf, vf, pf, up, vp, pp)
    assert np.array_equal(fields["speed_fem"], np.hypot(uf, vf))
    assert np.array_equal(fields["speed_pinn_seed1"], np.hypot(up, vp))
    assert np.array_equal(fields["velocity_error_seed1"], np.hypot(up - uf, vp - vf))
    assert np.array_equal(fields["u_error_seed1"], np.abs(up - uf))
    assert np.array_equal(fields["v_error_seed1"], np.abs(vp - vf))
    assert np.array_equal(fields["pressure_error_seed1"], np.abs(pp - pf))


def test_comparison_grid_accepts_exact_and_roundoff_scale_y():
    x = np.asarray([[0.0, 0.5]], dtype=np.float64)
    eta = np.asarray([[-0.5, 0.5]], dtype=np.float64)
    y = np.asarray([[-0.1, 0.1]], dtype=np.float64)
    figures.validate_comparison_grid(x, eta, y, x.copy(), eta.copy(), y.copy())
    roundoff_y = y.copy()
    roundoff_y[0, 0] += 2.7755575615628914e-17
    figures.validate_comparison_grid(x, eta, roundoff_y, x, eta, y)


def test_periodic_mapped_scalar_closes_with_the_first_column():
    x = np.asarray([[0.0, 0.5], [0.0, 0.5]])
    y = np.asarray([[-0.2, -0.1], [0.2, 0.1]])
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    closed_x, closed_y, closed_values = figures.close_periodic_mapped_scalar(
        x, y, values
    )
    assert closed_x.shape == closed_y.shape == closed_values.shape == (2, 3)
    assert np.array_equal(closed_x[:, -1], np.full(2, figures.PHYSICAL.L))
    assert np.array_equal(closed_y[:, -1], y[:, 0])
    assert np.array_equal(closed_values[:, -1], values[:, 0])


@pytest.mark.parametrize("coordinate", ("x", "eta", "y"))
def test_comparison_grid_rejects_material_coordinate_mismatch(coordinate):
    x = np.asarray([[0.0, 0.5]], dtype=np.float64)
    eta = np.asarray([[-0.5, 0.5]], dtype=np.float64)
    y = np.asarray([[-0.1, 0.1]], dtype=np.float64)
    stored = {"x": x.copy(), "eta": eta.copy(), "y": y.copy()}
    stored[coordinate][0, 0] += 1.0e-10
    with pytest.raises(
        ValueError,
        match=rf"stored FEM {coordinate} grid differs from validation convention",
    ):
        figures.validate_comparison_grid(
            stored["x"], stored["eta"], stored["y"], x, eta, y
        )


def test_each_solution_pair_has_one_shared_color_limit():
    fem = np.asarray([[-2.0, 1.0], [0.0, 3.0]])
    pinn = np.asarray([[-1.0, 2.0], [0.5, 4.0]])
    assert figures.paired_color_limits(fem, pinn, mode="range") == (-2.0, 4.0)
    assert figures.paired_color_limits(np.abs(fem), np.abs(pinn), mode="nonnegative") == (0.0, 4.0)
    assert figures.paired_color_limits(fem, pinn, mode="symmetric") == (-4.0, 4.0)


def test_only_speed_solution_panels_are_configured_for_streamlines():
    assert figures.STREAMLINE_PANELS == frozenset(
        {"speed_fem", "speed_pinn_seed1"}
    )


def test_cli_groups_cover_all_panels_once():
    assert figures.selected_panels("speed") == figures.GROUPS["speed"]
    assert figures.selected_panels("pressure") == figures.GROUPS["pressure"]
    assert figures.selected_panels("components") == figures.GROUPS["components"]
    assert figures.selected_panels("all") == figures.PANEL_NAMES
    grouped = sum((figures.GROUPS[name] for name in ("speed", "pressure", "components")), ())
    assert set(grouped) == set(figures.PANEL_NAMES)
    assert len(grouped) == len(set(grouped)) == 12


def test_import_is_side_effect_free():
    before = {
        name: (figures.OUTPUT_DIR / name).stat().st_mtime_ns
        if (figures.OUTPUT_DIR / name).exists() else None
        for name in figures.EXPECTED_FILENAMES
    }
    second_spec = importlib.util.spec_from_file_location(
        "final_validation_figures_second", SCRIPT
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


def test_generate_panels_uses_one_custom_root_for_all_inputs_and_outputs(monkeypatch):
    root = Path("custom/root")
    calls = []
    dataset = {"fields": {}}
    monkeypatch.setattr(
        figures,
        "load_comparison_dataset",
        lambda selected: calls.append(("input", selected)) or dataset,
    )
    monkeypatch.setattr(figures, "solution_color_limits", lambda fields: {})
    monkeypatch.setattr(figures, "selected_panels", lambda selection: ("speed_fem",))
    monkeypatch.setattr(
        figures,
        "render_panel",
        lambda name, loaded, limits, stems: calls.append(
            ("output", stems[name])
        )
        or (Path("panel.png"), Path("panel.pdf")),
    )
    assert figures.generate_panels("speed", root) == [
        Path("panel.png"), Path("panel.pdf")
    ]
    assert calls == [
        ("input", root),
        ("output", Path("custom/root/figures/validation/speed_fem")),
    ]
