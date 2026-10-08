"""Inexpensive checks for the stored-reference FEM figure module."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "figures" / "fem.py"
SPEC = importlib.util.spec_from_file_location("final_fem_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
figures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = figures
SPEC.loader.exec_module(figures)


def test_expected_output_paths_are_exact():
    assert figures.EXPECTED_FILENAMES == (
        "mesh.png", "mesh.pdf", "u.png", "u.pdf", "v.png", "v.pdf",
        "speed_streamlines.png", "speed_streamlines.pdf",
        "pressure_tilde.png", "pressure_tilde.pdf",
        "flow_overview.png", "flow_overview.pdf",
    )
    assert figures.OUTPUT_DIR == Path("reproduction/figures/fem")
    assert figures.OUTPUT_STEMS["mesh"] == figures.OUTPUT_DIR / "mesh"
    assert figures.OUTPUT_STEMS["flow_overview"] == figures.OUTPUT_DIR / "flow_overview"
    assert figures.reference_paths(Path("results")) == (
        Path("results/fem/reference.npz"), Path("results/fem/reference.json")
    )
    assert figures.output_stems(Path("custom"))["u"] == Path(
        "custom/figures/fem/u"
    )


def test_mesh_protocol_uses_16_by_8_target_physical_geometry(monkeypatch):
    calls = []
    sentinel = {"mesh": object()}

    def fake_make_mapped_mesh(*, physical, nx, ny):
        calls.append((physical, nx, ny))
        return sentinel

    monkeypatch.setattr(figures, "make_mapped_mesh", fake_make_mapped_mesh)
    assert figures.load_illustrative_mesh() is sentinel
    assert calls == [(figures.PHYSICAL, 16, 8)]
    assert figures.MESH_NX == 16
    assert figures.MESH_NY == 8


def test_import_is_side_effect_free():
    before = {
        name: (figures.OUTPUT_DIR / name).stat().st_mtime_ns
        if (figures.OUTPUT_DIR / name).exists() else None
        for name in figures.EXPECTED_FILENAMES
    }
    second_spec = importlib.util.spec_from_file_location("final_fem_figures_second", SCRIPT)
    assert second_spec is not None and second_spec.loader is not None
    second = importlib.util.module_from_spec(second_spec)
    second_spec.loader.exec_module(second)
    after = {
        name: (figures.OUTPUT_DIR / name).stat().st_mtime_ns
        if (figures.OUTPUT_DIR / name).exists() else None
        for name in figures.EXPECTED_FILENAMES
    }
    assert after == before


def test_reference_load_and_cartesian_mask_are_finite_inside(tmp_path):
    reference_npz, reference_json = figures.reference_paths(Path("results"))
    metadata_payload = json.loads(reference_json.read_text(encoding="utf-8"))
    metadata_payload["fem"].update(nx=256, ny=128)
    metadata_path = tmp_path / "reference.json"
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")
    metadata = figures.load_reference_metadata(metadata_path)
    reference = figures.load_reference_grid(reference_npz)
    assert metadata["fem"]["nx"] == 256
    assert reference["u"].shape == (96, 257)
    pressure = figures.weighted_zero_mean(
        reference["p_tilde"], reference["quadrature_weights"]
    )
    assert abs(np.sum(reference["quadrature_weights"] * pressure)) < 1.0e-14
    cartesian = figures.cartesian_velocity_grid(reference, y_points=41)
    assert cartesian["u"].shape == (41, 257)
    assert np.isfinite(cartesian["u"]).any()
    assert np.isnan(cartesian["u"][0]).all()
    assert np.isnan(cartesian["u"][-1]).all()


def test_module_uses_only_stored_fem_reference_not_solver_or_pinn():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "reference.npz" in source
    assert "reference.json" in source
    assert "solve_" + "stokes" not in source
    assert "corrugated_stokes.fem" not in source
    assert "corrugated_stokes.pinn" not in source


def test_flow_overview_tiles_exactly_three_periods():
    x = np.array([0.0, 0.25, 0.5, 0.75])
    y = np.array([-0.2, 0.0, 0.2])
    u = np.arange(y.size * x.size, dtype=np.float64).reshape(y.size, x.size)
    field = {"x": x, "y": y, "u": u, "v": -u, "speed": np.abs(u)}
    tiled = figures.tile_periodic_velocity_grid(field)

    assert figures.FLOW_OVERVIEW_PERIODS == 3
    assert tiled["x"].shape == (3 * x.size + 1,)
    assert tiled["x"][0] == 0.0
    assert tiled["x"][-1] == 3.0 * figures.PHYSICAL.L
    for name in ("u", "v", "speed"):
        assert tiled[name].shape == (y.size, 3 * x.size + 1)
        assert np.array_equal(tiled[name][:, :x.size], field[name])
        assert np.array_equal(tiled[name][:, -1], field[name][:, 0])


def test_streamline_starts_are_symmetric_and_deterministic():
    first = figures.public_streamline_start_points()
    second = figures.public_streamline_start_points()
    assert figures.PUBLIC_STREAMLINE_COUNT == 7
    assert figures.PUBLIC_STREAMLINE_WALL_MARGIN_FRACTION == 0.10
    assert np.array_equal(first, second)
    assert first.shape == (7, 2)
    assert np.array_equal(first[:, 0], np.zeros(7))
    assert np.allclose(first[:, 1], -first[::-1, 1])
    assert first[3, 1] == 0.0
    assert np.allclose(first[:, 1], np.linspace(-0.04, 0.04, 7))


def test_periodic_closure_gives_matching_exact_endpoints():
    x = np.array([0.0, 0.25, 0.5, 0.75])
    y = np.array([-0.2, 0.0, 0.2])
    values = np.arange(y.size * x.size, dtype=np.float64).reshape(y.size, x.size)
    closed = figures.close_periodic_velocity_grid(
        {"x": x, "y": y, "u": values, "v": -values, "speed": np.abs(values)}
    )
    assert closed["x"][0] == 0.0
    assert closed["x"][-1] == figures.PHYSICAL.L
    for name in ("u", "v", "speed"):
        assert np.array_equal(closed[name][:, 0], closed[name][:, -1])


def test_flow_overview_selector_is_focused(monkeypatch):
    calls = []
    overview = [Path("flow_overview.png"), Path("flow_overview.pdf")]
    monkeypatch.setattr(
        figures,
        "generate_flow_overview",
        lambda root: calls.append(("overview", root)) or overview,
    )
    monkeypatch.setattr(
        figures, "generate_all", lambda root: calls.append(("all", root)) or []
    )

    assert figures.main(["--only", "flow-overview"]) == 0
    assert calls == [("overview", Path("reproduction"))]


def test_default_generation_includes_flow_overview(monkeypatch):
    calls = []
    mesh = [Path("mesh.png"), Path("mesh.pdf")]
    fields = [Path("fields.png")]
    overview = [Path("flow_overview.png"), Path("flow_overview.pdf")]
    monkeypatch.setattr(
        figures,
        "make_mesh_figure",
        lambda stem: calls.append(("mesh", stem)) or mesh,
    )
    monkeypatch.setattr(
        figures,
        "generate_field_figures",
        lambda root: calls.append(("fields", root)) or fields,
    )
    monkeypatch.setattr(
        figures,
        "generate_flow_overview",
        lambda root: calls.append(("overview", root)) or overview,
    )

    assert figures.generate_all(Path("custom-results")) == [
        *mesh,
        *fields,
        *overview,
    ]
    assert calls == [
        ("mesh", Path("custom-results/figures/fem/mesh")),
        ("fields", Path("custom-results")),
        ("overview", Path("custom-results")),
    ]


def test_flow_overview_loads_only_the_stored_reference(monkeypatch):
    calls = []
    reference = {"stored": np.array([1.0])}
    outputs = (Path("flow_overview.png"), Path("flow_overview.pdf"))
    monkeypatch.setattr(
        figures,
        "load_reference_metadata",
        lambda path: calls.append(("metadata", path)),
    )
    monkeypatch.setattr(
        figures,
        "load_reference_grid",
        lambda path: calls.append(("reference", path)) or reference,
    )
    monkeypatch.setattr(
        figures,
        "make_flow_overview_figure",
        lambda value, stem: calls.append(("render", value, stem)) or outputs,
    )

    assert figures.generate_flow_overview() == list(outputs)
    assert calls == [
        ("metadata", Path("reproduction/fem/reference.json")),
        ("reference", Path("reproduction/fem/reference.npz")),
        ("render", reference, Path("reproduction/figures/fem/flow_overview")),
    ]


def test_speed_streamline_selector_is_focused(monkeypatch):
    calls = []
    outputs = [Path("speed_streamlines.png"), Path("speed_streamlines.pdf")]
    monkeypatch.setattr(
        figures,
        "generate_speed_streamlines",
        lambda root: calls.append(("speed-streamlines", root)) or outputs,
    )
    monkeypatch.setattr(
        figures, "generate_all", lambda root: calls.append(("all", root)) or []
    )

    assert figures.main(["--only", "speed-streamlines"]) == 0
    assert calls == [("speed-streamlines", Path("reproduction"))]


def test_explicit_results_mesh_selector_routes_only_its_output(monkeypatch):
    calls = []
    monkeypatch.setattr(
        figures,
        "make_mesh_figure",
        lambda stem: calls.append(stem) or (stem.with_suffix(".png"), stem.with_suffix(".pdf")),
    )
    assert figures.main(["--canonical", "--only", "mesh"]) == 0
    assert calls == [Path("results/figures/fem/mesh")]
