"""Safe-help tests for parameterless calculation scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    """Import one script without executing its main function."""

    path = ROOT / relative_path
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "relative_path", "guard_name"),
    (
        ("help_sweep", "scripts/run_analytical_fem_sweep.py", "write_study"),
        ("help_summary", "scripts/summarize_results.py", "write_summary"),
        ("help_problem_figure", "scripts/figures/problem.py", "make_geometry_figure"),
        ("help_analytical_figure", "scripts/figures/analytical.py", "load_payload"),
    ),
)
def test_help_exits_before_computation_or_writes(
    name, relative_path, guard_name, monkeypatch, capsys
):
    module = _load(name, relative_path)

    def forbidden(*args, **kwargs):
        pytest.fail("--help reached a computation or write path")

    monkeypatch.setattr(module, guard_name, forbidden)
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_problem_figure_cli_routes_default_and_results_outputs(monkeypatch):
    module = _load("problem_root_routing", "scripts/figures/problem.py")
    calls = []

    def fake_render(stem):
        calls.append(stem)
        return stem.with_suffix(".png"), stem.with_suffix(".pdf")

    monkeypatch.setattr(module, "make_geometry_figure", fake_render)
    assert module.main([]) == 0
    assert module.main(["--canonical"]) == 0
    assert calls == [
        Path("reproduction/figures/problem/geometry"),
        Path("results/figures/problem/geometry"),
    ]
