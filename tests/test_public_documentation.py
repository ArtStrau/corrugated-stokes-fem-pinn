"""Small integrity checks for the public Markdown documentation."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS = tuple(
    ROOT / "docs" / f"0{index}_{name}.md"
    for index, name in enumerate(
        ("problem", "analytical", "fem", "pinn", "validation"), start=1
    )
)
PUBLIC_MARKDOWN = (ROOT / "README.md", *DOCS, ROOT / "docs/index.md")


def test_required_public_documents_and_local_links_exist():
    assert all(path.is_file() for path in PUBLIC_MARKDOWN)
    for path in PUBLIC_MARKDOWN:
        text = path.read_text(encoding="utf-8")
        targets = re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text)
        targets += re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        targets += re.findall(r'<img\s+[^>]*src="([^"]+)"', text)
        for target in targets:
            target = target.split("#", 1)[0]
            if target and not target.startswith(("http://", "https://", "mailto:")):
                assert (path.parent / target).resolve().exists(), (path, target)


def test_principal_documented_values_match_stored_results():
    reference = json.loads((ROOT / "results/fem/reference.json").read_text())
    sweep = json.loads((ROOT / "results/analytical_fem/sweep.json").read_text())
    summary = json.loads((ROOT / "results/summary.json").read_text())
    target = next(point for point in sweep["points"] if point["delta"] == 0.2)
    assert reference["diagnostics"]["mean_flux"] == pytest.approx(
        summary["fem_reference"]["mean_flux"], rel=0, abs=0
    )
    assert target["relative_flux_error"] == pytest.approx(0.09933535615497897)
    assert [record["seed"] for record in summary["seeds"]] == [0, 1, 2]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_MARKDOWN)
    for value in (
        "3.85231\\times10^{-4}",
        "0.258\\%",
        "0.138\\%",
        "0.077\\%",
        "9.93\\%",
    ):
        assert value in combined


def test_pinn_documentation_contains_the_eight_loss_terms_and_soft_periodicity():
    text = (ROOT / "docs/04_pinn.md").read_text(encoding="utf-8")
    for term in (
        r"\mathcal L_x",
        r"\mathcal L_y",
        r"\mathcal L_c",
        r"\mathcal L_{\mathrm{wall},U}",
        r"\mathcal L_{\mathrm{wall},V}",
        r"\mathcal L_{\mathrm{per},U}",
        r"\mathcal L_{\mathrm{per},V}",
        r"\mathcal L_{\mathrm{per},\Pi}",
    ):
        assert term in text
    normalized = " ".join(text.lower().split())
    assert "periodicity is imposed softly" in normalized
    assert "derivative periodicity is not" in normalized
