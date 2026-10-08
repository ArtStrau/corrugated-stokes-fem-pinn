"""Low-level output helpers shared by figure renderers.

This helper keeps paired PNG/PDF writing consistent. Geometry- or
solver-specific plotting logic belongs in the individual renderer modules
that use it.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_figure_pair(fig, output_stem: Path) -> tuple[Path, Path]:
    """Save one Matplotlib figure as deterministic PNG/PDF presentation files."""
    import matplotlib.pyplot as plt

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "corrugated-stokes"},
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "corrugated-stokes",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png_path, pdf_path
