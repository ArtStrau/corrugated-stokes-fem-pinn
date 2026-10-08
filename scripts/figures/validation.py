"""Generate standalone read-only FEM–PINN comparison panels for F5.

All physical fields come from the same selected-root FEM reference and
completed seed-1 PINN evaluated on the defined validation grid. The
module never trains, constructs an optimizer, solves FEM, reruns validation,
or writes scientific data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _native_environment import configure_native_library_environment


configure_native_library_environment(os.name, os.environ)

PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC = PROJECT_ROOT / "src"
for search_path in (SCRIPT_DIR, SCRIPTS_DIR, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from _result_root import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    add_result_root_arguments,
    parse_result_root,
)

import numpy as np  # noqa: E402

from _common import save_figure_pair  # noqa: E402
from fem import (  # noqa: E402
    PUBLIC_VERTICAL_COLORBAR_PAD,
    cartesian_velocity_grid,
    close_periodic_velocity_grid,
    draw_public_streamlines,
    load_reference_grid,
    load_reference_metadata,
    reference_paths,
    weighted_zero_mean,
)
from corrugated_stokes.config import (  # noqa: E402
    FEM_REFERENCE,
    PHYSICAL,
    PINN_SCALES,
    VALIDATION,
)
from corrugated_stokes.geometry import (  # noqa: E402
    discrete_wall_bounds,
    lower_wall,
    make_mapped_mesh,
    upper_wall,
)
from corrugated_stokes.pinn import model_state_hash  # noqa: E402
from corrugated_stokes.validation import (  # noqa: E402
    evaluate_fields_chunked,
    verify_training_results,
)


REPRESENTATIVE_SEED = 1


def input_paths(results_root: Path) -> dict[str, Path]:
    """Return all FEM and representative PINN inputs below one selected root."""

    reference_npz, reference_json = reference_paths(results_root)
    seed_root = Path(results_root) / "pinn" / f"seed_{REPRESENTATIVE_SEED}"
    return {
        "reference_npz": reference_npz,
        "reference_json": reference_json,
        "model": seed_root / "model.pt",
        "training": seed_root / "training.json",
    }


def output_stems(results_root: Path) -> dict[str, Path]:
    """Return all standalone validation panel stems below one selected root."""

    output_dir = Path(results_root) / "figures" / "validation"
    return {name: output_dir / name for name in PANEL_NAMES}


_DEFAULT_INPUTS = input_paths(DEFAULT_RESULT_ROOT)
MODEL_PATH = _DEFAULT_INPUTS["model"]
TRAINING_PATH = _DEFAULT_INPUTS["training"]
OUTPUT_DIR = Path(DEFAULT_RESULT_ROOT) / "figures" / "validation"
COMPARISON_SHAPE = (
    VALIDATION.fem_comparison_eta_order,
    VALIDATION.fem_comparison_nx,
)

PANEL_NAMES = (
    "speed_fem",
    "speed_pinn_seed1",
    "velocity_error_seed1",
    "pressure_fem",
    "pressure_pinn_seed1",
    "pressure_error_seed1",
    "u_fem",
    "u_pinn_seed1",
    "u_error_seed1",
    "v_fem",
    "v_pinn_seed1",
    "v_error_seed1",
)
OUTPUT_STEMS = {name: OUTPUT_DIR / name for name in PANEL_NAMES}


def validate_comparison_grid(
    stored_x: np.ndarray,
    stored_eta: np.ndarray,
    stored_y: np.ndarray,
    expected_x: np.ndarray,
    expected_eta: np.ndarray,
    expected_y: np.ndarray,
) -> None:
    """Require exact logical coordinates and roundoff-level physical coordinates."""

    for name, stored, expected in (
        ("x", stored_x, expected_x),
        ("eta", stored_eta, expected_eta),
    ):
        if not np.array_equal(stored, expected):
            raise ValueError(
                f"stored FEM {name} grid differs from validation convention"
            )
    y_tolerance = 8.0 * np.finfo(np.float64).eps
    if (
        stored_y.shape != expected_y.shape
        or not np.isfinite(stored_y).all()
        or not np.isfinite(expected_y).all()
        or not np.allclose(stored_y, expected_y, rtol=0.0, atol=y_tolerance)
    ):
        raise ValueError("stored FEM y grid differs from validation convention")
EXPECTED_FILENAMES = tuple(
    f"{name}.{suffix}" for name in PANEL_NAMES for suffix in ("png", "pdf")
)
GROUPS = {
    "speed": ("speed_fem", "speed_pinn_seed1", "velocity_error_seed1"),
    "pressure": (
        "pressure_fem",
        "pressure_pinn_seed1",
        "pressure_error_seed1",
    ),
    "components": (
        "u_fem",
        "u_pinn_seed1",
        "u_error_seed1",
        "v_fem",
        "v_pinn_seed1",
        "v_error_seed1",
    ),
}
STREAMLINE_PANELS = frozenset({"speed_fem", "speed_pinn_seed1"})


def derive_plot_fields(
    u_fem: np.ndarray,
    v_fem: np.ndarray,
    pressure_fem_zero: np.ndarray,
    u_pinn: np.ndarray,
    v_pinn: np.ndarray,
    pressure_pinn_zero: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the six solution and six direct difference fields."""
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            u_fem,
            v_fem,
            pressure_fem_zero,
            u_pinn,
            v_pinn,
            pressure_pinn_zero,
        )
    )
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("all FEM/PINN solution fields must have identical shapes")
    if not all(np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("FEM/PINN solution fields must be finite")
    uf, vf, pf, up, vp, pp = arrays
    fields = {
        "u_fem": uf,
        "v_fem": vf,
        "pressure_fem": pf,
        "u_pinn_seed1": up,
        "v_pinn_seed1": vp,
        "pressure_pinn_seed1": pp,
        "speed_fem": np.hypot(uf, vf),
        "speed_pinn_seed1": np.hypot(up, vp),
        "velocity_error_seed1": np.hypot(up - uf, vp - vf),
        "u_error_seed1": np.abs(up - uf),
        "v_error_seed1": np.abs(vp - vf),
        "pressure_error_seed1": np.abs(pp - pf),
    }
    error_names = (
        "velocity_error_seed1",
        "u_error_seed1",
        "v_error_seed1",
        "pressure_error_seed1",
    )
    if not all(np.isfinite(fields[name]).all() for name in fields):
        raise FloatingPointError("derived validation fields must be finite")
    if any(np.any(fields[name] < 0.0) for name in error_names):
        raise ValueError("validation error fields must be nonnegative")
    return fields


def paired_color_limits(
    fem_values: np.ndarray,
    pinn_values: np.ndarray,
    *,
    mode: str,
) -> tuple[float, float]:
    """Return one shared, unclipped scale for a FEM/PINN solution pair."""
    combined = np.concatenate(
        (
            np.asarray(fem_values, dtype=np.float64).ravel(),
            np.asarray(pinn_values, dtype=np.float64).ravel(),
        )
    )
    if not np.isfinite(combined).all():
        raise FloatingPointError("paired solution values must be finite")
    if mode == "nonnegative":
        limits = (0.0, float(np.max(combined)))
    elif mode == "symmetric":
        bound = float(np.max(np.abs(combined)))
        limits = (-bound, bound)
    elif mode == "range":
        limits = (float(np.min(combined)), float(np.max(combined)))
    else:
        raise ValueError("color-limit mode must be nonnegative, symmetric, or range")
    if limits[1] <= limits[0]:
        limits = (limits[0], limits[0] + np.finfo(np.float64).eps)
    return limits


def error_color_limits(values: np.ndarray) -> tuple[float, float]:
    """Return the required zero-based finite scale for one error field."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("error field must be finite and nonnegative")
    upper = float(np.max(values))
    return 0.0, max(upper, np.finfo(np.float64).eps)


def load_comparison_dataset(
    results_root: Path = DEFAULT_RESULT_ROOT,
) -> dict[str, object]:
    """Load both stored solutions on the defined comparison grid."""
    paths = input_paths(results_root)
    load_reference_metadata(paths["reference_json"])
    reference = load_reference_grid(paths["reference_npz"])
    if reference["u"].shape != COMPARISON_SHAPE:
        raise ValueError(
            f"stored comparison shape {reference['u'].shape} != {COMPARISON_SHAPE}"
        )

    eta, _eta_weights = np.polynomial.legendre.leggauss(
        VALIDATION.fem_comparison_eta_order
    )
    x_values = np.linspace(
        0.0,
        PHYSICAL.L,
        VALIDATION.fem_comparison_nx,
        endpoint=False,
    )
    expected_x, expected_eta = np.meshgrid(x_values, eta)
    geometry = make_mapped_mesh(
        PHYSICAL,
        FEM_REFERENCE.nx,
        FEM_REFERENCE.ny,
    )
    lower, upper = discrete_wall_bounds(expected_x.ravel(), geometry)
    width = upper - lower
    if not np.isfinite(width).all() or np.any(width <= 0.0):
        raise ValueError("comparison grid has non-positive discrete width")
    expected_y = (
        lower + 0.5 * (expected_eta.ravel() + 1.0) * width
    ).reshape(COMPARISON_SHAPE)
    validate_comparison_grid(
        reference["x"],
        reference["eta"],
        reference["y"],
        expected_x,
        expected_eta,
        expected_y,
    )

    model, _training, _consistency = verify_training_results(
        paths["model"],
        paths["training"],
        REPRESENTATIVE_SEED,
    )
    state_before = model_state_hash(model)
    scaled_x = (reference["x"].ravel() - 0.5 * PHYSICAL.L) / PINN_SCALES.mean_width
    scaled_y = reference["y"].ravel() / PINN_SCALES.mean_width
    pinn = evaluate_fields_chunked(
        model,
        scaled_x,
        scaled_y,
        chunk_size=VALIDATION.chunk_size,
        full=False,
    )
    if model_state_hash(model) != state_before:
        raise RuntimeError("figure inference changed the completed seed-1 model")

    weights = np.asarray(reference["quadrature_weights"], dtype=np.float64)
    p_fem_zero = weighted_zero_mean(reference["p_tilde"], weights)
    p_pinn_zero = weighted_zero_mean(
        (PINN_SCALES.P0 * pinn["Pi"]).reshape(COMPARISON_SHAPE),
        weights,
    )
    fields = derive_plot_fields(
        reference["u"],
        reference["v"],
        p_fem_zero,
        (PINN_SCALES.U0 * pinn["U"]).reshape(COMPARISON_SHAPE),
        (PINN_SCALES.U0 * pinn["V"]).reshape(COMPARISON_SHAPE),
        p_pinn_zero,
    )
    return {
        "x": reference["x"].copy(),
        "y": reference["y"].copy(),
        "eta": reference["eta"].copy(),
        "quadrature_weights": weights.copy(),
        "discrete_width": width.reshape(COMPARISON_SHAPE),
        "fields": fields,
    }


def solution_color_limits(fields: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Build the four exact shared FEM/PINN scales used by panel pairs."""
    return {
        "speed": paired_color_limits(
            fields["speed_fem"], fields["speed_pinn_seed1"], mode="nonnegative"
        ),
        "pressure": paired_color_limits(
            fields["pressure_fem"], fields["pressure_pinn_seed1"], mode="symmetric"
        ),
        "u": paired_color_limits(
            fields["u_fem"], fields["u_pinn_seed1"], mode="range"
        ),
        "v": paired_color_limits(
            fields["v_fem"], fields["v_pinn_seed1"], mode="symmetric"
        ),
    }


def _format_axes(ax) -> None:
    """Apply the accepted FEM physical-wall and Cartesian-axis presentation."""
    wall_x = np.linspace(0.0, PHYSICAL.L, 1001)
    ax.plot(wall_x, lower_wall(wall_x), color="black", linewidth=0.9)
    ax.plot(wall_x, upper_wall(wall_x), color="black", linewidth=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, PHYSICAL.L)
    ax.set_ylim(-0.55 * PHYSICAL.width_max, 0.55 * PHYSICAL.width_max)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def _compact_colorbar(
    fig, ax, artist, label: str, *, readable_error_ticks: bool = False
) -> None:
    """Attach one rectangular colorbar exactly aligned to the panel height."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes(
        "right", size="4.5%", pad=PUBLIC_VERTICAL_COLORBAR_PAD
    )
    colorbar = fig.colorbar(artist, cax=colorbar_ax, label=label)
    if readable_error_ticks:
        colorbar.locator = MaxNLocator(nbins=5, min_n_ticks=3)
        colorbar.formatter = ScalarFormatter(useMathText=True)
        colorbar.update_ticks()


def close_periodic_mapped_scalar(
    x: np.ndarray, y: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append the periodic endpoint by duplicating the stored first column."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape != values.shape:
        raise ValueError("mapped scalar coordinates and values must share a 2D shape")
    if not all(np.isfinite(array).all() for array in (x, y, values)):
        raise FloatingPointError("mapped scalar coordinates and values must be finite")
    closed_x = np.concatenate(
        (x, np.full_like(x[:, :1], PHYSICAL.L)), axis=1
    )
    return (
        closed_x,
        np.concatenate((y, y[:, :1]), axis=1),
        np.concatenate((values, values[:, :1]), axis=1),
    )


def _mapped_panel(
    dataset: dict[str, object],
    values: np.ndarray,
    limits: tuple[float, float],
    *,
    cmap: str,
    label: str,
    output_stem: Path,
    readable_error_ticks: bool = False,
) -> tuple[Path, Path]:
    """Render one periodically closed mapped scalar without altering its samples."""
    import matplotlib.pyplot as plt

    levels = np.linspace(limits[0], limits[1], 49)
    plot_x, plot_y, plot_values = close_periodic_mapped_scalar(
        dataset["x"], dataset["y"], values
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    artist = ax.contourf(
        plot_x,
        plot_y,
        plot_values,
        levels=levels,
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
    )
    _compact_colorbar(
        fig, ax, artist, label, readable_error_ticks=readable_error_ticks
    )
    _format_axes(ax)
    return save_figure_pair(fig, output_stem)


def _speed_panel(
    dataset: dict[str, object],
    *,
    source: str,
    limits: tuple[float, float],
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render speed and streamlines using the accepted FEM interpolation logic."""
    import matplotlib.pyplot as plt

    fields = dataset["fields"]
    suffix = "fem" if source == "fem" else "pinn_seed1"
    cartesian = close_periodic_velocity_grid(cartesian_velocity_grid(
        {
            "x": dataset["x"],
            "y": dataset["y"],
            "eta": dataset["eta"],
            "u": fields[f"u_{suffix}"],
            "v": fields[f"v_{suffix}"],
        }
    ))
    xx, yy = np.meshgrid(cartesian["x"], cartesian["y"])
    speed = np.ma.masked_invalid(cartesian["speed"])
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    artist = ax.pcolormesh(
        xx,
        yy,
        speed,
        shading="auto",
        cmap="viridis",
        vmin=limits[0],
        vmax=limits[1],
    )
    draw_public_streamlines(ax, cartesian)
    _compact_colorbar(fig, ax, artist, r"$|\mathbf{u}|$")
    _format_axes(ax)
    return save_figure_pair(fig, output_stem)


def render_panel(
    name: str,
    dataset: dict[str, object],
    limits: dict[str, tuple[float, float]],
    stems: dict[str, Path] | None = None,
) -> tuple[Path, Path]:
    """Render exactly one named standalone panel from the common dataset."""
    stems = OUTPUT_STEMS if stems is None else stems
    if name not in stems:
        raise ValueError(f"unknown validation panel: {name}")
    if name in STREAMLINE_PANELS:
        source = "fem" if name == "speed_fem" else "pinn"
        return _speed_panel(
            dataset,
            source=source,
            limits=limits["speed"],
            output_stem=stems[name],
        )

    fields = dataset["fields"]
    if name.startswith("pressure_"):
        scale = limits["pressure"] if "error" not in name else error_color_limits(fields[name])
        cmap, label = ("coolwarm", r"$\widetilde p^0$") if "error" not in name else ("magma", r"$|\Delta\widetilde p^0|$")
    elif name.startswith("u_"):
        scale = limits["u"] if "error" not in name else error_color_limits(fields[name])
        cmap, label = ("viridis", "$u$") if "error" not in name else ("magma", r"$|\Delta u|$")
    elif name.startswith("v_"):
        scale = limits["v"] if "error" not in name else error_color_limits(fields[name])
        cmap, label = ("coolwarm", "$v$") if "error" not in name else ("magma", r"$|\Delta v|$")
    else:
        scale, cmap, label = error_color_limits(fields[name]), "magma", r"$|\Delta\mathbf{u}|$"
    return _mapped_panel(
        dataset,
        fields[name],
        scale,
        cmap=cmap,
        label=label,
        output_stem=stems[name],
        readable_error_ticks="error" in name,
    )


def selected_panels(selection: str) -> tuple[str, ...]:
    """Return the exact panel names selected by the compact CLI."""
    if selection == "all":
        return PANEL_NAMES
    return GROUPS[selection]


def generate_panels(
    selection: str = "all", results_root: Path = DEFAULT_RESULT_ROOT
) -> list[Path]:
    """Load the comparison data once and generate the selected panel pairs."""
    dataset = load_comparison_dataset(results_root)
    limits = solution_color_limits(dataset["fields"])
    stems = output_stems(results_root)
    outputs: list[Path] = []
    for name in selected_panels(selection):
        outputs.extend(render_panel(name, dataset, limits, stems))
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the four deliberately small validation-panel groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("speed", "pressure", "components", "all"),
        default="all",
    )
    add_result_root_arguments(parser)
    args = parser.parse_args(argv)
    args.selected_result_root = parse_result_root(parser, args)
    return args


def main(argv: list[str] | None = None) -> int:
    """Generate only the requested standalone validation panels."""
    args = parse_args(argv)
    outputs = generate_panels(args.only, args.selected_result_root)
    print("Created validation figure outputs:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
