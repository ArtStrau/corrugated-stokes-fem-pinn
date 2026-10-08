"""Generate final FEM field figures and the illustrative mapped mesh.

The mapped scalar-field and Cartesian streamline logic adapts the earlier
accepted FEM postprocessing implementation.  No FEM system is assembled or
solved by this module.  The optional README overview periodically repeats the
stored single-period solution; it is a visualization, not a new computation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = SCRIPT_DIR.parent
SRC = PROJECT_ROOT / "src"
for search_path in (SCRIPT_DIR, SCRIPTS_DIR, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from _common import save_figure_pair  # noqa: E402
from _result_root import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    add_result_root_arguments,
    parse_result_root,
)
from corrugated_stokes.config import FEM_REFERENCE, PHYSICAL  # noqa: E402
from corrugated_stokes.geometry import (  # noqa: E402
    lower_wall,
    make_mapped_mesh,
    upper_wall,
)


def reference_paths(results_root: Path) -> tuple[Path, Path]:
    """Return the NPZ and JSON FEM references below one selected root."""

    fem_root = Path(results_root) / "fem"
    return fem_root / "reference.npz", fem_root / "reference.json"


def output_stems(results_root: Path) -> dict[str, Path]:
    """Return every FEM figure stem below one selected root."""

    output_dir = Path(results_root) / "figures" / "fem"
    return {
        "mesh": output_dir / "mesh",
        "u": output_dir / "u",
        "v": output_dir / "v",
        "speed_streamlines": output_dir / "speed_streamlines",
        "pressure_tilde": output_dir / "pressure_tilde",
        "flow_overview": output_dir / "flow_overview",
    }


REFERENCE_NPZ, REFERENCE_JSON = reference_paths(DEFAULT_RESULT_ROOT)
OUTPUT_STEMS = output_stems(DEFAULT_RESULT_ROOT)
OUTPUT_DIR = OUTPUT_STEMS["mesh"].parent
EXPECTED_FILENAMES = tuple(
    f"{name}.{suffix}"
    for name in (
        "mesh",
        "u",
        "v",
        "speed_streamlines",
        "pressure_tilde",
        "flow_overview",
    )
    for suffix in ("png", "pdf")
)
MESH_NX = 16
MESH_NY = 8
FLOW_OVERVIEW_PERIODS = 3
PUBLIC_STREAMLINE_COUNT = 7
PUBLIC_STREAMLINE_WALL_MARGIN_FRACTION = 0.10
PUBLIC_STREAMLINE_BUFFER_INTERVALS = 3
# With explicit starts, density controls only Matplotlib's integration mask;
# this fine mask prevents adjacent prescribed starts from being discarded.
PUBLIC_STREAMLINE_INTEGRATION_MASK_DENSITY = 3.0
PUBLIC_VERTICAL_COLORBAR_PAD = 0.20
PUBLIC_STREAMLINE_LINEWIDTH = 0.7
# The README overview alone uses the selected Option 10 presentation.  The
# default remains black so every technical streamline panel is unchanged.
HERO_STREAMLINE_COLOR = "#A0A7AE"
HERO_STREAMLINE_OUTLINE_COLOR = "#586068"
HERO_STREAMLINE_OUTLINE_LINEWIDTH = 1.05


def _require_finite(value: Any, location: str = "metadata") -> None:
    """Reject non-finite JSON numbers recursively."""
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite(item, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {location}")


def load_reference_metadata(path: Path = REFERENCE_JSON) -> dict[str, Any]:
    """Load and validate the reference settings."""
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("FEM reference metadata must be a JSON object")
    _require_finite(metadata)
    expected_physical = {
        "L": PHYSICAL.L,
        "Re": PHYSICAL.Re,
        "delta_p": PHYSICAL.delta_p,
        "mu": PHYSICAL.mu,
        "width_max": PHYSICAL.width_max,
        "width_min": PHYSICAL.width_min,
    }
    if metadata.get("physical") != expected_physical:
        raise ValueError("FEM reference physical configuration mismatch")
    if metadata.get("fem") != {
        "nx": FEM_REFERENCE.nx,
        "ny": FEM_REFERENCE.ny,
        "integration_order": FEM_REFERENCE.integration_order,
        "flux_quadrature_order": FEM_REFERENCE.flux_quadrature_order,
    }:
        raise ValueError("FEM reference discretization mismatch")
    return metadata


def load_reference_grid(path: Path = REFERENCE_NPZ) -> dict[str, np.ndarray]:
    """Load the stored mapped Gauss-eta by endpoint-excluded-x field grid."""
    required = ("x", "y", "u", "v", "p_tilde", "eta", "quadrature_weights")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"FEM reference is missing arrays: {', '.join(missing)}")
        flat = {name: np.asarray(archive[name], dtype=np.float64).ravel() for name in required}

    size = flat["x"].size
    if size == 0 or any(values.size != size for values in flat.values()):
        raise ValueError("FEM reference field arrays must be non-empty and equally sized")
    if not all(np.isfinite(values).all() for values in flat.values()):
        raise ValueError("FEM reference contains non-finite field values")
    if np.any(flat["quadrature_weights"] <= 0.0):
        raise ValueError("FEM quadrature weights must be positive")

    eta_changes = np.flatnonzero(np.diff(flat["eta"]) != 0.0)
    if eta_changes.size == 0:
        raise ValueError("FEM reference does not contain multiple eta rows")
    nx = int(eta_changes[0] + 1)
    if size % nx:
        raise ValueError("FEM reference cannot be reshaped as an x/eta grid")
    neta = size // nx
    grid = {name: values.reshape(neta, nx) for name, values in flat.items()}
    if not np.all(grid["x"] == grid["x"][0:1, :]):
        raise ValueError("FEM x coordinates do not repeat across eta rows")
    if not np.all(grid["eta"] == grid["eta"][:, 0:1]):
        raise ValueError("FEM eta coordinates are not constant along x rows")
    if np.any(np.diff(grid["x"][0]) <= 0.0) or np.any(np.diff(grid["eta"][:, 0]) <= 0.0):
        raise ValueError("FEM structured coordinates must be strictly ordered")
    return grid


def weighted_zero_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Select the physical-area-weighted zero-mean pressure representative."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or np.any(weights <= 0.0):
        raise ValueError("pressure and positive physical-area weights must match")
    return values - np.sum(weights * values) / np.sum(weights)


def cartesian_velocity_grid(
    reference: dict[str, np.ndarray],
    y_points: int = 161,
) -> dict[str, np.ndarray]:
    """Interpolate stored sections to the earlier Cartesian streamline grid.

    The reference uses uniform periodic x locations and mapped Gauss nodes in
    eta.  Each section is interpolated only in y, with exact zero velocity
    appended at its represented walls.  Values outside the physical channel
    remain NaN and are masked before plotting.
    """
    if not isinstance(y_points, int) or y_points < 3:
        raise ValueError("y_points must be an integer of at least three")
    x = reference["x"][0].copy()
    eta = reference["eta"][:, 0]
    y = np.linspace(-0.5 * PHYSICAL.width_max, 0.5 * PHYSICAL.width_max, y_points)
    u = np.full((y.size, x.size), np.nan, dtype=np.float64)
    v = np.full_like(u, np.nan)
    exact_lower = lower_wall(x)
    exact_upper = upper_wall(x)

    centered_eta = eta - np.mean(eta)
    eta_scale = float(np.dot(centered_eta, centered_eta))
    for column in range(x.size):
        stored_y = reference["y"][:, column]
        half_width = float(np.dot(centered_eta, stored_y - np.mean(stored_y)) / eta_scale)
        center = float(np.mean(stored_y) - half_width * np.mean(eta))
        represented_lower, represented_upper = center - half_width, center + half_width
        lower = max(represented_lower, float(exact_lower[column]))
        upper = min(represented_upper, float(exact_upper[column]))
        inside = (y >= lower) & (y <= upper)
        interpolation_y = np.concatenate(([represented_lower], stored_y, [represented_upper]))
        interpolation_u = np.concatenate(([0.0], reference["u"][:, column], [0.0]))
        interpolation_v = np.concatenate(([0.0], reference["v"][:, column], [0.0]))
        u[inside, column] = np.interp(y[inside], interpolation_y, interpolation_u)
        v[inside, column] = np.interp(y[inside], interpolation_y, interpolation_v)
    return {"x": x, "y": y, "u": u, "v": v, "speed": np.hypot(u, v)}


def close_periodic_velocity_grid(field: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Append the periodic right endpoint for visually symmetric streamlines."""
    x = np.asarray(field["x"], dtype=np.float64)
    y = np.asarray(field["y"], dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("periodic field coordinates must be ordered one-dimensional arrays")
    if not np.isclose(x[0], 0.0) or x[-1] >= PHYSICAL.L:
        raise ValueError("periodic one-cell field must be endpoint-excluded on [0,L)")
    closed = {"x": np.concatenate((x, [PHYSICAL.L])), "y": y.copy()}
    for name in ("u", "v", "speed"):
        values = np.asarray(field[name], dtype=np.float64)
        if values.shape != (y.size, x.size):
            raise ValueError(f"periodic field {name} has an incompatible shape")
        closed[name] = np.concatenate((values, values[:, :1]), axis=1)
    return closed


def public_streamline_start_points(x_start: float = 0.0) -> np.ndarray:
    """Return the seven selected symmetric streamline start points."""
    x_start = float(x_start)
    lower = float(lower_wall(np.asarray([x_start]))[0])
    upper = float(upper_wall(np.asarray([x_start]))[0])
    margin = PUBLIC_STREAMLINE_WALL_MARGIN_FRACTION * (upper - lower)
    y_starts = np.linspace(
        lower + margin,
        upper - margin,
        PUBLIC_STREAMLINE_COUNT,
    )
    points = np.column_stack((np.full(PUBLIC_STREAMLINE_COUNT, x_start), y_starts))
    if not np.isclose(points[PUBLIC_STREAMLINE_COUNT // 2, 1], 0.0):
        raise RuntimeError("streamline starts must include the centerline")
    if not np.allclose(points[:, 1], -points[::-1, 1]):
        raise RuntimeError("streamline starts must be symmetric about y=0")
    return points


def draw_public_streamlines(
    ax,
    field: dict[str, np.ndarray],
    *,
    periods: int = 1,
    arrowsize: float = 0.8,
    color: str = "black",
    outline_color: str | None = None,
    outline_linewidth: float | None = None,
) -> None:
    """Draw seven forward-integrated streamlines in every displayed period.

    Each period is integrated separately from the same physical inlet y
    positions.  This gives exact left/right endpoint coverage and ensures that
    the three-period README overview has visible arrowheads in every cell.
    """
    if not isinstance(periods, int) or periods < 1:
        raise ValueError("periods must be a positive integer")
    if (outline_color is None) != (outline_linewidth is None):
        raise ValueError("streamline outline color and linewidth must be provided together")
    if outline_linewidth is not None and outline_linewidth <= PUBLIC_STREAMLINE_LINEWIDTH:
        raise ValueError("streamline outline must be wider than the visible inner line")
    x = np.asarray(field["x"], dtype=np.float64)
    y = np.asarray(field["y"], dtype=np.float64)
    u = np.ma.masked_invalid(field["u"])
    v = np.ma.masked_invalid(field["v"])
    tolerance = 16.0 * np.finfo(np.float64).eps * max(1.0, periods * PHYSICAL.L)
    for period in range(periods):
        left = period * PHYSICAL.L
        right = (period + 1) * PHYSICAL.L
        section = (x >= left - tolerance) & (x <= right + tolerance)
        section_x = x[section]
        if section_x.size < 2 or not np.isclose(section_x[0], left) or not np.isclose(section_x[-1], right):
            raise ValueError("streamline field must include both endpoints of every period")
        spacing = float(section_x[1] - section_x[0])
        if spacing <= 0.0 or not np.allclose(np.diff(section_x), spacing):
            raise ValueError("streamline integration grid must be uniformly spaced")
        section_u = u[:, section]
        section_v = v[:, section]
        buffer_count = PUBLIC_STREAMLINE_BUFFER_INTERVALS
        if section_x.size <= buffer_count:
            raise ValueError("streamline integration grid is too short for its buffer")
        # Matplotlib does not render the seed-to-first-step interval.  Extending
        # three periodic intervals beyond both boundaries ensures that its first
        # rendered segment already crosses the visible inlet, with no inset.
        left_buffer_x = left + spacing * np.arange(-buffer_count, 0)
        right_buffer_x = right + spacing * np.arange(1, buffer_count + 1)
        integration_x = np.concatenate((left_buffer_x, section_x, right_buffer_x))
        integration_u = np.ma.concatenate(
            (section_u[:, -(buffer_count + 1):-1], section_u, section_u[:, 1:buffer_count + 1]),
            axis=1,
        )
        integration_v = np.ma.concatenate(
            (section_v[:, -(buffer_count + 1):-1], section_v, section_v[:, 1:buffer_count + 1]),
            axis=1,
        )
        starts = public_streamline_start_points(left)
        starts[:, 0] = integration_x[0]
        patch_count = len(ax.patches) if outline_color is not None else 0
        stream = ax.streamplot(
            integration_x,
            y,
            integration_u,
            integration_v,
            start_points=starts,
            integration_direction="forward",
            density=PUBLIC_STREAMLINE_INTEGRATION_MASK_DENSITY,
            broken_streamlines=False,
            color=color,
            linewidth=PUBLIC_STREAMLINE_LINEWIDTH,
            arrowsize=arrowsize,
        )
        if outline_color is not None:
            import matplotlib.patheffects as path_effects

            effects = [
                path_effects.Stroke(
                    linewidth=outline_linewidth, foreground=outline_color
                ),
                path_effects.Normal(),
            ]
            stream.lines.set_path_effects(effects)
            stream.arrows.set_path_effects(effects)
            # Matplotlib adds individual FancyArrowPatch artists to the axes;
            # applying the same path effect keeps their heads as legible as the
            # outlined streamline curves.
            for arrow in ax.patches[patch_count:]:
                arrow.set_path_effects(effects)


def _format_axes(ax, periods: int = 1) -> None:
    """Apply the accepted physical-wall and Cartesian-axis presentation."""
    if not isinstance(periods, int) or periods < 1:
        raise ValueError("periods must be a positive integer")
    x_wall = np.linspace(0.0, periods * PHYSICAL.L, 1000 * periods + 1)
    ax.plot(x_wall, lower_wall(x_wall), color="black", linewidth=0.9)
    ax.plot(x_wall, upper_wall(x_wall), color="black", linewidth=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, periods * PHYSICAL.L)
    ax.set_ylim(-0.55 * PHYSICAL.width_max, 0.55 * PHYSICAL.width_max)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def _compact_colorbar(fig, ax, artist, label: str) -> None:
    """Attach one rectangular colorbar exactly aligned to the panel height."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes(
        "right", size="4.5%", pad=PUBLIC_VERTICAL_COLORBAR_PAD
    )
    fig.colorbar(artist, cax=colorbar_ax, label=label)


def load_illustrative_mesh() -> dict[str, Any]:
    """Build the prescribed 16-by-8 mapped mesh without assembling a FEM system."""
    return make_mapped_mesh(physical=PHYSICAL, nx=MESH_NX, ny=MESH_NY)


def make_mesh_figure(
    output_stem: Path = OUTPUT_STEMS["mesh"],
) -> tuple[Path, Path]:
    """Render the actual mapped quadrilateral edges for the illustrative level."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    mesh = load_illustrative_mesh()["mesh"]
    segments = mesh.p[:, mesh.facets].transpose(2, 1, 0)
    fig, ax = plt.subplots(figsize=(8.4, 3.7), constrained_layout=True)
    ax.add_collection(
        LineCollection(segments, colors="0.45", linewidths=0.55, zorder=1)
    )
    _format_axes(ax)
    return save_figure_pair(fig, output_stem)


def make_mapped_field_figure(
    reference: dict[str, np.ndarray],
    values: np.ndarray,
    output_stem: Path,
    label: str,
    cmap: str,
    signed: bool = False,
) -> tuple[Path, Path]:
    """Render one stored scalar field directly on its curved mapped grid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 3.7), constrained_layout=True)
    if signed:
        limit = max(float(np.max(np.abs(values))), np.finfo(np.float64).eps)
        levels = np.linspace(-limit, limit, 49)
    else:
        lower = float(np.min(values))
        upper = float(np.max(values))
        if upper <= lower:
            upper = lower + np.finfo(np.float64).eps
        levels = np.linspace(lower, upper, 49)
    # Reuse the mapped-grid contour rendering. Unlike
    # pcolormesh, contourf does not infer misleading Cartesian cell edges from
    # the non-monotone physical y coordinates of adjacent mapped sections.
    artist = ax.contourf(reference["x"], reference["y"], values, levels=levels, cmap=cmap)
    _compact_colorbar(fig, ax, artist, label)
    _format_axes(ax)
    return save_figure_pair(fig, output_stem)


def make_speed_streamline_figure(
    reference: dict[str, np.ndarray],
    output_stem: Path = OUTPUT_STEMS["speed_streamlines"],
) -> tuple[Path, Path]:
    """Render speed with the accepted masked Cartesian streamline approach."""
    import matplotlib.pyplot as plt

    field = close_periodic_velocity_grid(cartesian_velocity_grid(reference))
    speed = np.ma.masked_invalid(field["speed"])
    xx, yy = np.meshgrid(field["x"], field["y"])
    fig, ax = plt.subplots(figsize=(8.4, 3.7), constrained_layout=True)
    artist = ax.pcolormesh(xx, yy, speed, shading="auto", cmap="viridis")
    draw_public_streamlines(ax, field)
    _compact_colorbar(fig, ax, artist, "$|\\mathbf{u}|$")
    _format_axes(ax)
    return save_figure_pair(fig, output_stem)


def tile_periodic_velocity_grid(
    field: dict[str, np.ndarray], periods: int = FLOW_OVERVIEW_PERIODS
) -> dict[str, np.ndarray]:
    """Repeat one endpoint-excluded periodic field over consecutive periods."""
    if periods != FLOW_OVERVIEW_PERIODS:
        raise ValueError(f"the README overview requires exactly {FLOW_OVERVIEW_PERIODS} periods")
    x = np.asarray(field["x"], dtype=np.float64)
    y = np.asarray(field["y"], dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("overview coordinates must be ordered one-dimensional arrays")
    tiled = {
        "x": np.concatenate(
            [*(x + period * PHYSICAL.L for period in range(periods)), [periods * PHYSICAL.L]]
        ),
        "y": y.copy(),
    }
    for name in ("u", "v", "speed"):
        values = np.asarray(field[name], dtype=np.float64)
        if values.shape != (y.size, x.size):
            raise ValueError(f"overview field {name} has an incompatible shape")
        tiled[name] = np.concatenate((np.tile(values, (1, periods)), values[:, :1]), axis=1)
    return tiled


def make_flow_overview_figure(
    reference: dict[str, np.ndarray],
    output_stem: Path = OUTPUT_STEMS["flow_overview"],
) -> tuple[Path, Path]:
    """Render three periodic copies of the stored one-period FEM velocity field."""
    import matplotlib.pyplot as plt
    one_period = cartesian_velocity_grid(reference)
    field = tile_periodic_velocity_grid(one_period)
    speed = np.ma.masked_invalid(field["speed"])
    xx, yy = np.meshgrid(field["x"], field["y"])
    speed_min = float(np.nanmin(one_period["speed"]))
    speed_max = float(np.nanmax(one_period["speed"]))
    fig = plt.figure(figsize=(13.0, 2.8))
    ax = fig.add_axes([0.015, 0.20, 0.97, 0.78])
    artist = ax.pcolormesh(
        xx,
        yy,
        speed,
        shading="auto",
        cmap="viridis",
        vmin=speed_min,
        vmax=speed_max,
    )
    draw_public_streamlines(
        ax,
        field,
        periods=FLOW_OVERVIEW_PERIODS,
        arrowsize=0.9,
        color=HERO_STREAMLINE_COLOR,
        outline_color=HERO_STREAMLINE_OUTLINE_COLOR,
        outline_linewidth=HERO_STREAMLINE_OUTLINE_LINEWIDTH,
    )
    x_wall = np.linspace(
        0.0, FLOW_OVERVIEW_PERIODS * PHYSICAL.L, 1000 * FLOW_OVERVIEW_PERIODS + 1
    )
    ax.plot(x_wall, lower_wall(x_wall), color="black", linewidth=0.9)
    ax.plot(x_wall, upper_wall(x_wall), color="black", linewidth=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, FLOW_OVERVIEW_PERIODS * PHYSICAL.L)
    ax.set_ylim(-0.55 * PHYSICAL.width_max, 0.55 * PHYSICAL.width_max)
    ax.set_axis_off()

    colorbar_ax = fig.add_axes([0.35, 0.07, 0.30, 0.045])
    colorbar = fig.colorbar(artist, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([speed_min, speed_max])
    colorbar.set_ticklabels(["slower", "faster"])
    colorbar.set_label("Flow speed", labelpad=2)
    colorbar.ax.tick_params(labelsize=8, length=2.5, pad=1.5)
    return save_figure_pair(fig, output_stem)


def generate_flow_overview(results_root: Path = DEFAULT_RESULT_ROOT) -> list[Path]:
    """Load the stored FEM reference and render only the README overview pair."""
    reference_npz, reference_json = reference_paths(results_root)
    load_reference_metadata(reference_json)
    reference = load_reference_grid(reference_npz)
    return list(make_flow_overview_figure(reference, output_stems(results_root)["flow_overview"]))


def generate_speed_streamlines(results_root: Path = DEFAULT_RESULT_ROOT) -> list[Path]:
    """Load the stored FEM reference and render only the technical speed pair."""
    reference_npz, reference_json = reference_paths(results_root)
    load_reference_metadata(reference_json)
    reference = load_reference_grid(reference_npz)
    return list(
        make_speed_streamline_figure(
            reference, output_stems(results_root)["speed_streamlines"]
        )
    )


def generate_field_figures(results_root: Path = DEFAULT_RESULT_ROOT) -> list[Path]:
    """Load the selected-root reference once and generate exactly four field pairs."""
    reference_npz, reference_json = reference_paths(results_root)
    stems = output_stems(results_root)
    load_reference_metadata(reference_json)
    reference = load_reference_grid(reference_npz)
    outputs: list[Path] = []
    outputs.extend(
        make_mapped_field_figure(
            reference, reference["u"], stems["u"], "$u$", "viridis"
        )
    )
    outputs.extend(
        make_mapped_field_figure(
            reference, reference["v"], stems["v"], "$v$", "coolwarm", signed=True
        )
    )
    outputs.extend(make_speed_streamline_figure(reference, stems["speed_streamlines"]))
    pressure = weighted_zero_mean(reference["p_tilde"], reference["quadrature_weights"])
    outputs.extend(
        make_mapped_field_figure(
            reference,
            pressure,
            stems["pressure_tilde"],
            "$\\widetilde p$",
            "coolwarm",
            signed=True,
        )
    )
    return outputs


def generate_all(results_root: Path = DEFAULT_RESULT_ROOT) -> list[Path]:
    """Generate the complete public FEM figure set from stored reference data."""
    return [
        *make_mesh_figure(output_stems(results_root)["mesh"]),
        *generate_field_figures(results_root),
        *generate_flow_overview(results_root),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the isolated mesh/field/speed/README generation selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "mesh", "fields", "speed-streamlines", "flow-overview"),
        default="all",
        help=(
            "generate the stored figures, only the mesh, only the four "
            "field figures, only the technical speed/streamline figure, or only "
            "the three-period README flow overview"
        ),
    )
    add_result_root_arguments(parser)
    args = parser.parse_args(argv)
    args.selected_result_root = parse_result_root(parser, args)
    return args


def main(argv: list[str] | None = None) -> int:
    """Generate only the requested deterministic FEM figure files."""
    args = parse_args(argv)
    selection = args.only
    results_root = args.selected_result_root
    if selection == "mesh":
        outputs = list(make_mesh_figure(output_stems(results_root)["mesh"]))
    elif selection == "fields":
        outputs = generate_field_figures(results_root)
    elif selection == "speed-streamlines":
        outputs = generate_speed_streamlines(results_root)
    elif selection == "flow-overview":
        outputs = generate_flow_overview(results_root)
    else:
        outputs = generate_all(results_root)
    print("Created FEM figure outputs:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
