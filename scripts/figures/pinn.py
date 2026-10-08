"""Generate final PINN explanatory figures without training or model loading.

The collocation renderer calls the deterministic point generator
directly.  Importing this module creates no figures or scientific objects.
"""

from __future__ import annotations

import argparse
import json
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
import torch  # noqa: E402

from _common import save_figure_pair  # noqa: E402
from corrugated_stokes.config import ADAM_POINTS, PHYSICAL, PINN_SCALES  # noqa: E402
from corrugated_stokes.pinn import (  # noqa: E402
    dimensionless_lower_wall,
    dimensionless_upper_wall,
    generate_point_bundle,
)


def output_stems(results_root: Path) -> dict[str, Path]:
    """Return all PINN figure stems below one selected result root."""

    output_dir = Path(results_root) / "figures" / "pinn"
    return {
        "collocation": output_dir / "collocation_adam",
        "adam_total": output_dir / "adam_total",
        "lbfgs_total": output_dir / "lbfgs_total",
    }


def training_paths(results_root: Path) -> dict[int, Path]:
    """Return the three stored histories below one selected result root."""

    return {
        seed: Path(results_root) / "pinn" / f"seed_{seed}" / "training.json"
        for seed in (0, 1, 2)
    }


_DEFAULT_STEMS = output_stems(DEFAULT_RESULT_ROOT)
OUTPUT_DIR = _DEFAULT_STEMS["collocation"].parent
COLLOCATION_STEM = _DEFAULT_STEMS["collocation"]
ADAM_TOTAL_STEM = _DEFAULT_STEMS["adam_total"]
LBFGS_TOTAL_STEM = _DEFAULT_STEMS["lbfgs_total"]
TRAINING_PATHS = training_paths(DEFAULT_RESULT_ROOT)
EXPECTED_FILENAMES = tuple(
    f"{name}.{suffix}"
    for name in (
        "collocation_adam",
        "adam_total",
        "lbfgs_total",
    )
    for suffix in ("png", "pdf")
)
HISTORY_LENGTHS = {"adam": 16500, "lbfgs": 500}
HISTORY_LINEWIDTH = 1.05
SEED_STYLES = {
    0: {"color": "#67B7E1", "linestyle": "-", "marker": None},
    1: {"color": "#2D3138", "linestyle": "-", "marker": None},
    2: {"color": "#6FA85B", "linestyle": "-", "marker": None},
}


def load_adam_collocation():
    """Return the exact fixed Adam point bundle used by scientific training."""
    return generate_point_bundle(ADAM_POINTS)


def extract_training_history(
    training: dict, stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return true stored indices and unsmoothed totals using earlier semantics."""
    if stage not in HISTORY_LENGTHS:
        raise ValueError("stage must be 'adam' or 'lbfgs'")
    entries = training.get(f"{stage}_history")
    expected_length = HISTORY_LENGTHS[stage]
    if not isinstance(entries, list) or len(entries) != expected_length:
        raise ValueError(f"{stage} history must contain exactly {expected_length} entries")
    if any(not isinstance(entry, dict) or entry.get("stage") != stage for entry in entries):
        raise ValueError(f"incorrect stage label in {stage} history")
    evaluations = np.asarray([entry.get("evaluation") for entry in entries], dtype=np.int64)
    losses = np.asarray([entry.get("total") for entry in entries], dtype=np.float64)
    expected = np.arange(1, expected_length + 1, dtype=np.int64)
    if not np.array_equal(evaluations, expected):
        raise ValueError(f"{stage} evaluations must be exactly 1 through {expected_length}")
    if not np.isfinite(losses).all() or np.any(losses <= 0.0):
        raise ValueError(f"{stage} totals must be finite and strictly positive")
    return evaluations, losses


def load_training_records(
    paths: dict[int, Path] | None = None,
) -> dict[int, dict]:
    """Load and validate the three completed stored-history JSON files."""
    paths = TRAINING_PATHS if paths is None else paths
    if set(paths) != {0, 1, 2}:
        raise ValueError("training sources must contain exactly seeds 0, 1, and 2")
    records: dict[int, dict] = {}
    for seed in (0, 1, 2):
        path = Path(paths[seed])
        if not path.is_file():
            raise FileNotFoundError(f"missing training history source: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("seed") != seed:
            raise ValueError(f"training history seed mismatch in {path}")
        extract_training_history(payload, "adam")
        extract_training_history(payload, "lbfgs")
        records[seed] = payload
    return records


def scaled_wall_curves(samples: int = 1001) -> dict[str, np.ndarray]:
    """Evaluate exact scaled walls and the optional near-wall thresholds."""
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer of at least two")
    X = torch.linspace(
        -0.5 * PINN_SCALES.ell,
        0.5 * PINN_SCALES.ell,
        samples,
        dtype=torch.float64,
    ).reshape(-1, 1)
    upper = dimensionless_upper_wall(X).detach().cpu().numpy().ravel()
    lower = dimensionless_lower_wall(X).detach().cpu().numpy().ravel()
    return {
        "x": X.detach().cpu().numpy().ravel(),
        "upper": upper,
        "lower": lower,
        "near_upper": 0.9 * upper,
        "near_lower": 0.9 * lower,
    }


def make_collocation_figure(
    output_stem: Path = COLLOCATION_STEM,
) -> tuple[Path, Path]:
    """Plot the exact deterministic Adam interior, wall, and periodic points."""
    import matplotlib.pyplot as plt

    interior, walls, periodic = load_adam_collocation()
    x_interior = interior["x"].detach().cpu().numpy().ravel()
    y_interior = interior["y"].detach().cpu().numpy().ravel()
    region = interior["region"].detach().cpu().numpy().ravel()
    ordinary = region == 0
    near_wall = region == 1
    wall_x = walls["x"].detach().cpu().numpy().ravel()
    wall_y = walls["y"].detach().cpu().numpy().ravel()
    left_x = periodic["x_left"].detach().cpu().numpy().ravel()
    left_y = periodic["y_left"].detach().cpu().numpy().ravel()
    right_x = periodic["x_right"].detach().cpu().numpy().ravel()
    right_y = periodic["y_right"].detach().cpu().numpy().ravel()
    curves = scaled_wall_curves()

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(curves["x"], curves["upper"], color="black", linewidth=1.2, zorder=5)
    ax.plot(curves["x"], curves["lower"], color="black", linewidth=1.2, zorder=5)
    ax.plot(
        curves["x"], curves["near_upper"], color="0.55", linewidth=0.7,
        linestyle="--", zorder=1, label=r"$|\eta|=0.9$",
    )
    ax.plot(
        curves["x"], curves["near_lower"], color="0.55", linewidth=0.7,
        linestyle="--", zorder=1,
    )
    ax.scatter(
        x_interior[ordinary], y_interior[ordinary], s=6, marker="o",
        color="#3973ac", alpha=0.62, linewidths=0, zorder=2,
        label="ordinary interior",
    )
    ax.scatter(
        x_interior[near_wall], y_interior[near_wall], s=8, marker="o",
        color="#dd6b20", alpha=0.78, linewidths=0, zorder=3,
        label="near-wall interior",
    )
    ax.scatter(
        wall_x, wall_y, s=10, marker="o", facecolors="white",
        edgecolors="black", linewidths=0.55, zorder=6, label="wall",
    )
    ax.scatter(
        left_x, left_y, s=11, marker="s", color="#238b45", alpha=0.8,
        linewidths=0, zorder=4, label="periodic boundaries",
    )
    ax.scatter(
        right_x, right_y, s=11, marker="s", color="#238b45", alpha=0.8,
        linewidths=0, zorder=4,
    )

    maximum_half_height = 0.5 * (1.0 + PINN_SCALES.relative_amplitude)
    ax.set_xlim(-0.5 * PINN_SCALES.ell, 0.5 * PINN_SCALES.ell)
    ax.set_ylim(-1.08 * maximum_half_height, 1.08 * maximum_half_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
        frameon=False, fontsize=10, handletextpad=0.45, columnspacing=1.0,
    )
    return save_figure_pair(fig, output_stem)


def _make_total_history_figure(
    stage: str,
    output_stem: Path,
    trainings: dict[int, dict] | None = None,
) -> tuple[Path, Path]:
    """Adapt the earlier three-seed unsmoothed semilog history plot."""
    import matplotlib.pyplot as plt

    records = load_training_records() if trainings is None else trainings
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for seed in (0, 1, 2):
        evaluations, totals = extract_training_history(records[seed], stage)
        ax.semilogy(
            evaluations,
            totals,
            linewidth=HISTORY_LINEWIDTH,
            **SEED_STYLES[seed],
            label=f"seed {seed}",
        )
    ax.set_xlabel("Adam step" if stage == "adam" else "L-BFGS closure evaluation")
    ax.set_ylabel("Total loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return save_figure_pair(fig, output_stem)


def make_adam_total_figure(
    output_stem: Path = ADAM_TOTAL_STEM,
    trainings: dict[int, dict] | None = None,
) -> tuple[Path, Path]:
    """Plot the three exact stored Adam total-loss histories."""
    return _make_total_history_figure("adam", output_stem, trainings)


def make_lbfgs_total_figure(
    output_stem: Path = LBFGS_TOTAL_STEM,
    trainings: dict[int, dict] | None = None,
) -> tuple[Path, Path]:
    """Plot the three exact stored Strong-Wolfe closure-loss histories."""
    return _make_total_history_figure("lbfgs", output_stem, trainings)


def generate_history_figures(results_root: Path = DEFAULT_RESULT_ROOT) -> list[Path]:
    """Generate the two stored total-history figure pairs."""
    trainings = load_training_records(training_paths(results_root))
    stems = output_stems(results_root)
    outputs: list[Path] = []
    outputs.extend(make_adam_total_figure(stems["adam_total"], trainings))
    outputs.extend(make_lbfgs_total_figure(stems["lbfgs_total"], trainings))
    return outputs


def generate_all_figures(results_root: Path = DEFAULT_RESULT_ROOT) -> list[Path]:
    """Generate the collocation pair followed by both history pairs."""
    return [
        *make_collocation_figure(output_stems(results_root)["collocation"]),
        *generate_history_figures(results_root),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit PINN figure generation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=(
            "collocation",
            "adam-total",
            "lbfgs-total",
            "histories",
        ),
        default=None,
    )
    add_result_root_arguments(parser)
    args = parser.parse_args(argv)
    args.selected_result_root = parse_result_root(parser, args)
    return args


def main(argv: list[str] | None = None) -> int:
    """Generate only the requested PINN figure pair or history group."""
    args = parse_args(argv)
    selection = args.only
    results_root = args.selected_result_root
    stems = output_stems(results_root)
    if selection is None:
        outputs = generate_all_figures(results_root)
    elif selection == "collocation":
        outputs = list(make_collocation_figure(stems["collocation"]))
    elif selection == "adam-total":
        outputs = list(
            make_adam_total_figure(
                stems["adam_total"], load_training_records(training_paths(results_root))
            )
        )
    elif selection == "lbfgs-total":
        outputs = list(
            make_lbfgs_total_figure(
                stems["lbfgs_total"], load_training_records(training_paths(results_root))
            )
        )
    else:
        outputs = generate_history_figures(results_root)
    print("Created PINN figure outputs:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
