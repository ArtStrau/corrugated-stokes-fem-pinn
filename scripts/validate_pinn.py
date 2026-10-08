"""Read-only numerical validation of one completed PINN result against FEM."""

from __future__ import annotations

import os

try:
    from scripts._native_environment import configure_native_library_environment
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _native_environment import configure_native_library_environment

configure_native_library_environment(os.name, os.environ)

import argparse
from pathlib import Path

from corrugated_stokes.config import FEM_REFERENCE
from corrugated_stokes.fem import solve_stokes
from corrugated_stokes.pinn import model_state_hash
from corrugated_stokes.validation import (
    atomic_json_write, compare_with_fem, evaluate_pinn_diagnostics,
    verify_training_results,
)

try:
    from scripts._result_root import add_result_root_arguments, parse_result_root
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _result_root import add_result_root_arguments, parse_result_root


def result_paths(results_root: Path, seed: int) -> dict[str, Path]:
    """Return the read-only input and validation paths for one seed."""

    root = Path(results_root) / "pinn" / f"seed_{seed}"
    return {
        "root": root,
        "model": root / "model.pt",
        "training": root / "training.json",
        "validation": root / "validation.json",
    }


def validate(seed: int, results_root: Path) -> dict:
    """Evaluate PINN diagnostics and an independent FEM comparison read-only."""
    paths = result_paths(results_root, seed)
    model_path, training_path = paths["model"], paths["training"]
    output = paths["validation"]
    if output.exists():
        raise FileExistsError(f"validation already exists: {output}")
    model, _, consistency = verify_training_results(
        model_path, training_path, seed
    )
    state_before = model_state_hash(model)
    physics = evaluate_pinn_diagnostics(model)
    if model_state_hash(model) != state_before:
        raise RuntimeError("PINN diagnostics changed the model state")
    fem_model = solve_stokes(settings=FEM_REFERENCE)
    fem = compare_with_fem(model, fem_model)
    if model_state_hash(model) != state_before:
        raise RuntimeError("FEM comparison changed the model state")
    result = {
        "seed": seed,
        "model_file": model_path.as_posix(),
        "training_file": training_path.as_posix(),
        "training_performed": False,
        "optimizer_created": False,
        "training_consistency": consistency,
        "physics": physics,
        "fem": fem,
    }
    atomic_json_write(output, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    add_result_root_arguments(parser)
    args = parser.parse_args(argv)
    result = validate(args.seed, parse_result_root(parser, args))
    print(f"VALIDATION_COMPLETE seed={result['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
