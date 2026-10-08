"""Train one deterministic corrugated-channel Stokes PINN.

The script implements the fixed Adam schedule, periodic recovery
checkpoints, and one fresh L-BFGS stage. Validation is a separate read-only
operation in ``scripts/validate_pinn.py``.
"""

from __future__ import annotations

import os

# Set native-library controls before NumPy or PyTorch enters the process.
try:
    from scripts._native_environment import configure_native_library_environment
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _native_environment import configure_native_library_environment

configure_native_library_environment(os.name, os.environ)

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

try:
    from scripts._result_root import add_result_root_arguments, parse_result_root
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _result_root import add_result_root_arguments, parse_result_root

from corrugated_stokes.config import (
    ADAM_POINTS,
    ADAM_SCHEDULE,
    LBFGS,
    RECOVERY_INTERVAL,
    REFINED_POINTS,
)
from corrugated_stokes.pinn import (
    LOSS_COMPONENT_ORDER,
    POINT_HASH_SCHEME,
    adam_learning_rate,
    build_network,
    component_values,
    generate_point_bundle,
    loss_components,
    model_state_hash,
    point_bundle_hash,
    protocol_hash,
    protocol_record,
)


FINAL_ADAM_STEP = ADAM_SCHEDULE[-1][1]


class _LBFGSStop(RuntimeError):
    """Stop before an over-cap or non-finite closure is accepted."""


def result_paths(results_root: Path, seed: int) -> dict[str, Path]:
    """Return seed-isolated recovery and final-result paths."""

    root = Path(results_root) / "pinn" / f"seed_{seed}"
    return {
        "root": root,
        "recovery": root / "recovery",
        "model": root / "model.pt",
        "training": root / "training.json",
    }


def _atomic_torch_write_new(path: Path, payload: dict[str, Any]) -> None:
    """Write one new Torch file through a sibling temporary file."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"unresolved temporary file exists: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_write_new(path: Path, payload: dict[str, Any]) -> None:
    """Write finite JSON through a sibling temporary file."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"unresolved temporary file exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def latest_recovery(recovery_dir: Path) -> Path | None:
    """Return the highest numbered Adam recovery checkpoint."""

    candidates = list(recovery_dir.glob("adam_step_*.pt")) if recovery_dir.exists() else []
    return max(candidates, key=lambda path: int(path.stem.split("_")[-1])) if candidates else None


def _print_adam_checkpoint_progress(step: int) -> None:
    percentage = 100.0 * step / FINAL_ADAM_STEP
    print(
        f"Adam {step}/{FINAL_ADAM_STEP} [{percentage:.1f}%] checkpoint saved",
        flush=True,
    )


def _print_lbfgs_progress(evaluation: int) -> None:
    if evaluation % 50 == 0:
        print(f"L-BFGS {evaluation}/{LBFGS.maximum_closure_evaluations}", flush=True)


def _finite_parameters(model) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def _finite_history_entry(entry: dict[str, Any], evaluation: int) -> bool:
    components = entry.get("components", {})
    return (
        entry.get("evaluation") == evaluation
        and set(components) == set(LOSS_COMPONENT_ORDER)
        and all(
            math.isfinite(float(value))
            for value in (entry.get("total", math.nan), *components.values())
        )
    )


def validate_adam_history(history: list[dict[str, Any]], step: int) -> None:
    """Check the history required to continue the fixed Adam schedule."""

    if not 1 <= step <= FINAL_ADAM_STEP or len(history) != step:
        raise ValueError("Adam recovery step/history length is invalid")
    for expected, entry in enumerate(history, 1):
        if entry.get("stage") != "adam" or not _finite_history_entry(entry, expected):
            raise ValueError(f"invalid Adam history entry {expected}")
        if float(entry.get("learning_rate", math.nan)) != adam_learning_rate(expected):
            raise ValueError(f"Adam learning rate mismatch at step {expected}")


def adam_update(model, optimizer, points, step: int) -> tuple[float, dict[str, float]]:
    """Apply one full-batch Adam update with finite-value safeguards."""

    rate = adam_learning_rate(step)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    total, components = loss_components(model, *points)
    if not bool(torch.isfinite(total)) or not all(
        bool(torch.isfinite(value)) for value in components.values()
    ):
        raise FloatingPointError(f"non-finite Adam loss at step {step}")
    total.backward()
    if not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise FloatingPointError(f"non-finite Adam gradient at step {step}")
    optimizer.step()
    if not _finite_parameters(model):
        raise FloatingPointError(f"non-finite Adam parameter at step {step}")
    return float(total.detach()), component_values(components)


def save_recovery(
    path: Path,
    model,
    optimizer,
    seed: int,
    step: int,
    history: list[dict[str, Any]],
    points_hash: str,
    adam_elapsed_seconds: float,
) -> None:
    """Save the model, Adam state, history, and elapsed time needed to resume."""

    validate_adam_history(history, step)
    if not _finite_parameters(model) or not math.isfinite(adam_elapsed_seconds):
        raise ValueError("recovery state must be finite")
    payload = {
        "kind": "adam_recovery",
        "seed": seed,
        "step": step,
        "protocol_hash": protocol_hash(seed),
        "point_hash_scheme": POINT_HASH_SCHEME,
        "point_bundle_hash": points_hash,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "adam_elapsed_seconds": float(adam_elapsed_seconds),
    }
    _atomic_torch_write_new(path, payload)
    _print_adam_checkpoint_progress(step)


def load_recovery(path: Path, model, optimizer, seed: int, points_hash: str):
    """Load one compatible Adam recovery checkpoint."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("kind") != "adam_recovery" or state.get("seed") != seed:
        raise RuntimeError("recovery checkpoint seed/type mismatch")
    if state.get("protocol_hash") != protocol_hash(seed):
        raise RuntimeError("recovery checkpoint protocol mismatch")
    if state.get("point_hash_scheme") != POINT_HASH_SCHEME:
        raise RuntimeError("recovery checkpoint point-hash scheme mismatch")
    if state.get("point_bundle_hash") != points_hash:
        raise RuntimeError("recovery checkpoint point mismatch")
    step = state.get("step")
    history = state.get("history")
    elapsed = state.get("adam_elapsed_seconds")
    if not isinstance(step, int) or not isinstance(history, list):
        raise RuntimeError("recovery checkpoint step/history is invalid")
    validate_adam_history(history, step)
    if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)):
        raise RuntimeError("recovery checkpoint elapsed time is invalid")
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if not _finite_parameters(model):
        raise RuntimeError("recovery checkpoint contains non-finite parameters")
    return step, copy.deepcopy(history), float(elapsed)


def run_lbfgs_with_strict_cap(model, points, controls=LBFGS) -> dict[str, Any]:
    """Run one fresh L-BFGS optimizer without exceeding its closure cap."""

    limit = int(controls.maximum_closure_evaluations)
    if limit <= 0:
        raise ValueError("closure cap must be positive")
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=controls.learning_rate,
        max_iter=limit,
        max_eval=limit,
        history_size=controls.history_size,
        line_search_fn=controls.line_search,
        tolerance_grad=controls.tolerance_grad,
        tolerance_change=controls.tolerance_change,
    )
    history: list[dict[str, Any]] = []
    last_finite = copy.deepcopy(model.state_dict())
    blocked_closure_requests = 0

    def closure() -> torch.Tensor:
        nonlocal last_finite, blocked_closure_requests
        if len(history) >= limit:
            blocked_closure_requests += 1
            raise _LBFGSStop("evaluation_cap")
        optimizer.zero_grad(set_to_none=True)
        total, components = loss_components(model, *points)
        if not bool(torch.isfinite(total)) or not all(
            bool(torch.isfinite(value)) for value in components.values()
        ):
            raise _LBFGSStop("nonfinite_loss")
        total.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise _LBFGSStop("nonfinite_gradient")
        history.append({
            "stage": "lbfgs",
            "evaluation": len(history) + 1,
            "total": float(total.detach()),
            "components": component_values(components),
        })
        _print_lbfgs_progress(len(history))
        last_finite = copy.deepcopy(model.state_dict())
        return total

    explicit_stop = None
    try:
        optimizer.step(closure)
    except _LBFGSStop as error:
        explicit_stop = str(error)
        model.load_state_dict(last_finite, strict=True)
        if explicit_stop != "evaluation_cap":
            raise FloatingPointError(explicit_stop) from error
    except Exception:
        model.load_state_dict(last_finite, strict=True)
        raise

    if not history or len(history) > limit:
        raise RuntimeError("L-BFGS closure-cap integrity failure")
    parameter_state = optimizer.state.get(next(iter(model.parameters())), {})
    iterations = int(parameter_state.get("n_iter", 0))
    if explicit_stop == "evaluation_cap" or len(history) >= limit:
        termination = "evaluation_cap"
    elif iterations >= limit:
        termination = "iteration_cap"
    else:
        termination = "tolerance_convergence"
    return {
        "history": history,
        "closure_evaluations": len(history),
        "termination": termination,
        "last_evaluated_model_state_hash": model_state_hash(model),
        "blocked_closure_requests": blocked_closure_requests,
    }


def _refuse_existing_final_result(paths: dict[str, Path]) -> None:
    """Prevent accidental replacement of a completed or partial final result."""

    candidates = (
        paths["model"],
        paths["training"],
        paths["model"].with_suffix(".pt.tmp"),
        paths["training"].with_suffix(".json.tmp"),
    )
    existing = [path for path in candidates if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite an existing final result: "
            + ", ".join(str(path) for path in existing)
        )


def train(seed: int, results_root: Path, resume: bool) -> dict[str, Any]:
    """Execute the fixed Adam schedule and one fresh refined-grid L-BFGS stage."""

    paths = result_paths(results_root, seed)
    paths["root"].mkdir(parents=True, exist_ok=True)
    _refuse_existing_final_result(paths)
    paths["recovery"].mkdir(parents=True, exist_ok=True)
    print(f"Seed {seed} training started", flush=True)

    model = build_network(seed)
    adam_points = generate_point_bundle(ADAM_POINTS)
    adam_hash = point_bundle_hash(adam_points)
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_SCHEDULE[0][2])
    step, history, prior_adam_seconds = 0, [], 0.0
    recovery = latest_recovery(paths["recovery"])
    if resume:
        if recovery is None:
            raise FileNotFoundError("--resume requested but no recovery checkpoint exists")
        step, history, prior_adam_seconds = load_recovery(
            recovery, model, optimizer, seed, adam_hash
        )
    elif recovery is not None:
        raise FileExistsError(
            "recovery checkpoints exist; use --resume or another result directory"
        )

    adam_started = time.perf_counter()
    for current in range(step + 1, FINAL_ADAM_STEP + 1):
        total, components = adam_update(model, optimizer, adam_points, current)
        history.append({
            "stage": "adam",
            "evaluation": current,
            "learning_rate": adam_learning_rate(current),
            "total": total,
            "components": components,
        })
        if current % RECOVERY_INTERVAL == 0 or current == FINAL_ADAM_STEP:
            elapsed = prior_adam_seconds + time.perf_counter() - adam_started
            save_recovery(
                paths["recovery"] / f"adam_step_{current:05d}.pt",
                model,
                optimizer,
                seed,
                current,
                history,
                adam_hash,
                elapsed,
            )

    adam_seconds = prior_adam_seconds + time.perf_counter() - adam_started
    final_adam_total, final_adam_components = loss_components(model, *adam_points)
    if not bool(torch.isfinite(final_adam_total)):
        raise FloatingPointError("non-finite final Adam objective")
    print("Adam complete; starting L-BFGS", flush=True)

    refined_points = generate_point_bundle(REFINED_POINTS)
    lbfgs_started = time.perf_counter()
    lbfgs_result = run_lbfgs_with_strict_cap(model, refined_points)
    lbfgs_seconds = time.perf_counter() - lbfgs_started
    final_total, final_components = loss_components(model, *refined_points)
    final_values = component_values(final_components)
    if not bool(torch.isfinite(final_total)) or not all(
        math.isfinite(value) for value in final_values.values()
    ):
        raise FloatingPointError("non-finite final L-BFGS objective")

    metadata = {
        "state": "TRAINING_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "protocol": protocol_record(seed),
        "protocol_hash": protocol_hash(seed),
        "model_state_hash": model_state_hash(model),
        "point_hash_scheme": POINT_HASH_SCHEME,
        "adam_point_hash": adam_hash,
        "refined_point_hash": point_bundle_hash(refined_points),
        "adam_steps": FINAL_ADAM_STEP,
        "lbfgs_closure_evaluations": lbfgs_result["closure_evaluations"],
        "lbfgs_termination": lbfgs_result["termination"],
        "lbfgs_blocked_closure_requests": lbfgs_result["blocked_closure_requests"],
        "adam_history": history,
        "lbfgs_history": lbfgs_result["history"],
        "final_adam_total": float(final_adam_total.detach()),
        "final_adam_components": component_values(final_adam_components),
        "final_total": float(final_total.detach()),
        "final_components": final_values,
        "training_seconds": adam_seconds + lbfgs_seconds,
        "adam_seconds": adam_seconds,
        "lbfgs_seconds": lbfgs_seconds,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "notation": {
            "width_max": "Delta Omega",
            "width_min": "Delta omega",
            "constriction_ratio": "delta=width_min/width_max",
            "epsilon": "(width_max-width_min)/L",
            "relative_amplitude": "a=A/H0",
        },
    }
    checkpoint = {
        "kind": "training_result",
        "state": "TRAINING_COMPLETE",
        "seed": seed,
        "protocol": protocol_record(seed),
        "protocol_hash": protocol_hash(seed),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "model_state_hash": metadata["model_state_hash"],
    }
    _atomic_torch_write_new(paths["model"], checkpoint)
    _atomic_json_write_new(paths["training"], metadata)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    add_result_root_arguments(parser)
    args = parser.parse_args(argv)
    result = train(args.seed, parse_result_root(parser, args), args.resume)
    print(
        f"TRAINING_COMPLETE seed={args.seed} "
        f"closures={result['lbfgs_closure_evaluations']} "
        f"termination={result['lbfgs_termination']}",
        flush=True,
    )
    root_option = (
        " --canonical"
        if args.canonical
        else f" --results-root {args.results_root}"
        if args.results_root is not None
        else ""
    )
    print(
        f"Run: {sys.executable} scripts/validate_pinn.py --seed {args.seed}{root_option}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
