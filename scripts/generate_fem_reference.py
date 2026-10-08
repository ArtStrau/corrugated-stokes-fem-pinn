"""Generate the mapped Q2--Q1 FEM reference files on demand."""

from __future__ import annotations

import os

try:
    from scripts._native_environment import configure_native_library_environment
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _native_environment import configure_native_library_environment

configure_native_library_environment(os.name, os.environ)

import argparse
import gc
from dataclasses import asdict
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import skfem

try:
    from scripts._result_root import add_result_root_arguments, parse_result_root
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from _result_root import add_result_root_arguments, parse_result_root

from corrugated_stokes.config import FEM_REFERENCE, PHYSICAL
from corrugated_stokes.fem import cross_sectional_flux, solve_stokes
from corrugated_stokes.geometry import discrete_wall_bounds


def deterministic_metadata(diagnostics: dict) -> dict:
    """Build timestamp-free scientific metadata for byte-reproducible output."""
    return {"physical": asdict(PHYSICAL), "fem": asdict(FEM_REFERENCE),
            "diagnostics": diagnostics,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "scikit_fem": skfem.__version__},
            "training_data": False}


def output_dir(results_root: Path) -> Path:
    """Return the FEM directory below the selected top-level result root."""

    return Path(results_root) / "fem"


def generate(output_dir: Path, *, force: bool = False) -> tuple[Path, Path]:
    """Solve the fixed target and save compact field/reference metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = output_dir / "reference.npz", output_dir / "reference.json"
    if (npz_path.exists() or json_path.exists()) and not force:
        raise FileExistsError("FEM reference exists; refusing silent overwrite")
    print("FEM solve started", flush=True)
    model = solve_stokes(settings=FEM_REFERENCE)
    print("FEM solve finished", flush=True)
    diagnostics = {
        name: value
        for name, value in model["diagnostics"].items()
        if np.isscalar(value)
    }
    evaluation_model = {
        "geometry": model["geometry"],
        "velocity_at": model["velocity_at"],
        "pressure_at": model["pressure_at"],
    }
    del model
    gc.collect()

    print("field sampling started", flush=True)
    eta, wg = np.polynomial.legendre.leggauss(96)
    x = np.linspace(0, PHYSICAL.L, 257, endpoint=False)
    xx, ee = np.meshgrid(x, eta)
    lo, hi = discrete_wall_bounds(xx.ravel(), evaluation_model["geometry"])
    y = lo + 0.5 * (ee.ravel() + 1) * (hi - lo)
    velocity = evaluation_model["velocity_at"](xx.ravel(), y)
    pressure = evaluation_model["pressure_at"](xx.ravel(), y)
    print("field sampling finished", flush=True)

    print("flux sampling started", flush=True)
    flux_x = np.linspace(0, PHYSICAL.L, 513, endpoint=False)
    flux = cross_sectional_flux(
        evaluation_model, flux_x, FEM_REFERENCE.flux_quadrature_order
    )
    print("flux sampling finished", flush=True)
    np.savez(npz_path, x=xx.ravel(), y=y, u=velocity[0], v=velocity[1], p_tilde=pressure,
             eta=ee.ravel(), quadrature_weights=np.repeat(wg, x.size) * 0.5 * (hi-lo) * PHYSICAL.L/x.size,
             flux_x=flux_x, flux=flux)
    metadata = deterministic_metadata(diagnostics)
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print("reference written", flush=True)
    return npz_path, json_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_result_root_arguments(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing reference under the explicitly selected root",
    )
    args = parser.parse_args(argv)
    results_root = parse_result_root(parser, args)
    npz, metadata = generate(output_dir(results_root), force=args.force)
    print(npz)
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
