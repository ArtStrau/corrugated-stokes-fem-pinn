# Technical documentation

[README.md](../README.md) is the project landing page and headline scientific
summary. The numbered documents below contain the detailed methodology and
results. This page also documents the stored numerical results used for
reproducibility, validation, and figure generation.

## Documentation guide

| Document | Purpose |
|---|---|
| [Physical problem](01_problem.md) | Defines the periodic corrugated geometry, pressure-driven Stokes equations, boundary conditions, pressure gauge, and target parameters. |
| [Analytical reference](02_analytical.md) | Derives the straight-channel Poiseuille limit and leading-order lubrication approximation, then compares analytical and FEM fluxes over the constriction-ratio sweep. |
| [Finite-element reference](03_fem.md) | Describes the mapped $Q_2$–$Q_1$ Taylor–Hood discretization, periodic reduction, convergence study, and reference FEM fields. |
| [Physics-informed neural network](04_pinn.md) | Documents the scaled streamfunction–pressure network, eight-term physics loss, deterministic collocation, optimizer schedule, and three completed training runs. |
| [Post-training validation](05_validation.md) | Reports independent physics diagnostics and the three-seed comparison with the FEM reference, including representative field figures. |

## Computational workflow and outputs

The table lists the included outputs under `results/`. With no root-selection
option, the same calculation and post-processing scripts use the
corresponding paths under the optional top-level `reproduction/` tree instead.
`--canonical` selects `results/`, while `--results-root PATH` selects another
top-level root; the selectors are mutually exclusive, and a custom root cannot
target `results/` or its descendants. Inputs and outputs remain within the same
selected root. This documentation and its included figures always refer to
`results/`.

| Stage | Script | Stored output | Used by / represented in |
|---|---|---|---|
| Analytical/FEM sweep | `scripts/run_analytical_fem_sweep.py` | `results/analytical_fem/sweep.json` | `scripts/figures/analytical.py`; [analytical documentation](02_analytical.md) and analytical figure |
| Analytical/FEM resolution check | `scripts/check_analytical_fem_resolution.py` | `results/analytical_fem/resolution_check.json` | Supporting mesh-resolution evidence in [analytical documentation](02_analytical.md) |
| FEM reference | `scripts/generate_fem_reference.py` | `results/fem/reference.npz`, `results/fem/reference.json` | FEM and validation figure renderers; [FEM documentation](03_fem.md) and [validation documentation](05_validation.md) |
| FEM convergence | `scripts/run_fem_convergence.py` | `results/fem/convergence.json` | Convergence tables in [FEM documentation](03_fem.md); an existing valid file is checked without a solve unless `--force` is supplied |
| PINN training | `scripts/train_pinn.py` | `results/pinn/seed_N/model.pt`, `results/pinn/seed_N/training.json` | Read-only validator, PINN history figures, representative validation figures, and [PINN documentation](04_pinn.md) |
| PINN validation | `scripts/validate_pinn.py` | `results/pinn/seed_N/validation.json` | Result summarizer and tables in [validation documentation](05_validation.md) |
| Result summary | `scripts/summarize_results.py` | `results/summary.json` | README and validation summaries |
| Figure rendering | `scripts/figures/problem.py`, `scripts/figures/analytical.py`, `scripts/figures/fem.py`, `scripts/figures/pinn.py`, `scripts/figures/validation.py` | PNG/PDF files under `results/figures/` | README and the five numbered documents; renderers read stored results and do not train PINNs or solve FEM |

## Stored results

The files under `results/` are the retained numerical outputs of the completed
calculations. They provide the data and trained models used by documentation
figures, post-processing, consistency checks, and validation. The tables below
define the contents and role of each stored file. Binary NPZ and PT files are
machine-readable rather than intended for direct visual inspection.

### Analytical/FEM sweep

**`results/analytical_fem/sweep.json` — analytical/FEM sweep record**

This deterministic JSON file is written by
`scripts/run_analytical_fem_sweep.py` and read by
`scripts/figures/analytical.py`.

| Top-level field | Stored structure and meaning |
|---|---|
| `study` | `fixed_parameters` stores $L$, `width_max`, $\mu$, $\Delta p$, and $Re$; `delta_values` stores the ten ordered values from $1.0$ to $0.1$; `fem_settings` stores $256\times128$, integration order 6, and flux quadrature order 64. It also names the swept parameter, analytical implementation, and relative-error definition. |
| `points` | Ten records. Every record contains `delta`, `epsilon`, `width_max`, `width_min`, `mean_width`, `relative_amplitude`, `analytical_flux`, `fem_flux`, `relative_flux_error`, and `fem_relative_flux_variation`. |
| `checks` | Relative analytical-versus-exact and FEM-versus-exact flux errors for the straight-channel point $\delta=1$. |

The corresponding table and interpretation are in [Analytical reference](02_analytical.md),
and the stored points are plotted in
[`relative_flux_error.png`](../results/figures/analytical/relative_flux_error.png)
and its PDF counterpart.

**`results/analytical_fem/resolution_check.json` — mesh-resolution check**

This deterministic JSON file stores the complete independently recomputed
$128\times64$ coarse sweep and ten pointwise relative coarse–fine flux
differences against the corresponding $256\times128$ fine-sweep FEM values in
the analytical sweep stored in
`sweep.json`. It
records both FEM protocols, straight-channel checks, and compact summary
diagnostics without duplicating the complete fine sweep. Its
`fine_sweep_source` field records the selected-root `sweep.json` used for the
comparison.

### FEM results

The FEM part contains two logically distinct stored results: the reference
solution and the independent mesh-convergence study.

#### Reference solution

The FEM reference produced by `scripts/generate_fem_reference.py`
consists of two companion files. `results/fem/reference.npz` stores the
sampled numerical fields, coordinates, quadrature weights, and flux data;
`results/fem/reference.json` stores the physical parameters, FEM settings,
diagnostics, software versions, and training-data flag.

**`results/fem/reference.npz` — sampled FEM field data**

This NumPy archive contains field samples flattened in row-major order from a
$96\times257$ grid: 96 Gauss–Legendre values of the logical transverse
coordinate $\eta$, each containing 257 uniform endpoint-excluded physical
$x$ locations in $[0,L)$. The physical $y$ locations use the discrete mapped
FEM wall bounds.

The archive stores the following arrays:

| Array key | Shape | Dtype | Meaning |
|---|---:|---|---|
| `x` | `(24672,)` | `float64` | Physical longitudinal coordinate; the same 257-point row is repeated for each $\eta$. |
| `y` | `(24672,)` | `float64` | Physical transverse coordinate obtained by mapping each $\eta$ to the discrete FEM cross-section. |
| `u` | `(24672,)` | `float64` | Physical longitudinal velocity component evaluated at `(x, y)`. |
| `v` | `(24672,)` | `float64` | Physical transverse velocity component evaluated at `(x, y)`. |
| `p_tilde` | `(24672,)` | `float64` | Periodic pressure correction $\widetilde p$ in the FEM solver gauge. Plotting and comparisons form a weighted zero-mean representative. |
| `eta` | `(24672,)` | `float64` | Logical Gauss–Legendre transverse coordinate in $[-1,1]$. |
| `quadrature_weights` | `(24672,)` | `float64` | Positive physical-area weights: transverse Gauss weight times local physical half-width times the uniform $x$ measure $L/257$. |
| `flux_x` | `(513,)` | `float64` | Uniform endpoint-excluded physical locations used for the stored flux diagnostic. |
| `flux` | `(513,)` | `float64` | Cross-sectional physical volume flux at `flux_x`, evaluated with quadrature order 64. |

**`results/fem/reference.json` — FEM reference metadata**

This UTF-8 JSON file contains:

| Top-level field | Stored structure and meaning |
|---|---|
| `physical` | Exact physical inputs: $L$, $Re$, $\Delta p$, $\mu$, maximum width, and minimum width. |
| `fem` | Mesh dimensions $256\times128$, integration order 6, and flux quadrature order 64. |
| `diagnostics` | Mean flux, relative flux variation, periodic-coordinate mismatch, and maximum wall velocity degree of freedom. |
| `versions` | Python, NumPy, SciPy, and scikit-fem versions used to generate the reference. |
| `training_data` | `false`, recording that this is an independent FEM result rather than PINN training data. |

`scripts/figures/fem.py` reads both files to render the FEM field figures.
`scripts/figures/validation.py` uses the stored grid and fields as the FEM side
of the representative seed-1 comparison; it converts FEM and PINN pressures
independently to weighted zero mean. The fields are presented in
[Finite-element reference](03_fem.md) and under
[`results/figures/fem/`](../results/figures/fem/); comparison panels appear in
[Post-training validation](05_validation.md) and under
[`results/figures/validation/`](../results/figures/validation/).

#### Convergence study

**`results/fem/convergence.json` — FEM convergence record**

This JSON file records the independent mesh-refinement evidence and
retained-side consistency checks. It is written by
`scripts/run_fem_convergence.py` and has this structure:

| Top-level field | Stored structure and meaning |
|---|---|
| `study` | Physical inputs, the five refinement levels $16\times8$, $32\times16$, $64\times32$, $128\times64$, and $256\times128$, integration and flux-quadrature orders, and the common-grid definition for successive comparisons. The latter records 129 endpoint-excluded $x$ points, 64-point transverse quadrature, the adjacent-domain intersection, physical-area weighting, and separate weighted zero-mean pressure gauges. |
| `levels` | Five records containing `nx`, `ny`, `mean_flux`, `relative_flux_variation`, `divergence_l2_norm`, and `maximum_wall_geometry_error`. |
| `successive_changes` | Four adjacent-level records containing the coarse/fine mesh dimensions and relative changes in velocity, zero-mean pressure, and mean flux. |
| `finest_retained_side` | Maximum velocity and zero-mean-pressure differences between left- and right-retained periodic reductions, plus maximum periodic degree-of-freedom mismatch. |

The convergence tables and interpretation are in
[Finite-element reference](03_fem.md). No separate convergence figure is used.

### PINN results

The directories `results/pinn/seed_0/`, `results/pinn/seed_1/`, and
`results/pinn/seed_2/` use the same structure. Each directory separates two
logical stages: completed training and subsequent read-only post-training
validation.

#### Trained model and training record

`model.pt` and `training.json` are companion outputs produced by
`scripts/train_pinn.py`. `model.pt` is the completed trained checkpoint and
stores the seed, fixed scientific protocol identity, final network parameters,
and deterministic model-state hash. `training.json` is the corresponding
completed training record; it stores the fixed protocol, reproducibility
hashes and metadata, optimizer histories and counts, final losses, and runtime.

`scripts/validate_pinn.py` loads each pair through
`corrugated_stokes.validation.verify_training_results`, reconstructs the
network, and loads `model_state_dict` strictly on the CPU.
`scripts/figures/pinn.py` reads the three training records for the optimizer
history figures, `scripts/summarize_results.py` extracts the completed step
counts, losses, and runtimes, and `scripts/figures/validation.py` uses the same
verified loader for the representative seed-1 fields. The human-readable
training protocol, histories, and final losses are in
[Physics-informed neural network](04_pinn.md) and
[`results/figures/pinn/`](../results/figures/pinn/).

#### Post-training validation

**`results/pinn/seed_N/validation.json` — post-training validation record**

With the same schema for $N=0,1,2$, this JSON file is not produced during
optimization. It is created afterward by the separate read-only validator and
records independent PINN physics diagnostics, boundary and flux diagnostics,
and FEM-comparison metrics. It also records the completed-result consistency
checks and confirms that validation performed no training, created no
optimizer, and left the model state unchanged.

`scripts/summarize_results.py` consumes all three validation records. Their
diagnostic tables are presented in [Post-training validation](05_validation.md).
The representative field panels in
[`results/figures/validation/`](../results/figures/validation/) are generated
from the verified seed-1 checkpoint and stored FEM field grid rather than from
field arrays in `validation.json`.

### Consolidated summary

**`results/summary.json` — consolidated result summary**

This deterministic JSON file is rebuilt by `scripts/summarize_results.py` from
`results/fem/reference.json` and the three `training.json` and
`validation.json` pairs. It does not contain field arrays or optimizer
histories.

| Top-level field | Stored structure and meaning |
|---|---|
| `fem_reference` | Physical parameters, FEM mesh and quadrature controls, mean flux, and relative flux variation from the reference metadata. |
| `seeds` | Three compact records with seed, Adam/L-BFGS counts and final losses, runtime, FEM comparison errors, flux variation, periodic RMS values, wall-speed RMS/maximum, and unseen horizontal/vertical momentum RMS values. |
| `aggregate` | Mean, population standard deviation, minimum, and maximum of the three relative FEM–PINN velocity, zero-mean-pressure, and mean-flux differences, plus the stated standard-deviation convention. |

The file supplies the compact three-seed values used in the README headline
table and in [Post-training validation](05_validation.md).

## File formats and inspection

JSON files are UTF-8 plain-text structured records. NPZ files are NumPy binary
archives for numerical arrays, and PT files are PyTorch serialized
checkpoints. The NPZ archive can be inspected read-only without loading
pickled objects:

```python
import numpy as np

with np.load("results/fem/reference.npz", allow_pickle=False) as data:
    print({name: (data[name].shape, data[name].dtype) for name in data.files})
```

The trusted project checkpoint can be inspected on the CPU without building or
running the network:

```python
import torch

checkpoint = torch.load(
    "results/pinn/seed_0/model.pt", map_location="cpu", weights_only=False
)
print(checkpoint.keys())
print({name: tuple(value.shape) for name, value in checkpoint["model_state_dict"].items()})
```
