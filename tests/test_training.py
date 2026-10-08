"""Low-cost tests for resumable PINN training."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from corrugated_stokes.config import LBFGS
from corrugated_stokes.pinn import LOSS_COMPONENT_ORDER, build_network, model_state_hash


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "training_script", ROOT / "scripts/train_pinn.py"
)
TRAINING = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = TRAINING
SPEC.loader.exec_module(TRAINING)


def _cheap_loss(model, *_points):
    active = sum(torch.mean(parameter**2) for parameter in model.parameters())
    zero = 0.0 * active
    components = {name: zero for name in LOSS_COMPONENT_ORDER}
    components[LOSS_COMPONENT_ORDER[0]] = active
    return sum(components.values()), components


def _advance(model, optimizer, first, last, history):
    for step in range(first, last + 1):
        total, components = TRAINING.adam_update(
            model, optimizer, (None, None, None), step
        )
        history.append({
            "stage": "adam",
            "evaluation": step,
            "learning_rate": TRAINING.adam_learning_rate(step),
            "total": total,
            "components": components,
        })


def test_progress_output_uses_recovery_and_fifty_closure_milestones(capsys):
    TRAINING._print_adam_checkpoint_progress(1500)
    TRAINING._print_adam_checkpoint_progress(16500)
    for evaluation in (1, 49, 50, 99, 100):
        TRAINING._print_lbfgs_progress(evaluation)
    assert capsys.readouterr().out.splitlines() == [
        "Adam 1500/16500 [9.1%] checkpoint saved",
        "Adam 16500/16500 [100.0%] checkpoint saved",
        "L-BFGS 50/500",
        "L-BFGS 100/500",
    ]


def test_simple_adam_checkpoint_resume_matches_uninterrupted(monkeypatch, tmp_path):
    monkeypatch.setattr(TRAINING, "loss_components", _cheap_loss)
    uninterrupted = build_network(1)
    uninterrupted_optimizer = torch.optim.Adam(uninterrupted.parameters(), lr=5e-4)
    uninterrupted_history = []
    _advance(uninterrupted, uninterrupted_optimizer, 1, 4, uninterrupted_history)

    interrupted = build_network(1)
    interrupted_optimizer = torch.optim.Adam(interrupted.parameters(), lr=5e-4)
    interrupted_history = []
    _advance(interrupted, interrupted_optimizer, 1, 2, interrupted_history)
    checkpoint = tmp_path / "adam_step_00002.pt"
    TRAINING.save_recovery(
        checkpoint,
        interrupted,
        interrupted_optimizer,
        1,
        2,
        interrupted_history,
        "POINTS",
        1.25,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert set(payload) == {
        "kind", "seed", "step", "protocol_hash", "point_hash_scheme",
        "point_bundle_hash", "model_state_dict", "optimizer_state_dict",
        "history", "adam_elapsed_seconds",
    }
    assert payload["point_hash_scheme"] == TRAINING.POINT_HASH_SCHEME

    resumed = build_network(1)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=5e-4)
    step, resumed_history, elapsed = TRAINING.load_recovery(
        checkpoint, resumed, resumed_optimizer, 1, "POINTS"
    )
    assert (step, elapsed) == (2, 1.25)
    _advance(resumed, resumed_optimizer, 3, 4, resumed_history)
    assert resumed_history == uninterrupted_history
    assert model_state_hash(resumed) == model_state_hash(uninterrupted)


def test_recovery_rejects_wrong_protocol_or_points(monkeypatch, tmp_path):
    monkeypatch.setattr(TRAINING, "loss_components", _cheap_loss)
    model = build_network(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    history = []
    _advance(model, optimizer, 1, 1, history)
    checkpoint = tmp_path / "adam_step_00001.pt"
    TRAINING.save_recovery(
        checkpoint, model, optimizer, 1, 1, history, "POINTS", 0.5
    )
    target = build_network(1)
    target_optimizer = torch.optim.Adam(target.parameters(), lr=5e-4)
    with pytest.raises(RuntimeError, match="point mismatch"):
        TRAINING.load_recovery(
            checkpoint, target, target_optimizer, 1, "OTHER_POINTS"
        )


def test_fresh_lbfgs_obeys_closure_cap(monkeypatch):
    model = torch.nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight[:, 0] = torch.tensor([-1.2, 1.0], dtype=torch.float64)

    def rosenbrock(candidate, *_points):
        x, y = candidate.weight[:, 0]
        active = (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2
        zero = 0.0 * active
        components = {name: zero for name in LOSS_COMPONENT_ORDER}
        components[LOSS_COMPONENT_ORDER[0]] = active
        return sum(components.values()), components

    monkeypatch.setattr(TRAINING, "loss_components", rosenbrock)
    controls = replace(
        LBFGS,
        maximum_closure_evaluations=3,
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )
    result = TRAINING.run_lbfgs_with_strict_cap(
        model, (None, None, None), controls
    )
    assert result["closure_evaluations"] == 3
    assert result["termination"] == "evaluation_cap"
    assert model_state_hash(model) == result["last_evaluated_model_state_hash"]


def test_final_result_paths_are_non_overwriting(tmp_path):
    paths = TRAINING.result_paths(tmp_path, 1)
    paths["root"].mkdir(parents=True)
    paths["model"].write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        TRAINING._refuse_existing_final_result(paths)
