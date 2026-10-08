"""Reduced-cost tests for the PINN implementation."""

import numpy as np
import torch

from corrugated_stokes.config import PHYSICAL, PINN_SCALES, PointConfig
from corrugated_stokes.pinn import (
    LOSS_COMPONENT_ORDER, POINT_HASH_DECIMALS, adam_learning_rate, build_network,
    dimensionless_height, generate_point_bundle, loss_components,
    network_fields, network_kinematics, point_bundle_hash,
)


SMALL = PointConfig(48, 16, 8)


def flattened(model):
    return torch.cat([parameter.detach().ravel() for parameter in model.parameters()])


def test_same_seed_is_identical_and_different_seed_differs():
    assert torch.equal(flattened(build_network(1)), flattened(build_network(1)))
    assert not torch.equal(flattened(build_network(1)), flattened(build_network(2)))


def test_scaled_geometry_uses_relative_amplitude_not_prl_epsilon():
    X = torch.tensor([[0.0]], dtype=torch.float64)
    assert torch.allclose(dimensionless_height(X), torch.tensor([[1 + PHYSICAL.relative_amplitude]], dtype=torch.float64))
    assert PHYSICAL.epsilon != PINN_SCALES.relative_amplitude


def test_point_generation_is_deterministic_and_has_fixed_small_shapes():
    first, second = generate_point_bundle(SMALL), generate_point_bundle(SMALL)
    assert point_bundle_hash(first) == point_bundle_hash(second)
    assert first[0]["x"].shape == (48, 1)
    assert first[1]["x"].shape == (16, 1)
    assert first[2]["x_left"].shape == (8, 1)


def test_portable_point_hash_ignores_roundoff_but_detects_meaningful_changes():
    points = (
        {
            "x": torch.tensor([[0.0], [0.25]], dtype=torch.float64),
            "index": torch.tensor([1, 2], dtype=torch.int64),
        },
        {"y": torch.tensor([[-0.5], [0.5]], dtype=torch.float64)},
        {"pair": torch.tensor([[0.125]], dtype=torch.float64)},
    )
    clone = lambda bundle: tuple(
        {name: value.clone() for name, value in group.items()} for group in bundle
    )
    baseline = point_bundle_hash(points)
    assert baseline == point_bundle_hash(clone(points))

    roundoff = clone(points)
    roundoff[0]["x"][0, 0] += 10.0 ** -(POINT_HASH_DECIMALS + 2)
    assert point_bundle_hash(roundoff) == baseline

    changed = clone(points)
    changed[0]["x"][0, 0] += 1.0e-8
    assert point_bundle_hash(changed) != baseline
    changed_integer = clone(points)
    changed_integer[0]["index"][0] += 1
    assert point_bundle_hash(changed_integer) != baseline


def test_streamfunction_velocity_and_residuals_are_finite_and_continuous():
    model = build_network(1)
    X = torch.tensor([[-0.7], [0.1], [1.2]], dtype=torch.float64, requires_grad=True)
    Y = torch.tensor([[0.0], [0.05], [-0.04]], dtype=torch.float64, requires_grad=True)
    field = network_fields(model, X, Y)
    assert all(value.shape == (3, 1) for value in field.values())
    assert all(torch.isfinite(value).all() for value in field.values())
    assert torch.max(torch.abs(field["R_c"])).item() < 1e-13


def test_eight_loss_terms_sum_exactly_and_smoke_adam_decreases_loss():
    model, points = build_network(1), generate_point_bundle(SMALL)
    initial, components = loss_components(model, *points)
    assert tuple(components) == LOSS_COMPONENT_ORDER
    assert torch.equal(initial, sum(components.values()))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        total, _ = loss_components(model, *points)
        total.backward()
        optimizer.step()
    final, _ = loss_components(model, *points)
    assert torch.isfinite(final)
    assert final < initial


def test_fixed_adam_schedule_boundaries():
    assert adam_learning_rate(1) == adam_learning_rate(3000) == 5e-4
    assert adam_learning_rate(3001) == adam_learning_rate(9000) == 1e-4
    assert adam_learning_rate(9001) == adam_learning_rate(16000) == 1e-5
    assert adam_learning_rate(16001) == adam_learning_rate(16500) == 1e-6
