from __future__ import annotations

import torch
from pytest import fixture, raises

from experiments.gnn.message_block import A, PathSpec, SourceSpec
from experiments.gnn.one_block_model import (
    OneBlockForceGNN,
    benzene_generators,
)
from TFENN.tensor_math import CSignature, CSlot, compile_covariant_basis


@fixture(scope="module")
def model() -> OneBlockForceGNN:
    torch.manual_seed(20260824)
    return OneBlockForceGNN(0.7)


def _inputs(
    batch_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(314159)
    centers = torch.randn(
        batch_size,
        5,
        3,
        dtype=dtype,
        generator=generator,
    )
    vectors = torch.randn(
        batch_size,
        5,
        3,
        dtype=dtype,
        generator=generator,
    )
    x, y, z = vectors.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(batch_size, 5, 3, 3)
    rotations = torch.matrix_exp(skew)
    return centers, rotations


def test_one_block_graph_shapes_and_zero_baseline(
    model: OneBlockForceGNN,
) -> None:
    centers, rotations = _inputs(3, torch.float64)
    messages = model.messages(centers, rotations)
    prediction = model(centers, rotations)
    assert model.message_block_count == 1
    assert model.trainable_parameter_count == 550
    assert model.edge_count == 20
    assert torch.all(model.sender != model.receiver)
    assert torch.equal(
        torch.bincount(model.receiver, minlength=5),
        torch.full((5,), 4, dtype=torch.long),
    )
    assert messages.edge_a.shape == (3, 20, 1, 3)
    assert messages.node_a.shape == (3, 5, 1, 3)
    assert tuple(value.shape for value in messages.edge_b.values()) == (
        (3, 20, 1, 5),
        (3, 20, 1, 13),
    )
    assert tuple(value.shape for value in messages.node_b.values()) == (
        (3, 5, 1, 5),
        (3, 5, 1, 13),
    )
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))


def test_aggregation_is_the_incoming_typed_sum(
    model: OneBlockForceGNN,
) -> None:
    centers, rotations = _inputs(2, torch.float64)
    projection = model.message_update.a_out.updates["A"].project.weight
    saved = projection.detach().clone()
    with torch.no_grad():
        projection.fill_(0.2)
    try:
        messages = model.messages(centers, rotations)
        expected_a = torch.stack(
            tuple(
                messages.edge_a[:, model.receiver == index].sum(dim=1)
                for index in range(5)
            ),
            dim=1,
        )
        torch.testing.assert_close(messages.node_a, expected_a)
        for key, edge_value in messages.edge_b.items():
            expected_b = torch.stack(
                tuple(
                    edge_value[:, model.receiver == index].sum(dim=1)
                    for index in range(5)
                ),
                dim=1,
            )
            torch.testing.assert_close(messages.node_b[key], expected_b)
    finally:
        with torch.no_grad():
            projection.copy_(saved)


def test_complete_graph_is_permutation_equivariant(
    model: OneBlockForceGNN,
) -> None:
    centers, rotations = _inputs(2, torch.float64)
    projection = model.message_update.a_out.updates["A"].project.weight
    saved = projection.detach().clone()
    with torch.no_grad():
        projection.fill_(0.2)
    try:
        original = model(centers, rotations)
        permutation = torch.tensor((0, 3, 1, 4, 2))
        permuted = model(
            centers[:, permutation],
            rotations[:, permutation],
        )
        torch.testing.assert_close(
            permuted,
            original[:, permutation],
            atol=2e-10,
            rtol=2e-10,
        )
    finally:
        with torch.no_grad():
            projection.copy_(saved)


def test_registered_generator_covariance_and_translation_invariance(
    model: OneBlockForceGNN,
) -> None:
    centers, rotations = _inputs(2, torch.float64)
    projection = model.message_update.a_out.updates["A"].project.weight
    saved = projection.detach().clone()
    with torch.no_grad():
        projection.fill_(0.15)
    try:
        original = model(centers, rotations)
        assert float(original.detach().abs().max()) > 1e-8
        generators = benzene_generators()
        for generator in generators:
            transformed_centers = centers @ generator.mT
            transformed_rotations = generator @ rotations
            transformed = model(transformed_centers, transformed_rotations)
            torch.testing.assert_close(
                transformed,
                original @ generator.mT,
                atol=2e-9,
                rtol=2e-9,
            )
        right_actions = torch.stack(
            (
                generators[0],
                generators[1],
                generators[0] @ generators[1],
                generators[1] @ generators[0],
                torch.eye(3, dtype=torch.float64),
            )
        )
        gauge_changed = model(centers, rotations @ right_actions)
        torch.testing.assert_close(
            gauge_changed,
            original,
            atol=2e-9,
            rtol=2e-9,
        )
        shift = torch.tensor((1.2, -0.7, 0.4), dtype=torch.float64)
        shifted = model(centers + shift, rotations)
        torch.testing.assert_close(
            shifted,
            original,
            atol=2e-10,
            rtol=2e-10,
        )
    finally:
        with torch.no_grad():
            projection.copy_(saved)


def test_float32_forward_and_backward_are_finite() -> None:
    torch.manual_seed(271828)
    model = OneBlockForceGNN(0.7, dtype=torch.float32)
    centers, rotations = _inputs(2, torch.float32)
    projection = model.message_update.a_out.updates["A"].project.weight
    with torch.no_grad():
        projection.fill_(0.1)
    loss = model(centers, rotations).square().mean()
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_path_and_channel_contracts_reject_mismatches(
    model: OneBlockForceGNN,
) -> None:
    signature = CSignature(A, (CSlot("value", A),))
    artifact = compile_covariant_basis(model.catalog, signature)
    with raises(ValueError, match="input count"):
        PathSpec(signature, (), artifact)
    renamed = CSignature(A, (CSlot("renamed", A),))
    with raises(ValueError, match="artifact signature"):
        PathSpec(renamed, ("d",), artifact)
    with raises(ValueError, match="positive integer"):
        SourceSpec(A, 0)
