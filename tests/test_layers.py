from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from TFENN.nn import (
    D6BiTensorLinearV1,
    D6BiTensorNormGateV1,
    D6BiTensorSoftplusResidualGateV2,
    D6BiTensorSpectralGateV1,
    D6BiTensorTanhResidualGateV1,
    D6TensorToVectorBasisAverageV1,
    D6TensorToVectorLinearV1,
    D6VectorLinearV1,
    D6VectorNormGateV1,
)
from TFENN.symmetry import d6_rotations


ATOL = 2e-11
RTOL = 2e-11


def _transform_bi_tensor(
    value: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    return left.T @ value @ right


def test_tensor_basis_linear_layers_have_expected_shapes() -> None:
    generator = torch.Generator().manual_seed(7)
    vector = torch.randn(5, 2, 3, generator=generator, dtype=torch.float64)
    tensor = torch.randn(5, 2, 3, 3, generator=generator, dtype=torch.float64)

    assert D6VectorLinearV1(2, 4).double()(vector).shape == (5, 4, 3)
    assert D6BiTensorLinearV1(2, 4).double()(tensor).shape == (5, 4, 3, 3)
    assert D6TensorToVectorLinearV1(2, 4).double()(tensor).shape == (5, 4, 3)
    assert D6TensorToVectorBasisAverageV1(2, 4).double()(tensor).shape == (
        5,
        4,
        3,
    )


def test_tensor_basis_linear_layers_preserve_their_representations() -> None:
    torch.manual_seed(11)
    rotations = d6_rotations(dtype=torch.float64)
    left = rotations[5]
    right = rotations[8]
    vector = torch.randn(3, 2, 3, dtype=torch.float64)
    tensor = torch.randn(3, 2, 3, 3, dtype=torch.float64)

    vector_layer = D6VectorLinearV1(2, 4).double()
    vector_value = vector_layer(vector)
    torch.testing.assert_close(
        vector_layer(vector @ left),
        vector_value @ left,
        atol=ATOL,
        rtol=RTOL,
    )

    tensor_layer = D6BiTensorLinearV1(2, 4).double()
    tensor_value = tensor_layer(tensor)
    torch.testing.assert_close(
        tensor_layer(_transform_bi_tensor(tensor, left, right)),
        _transform_bi_tensor(tensor_value, left, right),
        atol=ATOL,
        rtol=RTOL,
    )

    left_layer = D6TensorToVectorLinearV1(2, 4).double()
    left_value = left_layer(tensor)
    torch.testing.assert_close(
        left_layer(left.T @ tensor),
        left_value @ left,
        atol=ATOL,
        rtol=RTOL,
    )

    averaged_layer = D6TensorToVectorBasisAverageV1(2, 4).double()
    averaged_value = averaged_layer(tensor)
    torch.testing.assert_close(
        averaged_layer(_transform_bi_tensor(tensor, left, right)),
        averaged_value @ left,
        atol=ATOL,
        rtol=RTOL,
    )


def test_vector_gate_is_equivariant() -> None:
    torch.manual_seed(13)
    rotation = d6_rotations(dtype=torch.float64)[7]
    vector = torch.randn(4, 3, 3, dtype=torch.float64)
    gate = D6VectorNormGateV1(activation=nn.SiLU).double()

    torch.testing.assert_close(
        gate(vector @ rotation),
        gate(vector) @ rotation,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "factory",
    [
        D6BiTensorNormGateV1,
        D6BiTensorTanhResidualGateV1,
        D6BiTensorSoftplusResidualGateV2,
        D6BiTensorSpectralGateV1,
    ],
)
def test_bi_tensor_gates_are_equivariant(
    factory: Callable[[], nn.Module],
) -> None:
    torch.manual_seed(17)
    rotations = d6_rotations(dtype=torch.float64)
    left = rotations[3]
    right = rotations[10]
    tensor = torch.randn(3, 2, 3, 3, dtype=torch.float64)
    gate = factory().double()

    torch.testing.assert_close(
        gate(_transform_bi_tensor(tensor, left, right)),
        _transform_bi_tensor(gate(tensor), left, right),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    ("factory", "shape"),
    [
        (D6VectorNormGateV1, (2, 3, 3)),
        (D6BiTensorNormGateV1, (2, 3, 3, 3)),
        (D6BiTensorTanhResidualGateV1, (2, 3, 3, 3)),
        (D6BiTensorSoftplusResidualGateV2, (2, 3, 3, 3)),
        (D6BiTensorSpectralGateV1, (2, 3, 3, 3)),
    ],
)
def test_gates_have_finite_zero_forward_and_backward(
    factory: Callable[[], nn.Module],
    shape: tuple[int, ...],
) -> None:
    gate = factory().double()
    value = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    result = gate(value)
    result.sum().backward()

    assert torch.isfinite(result).all()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    for parameter in gate.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "factory",
    [
        D6BiTensorTanhResidualGateV1,
        D6BiTensorSoftplusResidualGateV2,
        D6BiTensorSpectralGateV1,
    ],
)
def test_trainable_gates_receive_a_nonzero_finite_gradient(
    factory: Callable[[], nn.Module],
) -> None:
    torch.manual_seed(19)
    gate = factory().double()
    value = torch.randn(3, 2, 3, 3, dtype=torch.float64, requires_grad=True)
    gate(value).square().mean().backward()

    gradients = [parameter.grad for parameter in gate.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().max().item() > 1e-14 for gradient in gradients)
