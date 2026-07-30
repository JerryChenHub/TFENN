"""D6 layers constructed by explicit finite group averaging."""

from __future__ import annotations

import torch
from torch import nn

from ..symmetry import d6_rotations
from ._common import (
    ActivationSpec,
    make_activation,
    require_bi_tensor,
    require_vector,
    validate_positive_channels,
)
from .gates import D6VectorNormGateV1
from .tensor_basis import D6TensorToVectorLinearV1


class D6VectorGroupAverageV1(nn.Module):
    """Symmetrize a dense vector map under the stored row vector action."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationSpec = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        validate_positive_channels(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner = nn.Linear(in_channels * 3, out_channels * 3, bias=bias)
        self.activation = make_activation(activation, nn.LeakyReLU)
        self.register_buffer("rotations", d6_rotations(dtype=torch.float64))

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        require_vector(vector, self.in_channels)
        batch_size = vector.shape[0]
        rotations = self.rotations.to(dtype=vector.dtype, device=vector.device)
        group_size = rotations.shape[0]
        transformed = torch.einsum("bci,gij->bgcj", vector, rotations)
        values = self.inner(transformed.reshape(group_size * batch_size, -1))
        values = self.activation(values).reshape(
            batch_size, group_size, self.out_channels, 3
        )
        restored = torch.einsum(
            "bgci,gij->bgcj", values, rotations.transpose(-1, -2)
        )
        return restored.mean(dim=1)


class D6BiTensorGroupAverageV1(nn.Module):
    """Symmetrize a dense tensor map under independent left and right actions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationSpec = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        validate_positive_channels(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner = nn.Linear(in_channels * 9, out_channels * 9, bias=bias)
        self.activation = make_activation(activation, nn.LeakyReLU)
        self.register_buffer("rotations", d6_rotations(dtype=torch.float64))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        require_bi_tensor(tensor, self.in_channels)
        batch_size = tensor.shape[0]
        rotations = self.rotations.to(dtype=tensor.dtype, device=tensor.device)
        group_size = rotations.shape[0]
        inverse = rotations.transpose(-1, -2)
        left = torch.einsum("gij,bcjk->bgcik", inverse, tensor)
        transformed = torch.einsum("bgcij,hjk->bghcik", left, rotations)
        values = self.inner(
            transformed.reshape(group_size * group_size * batch_size, -1)
        )
        values = self.activation(values).reshape(
            batch_size, group_size, group_size, self.out_channels, 3, 3
        )
        restored_left = torch.einsum("gij,bghcjk->bghcik", rotations, values)
        restored = torch.einsum("bghcij,hjk->bghcik", restored_left, inverse)
        return restored.mean(dim=(1, 2))


class D6TensorToVectorGroupAverageV1(nn.Module):
    """Dense tensor to vector map that is left equivariant and right invariant."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationSpec = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        validate_positive_channels(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner = nn.Linear(in_channels * 9, out_channels * 3, bias=bias)
        self.activation = make_activation(activation, nn.LeakyReLU)
        self.register_buffer("rotations", d6_rotations(dtype=torch.float64))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        require_bi_tensor(tensor, self.in_channels)
        batch_size = tensor.shape[0]
        rotations = self.rotations.to(dtype=tensor.dtype, device=tensor.device)
        group_size = rotations.shape[0]
        inverse = rotations.transpose(-1, -2)
        left = torch.einsum("gij,bcjk->bgcik", inverse, tensor)
        transformed = torch.einsum("bgcij,hjk->bghcik", left, rotations)
        values = self.inner(
            transformed.reshape(group_size * group_size * batch_size, -1)
        )
        values = self.activation(values).reshape(
            batch_size, group_size, group_size, self.out_channels, 3
        )
        restored = torch.einsum("bghci,gij->bghcj", values, inverse)
        return restored.mean(dim=(1, 2))


class D6TensorToVectorBasisAverageV1(nn.Module):
    """Tensor basis map followed by nonlinear right group averaging."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationSpec = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = D6TensorToVectorLinearV1(in_channels, out_channels)
        self.gate = D6VectorNormGateV1(activation=activation, eps=eps)
        self.register_buffer("rotations", d6_rotations(dtype=torch.float64))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        require_bi_tensor(tensor, self.in_channels)
        batch_size = tensor.shape[0]
        rotations = self.rotations.to(dtype=tensor.dtype, device=tensor.device)
        group_size = rotations.shape[0]
        transformed = torch.einsum("bcij,gjk->bgcik", tensor, rotations)
        transformed = transformed.reshape(
            group_size * batch_size, self.in_channels, 3, 3
        )
        values = self.gate(self.linear(transformed))
        return values.reshape(
            batch_size, group_size, self.out_channels, 3
        ).mean(dim=1)


__all__ = [
    "D6BiTensorGroupAverageV1",
    "D6TensorToVectorBasisAverageV1",
    "D6TensorToVectorGroupAverageV1",
    "D6VectorGroupAverageV1",
]
