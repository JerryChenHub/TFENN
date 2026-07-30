"""Private helpers shared by TFENN neural network layers."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import torch
from torch import nn

from ..symmetry import d6_projectors


ActivationSpec = nn.Module | Callable[[], nn.Module] | None


def make_activation(
    activation: ActivationSpec,
    default_factory: Callable[[], nn.Module],
) -> nn.Module:
    if activation is None:
        return default_factory()
    if isinstance(activation, nn.Module):
        return copy.deepcopy(activation)
    result = activation()
    if not isinstance(result, nn.Module):
        raise TypeError("activation factory must return torch.nn.Module")
    return result


def require_vector(
    value: torch.Tensor,
    in_channels: int | None = None,
    name: str = "vector",
) -> None:
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (batch, channels, 3), got {tuple(value.shape)}")
    if in_channels is not None and value.shape[1] != in_channels:
        raise ValueError(
            f"{name} has {value.shape[1]} channels, expected {in_channels}"
        )


def require_bi_tensor(
    value: torch.Tensor,
    in_channels: int | None = None,
    name: str = "tensor",
) -> None:
    if value.ndim != 4 or value.shape[-2:] != (3, 3):
        raise ValueError(
            f"{name} must have shape (batch, channels, 3, 3), got {tuple(value.shape)}"
        )
    if in_channels is not None and value.shape[1] != in_channels:
        raise ValueError(
            f"{name} has {value.shape[1]} channels, expected {in_channels}"
        )


def projector_buffer() -> torch.Tensor:
    return d6_projectors(dtype=torch.float64)


def split_vector_blocks(
    vector: torch.Tensor,
    projectors: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("bci,aij->bcaj", vector, projectors)


def split_bi_tensor_blocks(
    tensor: torch.Tensor,
    projectors: torch.Tensor,
) -> torch.Tensor:
    left = torch.einsum("aij,bcjk->bcaik", projectors, tensor)
    blocks = torch.einsum("bcaij,djk->bcadik", left, projectors)
    return blocks.reshape(*tensor.shape[:2], 4, 3, 3)


def stable_norm(
    value: torch.Tensor,
    dimensions: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    return torch.sqrt(value.square().sum(dim=dimensions, keepdim=True) + eps * eps)


def block_cardinality_scale(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor((2.0, math.sqrt(2.0), math.sqrt(2.0), 1.0)).view(
        1, 1, 4, 1, 1
    )


def validate_positive_channels(in_channels: int, out_channels: int) -> None:
    if in_channels < 1 or out_channels < 1:
        raise ValueError("in_channels and out_channels must be positive")
