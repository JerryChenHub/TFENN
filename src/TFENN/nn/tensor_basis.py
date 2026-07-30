"""Tensor basis linear layers for the D6 representations."""

from __future__ import annotations

import torch
from torch import nn

from ._common import (
    projector_buffer,
    require_bi_tensor,
    require_vector,
    split_bi_tensor_blocks,
    split_vector_blocks,
    validate_positive_channels,
)


class D6VectorLinearV1(nn.Module):
    """Linear map between D6 vector channels without a bias."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        validate_positive_channels(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 2))
        self.register_buffer("projectors", projector_buffer())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        require_vector(vector, self.in_channels)
        projectors = self.projectors.to(dtype=vector.dtype, device=vector.device)
        blocks = split_vector_blocks(vector, projectors)
        return torch.einsum("bcaj,oca->boj", blocks, self.weight)


class D6BiTensorLinearV1(nn.Module):
    """Linear map between tensors carrying independent left and right D6 actions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        validate_positive_channels(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 4))
        self.register_buffer("projectors", projector_buffer())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        require_bi_tensor(tensor, self.in_channels)
        projectors = self.projectors.to(dtype=tensor.dtype, device=tensor.device)
        blocks = split_bi_tensor_blocks(tensor, projectors)
        return torch.einsum("bcfij,ocf->boij", blocks, self.weight)


class D6TensorToVectorLinearV1(nn.Module):
    """Left equivariant tensor to vector basis map without right averaging."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        validate_positive_channels(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 2))
        self.register_buffer("projectors", projector_buffer())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        require_bi_tensor(tensor, self.in_channels)
        projectors = self.projectors.to(dtype=tensor.dtype, device=tensor.device)
        components = torch.einsum("aij,bcjk->bcika", projectors, tensor)
        return torch.einsum("bcika,ocka->boi", components, self.weight)


__all__ = [
    "D6BiTensorLinearV1",
    "D6TensorToVectorLinearV1",
    "D6VectorLinearV1",
]
