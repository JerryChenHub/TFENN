"""Numerically stable nonlinear gates for D6 tensor basis features."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

from ._common import (
    ActivationSpec,
    block_cardinality_scale,
    make_activation,
    projector_buffer,
    require_bi_tensor,
    require_vector,
    split_bi_tensor_blocks,
    split_vector_blocks,
    stable_norm,
)


class D6VectorNormGateV1(nn.Module):
    """Apply independent smooth radial modulation to planar and axial blocks."""

    def __init__(
        self,
        activation: ActivationSpec = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.activation = make_activation(activation, nn.Tanh)
        self.eps = eps
        self.register_buffer("projectors", projector_buffer())
        self.register_buffer(
            "scale",
            torch.tensor((math.sqrt(2.0), 1.0), dtype=torch.float64),
        )

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        require_vector(vector)
        projectors = self.projectors.to(dtype=vector.dtype, device=vector.device)
        blocks = split_vector_blocks(vector, projectors)
        norms = stable_norm(blocks, (-1,), self.eps)
        scale = self.scale.to(dtype=vector.dtype, device=vector.device).view(1, 1, 2, 1)
        radius = norms / scale
        modulation = (
            1.0
            + self.activation(radius)
            - self.activation(torch.zeros_like(radius))
        )
        return (modulation * blocks).sum(dim=2)


class D6BiTensorNormGateV1(nn.Module):
    """Apply independent smooth radial modulation to four tensor blocks."""

    def __init__(
        self,
        activation: ActivationSpec = None,
        eps: float = 1e-8,
        use_cardinality_scale: bool = True,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.activation = make_activation(activation, nn.Tanh)
        self.eps = eps
        self.use_cardinality_scale = use_cardinality_scale
        self.register_buffer("projectors", projector_buffer())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        require_bi_tensor(tensor)
        projectors = self.projectors.to(dtype=tensor.dtype, device=tensor.device)
        blocks = split_bi_tensor_blocks(tensor, projectors)
        radius = stable_norm(blocks, (-1, -2), self.eps)
        if self.use_cardinality_scale:
            radius = radius / block_cardinality_scale(tensor)
        modulation = (
            1.0
            + self.activation(radius)
            - self.activation(torch.zeros_like(radius))
        )
        return (modulation * blocks).sum(dim=2)


class D6BiTensorTanhResidualGateV1(nn.Module):
    """Learnable tanh modulation with trainable parameters active at initialization."""

    def __init__(
        self,
        eps: float = 1e-8,
        use_cardinality_scale: bool = True,
        alpha_init: float = 1.0,
        beta_init: float = 0.0,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.use_cardinality_scale = use_cardinality_scale
        self.alpha = nn.Parameter(torch.full((4,), alpha_init))
        self.beta = nn.Parameter(torch.full((4,), beta_init))
        self.gamma = nn.Parameter(torch.full((4,), gamma_init))
        self.register_buffer("projectors", projector_buffer())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        require_bi_tensor(tensor)
        projectors = self.projectors.to(dtype=tensor.dtype, device=tensor.device)
        blocks = split_bi_tensor_blocks(tensor, projectors)
        radius = stable_norm(blocks, (-1, -2), self.eps)
        if self.use_cardinality_scale:
            radius = radius / block_cardinality_scale(tensor)
        parameter_shape = (1, 1, 4, 1, 1)
        alpha = self.alpha.to(dtype=tensor.dtype).view(parameter_shape)
        beta = self.beta.to(dtype=tensor.dtype).view(parameter_shape)
        gamma = self.gamma.to(dtype=tensor.dtype).view(parameter_shape)
        modulation = 1.0 + gamma * torch.tanh(alpha * radius + beta)
        return (modulation * blocks).sum(dim=2)


class D6BiTensorSoftplusResidualGateV2(nn.Module):
    """Positive softplus block modulation blended with the identity."""

    def __init__(
        self,
        eps: float = 1e-8,
        use_cardinality_scale: bool = True,
        slope_init: float = 0.1,
        bias_init: float = 0.0,
        eta_init: float = 0.05,
        max_modulation: float | None = None,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0.0 < eta_init < 1.0:
            raise ValueError("eta_init must be between zero and one")
        if max_modulation is not None and max_modulation <= 1.0:
            raise ValueError("max_modulation must be greater than one")
        self.eps = eps
        self.use_cardinality_scale = use_cardinality_scale
        self.max_modulation = max_modulation
        self.slope = nn.Parameter(torch.full((4,), slope_init))
        self.bias = nn.Parameter(torch.full((4,), bias_init))
        eta_logit = math.log(eta_init) - math.log1p(-eta_init)
        self.eta_logit = nn.Parameter(torch.tensor(eta_logit))
        self.register_buffer("projectors", projector_buffer())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        require_bi_tensor(tensor)
        projectors = self.projectors.to(dtype=tensor.dtype, device=tensor.device)
        blocks = split_bi_tensor_blocks(tensor, projectors)
        radius = stable_norm(blocks, (-1, -2), self.eps)
        if self.use_cardinality_scale:
            radius = radius / block_cardinality_scale(tensor)
        parameter_shape = (1, 1, 4, 1, 1)
        slope = self.slope.to(dtype=tensor.dtype).view(parameter_shape)
        bias = self.bias.to(dtype=tensor.dtype).view(parameter_shape)
        denominator = functional.softplus(bias).clamp_min(self.eps)
        modulation = functional.softplus(slope * radius + bias) / denominator
        if self.max_modulation is not None:
            span = self.max_modulation - 1.0
            modulation = 1.0 + span * torch.tanh((modulation - 1.0) / span)
        proposal = (modulation * blocks).sum(dim=2)
        eta = torch.sigmoid(self.eta_logit).to(dtype=tensor.dtype)
        return tensor + eta * (proposal - tensor)


class D6BiTensorSpectralGateV1(nn.Module):
    """SVD free spectral polynomial gate with an identity initialization."""

    def __init__(
        self,
        degree: int = 2,
        eta_init: float = 0.1,
    ) -> None:
        super().__init__()
        if degree < 1:
            raise ValueError("degree must be positive")
        if not 0.0 < eta_init < 1.0:
            raise ValueError("eta_init must be between zero and one")
        self.degree = degree
        self.coefficients = nn.Parameter(torch.zeros(4, degree))
        eta_logit = math.log(eta_init) - math.log1p(-eta_init)
        self.eta_logit = nn.Parameter(torch.tensor(eta_logit))
        self.register_buffer("projectors", projector_buffer())

    def _polynomial_delta(
        self,
        block: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        gram = block.transpose(-1, -2) @ block
        term = block
        result = torch.zeros_like(block)
        for index in range(self.degree):
            term = term @ gram
            result = result + coefficients[index] * term
        return result

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        require_bi_tensor(tensor)
        projectors = self.projectors.to(dtype=tensor.dtype, device=tensor.device)
        blocks = split_bi_tensor_blocks(tensor, projectors)
        deltas = [
            self._polynomial_delta(blocks[:, :, index], self.coefficients[index])
            for index in range(4)
        ]
        eta = torch.sigmoid(self.eta_logit).to(dtype=tensor.dtype)
        return tensor + eta * torch.stack(deltas, dim=2).sum(dim=2)


__all__ = [
    "D6BiTensorNormGateV1",
    "D6BiTensorSoftplusResidualGateV2",
    "D6BiTensorSpectralGateV1",
    "D6BiTensorTanhResidualGateV1",
    "D6VectorNormGateV1",
]
