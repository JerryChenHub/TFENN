from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .activations import activation_factory
from .nn import (
    D6BiTensorGroupAverageV1,
    D6BiTensorLinearV1,
    D6BiTensorNormGateV1,
    D6BiTensorSoftplusResidualGateV2,
    D6BiTensorSpectralGateV1,
    D6BiTensorTanhResidualGateV1,
    D6TensorToVectorBasisAverageV1,
    D6TensorToVectorGroupAverageV1,
    D6TensorToVectorLinearV1,
    D6VectorGroupAverageV1,
    D6VectorLinearV1,
    D6VectorNormGateV1,
)
from .symmetry import d6_rotations


_TENSOR_BASIS_LINEAR_TYPES = (
    D6VectorLinearV1,
    D6BiTensorLinearV1,
    D6TensorToVectorLinearV1,
)


def _normalize_inputs(
    displacement: torch.Tensor,
    relative_rotation: torch.Tensor,
    x_channels: int,
    r_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if displacement.ndim == 2 and displacement.shape[-1] == 3:
        displacement = displacement.unsqueeze(1)
    if relative_rotation.ndim == 3 and relative_rotation.shape[-2:] == (3, 3):
        relative_rotation = relative_rotation.unsqueeze(1)
    if displacement.ndim != 3 or displacement.shape[-1] != 3:
        raise ValueError(
            "displacement must have shape (batch, channels, 3), "
            f"got {tuple(displacement.shape)}"
        )
    if relative_rotation.ndim != 4 or relative_rotation.shape[-2:] != (3, 3):
        raise ValueError(
            "relative_rotation must have shape (batch, channels, 3, 3), "
            f"got {tuple(relative_rotation.shape)}"
        )
    if displacement.shape[:2] != (relative_rotation.shape[0], x_channels):
        raise ValueError(
            f"expected {x_channels} displacement channels and matching batches"
        )
    if relative_rotation.shape[1] != r_channels:
        raise ValueError(f"expected {r_channels} relative rotation channels")
    return displacement, relative_rotation


def _initialize_model(model: nn.Module, policy: str) -> None:
    if policy == "layer_default":
        return
    if policy != "xavier_uniform":
        raise ValueError(
            "init_policy must be either 'layer_default' or 'xavier_uniform'"
        )
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, _TENSOR_BASIS_LINEAR_TYPES):
            flattened = module.weight.reshape(module.weight.shape[0], -1)
            nn.init.xavier_uniform_(flattened)


def _matrix_gate(name: str, activation_name: str) -> nn.Module:
    if name == "block_norm":
        return D6BiTensorNormGateV1(
            activation=activation_factory(activation_name)
        )
    if name == "tanh_residual":
        return D6BiTensorTanhResidualGateV1()
    if name == "softplus_residual":
        return D6BiTensorSoftplusResidualGateV2()
    if name == "spectral":
        return D6BiTensorSpectralGateV1()
    if name == "identity":
        return nn.Identity()
    raise ValueError(
        "matrix_gate must be block_norm, tanh_residual, "
        "softplus_residual, spectral, or identity"
    )


class _D6TensorBasisNet(nn.Module):
    def __init__(
        self,
        *,
        tensor_to_vector_method: str,
        x_in_channels: int = 1,
        r_in_channels: int = 1,
        x_hidden_channels: int = 16,
        r_hidden_channels: int = 16,
        num_x_layers: int = 2,
        num_r_layers: int = 1,
        r_to_x_channels: int = 16,
        out_channels: int = 2,
        vector_activation: str = "sigmoid_centered",
        matrix_gate: str = "block_norm",
        head_activation: str = "leaky_relu",
        num_head_layers: int = 2,
        head_hidden_channels: int | None = 16,
        init_policy: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        if num_x_layers < 0 or num_r_layers < 0:
            raise ValueError("branch layer counts cannot be negative")
        if num_head_layers < 1:
            raise ValueError("num_head_layers must be positive")

        self.x_in_channels = x_in_channels
        self.r_in_channels = r_in_channels
        self.tensor_to_vector_method = tensor_to_vector_method

        vector_factory = activation_factory(vector_activation)
        head_factory = activation_factory(head_activation)

        self.x_branch = nn.ModuleList()
        x_channels = x_in_channels
        for _ in range(num_x_layers):
            self.x_branch.extend(
                (
                    D6VectorLinearV1(x_channels, x_hidden_channels),
                    D6VectorNormGateV1(activation=vector_factory),
                )
            )
            x_channels = x_hidden_channels

        self.r_branch = nn.ModuleList()
        r_channels = r_in_channels
        for _ in range(num_r_layers):
            self.r_branch.extend(
                (
                    D6BiTensorLinearV1(r_channels, r_hidden_channels),
                    _matrix_gate(matrix_gate, vector_activation),
                )
            )
            r_channels = r_hidden_channels

        if tensor_to_vector_method == "basis_right_average":
            tensor_to_vector_type = D6TensorToVectorBasisAverageV1
        elif tensor_to_vector_method == "full_group_average":
            tensor_to_vector_type = D6TensorToVectorGroupAverageV1
        else:
            raise ValueError(f"unknown tensor_to_vector_method {tensor_to_vector_method!r}")
        self.r_to_x = tensor_to_vector_type(
            r_channels,
            r_to_x_channels,
            activation=vector_factory,
        )

        fused_channels = x_channels + r_to_x_channels
        if head_hidden_channels is not None and head_hidden_channels < 1:
            raise ValueError("head_hidden_channels must be positive or None")
        hidden_channels = (
            fused_channels
            if head_hidden_channels is None
            else head_hidden_channels
        )
        self.head = nn.ModuleList()
        head_in = fused_channels
        for layer_index in range(num_head_layers):
            head_out = (
                out_channels
                if layer_index == num_head_layers - 1
                else hidden_channels
            )
            self.head.append(
                D6VectorGroupAverageV1(
                    head_in,
                    head_out,
                    activation=head_factory,
                )
            )
            head_in = head_out

        self.configuration = {
            "x_in_channels": x_in_channels,
            "r_in_channels": r_in_channels,
            "x_hidden_channels": x_hidden_channels,
            "r_hidden_channels": r_hidden_channels,
            "num_x_layers": num_x_layers,
            "num_r_layers": num_r_layers,
            "r_to_x_channels": r_to_x_channels,
            "out_channels": out_channels,
            "vector_activation": vector_activation,
            "matrix_gate": matrix_gate,
            "head_activation": head_activation,
            "num_head_layers": num_head_layers,
            "head_hidden_channels": head_hidden_channels,
            "init_policy": init_policy,
            "tensor_to_vector_method": tensor_to_vector_method,
        }
        _initialize_model(self, init_policy)

    def forward(
        self,
        displacement: torch.Tensor,
        relative_rotation: torch.Tensor,
    ) -> torch.Tensor:
        displacement, relative_rotation = _normalize_inputs(
            displacement,
            relative_rotation,
            self.x_in_channels,
            self.r_in_channels,
        )
        x_features = displacement
        for layer in self.x_branch:
            x_features = layer(x_features)
        r_features = relative_rotation
        for layer in self.r_branch:
            r_features = layer(r_features)
        fused = torch.cat((x_features, self.r_to_x(r_features)), dim=1)
        output = fused
        for layer in self.head:
            output = layer(output)
        return output

    def architecture_summary(self) -> dict[str, Any]:
        return {
            "vector_branch": [type(layer).__name__ for layer in self.x_branch],
            "tensor_branch": [type(layer).__name__ for layer in self.r_branch],
            "tensor_to_vector": type(self.r_to_x).__name__,
            "head": [type(layer).__name__ for layer in self.head],
        }


class D6TensorBasisNetV1(_D6TensorBasisNet):
    """Tensor basis network with a basis map and right group averaging."""

    def __init__(
        self,
        x_in_channels: int = 1,
        r_in_channels: int = 1,
        x_hidden_channels: int = 16,
        r_hidden_channels: int = 16,
        num_x_layers: int = 2,
        num_r_layers: int = 1,
        r_to_x_channels: int = 16,
        out_channels: int = 2,
        vector_activation: str = "sigmoid_centered",
        matrix_gate: str = "block_norm",
        head_activation: str = "leaky_relu",
        num_head_layers: int = 2,
        head_hidden_channels: int | None = 16,
        init_policy: str = "xavier_uniform",
    ) -> None:
        super().__init__(
            tensor_to_vector_method="basis_right_average",
            x_in_channels=x_in_channels,
            r_in_channels=r_in_channels,
            x_hidden_channels=x_hidden_channels,
            r_hidden_channels=r_hidden_channels,
            num_x_layers=num_x_layers,
            num_r_layers=num_r_layers,
            r_to_x_channels=r_to_x_channels,
            out_channels=out_channels,
            vector_activation=vector_activation,
            matrix_gate=matrix_gate,
            head_activation=head_activation,
            num_head_layers=num_head_layers,
            head_hidden_channels=head_hidden_channels,
            init_policy=init_policy,
        )


class D6TensorBasisNetV2(_D6TensorBasisNet):
    """Tensor basis network with full group averaging at tensor conversion."""

    def __init__(
        self,
        x_in_channels: int = 1,
        r_in_channels: int = 1,
        x_hidden_channels: int = 16,
        r_hidden_channels: int = 16,
        num_x_layers: int = 2,
        num_r_layers: int = 1,
        r_to_x_channels: int = 16,
        out_channels: int = 2,
        vector_activation: str = "sigmoid_centered",
        matrix_gate: str = "block_norm",
        head_activation: str = "leaky_relu",
        num_head_layers: int = 2,
        head_hidden_channels: int | None = 16,
        init_policy: str = "xavier_uniform",
    ) -> None:
        super().__init__(
            tensor_to_vector_method="full_group_average",
            x_in_channels=x_in_channels,
            r_in_channels=r_in_channels,
            x_hidden_channels=x_hidden_channels,
            r_hidden_channels=r_hidden_channels,
            num_x_layers=num_x_layers,
            num_r_layers=num_r_layers,
            r_to_x_channels=r_to_x_channels,
            out_channels=out_channels,
            vector_activation=vector_activation,
            matrix_gate=matrix_gate,
            head_activation=head_activation,
            num_head_layers=num_head_layers,
            head_hidden_channels=head_hidden_channels,
            init_policy=init_policy,
        )


class D6GroupAverageNetV1(nn.Module):
    """Reference network built from explicit group averaged dense maps."""

    def __init__(
        self,
        x_in_channels: int = 1,
        r_in_channels: int = 1,
        x_hidden_channels: int = 16,
        r_hidden_channels: int = 16,
        num_x_layers: int = 2,
        num_r_layers: int = 1,
        r_to_x_channels: int = 16,
        out_channels: int = 2,
        vector_activation: str = "sigmoid_centered",
        head_activation: str = "leaky_relu",
        num_head_layers: int = 2,
        head_hidden_channels: int | None = 16,
        init_policy: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        if num_x_layers < 0 or num_r_layers < 0:
            raise ValueError("branch layer counts cannot be negative")
        if num_head_layers < 1:
            raise ValueError("num_head_layers must be positive")
        self.x_in_channels = x_in_channels
        self.r_in_channels = r_in_channels
        vector_factory = activation_factory(vector_activation)
        head_factory = activation_factory(head_activation)

        self.x_branch = nn.ModuleList()
        x_channels = x_in_channels
        for _ in range(num_x_layers):
            self.x_branch.append(
                D6VectorGroupAverageV1(
                    x_channels,
                    x_hidden_channels,
                    activation=vector_factory,
                )
            )
            x_channels = x_hidden_channels

        self.r_branch = nn.ModuleList()
        r_channels = r_in_channels
        for _ in range(num_r_layers):
            self.r_branch.append(
                D6BiTensorGroupAverageV1(
                    r_channels,
                    r_hidden_channels,
                    activation=vector_factory,
                )
            )
            r_channels = r_hidden_channels

        self.r_to_x = D6TensorToVectorGroupAverageV1(
            r_channels,
            r_to_x_channels,
            activation=vector_factory,
        )
        fused_channels = x_channels + r_to_x_channels
        if head_hidden_channels is not None and head_hidden_channels < 1:
            raise ValueError("head_hidden_channels must be positive or None")
        hidden_channels = (
            fused_channels
            if head_hidden_channels is None
            else head_hidden_channels
        )
        self.head = nn.ModuleList()
        head_in = fused_channels
        for layer_index in range(num_head_layers):
            head_out = (
                out_channels
                if layer_index == num_head_layers - 1
                else hidden_channels
            )
            self.head.append(
                D6VectorGroupAverageV1(
                    head_in,
                    head_out,
                    activation=head_factory,
                )
            )
            head_in = head_out

        self.configuration = {
            "x_in_channels": x_in_channels,
            "r_in_channels": r_in_channels,
            "x_hidden_channels": x_hidden_channels,
            "r_hidden_channels": r_hidden_channels,
            "num_x_layers": num_x_layers,
            "num_r_layers": num_r_layers,
            "r_to_x_channels": r_to_x_channels,
            "out_channels": out_channels,
            "vector_activation": vector_activation,
            "head_activation": head_activation,
            "num_head_layers": num_head_layers,
            "head_hidden_channels": head_hidden_channels,
            "init_policy": init_policy,
        }
        _initialize_model(self, init_policy)

    def forward(
        self,
        displacement: torch.Tensor,
        relative_rotation: torch.Tensor,
    ) -> torch.Tensor:
        displacement, relative_rotation = _normalize_inputs(
            displacement,
            relative_rotation,
            self.x_in_channels,
            self.r_in_channels,
        )
        x_features = displacement
        for layer in self.x_branch:
            x_features = layer(x_features)
        r_features = relative_rotation
        for layer in self.r_branch:
            r_features = layer(r_features)
        output = torch.cat((x_features, self.r_to_x(r_features)), dim=1)
        for layer in self.head:
            output = layer(output)
        return output

    def architecture_summary(self) -> dict[str, Any]:
        return {
            "vector_branch": [type(layer).__name__ for layer in self.x_branch],
            "tensor_branch": [type(layer).__name__ for layer in self.r_branch],
            "tensor_to_vector": type(self.r_to_x).__name__,
            "head": [type(layer).__name__ for layer in self.head],
        }


class MLPBaselineV1(nn.Module):
    """Plain MLP for the flattened relative rotation and displacement."""

    def __init__(
        self,
        x_in_channels: int = 1,
        r_in_channels: int = 1,
        out_channels: int = 2,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        activation: str = "leaky_relu",
        init_policy: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be positive")
        self.x_in_channels = x_in_channels
        self.r_in_channels = r_in_channels
        self.out_channels = out_channels
        activation_type = activation_factory(activation)
        input_dim = x_in_channels * 3 + r_in_channels * 9
        layers: list[nn.Module] = []
        layer_in = input_dim
        for _ in range(num_hidden_layers):
            layers.extend((nn.Linear(layer_in, hidden_dim), activation_type()))
            layer_in = hidden_dim
        layers.append(nn.Linear(layer_in, out_channels * 3))
        self.network = nn.Sequential(*layers)
        self.configuration = {
            "x_in_channels": x_in_channels,
            "r_in_channels": r_in_channels,
            "out_channels": out_channels,
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_hidden_layers,
            "activation": activation,
            "init_policy": init_policy,
        }
        _initialize_model(self, init_policy)

    def forward(
        self,
        displacement: torch.Tensor,
        relative_rotation: torch.Tensor,
    ) -> torch.Tensor:
        displacement, relative_rotation = _normalize_inputs(
            displacement,
            relative_rotation,
            self.x_in_channels,
            self.r_in_channels,
        )
        features = torch.cat(
            (
                relative_rotation.flatten(start_dim=1),
                displacement.flatten(start_dim=1),
            ),
            dim=1,
        )
        return self.network(features).reshape(
            features.shape[0],
            self.out_channels,
            3,
        )

    def architecture_summary(self) -> dict[str, Any]:
        return {"network": [type(layer).__name__ for layer in self.network]}


class D6SymmetrizedMLPBaselineV1(nn.Module):
    """Plain MLP projected onto the required D6 input and output symmetry."""

    def __init__(
        self,
        x_in_channels: int = 1,
        r_in_channels: int = 1,
        out_channels: int = 2,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        activation: str = "leaky_relu",
        init_policy: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        self.x_in_channels = x_in_channels
        self.r_in_channels = r_in_channels
        self.out_channels = out_channels
        self.base_model = MLPBaselineV1(
            x_in_channels=x_in_channels,
            r_in_channels=r_in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            init_policy=init_policy,
        )
        self.configuration = dict(self.base_model.configuration)
        self.configuration["projection"] = "D6_left_equivariant_right_invariant"
        self.register_buffer("rotations", d6_rotations(dtype=torch.float64))

    def forward(
        self,
        displacement: torch.Tensor,
        relative_rotation: torch.Tensor,
    ) -> torch.Tensor:
        displacement, relative_rotation = _normalize_inputs(
            displacement,
            relative_rotation,
            self.x_in_channels,
            self.r_in_channels,
        )
        batch_size = displacement.shape[0]
        rotations = self.rotations.to(
            dtype=displacement.dtype,
            device=displacement.device,
        )
        inverse = rotations.transpose(-1, -2)
        transformed_x = torch.einsum(
            "bci,gij->bgcj",
            displacement,
            rotations,
        )
        left_r = torch.einsum(
            "gij,bcjk->bgcik",
            inverse,
            relative_rotation,
        )
        transformed_r = torch.einsum(
            "bgcij,hjk->bghcik",
            left_r,
            rotations,
        )
        expanded_x = transformed_x[:, :, None].expand(
            batch_size,
            12,
            12,
            self.x_in_channels,
            3,
        )
        values = self.base_model(
            expanded_x.reshape(-1, self.x_in_channels, 3),
            transformed_r.reshape(-1, self.r_in_channels, 3, 3),
        ).reshape(batch_size, 12, 12, self.out_channels, 3)
        restored = torch.einsum(
            "bghci,gij->bghcj",
            values,
            inverse,
        )
        return restored.mean(dim=(1, 2))

    def architecture_summary(self) -> dict[str, Any]:
        return {
            "projection": "D6_left_equivariant_right_invariant",
            "base_model": self.base_model.architecture_summary(),
        }


__all__ = [
    "D6GroupAverageNetV1",
    "D6SymmetrizedMLPBaselineV1",
    "D6TensorBasisNetV1",
    "D6TensorBasisNetV2",
    "MLPBaselineV1",
]
