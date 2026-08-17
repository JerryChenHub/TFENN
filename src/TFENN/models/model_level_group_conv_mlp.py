"""Model level Reynolds averaged MLP for an ordered benzene pair."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn


__all__ = [
    "ModelLevelGroupConvMLP",
    "ModelLevelGroupConvMLPConfig",
    "build_model_level_group_conv_mlp",
]


ActivationName = Literal["silu", "gelu", "tanh"]


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_widths(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("hidden_widths must be a sequence")
    result = tuple(value)
    if not result:
        raise ValueError("hidden_widths cannot be empty")
    if any(isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in result):
        raise ValueError("hidden_widths must contain positive integers")
    return result


@dataclass(frozen=True, slots=True)
class ModelLevelGroupConvMLPConfig:
    """Configure the unconstrained MLP inside the Reynolds projection."""

    hidden_widths: tuple[int, ...] = (96, 96, 96)
    activation: ActivationName = "silu"
    distance_scale: float = 6.0
    seed: int = 20260814

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_widths", _positive_widths(self.hidden_widths))
        if self.activation not in ("silu", "gelu", "tanh"):
            raise ValueError("activation must be silu, gelu, or tanh")
        object.__setattr__(
            self,
            "distance_scale",
            _positive_float(self.distance_scale, "distance_scale"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_widths": list(self.hidden_widths),
            "activation": self.activation,
            "distance_scale": self.distance_scale,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelLevelGroupConvMLPConfig":
        if not isinstance(value, Mapping):
            raise TypeError("configuration must be a mapping")
        return cls(
            hidden_widths=tuple(value.get("hidden_widths", (96, 96, 96))),
            activation=value.get("activation", "silu"),
            distance_scale=value.get("distance_scale", 6.0),
            seed=value.get("seed", 20260814),
        )


def _validate_rotation(value: Tensor, name: str, tolerance: float) -> None:
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    if not torch.allclose(value.mT @ value, identity, atol=tolerance, rtol=tolerance):
        raise ValueError(f"{name} must be orthogonal")
    if not torch.allclose(
        torch.linalg.det(value),
        value.new_tensor(1.0),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError(f"{name} must be a proper rotation")


def _compile_proper_d6_actions(generators: Tensor) -> Tensor:
    if (
        not isinstance(generators, Tensor)
        or generators.shape != (2, 3, 3)
        or not torch.is_floating_point(generators)
    ):
        raise ValueError("generators must be a floating tensor with shape two by three by three")
    work = generators.detach().to(device="cpu", dtype=torch.float64)
    tolerance = 2.0e-6
    sixfold, twofold = work.unbind(dim=0)
    _validate_rotation(sixfold, "sixfold generator", tolerance)
    _validate_rotation(twofold, "twofold generator", tolerance)
    identity = torch.eye(3, dtype=work.dtype)
    if not torch.allclose(
        torch.linalg.matrix_power(sixfold, 6), identity, atol=tolerance, rtol=tolerance
    ):
        raise ValueError("the first generator must have order six")
    if any(
        torch.allclose(
            torch.linalg.matrix_power(sixfold, exponent),
            identity,
            atol=tolerance,
            rtol=tolerance,
        )
        for exponent in range(1, 6)
    ):
        raise ValueError("the first generator must have exact order six")
    if not torch.allclose(twofold @ twofold, identity, atol=tolerance, rtol=tolerance):
        raise ValueError("the second generator must have order two")
    if torch.allclose(twofold, identity, atol=tolerance, rtol=tolerance):
        raise ValueError("the second generator must have exact order two")
    if not torch.allclose(
        twofold @ sixfold @ twofold,
        sixfold.mT,
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("generators do not satisfy the proper D6 relation")

    powers = [identity]
    for _index in range(5):
        powers.append(powers[-1] @ sixfold)
    actions = torch.stack((*powers, *(power @ twofold for power in powers)))
    distances = torch.linalg.matrix_norm(
        actions[:, None] - actions[None, :], dim=(-2, -1)
    )
    distances.fill_diagonal_(math.inf)
    if float(distances.min()) <= tolerance:
        raise ValueError("generators do not produce twelve distinct D6 actions")
    products = actions[:, None] @ actions[None, :]
    closure_error = torch.linalg.matrix_norm(
        products[:, :, None] - actions[None, None, :], dim=(-2, -1)
    ).amin(dim=-1)
    if float(closure_error.max()) > tolerance:
        raise ValueError("compiled D6 actions are not closed")
    return actions.to(dtype=generators.dtype, device=generators.device)


def _activation(value: Tensor, name: ActivationName) -> Tensor:
    if name == "silu":
        return torch.nn.functional.silu(value)
    if name == "gelu":
        return torch.nn.functional.gelu(value)
    return torch.tanh(value)


class ModelLevelGroupConvMLP(nn.Module):
    """Apply one D6 by D6 Reynolds projection around a complete MLP."""

    input_width = 12
    output_width = 3

    def __init__(
        self,
        group_actions: Tensor,
        config: ModelLevelGroupConvMLPConfig,
    ) -> None:
        super().__init__()
        if group_actions.shape != (12, 3, 3):
            raise ValueError("group_actions must have shape twelve by three by three")
        if not torch.is_floating_point(group_actions):
            raise TypeError("group_actions must use a floating dtype")
        self.config = config
        self.register_buffer("group_actions", group_actions.clone(), persistent=False)

        widths = (self.input_width, *config.hidden_widths, self.output_width)
        layers: list[nn.Linear] = []
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        for index, (input_width, output_width) in enumerate(
            zip(widths[:-1], widths[1:])
        ):
            layer = nn.Linear(
                input_width,
                output_width,
                bias=index < len(widths) - 2,
                dtype=group_actions.dtype,
                device=group_actions.device,
            )
            standard_deviation = math.sqrt(2.0 / input_width)
            if index == len(widths) - 2:
                standard_deviation = 1.0 / math.sqrt(input_width)
            with torch.no_grad():
                initialized = torch.randn(
                    layer.weight.shape,
                    dtype=layer.weight.dtype,
                    device="cpu",
                    generator=generator,
                )
                layer.weight.copy_(initialized.to(layer.weight.device) * standard_deviation)
                if layer.bias is not None:
                    layer.bias.zero_()
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "model_family": "model_level_group_conv_mlp",
            "symmetry_projection": "network_level_reynolds_average_d6xd6",
            "group_order": int(self.group_actions.shape[0]),
            "group_pair_count": int(self.group_actions.shape[0] ** 2),
            "input_width": self.input_width,
            "output_width": self.output_width,
            "trainable_parameter_count": self.trainable_parameter_count,
            "config": self.config.as_dict(),
        }

    def _base_mlp(self, value: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            value = _activation(layer(value), self.config.activation)
        return self.layers[-1](value)

    def _root_geometry(self, centers: Tensor, frames: Tensor) -> tuple[Tensor, Tensor]:
        if centers.shape[-2:] != (2, 3):
            raise ValueError("centers must describe one ordered pair")
        if frames.shape != centers.shape[:-2] + (2, 3, 3):
            raise ValueError("frames must match the ordered pair shape")
        if centers.dtype != frames.dtype or centers.device != frames.device:
            raise ValueError("centers and frames must share dtype and device")
        if centers.dtype != self.group_actions.dtype or centers.device != self.group_actions.device:
            raise ValueError("inputs must match network dtype and device")
        root = frames[..., 0, :, :]
        displacement = centers[..., 1, :] - centers[..., 0, :]
        local = torch.einsum("...ji,...j->...i", root, displacement)
        relative_frame = root.mT @ frames[..., 1, :, :]
        return local, relative_frame

    def force_from_local(self, displacement: Tensor, relative_frame: Tensor) -> Tensor:
        """Evaluate the complete model projection from local pair geometry."""
        if displacement.shape[-1:] != (3,):
            raise ValueError("displacement must end with dimension three")
        if relative_frame.shape != displacement.shape[:-1] + (3, 3):
            raise ValueError("relative_frame must match displacement")
        if (
            displacement.dtype != relative_frame.dtype
            or displacement.device != relative_frame.device
            or displacement.dtype != self.group_actions.dtype
            or displacement.device != self.group_actions.device
        ):
            raise ValueError("local inputs must match network dtype and device")

        actions = self.group_actions
        inverse_actions = actions.mT
        moved_displacement = torch.einsum(
            "hij,...j->...hi", inverse_actions, displacement
        )
        left_moved_frame = torch.einsum(
            "hij,...jk->...hik", inverse_actions, relative_frame
        )
        moved_frame = torch.einsum(
            "...hij,kjl->...hkil", left_moved_frame, actions
        )
        expanded_displacement = moved_displacement.unsqueeze(-2).expand(
            moved_frame.shape[:-2] + (3,)
        )
        features = torch.cat(
            (
                expanded_displacement / self.config.distance_scale,
                moved_frame.flatten(start_dim=-2),
            ),
            dim=-1,
        )
        unconstrained = self._base_mlp(features)
        restored = torch.einsum("hij,...hkj->...hki", actions, unconstrained)
        return restored.mean(dim=(-3, -2))

    def forward_local(self, centers: Tensor, frames: Tensor) -> Tensor:
        displacement, relative_frame = self._root_geometry(centers, frames)
        return self.force_from_local(displacement, relative_frame)

    def forward(self, centers: Tensor, frames: Tensor) -> Tensor:
        local = self.forward_local(centers, frames)
        return torch.einsum("...ij,...j->...i", frames[..., 0, :, :], local)


def build_model_level_group_conv_mlp(
    generators: Tensor,
    config: ModelLevelGroupConvMLPConfig | Mapping[str, Any] | None = None,
) -> ModelLevelGroupConvMLP:
    """Compile fixed proper D6 actions and construct the averaged MLP."""
    resolved = (
        ModelLevelGroupConvMLPConfig()
        if config is None
        else ModelLevelGroupConvMLPConfig.from_dict(config)
        if isinstance(config, Mapping)
        else config
    )
    if not isinstance(resolved, ModelLevelGroupConvMLPConfig):
        raise TypeError("config must be a ModelLevelGroupConvMLPConfig or mapping")
    return ModelLevelGroupConvMLP(_compile_proper_d6_actions(generators), resolved)
