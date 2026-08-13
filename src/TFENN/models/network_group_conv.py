"""Network level D6 averaging baseline for one benzene pair.

The same ordinary MLP is evaluated on every receiver and sender D6 orbit.
The receiver action is restored on each predicted vector before averaging.
This Reynolds construction is exactly sender invariant and receiver covariant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


__all__ = [
    "BenzenePairNetworkGroupConvMLP",
    "NetworkGroupConvConfig",
    "build_benzene_pair_network_group_conv_mlp",
]


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _positive_widths(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("hidden_widths must be a sequence of integers")
    widths = tuple(value)
    if not widths:
        raise ValueError("hidden_widths must not be empty")
    if any(
        isinstance(width, bool) or not isinstance(width, int) or width <= 0
        for width in widths
    ):
        raise ValueError("hidden_widths must contain positive integers")
    return widths


@dataclass(frozen=True, slots=True)
class NetworkGroupConvConfig:
    """Configure the shared scalar MLP used by the D6 Reynolds average."""

    hidden_widths: tuple[int, ...] = (96, 96, 96)
    activation: str = "silu"
    distance_scale: float = 10.0
    use_hidden_bias: bool = True
    architecture_id: str = "pair_network_group_conv_mlp_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_widths", _positive_widths(self.hidden_widths))
        if not isinstance(self.activation, str):
            raise TypeError("activation must be a string")
        activation = self.activation.lower()
        if activation not in _ACTIVATIONS:
            choices = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(f"activation must be one of {choices}")
        object.__setattr__(self, "activation", activation)
        if isinstance(self.distance_scale, bool) or not isinstance(
            self.distance_scale, (int, float)
        ):
            raise TypeError("distance_scale must be a real number")
        distance_scale = float(self.distance_scale)
        if not math.isfinite(distance_scale) or distance_scale <= 0.0:
            raise ValueError("distance_scale must be finite and positive")
        object.__setattr__(self, "distance_scale", distance_scale)
        if not isinstance(self.use_hidden_bias, bool):
            raise TypeError("use_hidden_bias must be bool")
        if not isinstance(self.architecture_id, str) or not self.architecture_id:
            raise ValueError("architecture_id must be a nonempty string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NetworkGroupConvConfig:
        if not isinstance(value, Mapping):
            raise TypeError("network group convolution config must be a mapping")
        return cls(
            hidden_widths=tuple(value.get("hidden_widths", (96, 96, 96))),
            activation=value.get("activation", "silu"),
            distance_scale=value.get("distance_scale", 10.0),
            use_hidden_bias=value.get("use_hidden_bias", True),
            architecture_id=value.get(
                "architecture_id",
                "pair_network_group_conv_mlp_v1",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_widths": list(self.hidden_widths),
            "activation": self.activation,
            "distance_scale": self.distance_scale,
            "use_hidden_bias": self.use_hidden_bias,
            "architecture_id": self.architecture_id,
        }


def _proper_d6_group() -> Tensor:
    """Return the twelve proper rotations of benzene in fixed order."""
    angle = math.pi / 3.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    sixfold = torch.tensor(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )
    twofold = torch.diag(torch.tensor((1.0, -1.0, -1.0), dtype=torch.float64))
    identity = torch.eye(3, dtype=torch.float64)
    powers = [identity]
    for _ in range(5):
        powers.append(powers[-1] @ sixfold)
    return torch.stack(tuple(powers) + tuple(value @ twofold for value in powers))


class BenzenePairNetworkGroupConvMLP(nn.Module):
    """Predict root force through a shared MLP and a complete D6 orbit average."""

    input_dimension = 12
    output_dimension = 3
    group_order = 12

    def __init__(self, config: NetworkGroupConvConfig | None = None) -> None:
        super().__init__()
        self.config = NetworkGroupConvConfig() if config is None else config
        if not isinstance(self.config, NetworkGroupConvConfig):
            raise TypeError("config must be a NetworkGroupConvConfig")

        layers: list[nn.Module] = []
        current_width = self.input_dimension
        for hidden_width in self.config.hidden_widths:
            linear = nn.Linear(
                current_width,
                hidden_width,
                bias=self.config.use_hidden_bias,
                dtype=torch.float64,
            )
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(_ACTIVATIONS[self.config.activation]())
            current_width = hidden_width
        output = nn.Linear(
            current_width,
            self.output_dimension,
            bias=False,
            dtype=torch.float64,
        )
        nn.init.xavier_uniform_(output.weight)
        layers.append(output)
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("_group", _proper_d6_group(), persistent=False)

    @property
    def parameter_count(self) -> int:
        """Return the number of learned scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def group(self) -> Tensor:
        """Return a detached copy of the fixed D6 actions."""
        return self._group.detach().clone()

    def zero_output_head(self) -> int:
        """Set the final shared MLP map to zero and return the affected head count."""
        output = self.mlp[-1]
        if not isinstance(output, nn.Linear):
            raise RuntimeError("the shared MLP does not end with a linear map")
        with torch.no_grad():
            output.weight.zero_()
        return 1

    def _root_geometry(self, centers: Tensor, frames: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(centers, Tensor) or not isinstance(frames, Tensor):
            raise TypeError("centers and frames must be tensors")
        if centers.shape[-2:] != (2, 3):
            raise ValueError("centers must end with shape two by three")
        if frames.shape != centers.shape[:-2] + (2, 3, 3):
            raise ValueError("frames must match centers and contain two matrices")
        if centers.dtype != frames.dtype or centers.device != frames.device:
            raise ValueError("centers and frames must share dtype and device")
        if centers.dtype != self._group.dtype:
            raise TypeError("inputs must match the network dtype")
        if centers.device != self._group.device:
            raise ValueError("inputs must match the network device")
        root = frames[..., 0, :, :]
        displacement = centers[..., 1, :] - centers[..., 0, :]
        local_displacement = torch.einsum("...ji,...j->...i", root, displacement)
        relative_frame = root.mT @ frames[..., 1, :, :]
        return local_displacement, relative_frame

    def force_from_geometry(
        self,
        local_displacement: Tensor,
        relative_frame: Tensor,
    ) -> Tensor:
        """Evaluate normalized local force from local pair geometry."""
        if not isinstance(local_displacement, Tensor) or not isinstance(
            relative_frame, Tensor
        ):
            raise TypeError("local_displacement and relative_frame must be tensors")
        if local_displacement.shape[-1:] != (3,):
            raise ValueError("local_displacement must end with shape three")
        if relative_frame.shape != local_displacement.shape[:-1] + (3, 3):
            raise ValueError("relative_frame must match the displacement prefix")
        if (
            local_displacement.dtype != relative_frame.dtype
            or local_displacement.device != relative_frame.device
        ):
            raise ValueError("local geometry tensors must share dtype and device")
        if local_displacement.dtype != self._group.dtype:
            raise TypeError("inputs must match the network dtype")
        if local_displacement.device != self._group.device:
            raise ValueError("inputs must match the network device")

        group = self._group
        scaled_displacement = local_displacement / self.config.distance_scale
        orbit_displacement = torch.einsum(
            "hji,...j->...hi",
            group,
            scaled_displacement,
        )
        left_frame = torch.einsum("hji,...jk->...hik", group, relative_frame)
        orbit_frame = torch.einsum("...hij,kjl->...hkil", left_frame, group)
        frame_features = orbit_frame.flatten(start_dim=orbit_frame.ndim - 2)
        displacement_features = orbit_displacement.unsqueeze(-2).expand(
            frame_features.shape[:-1] + (3,)
        )
        features = torch.cat((frame_features, displacement_features), dim=-1)
        orbit_force = self.mlp(features)
        restored_force = torch.einsum(
            "hij,...hkj->...hki",
            group,
            orbit_force,
        )
        return restored_force.mean(dim=(-3, -2))

    def forward_local(self, centers: Tensor, frames: Tensor) -> Tensor:
        """Return normalized force in the root benzene frame."""
        displacement, relative_frame = self._root_geometry(centers, frames)
        return self.force_from_geometry(displacement, relative_frame)

    def forward(self, centers: Tensor, frames: Tensor) -> Tensor:
        """Return normalized force in world coordinates."""
        local = self.forward_local(centers, frames)
        root = frames[..., 0, :, :]
        return torch.einsum("...ij,...j->...i", root, local)


def build_benzene_pair_network_group_conv_mlp(
    config: NetworkGroupConvConfig | Mapping[str, Any] | None = None,
) -> BenzenePairNetworkGroupConvMLP:
    """Build the benzene pair network level group convolution baseline."""
    resolved = (
        NetworkGroupConvConfig()
        if config is None
        else NetworkGroupConvConfig.from_dict(config)
        if isinstance(config, Mapping)
        else config
    )
    if not isinstance(resolved, NetworkGroupConvConfig):
        raise TypeError("config must be a NetworkGroupConvConfig or mapping")
    return BenzenePairNetworkGroupConvMLP(resolved)
