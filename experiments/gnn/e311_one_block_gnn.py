"""One block two benzene adapter for the E311 multibody message block."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from TFENN.models import (
    E311MessageBlockConfigV1,
    E311MultibodyMessageBlockV1,
    GraphMessageBlockOutputV1,
    build_e311_multibody_message_block_v1,
)
from TFENN.tensor_math import (
    GeneratorSystem,
    PoseEncoder,
    TypeKey,
    build_primitive_b_manifest,
    build_type_catalog,
    compile_anchors,
)


@contextmanager
def _fixed_geometry_compilation_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    if previous != 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if previous != 1:
            torch.set_num_threads(previous)


def benzene_generators(dtype: torch.dtype = torch.float64) -> Tensor:
    """Return the proper D6 generators used by the benzene pose encoder."""

    angle = math.pi / 3.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        (
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0),
            ),
        ),
        dtype=dtype,
    )


@dataclass(frozen=True, slots=True)
class E311OneBlockGNNConfig:
    """Serializable fixed two node configuration for one E311 message block."""

    molecule_count: int = 2
    hidden_b_channels: int = 1
    edge_a_channels: int = 1
    a_mid_channels: int = 1
    b_wide_channels: int = 2
    b_out_channels: int = 1
    gate_width: int = 8
    molecular_scalar_dim: int = 0
    distance_scale: float = 6.0
    rbf_centers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    rbf_width: float = 0.4
    inverse_powers: tuple[int, ...] = (1, 2, 3)
    distance_epsilon: float = 1.0e-12
    rms_epsilon: float = 1.0e-8
    max_constraint_entries: int = 10_000_000

    def __post_init__(self) -> None:
        if self.molecule_count != 2:
            raise ValueError("E311OneBlockGNNConfig fixes molecule_count at two")
        if (
            isinstance(self.hidden_b_channels, bool)
            or not isinstance(self.hidden_b_channels, int)
            or self.hidden_b_channels < 1
        ):
            raise ValueError("hidden_b_channels must be a positive integer")
        self.message_block_config()

    def message_block_config(self) -> E311MessageBlockConfigV1:
        """Return the exact configuration consumed by the reused block."""

        return E311MessageBlockConfigV1(
            edge_a_channels=self.edge_a_channels,
            a_mid_channels=self.a_mid_channels,
            b_wide_channels=self.b_wide_channels,
            b_out_channels=self.b_out_channels,
            gate_width=self.gate_width,
            molecular_scalar_dim=self.molecular_scalar_dim,
            distance_scale=self.distance_scale,
            rbf_centers=self.rbf_centers,
            rbf_width=self.rbf_width,
            inverse_powers=self.inverse_powers,
            distance_epsilon=self.distance_epsilon,
            rms_epsilon=self.rms_epsilon,
            max_constraint_entries=self.max_constraint_entries,
        )


class E311OneBlockGNN(nn.Module):
    """Apply one E311 multibody block to one unordered benzene pair."""

    def __init__(
        self,
        force_scale: float,
        config: E311OneBlockGNNConfig = E311OneBlockGNNConfig(),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not math.isfinite(force_scale) or force_scale <= 0.0:
            raise ValueError("force_scale must be finite and positive")
        if not isinstance(config, E311OneBlockGNNConfig):
            raise TypeError("config must be an E311OneBlockGNNConfig")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")
        self.config = config
        self.message_block_count = 1
        with _fixed_geometry_compilation_threads():
            generators = benzene_generators(torch.float64)
            anchors = compile_anchors(generators, output_ranks=(2, 6))
            manifest = build_primitive_b_manifest(anchors)
            catalog = build_type_catalog(
                GeneratorSystem(("sixfold", "twofold"), generators),
                manifest,
            )
            pose_encoder = PoseEncoder(anchors)
            hidden_channels = {
                item.key: config.hidden_b_channels for item in manifest
            }
            message_block = build_e311_multibody_message_block_v1(
                catalog,
                manifest,
                pose_encoder,
                hidden_channels,
                config.message_block_config(),
                dtype,
            )
        self.catalog = catalog
        self.manifest = manifest
        self.message_block: E311MultibodyMessageBlockV1 = message_block
        self.register_buffer(
            "pair_index",
            torch.tensor(((0,), (1,)), dtype=torch.int64),
        )
        self.register_buffer(
            "force_scale",
            torch.tensor(float(force_scale), dtype=dtype),
        )

    @property
    def pair_count(self) -> int:
        return int(self.pair_index.shape[1])

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def b_keys(self) -> tuple[TypeKey, ...]:
        return self.message_block.b_keys

    def reset_running_rms(self) -> None:
        """Reset all cumulative invariant RMS buffers to their initial state."""

        self.message_block.pair_kernel.reset_running_rms()

    def _validate_inputs(self, centers: Tensor, rotations: Tensor) -> None:
        if centers.ndim < 2 or centers.shape[-2:] != (
            self.config.molecule_count,
            3,
        ):
            raise ValueError("centers must have shape (..., 2, 3)")
        expected_rotations = centers.shape[:-2] + (
            self.config.molecule_count,
            3,
            3,
        )
        if rotations.shape != expected_rotations:
            raise ValueError("rotations must have shape (..., 2, 3, 3)")
        if centers.dtype != self.force_scale.dtype or rotations.dtype != centers.dtype:
            raise TypeError("centers and rotations must match the model dtype")
        if (
            centers.device != self.force_scale.device
            or rotations.device != centers.device
        ):
            raise ValueError("centers and rotations must be on the model device")

    def initial_hidden_b(self, centers: Tensor) -> dict[TypeKey, Tensor]:
        """Create the all zero node B state consumed by the only block."""

        batch_shape = centers.shape[:-2]
        node_count = self.config.molecule_count
        return {
            key: centers.new_zeros(
                batch_shape
                + (
                    node_count,
                    self.message_block.pair_kernel.hidden_channels[key],
                    self.catalog.resolve(key).representation_dim,
                )
            )
            for key in self.b_keys
        }

    def initial_edge_a_world(self, centers: Tensor) -> Tensor:
        """Create the all zero edge A state consumed by the only block."""

        return centers.new_zeros(
            centers.shape[:-2]
            + (
                self.pair_count,
                self.config.edge_a_channels,
                3,
            )
        )

    def one_block_output(
        self,
        centers: Tensor,
        rotations: Tensor,
        node_scalars: Tensor | None = None,
        *,
        collect_trace: bool = False,
    ) -> GraphMessageBlockOutputV1:
        """Return the complete output of the reused message block."""

        self._validate_inputs(centers, rotations)
        return self.message_block(
            centers,
            rotations,
            self.pair_index,
            self.initial_hidden_b(centers),
            self.initial_edge_a_world(centers),
            node_scalars,
            collect_trace=collect_trace,
        )

    def normalized_forces_and_pairs_world(
        self,
        centers: Tensor,
        rotations: Tensor,
    ) -> tuple[Tensor, Tensor]:
        output = self.one_block_output(centers, rotations)
        return output.node_force_world, output.pair_force_world

    def normalized_forces_world(
        self,
        centers: Tensor,
        rotations: Tensor,
    ) -> Tensor:
        return self.normalized_forces_and_pairs_world(centers, rotations)[0]

    def forward_world(self, centers: Tensor, rotations: Tensor) -> Tensor:
        return self.normalized_forces_world(centers, rotations) * self.force_scale

    def forward(self, centers: Tensor, rotations: Tensor) -> Tensor:
        return self.forward_world(centers, rotations)

    def block_configuration(self) -> Mapping[str, Any]:
        return {
            "message_block_count": self.message_block_count,
            **asdict(self.config),
        }


__all__ = [
    "E311OneBlockGNN",
    "E311OneBlockGNNConfig",
    "benzene_generators",
]
