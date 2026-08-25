from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from TFENN.tensor_math import (
    CSignature,
    CSlot,
    GeneratorSystem,
    PoseEncoder,
    TypeCatalog,
    TypeKey,
    build_primitive_b_manifest,
    build_type_catalog,
    compile_anchors,
    compile_covariant_basis,
    encode_typed_blocks,
)

from .message_block import (
    A,
    ABMessageUpdate,
    PathSpec,
    SourceSpec,
    TargetSpec,
    TypedStage,
    type_name,
)


@dataclass(frozen=True)
class AggregatedMessages:
    edge_a: Tensor
    edge_b: dict[TypeKey, Tensor]
    node_a: Tensor
    node_b: dict[TypeKey, Tensor]


def benzene_generators(dtype: torch.dtype = torch.float64) -> Tensor:
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


def _unary_path(
    catalog: TypeCatalog,
    target: TypeKey,
    source: TypeKey,
    source_name: str,
) -> PathSpec:
    signature = CSignature(target, (CSlot("value", source),))
    artifact = compile_covariant_basis(catalog, signature)
    if artifact.basis_dimension < 1:
        raise ValueError(
            f"unary path {source} to {target} has an empty Hom space"
        )
    return PathSpec(signature, (source_name,), artifact)


def _source_bank(
    b_keys: tuple[TypeKey, ...],
    raw_channels: dict[TypeKey, int],
) -> dict[str, SourceSpec]:
    sources = {
        "d": SourceSpec(A, 1),
        "e": SourceSpec(A, 1),
    }
    for key in b_keys:
        name = type_name(key)
        channels = raw_channels[key]
        sources[f"Bi_{name}"] = SourceSpec(key, channels)
        sources[f"Bj_{name}"] = SourceSpec(key, channels)
        sources[f"hi_{name}"] = SourceSpec(key, channels)
        sources[f"hj_{name}"] = SourceSpec(key, channels)
    return sources


def build_message_update(
    catalog: TypeCatalog,
    raw_channels: dict[TypeKey, int],
    invariant_dim: int,
    gate_width: int,
    dtype: torch.dtype,
) -> ABMessageUpdate:
    b_keys = tuple(key for key in catalog.blocks if key.stream == "B")
    base = _source_bank(b_keys, raw_channels)
    a_mid_paths = [_unary_path(catalog, A, A, "d")]
    for key in b_keys:
        name = type_name(key)
        a_mid_paths.extend(
            (
                _unary_path(catalog, A, key, f"Bi_{name}"),
                _unary_path(catalog, A, key, f"Bj_{name}"),
            )
        )
    a_mid = TypedStage(
        invariant_dim,
        base,
        {
            A: TargetSpec(
                1,
                ("d", "e"),
                tuple(a_mid_paths),
            )
        },
        gate_width,
        dtype,
    )
    after_a = dict(base)
    after_a["a1"] = SourceSpec(A, 1)
    wide_targets = {}
    for key in b_keys:
        name = type_name(key)
        wide_targets[key] = TargetSpec(
            2,
            (f"Bj_{name}", f"hj_{name}"),
            (
                _unary_path(catalog, key, A, "d"),
                _unary_path(catalog, key, A, "a1"),
            ),
        )
    b_wide = TypedStage(
        invariant_dim,
        after_a,
        wide_targets,
        gate_width,
        dtype,
    )
    after_wide = dict(after_a)
    for key in b_keys:
        after_wide[f"b1_{type_name(key)}"] = SourceSpec(key, 2)
    out_targets = {}
    for key in b_keys:
        name = type_name(key)
        out_targets[key] = TargetSpec(
            1,
            (f"Bj_{name}", f"hj_{name}", f"b1_{name}"),
            (
                _unary_path(catalog, key, A, "d"),
                _unary_path(catalog, key, A, "a1"),
            ),
        )
    b_out = TypedStage(
        invariant_dim,
        after_wide,
        out_targets,
        gate_width,
        dtype,
    )
    after_out = dict(after_wide)
    for key in b_keys:
        after_out[f"b2_{type_name(key)}"] = SourceSpec(key, 1)
    a_out = TypedStage(
        invariant_dim,
        after_out,
        {
            A: TargetSpec(
                1,
                ("d", "e", "a1"),
                tuple(
                    _unary_path(
                        catalog,
                        A,
                        key,
                        f"b2_{type_name(key)}",
                    )
                    for key in b_keys
                ),
            )
        },
        gate_width,
        dtype,
    )
    block = ABMessageUpdate(catalog, a_mid, b_wide, b_out, a_out)
    nn.init.zeros_(block.a_out.updates["A"].project.weight)
    return block


class OneBlockForceGNN(nn.Module):
    """Predict five molecular forces with one typed Message Block."""

    def __init__(
        self,
        force_scale: float,
        molecule_count: int = 5,
        distance_scale: float = 8.0,
        gate_width: int = 8,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if molecule_count < 2:
            raise ValueError("molecule_count must be at least two")
        if not math.isfinite(force_scale) or force_scale <= 0.0:
            raise ValueError("force_scale must be finite and positive")
        if not math.isfinite(distance_scale) or distance_scale <= 0.0:
            raise ValueError("distance_scale must be finite and positive")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")
        generators = benzene_generators(torch.float64)
        anchors = compile_anchors(generators, output_ranks=(2, 6))
        manifest = build_primitive_b_manifest(anchors)
        catalog = build_type_catalog(
            GeneratorSystem(("sixfold", "twofold"), generators),
            manifest,
        )
        b_keys = tuple(item.key for item in manifest)
        raw_channels = {
            item.key: len(item.anchor_columns) for item in manifest
        }
        for key in b_keys:
            actions = catalog.resolve(key).actions
            identity = torch.eye(actions.shape[-1], dtype=actions.dtype)
            residual = actions.mT @ actions - identity
            if not torch.allclose(residual, torch.zeros_like(residual), atol=1e-10, rtol=1e-10):
                raise ValueError("pose invariant metric is not the identity")
        self.molecule_count = molecule_count
        self.message_block_count = 1
        self.distance_scale = float(distance_scale)
        self.manifest = manifest
        self.catalog = catalog
        self.b_keys = b_keys
        self.pose_encoder = PoseEncoder(anchors).to(dtype=dtype)
        self.message_update = build_message_update(
            catalog,
            raw_channels,
            2 + len(b_keys),
            gate_width,
            dtype,
        )
        receiver = []
        sender = []
        for receiver_index in range(molecule_count):
            for sender_index in range(molecule_count):
                if sender_index != receiver_index:
                    receiver.append(receiver_index)
                    sender.append(sender_index)
        self.register_buffer(
            "receiver",
            torch.tensor(receiver, dtype=torch.long),
        )
        self.register_buffer(
            "sender",
            torch.tensor(sender, dtype=torch.long),
        )
        self.register_buffer(
            "force_scale",
            torch.tensor(float(force_scale), dtype=dtype),
        )

    @property
    def edge_count(self) -> int:
        return int(self.receiver.numel())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_inputs(self, centers: Tensor, rotations: Tensor) -> None:
        if centers.ndim != 3 or rotations.ndim != 4:
            raise ValueError("centers and rotations have the wrong rank")
        expected_centers = (centers.shape[0], self.molecule_count, 3)
        expected_rotations = (
            centers.shape[0],
            self.molecule_count,
            3,
            3,
        )
        if tuple(centers.shape) != expected_centers:
            raise ValueError("centers have the wrong shape")
        if tuple(rotations.shape) != expected_rotations:
            raise ValueError("rotations have the wrong shape")
        if centers.dtype != self.force_scale.dtype or rotations.dtype != centers.dtype:
            raise TypeError("inputs must match the model dtype")
        if centers.device != self.force_scale.device or rotations.device != centers.device:
            raise ValueError("inputs must match the model device")

    def _invariants(self, state: dict[str, Tensor]) -> Tensor:
        displacement = state["d"][..., 0, :]
        distance = torch.linalg.vector_norm(
            displacement,
            dim=-1,
            keepdim=True,
        )
        values = [torch.ones_like(distance), distance]
        for key in self.b_keys:
            name = type_name(key)
            receiver = state[f"Bi_{name}"]
            sender = state[f"Bj_{name}"]
            alignment = (receiver * sender).sum(dim=-1).mean(
                dim=-1,
                keepdim=True,
            )
            values.append(alignment)
        return torch.cat(values, dim=-1)

    def _sum_incoming(self, value: Tensor) -> Tensor:
        result = value.new_zeros(
            (value.shape[0], self.molecule_count) + value.shape[2:]
        )
        return result.index_add(1, self.receiver, value)

    def messages(
        self,
        centers: Tensor,
        rotations: Tensor,
    ) -> AggregatedMessages:
        self._validate_inputs(centers, rotations)
        displacement = (
            centers[:, self.sender] - centers[:, self.receiver]
        ).unsqueeze(-2)
        displacement = displacement / self.distance_scale
        node_b = encode_typed_blocks(
            self.pose_encoder,
            rotations,
            self.manifest,
        )
        receiver_b = {
            key: value[:, self.receiver] for key, value in node_b.items()
        }
        sender_b = {
            key: value[:, self.sender] for key, value in node_b.items()
        }

        def invariants(
            stage: str,
            state: dict[str, Tensor],
        ) -> Tensor:
            del stage
            return self._invariants(state)

        edge_a, edge_b = self.message_update(
            displacement,
            displacement,
            receiver_b,
            sender_b,
            receiver_b,
            sender_b,
            invariants,
        )
        aggregated_a = self._sum_incoming(edge_a)
        aggregated_b = {
            key: self._sum_incoming(value) for key, value in edge_b.items()
        }
        return AggregatedMessages(
            edge_a,
            edge_b,
            aggregated_a,
            aggregated_b,
        )

    def normalized_forces(
        self,
        centers: Tensor,
        rotations: Tensor,
    ) -> Tensor:
        return self.messages(centers, rotations).node_a.squeeze(-2)

    def forward(self, centers: Tensor, rotations: Tensor) -> Tensor:
        return self.normalized_forces(centers, rotations) * self.force_scale
