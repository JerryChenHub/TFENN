"""Receiver-local E311-derived message passing for symmetric rigid molecules.

The pair kernel keeps the historical E311 typed-stage mechanics while adding
the graph operations that the benzene-pair pipeline did not contain:

* two receiver-local directions with shared parameters;
* frame transport before both OddPair operations;
* independently evaluated wide-B streams and transported reverse-wide input;
* one evaluation per unordered graph edge;
* signed A aggregation and receiver-local B aggregation.

Node hidden-B values live in each node's own body frame.  A second block can
consume the first block's aggregated B bank, which gives later edge messages
a genuine many-body dependency without weakening exact pair antisymmetry.

The graph wrapper predicts forces directly.  Signed scatter gives exact zero
total force, but this block alone does not assert that the force is the
gradient of a scalar energy.  It currently assumes a fixed node count and one
shared unordered topology across each batch.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import TypeAlias

import torch
from torch import Tensor, nn

from TFENN.tensor_math import (
    BBlockManifest,
    CSignature,
    CSlot,
    CovariantCompilation,
    PoseEncoder,
    RegisteredCovariant,
    TRIVIAL_SCALAR,
    TypeCatalog,
    TypeKey,
    compile_covariant_basis,
    encode_typed_blocks,
    scalar_contraction,
    stf_representation,
    vector_covariant,
)


A = TypeKey("A")
TypedBank: TypeAlias = Mapping[TypeKey, Tensor]
TypedState: TypeAlias = Mapping[str, TypedBank]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _floating_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")


def _type_name(key: TypeKey) -> str:
    return "A" if key == A else f"B_{key.component}"


@dataclass(frozen=True, slots=True)
class E311MessageBlockConfigV1:
    """E311-derived channel schedule plus graph-specific input dimensions.

    The default one-channel A/B inputs match the fixed one-channel outputs and
    can therefore be fed directly into another default block.  Non-default
    input channel counts describe a transition block; the next block must be
    constructed for the one-channel output schema.
    """

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
        for name in (
            "edge_a_channels",
            "a_mid_channels",
            "b_wide_channels",
            "b_out_channels",
            "gate_width",
            "max_constraint_entries",
        ):
            _positive_integer(getattr(self, name), name)
        if self.a_mid_channels != 1 or self.b_out_channels != 1:
            raise ValueError("v1 fixes the E311 A1-B2-B1-A1 channel schedule")
        if (
            isinstance(self.molecular_scalar_dim, bool)
            or not isinstance(self.molecular_scalar_dim, int)
            or self.molecular_scalar_dim < 0
        ):
            raise ValueError("molecular_scalar_dim must be a nonnegative integer")
        for name in (
            "distance_scale",
            "rbf_width",
            "distance_epsilon",
            "rms_epsilon",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        centers = tuple(float(value) for value in self.rbf_centers)
        if not centers or any(not math.isfinite(value) for value in centers):
            raise ValueError("rbf_centers must contain finite values")
        object.__setattr__(self, "rbf_centers", centers)
        powers = tuple(_positive_integer(value, "inverse power") for value in self.inverse_powers)
        if len(set(powers)) != len(powers):
            raise ValueError("inverse_powers cannot contain duplicates")
        object.__setattr__(self, "inverse_powers", powers)


@dataclass(frozen=True, slots=True)
class _Endpoint:
    source: str
    key: TypeKey
    channels: int

    @property
    def is_raw(self) -> bool:
        return self.source in {"x", "r"}


@dataclass(frozen=True, slots=True)
class _PrimitiveSpec:
    role: str
    target: TypeKey | None
    endpoints: tuple[_Endpoint, ...]
    primitive_channels: int
    operator_fingerprint: str | None = None
    slot_names: tuple[str, ...] = ()
    shortcut_rank: int | None = None

    @property
    def bank(self) -> str:
        return "scalar" if self.target is None else "covariant"


class _PrimitiveCompiler:
    """Compile each mathematical signature once and reuse it across roles."""

    def __init__(self, catalog: TypeCatalog, max_constraint_entries: int) -> None:
        if not isinstance(catalog, TypeCatalog):
            raise TypeError("catalog must be a TypeCatalog")
        self.catalog = catalog
        self.max_constraint_entries = max_constraint_entries
        self._cache: dict[CSignature, CovariantCompilation] = {}
        self.artifacts: dict[str, CovariantCompilation] = {}

    def generic(
        self,
        role: str,
        target: TypeKey | None,
        endpoints: tuple[_Endpoint, ...],
        slots: tuple[CSlot, ...],
    ) -> _PrimitiveSpec | None:
        output = TRIVIAL_SCALAR if target is None else target
        signature = CSignature(output, slots)
        artifact = self._cache.get(signature)
        if artifact is None:
            artifact = compile_covariant_basis(
                self.catalog,
                signature,
                max_constraint_entries=self.max_constraint_entries,
            )
            self._cache[signature] = artifact
            self.artifacts[artifact.artifact_fingerprint] = artifact
        if artifact.basis_dimension == 0:
            return None
        return _PrimitiveSpec(
            role=role,
            target=target,
            endpoints=endpoints,
            primitive_channels=(
                math.prod(endpoint.channels for endpoint in endpoints)
                * artifact.basis_dimension
            ),
            operator_fingerprint=artifact.artifact_fingerprint,
            slot_names=tuple(slot.name for slot in slots),
        )

    @staticmethod
    def shortcut(
        role: str,
        target: TypeKey | None,
        x: _Endpoint,
        b: _Endpoint,
        rank: int,
    ) -> _PrimitiveSpec:
        if x.key != A or x.channels != 1 or b.key.stream != "B":
            raise ValueError("STF shortcuts require one A position and one B block")
        return _PrimitiveSpec(
            role=role,
            target=target,
            endpoints=(x, b),
            primitive_channels=b.channels,
            shortcut_rank=rank,
        )


class _PrimitiveLibrary(nn.Module):
    """Own frozen registered operators once per unique CSignature."""

    def __init__(
        self,
        artifacts: Mapping[str, CovariantCompilation],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        ordered = tuple(sorted(artifacts.items()))
        self._names = {
            fingerprint: f"operator_{index}"
            for index, (fingerprint, _artifact) in enumerate(ordered)
        }
        self.operators = nn.ModuleDict(
            {
                self._names[fingerprint]: RegisteredCovariant(artifact).to(dtype=dtype)
                for fingerprint, artifact in ordered
            }
        )

    def evaluate(
        self,
        spec: _PrimitiveSpec,
        state: TypedState,
        distance_scale: float,
    ) -> Tensor:
        values = tuple(state[item.source][item.key] for item in spec.endpoints)
        leading = values[0].shape[:-2]
        for endpoint, value in zip(spec.endpoints, values, strict=True):
            if value.shape[:-2] != leading:
                raise ValueError(f"{spec.role}: source leading axes do not match")
            if value.shape[-2] != endpoint.channels:
                raise ValueError(f"{spec.role}: source channel count does not match")
        if spec.shortcut_rank is not None:
            position = values[0][..., 0, :] / distance_scale
            pose = values[1].movedim(-2, -1)
            result = (
                scalar_contraction(position, pose, spec.shortcut_rank).unsqueeze(-1)
                if spec.target is None
                else vector_covariant(position, pose, spec.shortcut_rank)
            )
            return result
        if spec.operator_fingerprint is None:
            raise RuntimeError("generic primitive has no registered operator")
        scaled = tuple(
            value / distance_scale if endpoint.source == "x" else value
            for endpoint, value in zip(spec.endpoints, values, strict=True)
        )
        slot_count = len(scaled)
        inputs: dict[str, Tensor] = {}
        for index, (slot_name, value) in enumerate(
            zip(spec.slot_names, scaled, strict=True)
        ):
            channel_shape = (
                (1,) * index
                + (value.shape[-2],)
                + (1,) * (slot_count - index - 1)
            )
            inputs[slot_name] = value.reshape(
                leading + channel_shape + (value.shape[-1],)
            )
        operator = self.operators[self._names[spec.operator_fingerprint]]
        result = operator.evaluate_basis(inputs)
        return result.reshape(
            leading + (spec.primitive_channels, result.shape[-1])
        )


class _RunningRMS(nn.Module):
    """E311 cumulative per-schema RMS without learned affine parameters."""

    def __init__(self, epsilon: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean_square", torch.ones((), dtype=dtype))
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.int64))

    def reset(self) -> None:
        with torch.no_grad():
            self.mean_square.fill_(1.0)
            self.sample_count.zero_()

    def forward(self, value: Tensor) -> Tensor:
        if self.training:
            detached = value.detach()
            value_count = detached.numel()
            if value_count == 0:
                raise ValueError("cannot normalize an empty invariant schema")
            with torch.no_grad():
                total_count = self.sample_count + value_count
                total = total_count.to(dtype=self.mean_square.dtype)
                old_weight = self.sample_count.to(dtype=self.mean_square.dtype) / total
                new_weight = value_count / total
                self.mean_square.mul_(old_weight).add_(
                    detached.square().mean() * new_weight
                )
                self.sample_count.copy_(total_count)
        scale = torch.where(
            self.sample_count > 0,
            torch.sqrt(self.mean_square.clamp_min(0.0) + self.epsilon),
            torch.ones_like(self.mean_square),
        )
        return value / scale


class _RadialFeatures(nn.Module):
    def __init__(self, config: E311MessageBlockConfigV1, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("centers", torch.tensor(config.rbf_centers, dtype=dtype))

    @property
    def output_dim(self) -> int:
        return 2 + len(self.config.rbf_centers) + len(self.config.inverse_powers)

    def forward(self, displacement: Tensor) -> Tensor:
        distance = torch.sqrt(
            displacement.square().sum(dim=-1) + self.config.distance_epsilon
        )
        scaled = distance / self.config.distance_scale
        rbf = torch.exp(
            -((scaled.unsqueeze(-1) - self.centers) / self.config.rbf_width).square()
        )
        inverse = (1.0 + scaled).reciprocal()
        powers = (
            torch.stack(
                tuple(inverse.pow(power) for power in self.config.inverse_powers),
                dim=-1,
            )
            if self.config.inverse_powers
            else scaled.new_empty(scaled.shape + (0,))
        )
        return torch.cat(
            (torch.ones_like(scaled).unsqueeze(-1), scaled.unsqueeze(-1), rbf, powers),
            dim=-1,
        )


class _InvariantDescriptor(nn.Module):
    """Full E311 scalar bank: unary, Sym2, raw-mixed, and x-r STF."""

    def __init__(
        self,
        radial_dim: int,
        scalar_specs: Sequence[_PrimitiveSpec],
        scalar_dim: int,
        config: E311MessageBlockConfigV1,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.specs = tuple(scalar_specs)
        self.scalar_dim = scalar_dim
        schema_count = 1 + len(self.specs) + (2 if scalar_dim else 0)
        self.normalizers = nn.ModuleList(
            _RunningRMS(config.rms_epsilon, dtype) for _ in range(schema_count)
        )
        self.output_dim = (
            radial_dim
            + sum(spec.primitive_channels for spec in self.specs)
            + 2 * scalar_dim
        )

    def forward(
        self,
        state: TypedState,
        radial: Tensor,
        scalar_receiver: Tensor | None,
        scalar_sender: Tensor | None,
        library: _PrimitiveLibrary,
        distance_scale: float,
    ) -> Tensor:
        pieces = [self.normalizers[0](radial)]
        index = 1
        for spec in self.specs:
            primitive = library.evaluate(spec, state, distance_scale).squeeze(-1)
            pieces.append(self.normalizers[index](primitive))
            index += 1
        if self.scalar_dim:
            if scalar_receiver is None or scalar_sender is None:
                raise ValueError("receiver and sender molecular scalars are required")
            expected = radial.shape[:-1] + (self.scalar_dim,)
            if scalar_receiver.shape != expected or scalar_sender.shape != expected:
                raise ValueError("molecular scalar shapes do not match edge leading axes")
            pieces.append(self.normalizers[index](scalar_receiver))
            pieces.append(self.normalizers[index + 1](scalar_sender))
        elif scalar_receiver is not None or scalar_sender is not None:
            raise ValueError("this block was built without molecular scalars")
        return torch.cat(pieces, dim=-1)


class _ChannelProjection(nn.Module):
    """Mix channel multiplicities and leave representation coordinates intact."""

    def __init__(self, c_in: int, c_out: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in, dtype=dtype))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(c_in))

    def forward(self, value: Tensor) -> Tensor:
        return torch.einsum("oi,...id->...od", self.weight, value)


class _GatedPath(nn.Module):
    def __init__(
        self,
        spec: _PrimitiveSpec,
        output_channels: int,
        gate_width: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if spec.target is None:
            raise ValueError("a gated covariant path needs a typed target")
        self.spec = spec
        self.output_channels = output_channels
        self.head = nn.Linear(
            gate_width,
            output_channels * spec.primitive_channels,
            dtype=dtype,
        )

    def forward(
        self,
        state: TypedState,
        gate: Tensor,
        library: _PrimitiveLibrary,
        distance_scale: float,
    ) -> Tensor:
        primitive = library.evaluate(self.spec, state, distance_scale)
        gamma = self.head(gate).reshape(
            gate.shape[:-1] + (self.output_channels, self.spec.primitive_channels)
        )
        return torch.einsum("...op,...pd->...od", gamma, primitive)


class _TargetUpdate(nn.Module):
    def __init__(
        self,
        target: TypeKey,
        output_channels: int,
        carriers: Sequence[_Endpoint],
        paths: Sequence[_PrimitiveSpec],
        gate_width: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.target = target
        self.carriers = tuple(carriers)
        if any(endpoint.key != target for endpoint in self.carriers):
            raise ValueError("direct carriers must have the target TypeKey")
        self.paths = nn.ModuleList(
            _GatedPath(spec, output_channels, gate_width, dtype) for spec in paths
        )
        c_in = sum(item.channels for item in self.carriers) + output_channels * len(paths)
        if c_in < 1:
            raise ValueError("target update has neither a carrier nor a path")
        self.project = _ChannelProjection(c_in, output_channels, dtype)

    def forward(
        self,
        state: TypedState,
        gate: Tensor,
        library: _PrimitiveLibrary,
        distance_scale: float,
    ) -> Tensor:
        pieces = [state[item.source][item.key] for item in self.carriers]
        pieces.extend(
            path(state, gate, library, distance_scale) for path in self.paths
        )
        return self.project(torch.cat(pieces, dim=-2))


class _TypedStage(nn.Module):
    def __init__(
        self,
        name: str,
        descriptor: _InvariantDescriptor,
        updates: Mapping[TypeKey, _TargetUpdate],
        gate_width: int,
        config: E311MessageBlockConfigV1,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.name = name
        self.descriptor = descriptor
        self.keys = tuple(updates)
        self.trunk = nn.Sequential(
            nn.Linear(descriptor.output_dim, gate_width, dtype=dtype),
            nn.SiLU(),
        )
        self.updates = nn.ModuleDict(
            {_type_name(key): update for key, update in updates.items()}
        )
        self.distance_scale = config.distance_scale

    def forward(
        self,
        state: TypedState,
        radial: Tensor,
        scalar_receiver: Tensor | None,
        scalar_sender: Tensor | None,
        library: _PrimitiveLibrary,
    ) -> dict[TypeKey, Tensor]:
        invariants = self.descriptor(
            state,
            radial,
            scalar_receiver,
            scalar_sender,
            library,
            self.distance_scale,
        )
        gate = self.trunk(invariants)
        return {
            key: self.updates[_type_name(key)](
                state, gate, library, self.distance_scale
            )
            for key in self.keys
        }


def _endpoints(
    source_names: Sequence[str],
    layout: Mapping[str, Mapping[TypeKey, int]],
) -> tuple[_Endpoint, ...]:
    return tuple(
        _Endpoint(source, key, channels)
        for source in source_names
        for key, channels in layout[source].items()
    )


def _compile_scalar_bank(
    stage_name: str,
    endpoints: tuple[_Endpoint, ...],
    compiler: _PrimitiveCompiler,
    manifest: Sequence[BBlockManifest],
) -> tuple[_PrimitiveSpec, ...]:
    result: list[_PrimitiveSpec] = []
    for endpoint in endpoints:
        for power, family in ((1, "unary"), (2, "symmetric2")):
            slot = (
                CSlot("value", endpoint.key)
                if power == 1
                else CSlot("value", endpoint.key, 2, "symmetric_power")
            )
            spec = compiler.generic(
                f"{stage_name}.scalar.{family}.{endpoint.source}.{_type_name(endpoint.key)}",
                None,
                (endpoint,),
                (slot,),
            )
            if spec is not None:
                result.append(spec)
    for left, right in combinations(endpoints, 2):
        if not (left.is_raw or right.is_raw):
            continue
        spec = compiler.generic(
            (
                f"{stage_name}.scalar.pair.{left.source}.{_type_name(left.key)}."
                f"{right.source}.{_type_name(right.key)}"
            ),
            None,
            (left, right),
            (CSlot("left", left.key), CSlot("right", right.key)),
        )
        if spec is not None:
            result.append(spec)
    x = next(item for item in endpoints if item.source == "x" and item.key == A)
    for item in manifest:
        b = next(
            endpoint
            for endpoint in endpoints
            if endpoint.source == "r" and endpoint.key == item.key
        )
        result.append(
            compiler.shortcut(
                f"{stage_name}.scalar.stf.{item.stable_component_id}",
                None,
                x,
                b,
                item.stf_rank,
            )
        )
    return tuple(result)


def _compile_covariant_bank(
    stage_name: str,
    target: TypeKey,
    endpoints: tuple[_Endpoint, ...],
    compiler: _PrimitiveCompiler,
    manifest: Sequence[BBlockManifest],
) -> tuple[_PrimitiveSpec, ...]:
    result: list[_PrimitiveSpec] = []
    for endpoint in endpoints:
        for power, family in ((1, "unary"), (2, "symmetric2")):
            slot = (
                CSlot("value", endpoint.key)
                if power == 1
                else CSlot("value", endpoint.key, 2, "symmetric_power")
            )
            spec = compiler.generic(
                f"{stage_name}.{_type_name(target)}.{family}.{endpoint.source}.{_type_name(endpoint.key)}",
                target,
                (endpoint,),
                (slot,),
            )
            if spec is not None:
                result.append(spec)
    # E311 disables generic raw-mixed covariant pairs.  The controlled x-r
    # STF vector shortcut remains active only for A targets.
    if target == A:
        x = next(item for item in endpoints if item.source == "x" and item.key == A)
        for item in manifest:
            b = next(
                endpoint
                for endpoint in endpoints
                if endpoint.source == "r" and endpoint.key == item.key
            )
            result.append(
                compiler.shortcut(
                    f"{stage_name}.A.stf.{item.stable_component_id}",
                    A,
                    x,
                    b,
                    item.stf_rank,
                )
            )
    return tuple(result)


def _build_stage(
    name: str,
    source_names: Sequence[str],
    output_channels: Mapping[TypeKey, int],
    carrier_sources: Mapping[TypeKey, Sequence[str]],
    layout: Mapping[str, Mapping[TypeKey, int]],
    compiler: _PrimitiveCompiler,
    manifest: Sequence[BBlockManifest],
    radial_dim: int,
    config: E311MessageBlockConfigV1,
    dtype: torch.dtype,
) -> tuple[_TypedStage, tuple[_PrimitiveSpec, ...]]:
    names = tuple(source_names)
    if "x" not in names or "r" not in names:
        raise ValueError("every E311 stage requires raw x and r context")
    endpoints = _endpoints(names, layout)
    scalar_specs = _compile_scalar_bank(name, endpoints, compiler, manifest)
    descriptor = _InvariantDescriptor(
        radial_dim,
        scalar_specs,
        config.molecular_scalar_dim,
        config,
        dtype,
    )
    updates: dict[TypeKey, _TargetUpdate] = {}
    covariant_specs: list[_PrimitiveSpec] = []
    for target, channels in output_channels.items():
        paths = _compile_covariant_bank(
            name, target, endpoints, compiler, manifest
        )
        covariant_specs.extend(paths)
        carriers = tuple(
            _Endpoint(source, target, layout[source][target])
            for source in carrier_sources[target]
            if target in layout[source]
        )
        updates[target] = _TargetUpdate(
            target,
            channels,
            carriers,
            paths,
            config.gate_width,
            dtype,
        )
    return (
        _TypedStage(name, descriptor, updates, config.gate_width, config, dtype),
        (*scalar_specs, *covariant_specs),
    )


@dataclass(frozen=True, slots=True)
class PairTraceV1:
    a_mid_i_local: Tensor
    a_mid_j_local: Tensor
    a_out_i_local: Tensor
    a_out_j_local: Tensor
    wide_i_local: Mapping[TypeKey, Tensor]
    wide_j_local: Mapping[TypeKey, Tensor]
    reverse_wide_in_i: Mapping[TypeKey, Tensor]
    reverse_wide_in_j: Mapping[TypeKey, Tensor]


@dataclass(frozen=True, slots=True)
class PairMessageOutputV1:
    edge_a_i_local: Tensor
    edge_a_j_local: Tensor
    message_j_to_i_local: Mapping[TypeKey, Tensor]
    message_i_to_j_local: Mapping[TypeKey, Tensor]
    trace: PairTraceV1 | None = None


@dataclass(frozen=True, slots=True)
class GraphMessageBlockOutputV1:
    pair_force_world: Tensor
    node_force_world: Tensor
    edge_a_world: Tensor
    message_j_to_i_local: Mapping[TypeKey, Tensor]
    message_i_to_j_local: Mapping[TypeKey, Tensor]
    node_b_local: Mapping[TypeKey, Tensor]
    trace: PairTraceV1 | None = None


class E311PairMessageKernelV1(nn.Module):
    """Shared two-direction typed kernel in separate receiver body frames."""

    def __init__(
        self,
        catalog: TypeCatalog,
        manifest: Sequence[BBlockManifest],
        pose_encoder: PoseEncoder,
        hidden_channels: Mapping[TypeKey, int],
        config: E311MessageBlockConfigV1,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        _floating_dtype(dtype)
        if not isinstance(catalog, TypeCatalog):
            raise TypeError("catalog must be a TypeCatalog")
        manifests = tuple(manifest)
        if not manifests or any(not isinstance(item, BBlockManifest) for item in manifests):
            raise ValueError("manifest must contain BBlockManifest values")
        self.catalog = catalog
        self.manifest = manifests
        self.b_keys = tuple(key for key in catalog.blocks if key.stream == "B")
        if tuple(item.key for item in manifests) != self.b_keys:
            raise ValueError("manifest order must match the catalog B TypeKeys")
        hidden = dict(hidden_channels)
        if set(hidden) != set(self.b_keys):
            raise ValueError("hidden_channels must contain every B TypeKey")
        for key, channels in hidden.items():
            _positive_integer(channels, f"hidden_channels[{key}]")
        self.hidden_channels = MappingProxyType(hidden)
        self.config = config
        self.pose_encoder = copy.deepcopy(pose_encoder).to(dtype=dtype)
        self.radial = _RadialFeatures(config, dtype)
        self._rank_by_key = {
            item.key: item.stf_rank for item in self.manifest
        }
        raw_channels = {item.key: len(item.anchor_columns) for item in manifests}
        compiler = _PrimitiveCompiler(catalog, config.max_constraint_entries)

        layout: dict[str, dict[TypeKey, int]] = {
            "x": {A: 1},
            "e": {A: config.edge_a_channels},
            "r": raw_channels,
            "h_receiver": hidden,
            "h_sender": hidden,
        }
        base_names = ("x", "e", "r", "h_receiver", "h_sender")
        self.a_mid, a_mid_specs = _build_stage(
            "a_mid",
            base_names,
            {A: config.a_mid_channels},
            {A: ("x", "e")},
            layout,
            compiler,
            manifests,
            self.radial.output_dim,
            config,
            dtype,
        )
        layout["a_mid"] = {A: config.a_mid_channels}
        wide_names = (*base_names, "a_mid")
        b_outputs = {key: config.b_wide_channels for key in self.b_keys}
        b_carriers = {key: ("r", "h_sender") for key in self.b_keys}
        self.b_wide, wide_specs = _build_stage(
            "b_wide",
            wide_names,
            b_outputs,
            b_carriers,
            layout,
            compiler,
            manifests,
            self.radial.output_dim,
            config,
            dtype,
        )
        layout["b_wide"] = {
            key: config.b_wide_channels for key in self.b_keys
        }
        layout["b_reverse"] = {
            key: config.b_wide_channels for key in self.b_keys
        }
        narrow_names = (*wide_names, "b_wide", "b_reverse")
        narrow_outputs = {key: config.b_out_channels for key in self.b_keys}
        narrow_carriers = {
            key: ("r", "h_sender", "b_wide") for key in self.b_keys
        }
        self.b_out, narrow_specs = _build_stage(
            "b_out",
            narrow_names,
            narrow_outputs,
            narrow_carriers,
            layout,
            compiler,
            manifests,
            self.radial.output_dim,
            config,
            dtype,
        )
        layout["b_out"] = {key: config.b_out_channels for key in self.b_keys}
        layout["b_out_reverse"] = {
            key: config.b_out_channels for key in self.b_keys
        }
        out_names = (*narrow_names, "b_out", "b_out_reverse")
        self.a_out, out_specs = _build_stage(
            "a_out",
            out_names,
            {A: 1},
            {A: ("x", "e", "a_mid")},
            layout,
            compiler,
            manifests,
            self.radial.output_dim,
            config,
            dtype,
        )
        self.library = _PrimitiveLibrary(compiler.artifacts, dtype)
        all_specs = (*a_mid_specs, *wide_specs, *narrow_specs, *out_specs)
        self.path_manifest = tuple(
            MappingProxyType(
                {
                    "role": spec.role,
                    "bank": spec.bank,
                    "target": None if spec.target is None else _type_name(spec.target),
                    "sources": tuple(item.source for item in spec.endpoints),
                    "types": tuple(_type_name(item.key) for item in spec.endpoints),
                    "primitive_channels": spec.primitive_channels,
                    "shortcut_rank": spec.shortcut_rank,
                }
            )
            for spec in all_specs
        )
        self.register_buffer("_runtime_reference", torch.empty((), dtype=dtype))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def reset_running_rms(self) -> None:
        for module in self.modules():
            if isinstance(module, _RunningRMS):
                module.reset()

    def _action(self, key: TypeKey, rotation: Tensor) -> Tensor:
        if key == A:
            return rotation
        return stf_representation(rotation, self._rank_by_key[key], validate=False)

    def transport(self, value: Tensor, key: TypeKey, rotation: Tensor) -> Tensor:
        """Transport row-layout values by the active local-frame rotation."""

        return value @ self._action(key, rotation).mT

    def _transport_bank(
        self,
        bank: Mapping[TypeKey, Tensor],
        rotation: Tensor,
    ) -> dict[TypeKey, Tensor]:
        return {key: self.transport(bank[key], key, rotation) for key in self.b_keys}

    @staticmethod
    def _stack_bank(
        first: Mapping[TypeKey, Tensor],
        second: Mapping[TypeKey, Tensor],
    ) -> dict[TypeKey, Tensor]:
        return {
            key: torch.stack((first[key], second[key]), dim=-3)
            for key in first
        }

    @staticmethod
    def _split_bank(
        bank: Mapping[TypeKey, Tensor],
    ) -> tuple[dict[TypeKey, Tensor], dict[TypeKey, Tensor]]:
        first: dict[TypeKey, Tensor] = {}
        second: dict[TypeKey, Tensor] = {}
        for key, value in bank.items():
            first[key], second[key] = value.unbind(dim=-3)
        return first, second

    def _validate_hidden(
        self,
        name: str,
        bank: Mapping[TypeKey, Tensor],
        leading: torch.Size,
    ) -> None:
        if set(bank) != set(self.b_keys):
            raise ValueError(f"{name} must contain every B TypeKey")
        for key in self.b_keys:
            expected = (
                self.hidden_channels[key],
                self.catalog.resolve(key).representation_dim,
            )
            value = bank[key]
            if value.shape[:-2] != leading or value.shape[-2:] != expected:
                raise ValueError(f"{name}[{key}] has the wrong shape")
            if value.dtype != self._runtime_reference.dtype:
                raise TypeError(f"{name}[{key}] has the wrong dtype")
            if value.device != self._runtime_reference.device:
                raise ValueError(f"{name}[{key}] has the wrong device")

    def _odd_pair(
        self,
        candidate_i: Tensor,
        candidate_j: Tensor,
        rotation_ij: Tensor,
    ) -> tuple[Tensor, Tensor]:
        candidate_j_in_i = self.transport(candidate_j, A, rotation_ij)
        value_i = 0.5 * (candidate_i - candidate_j_in_i)
        value_j = -self.transport(value_i, A, rotation_ij.mT)
        return value_i, value_j

    def forward_local(
        self,
        displacement_i: Tensor,
        relative_rotation_ij: Tensor,
        edge_a_i: Tensor,
        hidden_i: Mapping[TypeKey, Tensor],
        hidden_j: Mapping[TypeKey, Tensor],
        scalar_i: Tensor | None = None,
        scalar_j: Tensor | None = None,
        *,
        collect_trace: bool = False,
    ) -> PairMessageOutputV1:
        """Evaluate one batch of unordered pairs rooted once at each endpoint."""

        leading = displacement_i.shape[:-1]
        if displacement_i.shape[-1:] != (3,):
            raise ValueError("displacement_i must end in Cartesian dimension three")
        if relative_rotation_ij.shape != leading + (3, 3):
            raise ValueError("relative_rotation_ij has the wrong shape")
        if edge_a_i.shape != leading + (self.config.edge_a_channels, 3):
            raise ValueError("edge_a_i has the wrong shape")
        if (
            displacement_i.dtype != self._runtime_reference.dtype
            or relative_rotation_ij.dtype != self._runtime_reference.dtype
            or edge_a_i.dtype != self._runtime_reference.dtype
        ):
            raise TypeError("pair geometry must match the block dtype")
        if (
            displacement_i.device != self._runtime_reference.device
            or relative_rotation_ij.device != self._runtime_reference.device
            or edge_a_i.device != self._runtime_reference.device
        ):
            raise ValueError("pair geometry must match the block device")
        self._validate_hidden("hidden_i", hidden_i, leading)
        self._validate_hidden("hidden_j", hidden_j, leading)

        rotation_ji = relative_rotation_ij.mT
        displacement_j = -self.transport(
            displacement_i.unsqueeze(-2), A, rotation_ji
        ).squeeze(-2)
        edge_a_j = -self.transport(edge_a_i, A, rotation_ji)
        raw_i = dict(
            encode_typed_blocks(
                self.pose_encoder, relative_rotation_ij, self.manifest
            )
        )
        raw_j = dict(
            encode_typed_blocks(self.pose_encoder, rotation_ji, self.manifest)
        )
        sender_in_i = self._transport_bank(hidden_j, relative_rotation_ij)
        sender_in_j = self._transport_bank(hidden_i, rotation_ji)

        state: dict[str, dict[TypeKey, Tensor]] = {
            "x": {A: torch.stack((displacement_i, displacement_j), dim=-2).unsqueeze(-2)},
            "e": {A: torch.stack((edge_a_i, edge_a_j), dim=-3)},
            "r": self._stack_bank(raw_i, raw_j),
            "h_receiver": self._stack_bank(hidden_i, hidden_j),
            "h_sender": self._stack_bank(sender_in_i, sender_in_j),
        }
        radial = torch.stack(
            (self.radial(displacement_i), self.radial(displacement_j)), dim=-2
        )
        if self.config.molecular_scalar_dim:
            if scalar_i is None or scalar_j is None:
                raise ValueError("both molecular scalar tensors are required")
            scalar_receiver = torch.stack((scalar_i, scalar_j), dim=-2)
            scalar_sender = torch.stack((scalar_j, scalar_i), dim=-2)
        else:
            if scalar_i is not None or scalar_j is not None:
                raise ValueError("this block was built without molecular scalars")
            scalar_receiver = None
            scalar_sender = None

        a_candidates = self.a_mid(
            state, radial, scalar_receiver, scalar_sender, self.library
        )[A]
        candidate_i, candidate_j = a_candidates.unbind(dim=-3)
        a_mid_i, a_mid_j = self._odd_pair(
            candidate_i, candidate_j, relative_rotation_ij
        )
        state["a_mid"] = {
            A: torch.stack((a_mid_i, a_mid_j), dim=-3)
        }

        # Both wide directions read one frozen pre-wide state.
        wide = self.b_wide(
            state, radial, scalar_receiver, scalar_sender, self.library
        )
        wide_i, wide_j = self._split_bank(wide)
        reverse_wide_i = self._transport_bank(wide_j, relative_rotation_ij)
        reverse_wide_j = self._transport_bank(wide_i, rotation_ji)
        state["b_wide"] = self._stack_bank(wide_i, wide_j)
        state["b_reverse"] = self._stack_bank(reverse_wide_i, reverse_wide_j)

        narrow = self.b_out(
            state, radial, scalar_receiver, scalar_sender, self.library
        )
        narrow_i, narrow_j = self._split_bank(narrow)
        reverse_narrow_i = self._transport_bank(narrow_j, relative_rotation_ij)
        reverse_narrow_j = self._transport_bank(narrow_i, rotation_ji)
        state["b_out"] = self._stack_bank(narrow_i, narrow_j)
        state["b_out_reverse"] = self._stack_bank(
            reverse_narrow_i, reverse_narrow_j
        )

        out_candidates = self.a_out(
            state, radial, scalar_receiver, scalar_sender, self.library
        )[A]
        out_i, out_j = out_candidates.unbind(dim=-3)
        edge_i, edge_j = self._odd_pair(out_i, out_j, relative_rotation_ij)
        trace = (
            PairTraceV1(
                a_mid_i,
                a_mid_j,
                edge_i,
                edge_j,
                MappingProxyType(wide_i),
                MappingProxyType(wide_j),
                MappingProxyType(reverse_wide_i),
                MappingProxyType(reverse_wide_j),
            )
            if collect_trace
            else None
        )
        return PairMessageOutputV1(
            edge_i,
            edge_j,
            MappingProxyType(narrow_i),
            MappingProxyType(narrow_j),
            trace,
        )


class E311MultibodyMessageBlockV1(nn.Module):
    """Fixed-topology graph wrapper with one entry per unordered pair."""

    def __init__(self, pair_kernel: E311PairMessageKernelV1) -> None:
        super().__init__()
        if not isinstance(pair_kernel, E311PairMessageKernelV1):
            raise TypeError("pair_kernel must be E311PairMessageKernelV1")
        self.pair_kernel = pair_kernel

    @property
    def b_keys(self) -> tuple[TypeKey, ...]:
        return self.pair_kernel.b_keys

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _validate_pair_index(pair_index: Tensor, node_count: int) -> None:
        if pair_index.dtype != torch.int64 or pair_index.ndim != 2:
            raise TypeError("pair_index must be an int64 tensor of shape (2, E)")
        if pair_index.shape[0] != 2 or pair_index.shape[1] < 1:
            raise ValueError("pair_index must contain at least one unordered edge")
        if bool(((pair_index < 0) | (pair_index >= node_count)).any()):
            raise ValueError("pair_index contains an out-of-range node index")
        first, second = pair_index
        if bool((first == second).any()):
            raise ValueError("self edges are not allowed")
        low = torch.minimum(first, second)
        high = torch.maximum(first, second)
        keys = low * node_count + high
        if torch.unique(keys).numel() != keys.numel():
            raise ValueError("pair_index contains a duplicate unordered edge")

    @staticmethod
    def _index_add_nodes(
        values: Tensor,
        indices: Tensor,
        node_count: int,
    ) -> Tensor:
        batch_shape = values.shape[:-3]
        result = values.new_zeros(batch_shape + (node_count,) + values.shape[-2:])
        return result.index_add(len(batch_shape), indices, values)

    @staticmethod
    def _index_add_vectors(
        values: Tensor,
        indices: Tensor,
        node_count: int,
    ) -> Tensor:
        batch_shape = values.shape[:-2]
        result = values.new_zeros(batch_shape + (node_count, 3))
        return result.index_add(len(batch_shape), indices, values)

    def forward(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
        pair_index: Tensor,
        hidden_b_local: Mapping[TypeKey, Tensor],
        edge_a_world: Tensor | None = None,
        node_scalars: Tensor | None = None,
        *,
        collect_trace: bool = False,
    ) -> GraphMessageBlockOutputV1:
        if centers_world.ndim < 2 or centers_world.shape[-1] != 3:
            raise ValueError("centers_world must have shape (..., N, 3)")
        batch_shape = centers_world.shape[:-2]
        node_count = centers_world.shape[-2]
        if frames_body_to_world.shape != batch_shape + (node_count, 3, 3):
            raise ValueError("frames_body_to_world has the wrong shape")
        if centers_world.dtype != self.pair_kernel._runtime_reference.dtype:
            raise TypeError("graph geometry must match the block dtype")
        if frames_body_to_world.dtype != centers_world.dtype:
            raise TypeError("centers and frames must have the same dtype")
        if centers_world.device != self.pair_kernel._runtime_reference.device:
            raise ValueError("graph geometry must match the block device")
        if frames_body_to_world.device != centers_world.device:
            raise ValueError("centers and frames must be on the same device")
        if pair_index.device != centers_world.device:
            raise ValueError("pair_index must be on the graph device")
        self._validate_pair_index(pair_index, node_count)
        if set(hidden_b_local) != set(self.b_keys):
            raise ValueError("hidden_b_local must contain every B TypeKey")
        for key in self.b_keys:
            expected = batch_shape + (
                node_count,
                self.pair_kernel.hidden_channels[key],
                self.pair_kernel.catalog.resolve(key).representation_dim,
            )
            if hidden_b_local[key].shape != expected:
                raise ValueError(f"hidden_b_local[{key}] has the wrong shape")
        scalar_dim = self.pair_kernel.config.molecular_scalar_dim
        if scalar_dim:
            if node_scalars is None or node_scalars.shape != batch_shape + (
                node_count,
                scalar_dim,
            ):
                raise ValueError("node_scalars has the wrong shape")
        elif node_scalars is not None:
            raise ValueError("this block was built without molecular scalars")

        receiver, sender = pair_index
        frame_i = frames_body_to_world[..., receiver, :, :]
        frame_j = frames_body_to_world[..., sender, :, :]
        displacement_world = (
            centers_world[..., sender, :] - centers_world[..., receiver, :]
        )
        displacement_i = torch.einsum(
            "...d,...de->...e", displacement_world, frame_i
        )
        relative_rotation = frame_i.mT @ frame_j
        edge_count = pair_index.shape[1]
        if edge_a_world is None:
            edge_a_world = centers_world.new_zeros(
                batch_shape
                + (edge_count, self.pair_kernel.config.edge_a_channels, 3)
            )
        expected_edge = batch_shape + (
            edge_count,
            self.pair_kernel.config.edge_a_channels,
            3,
        )
        if edge_a_world.shape != expected_edge:
            raise ValueError("edge_a_world has the wrong shape")
        edge_a_i = torch.einsum("...cd,...de->...ce", edge_a_world, frame_i)
        hidden_i = {key: hidden_b_local[key][..., receiver, :, :] for key in self.b_keys}
        hidden_j = {key: hidden_b_local[key][..., sender, :, :] for key in self.b_keys}
        scalar_i = None if node_scalars is None else node_scalars[..., receiver, :]
        scalar_j = None if node_scalars is None else node_scalars[..., sender, :]
        pair = self.pair_kernel.forward_local(
            displacement_i,
            relative_rotation,
            edge_a_i,
            hidden_i,
            hidden_j,
            scalar_i,
            scalar_j,
            collect_trace=collect_trace,
        )
        next_edge_world = torch.einsum(
            "...cd,...de->...ce", pair.edge_a_i_local, frame_i.mT
        )
        pair_force_world = next_edge_world[..., 0, :]
        node_force_world = self._index_add_vectors(
            pair_force_world, receiver, node_count
        ) + self._index_add_vectors(-pair_force_world, sender, node_count)
        node_b = {
            key: self._index_add_nodes(
                pair.message_j_to_i_local[key], receiver, node_count
            )
            + self._index_add_nodes(
                pair.message_i_to_j_local[key], sender, node_count
            )
            for key in self.b_keys
        }
        return GraphMessageBlockOutputV1(
            pair_force_world,
            node_force_world,
            next_edge_world,
            pair.message_j_to_i_local,
            pair.message_i_to_j_local,
            MappingProxyType(node_b),
            pair.trace,
        )


def build_e311_multibody_message_block_v1(
    catalog: TypeCatalog,
    manifest: Sequence[BBlockManifest],
    pose_encoder: PoseEncoder,
    hidden_channels: Mapping[TypeKey, int],
    config: E311MessageBlockConfigV1 = E311MessageBlockConfigV1(),
    dtype: torch.dtype = torch.float64,
) -> E311MultibodyMessageBlockV1:
    """Compile the E311-derived typed kernel and its graph wrapper."""

    kernel = E311PairMessageKernelV1(
        catalog,
        manifest,
        pose_encoder,
        hidden_channels,
        config,
        dtype,
    )
    return E311MultibodyMessageBlockV1(kernel)


__all__ = [
    "A",
    "E311MessageBlockConfigV1",
    "E311MultibodyMessageBlockV1",
    "E311PairMessageKernelV1",
    "GraphMessageBlockOutputV1",
    "PairMessageOutputV1",
    "PairTraceV1",
    "build_e311_multibody_message_block_v1",
]
