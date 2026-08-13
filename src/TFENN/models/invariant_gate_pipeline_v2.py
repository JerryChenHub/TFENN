"""Information preserving invariant gate pipeline, version two.

All representation dependent tensors are compiled before construction and are
registered as buffers.  Runtime evaluation only performs fixed contractions,
scalar normalization, scalar neural maps, and channel axis projections.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor, nn

from TFENN.tensor_math import (
    AnchorCompilation,
    BBlockManifest,
    CSignature,
    CSlot,
    CovariantCompilation,
    GeneratorSystem,
    PoseEncoder,
    RegisteredCovariant,
    TRIVIAL_SCALAR,
    TypeKey,
    build_primitive_b_manifest,
    build_type_catalog,
    compile_anchors,
    compile_covariant_basis,
    encode_typed_blocks,
    scalar_contraction,
    vector_covariant,
)


__all__ = [
    "CandidateAuditV2",
    "InvariantGatePipelineV2",
    "InvariantGatePipelineV2Config",
    "InvariantGateStageV2Config",
    "PipelineV2CompilationError",
    "PipelineV2Debug",
    "RadialFeaturesV2Config",
    "TypedStateV2",
    "build_invariant_gate_pipeline_v2",
    "default_invariant_gate_pipeline_v2_config",
]


A = TypeKey("A")
Stream = Literal["A", "B"]
CandidateStatus = Literal["compiled", "empty_hom", "over_budget", "failed"]
TypedStateV2: TypeAlias = dict[str, dict[TypeKey, Tensor]]
_RAW_SOURCES = frozenset(("x", "r"))
_RESERVED_STREAMS: dict[str, Stream] = {"x": "A", "r": "B"}


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _names(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class RadialFeaturesV2Config:
    """Configure the scalar radial schema."""

    distance_scale: float = 6.0
    rbf_centers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    rbf_width: float = 0.4
    inverse_powers: tuple[int, ...] = (1, 2, 3)
    distance_epsilon: float = 1.0e-12
    rms_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distance_scale",
            _positive_float(self.distance_scale, "distance_scale"),
        )
        centers = tuple(float(value) for value in self.rbf_centers)
        if not centers or any(not math.isfinite(value) for value in centers):
            raise ValueError("rbf_centers must contain finite values")
        object.__setattr__(self, "rbf_centers", centers)
        object.__setattr__(
            self, "rbf_width", _positive_float(self.rbf_width, "rbf_width")
        )
        powers = tuple(
            _positive_int(value, "inverse power") for value in self.inverse_powers
        )
        if len(set(powers)) != len(powers):
            raise ValueError("inverse_powers cannot contain duplicates")
        object.__setattr__(self, "inverse_powers", powers)
        object.__setattr__(
            self,
            "distance_epsilon",
            _positive_float(self.distance_epsilon, "distance_epsilon"),
        )
        object.__setattr__(
            self, "rms_epsilon", _positive_float(self.rms_epsilon, "rms_epsilon")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RadialFeaturesV2Config":
        return cls(
            distance_scale=value.get("distance_scale", 6.0),
            rbf_centers=tuple(value.get("rbf_centers", (0.0, 0.5, 1.0, 1.5, 2.0))),
            rbf_width=value.get("rbf_width", 0.4),
            inverse_powers=tuple(value.get("inverse_powers", (1, 2, 3))),
            distance_epsilon=value.get("distance_epsilon", 1.0e-12),
            rms_epsilon=value.get("rms_epsilon", 1.0e-8),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "distance_scale": self.distance_scale,
            "rbf_centers": list(self.rbf_centers),
            "rbf_width": self.rbf_width,
            "inverse_powers": list(self.inverse_powers),
            "distance_epsilon": self.distance_epsilon,
            "rms_epsilon": self.rms_epsilon,
        }


@dataclass(frozen=True, slots=True)
class InvariantGateStageV2Config:
    """Declare one typed stage and every source it is allowed to read."""

    name: str
    output_stream: Stream
    source_names: tuple[str, ...]
    channels: int
    invariant_source_names: tuple[str, ...] | None = None
    skip_source_names: tuple[str, ...] | None = None
    trunk_width: int = 32
    activation: Literal["silu", "tanh", "relu", "gelu"] = "silu"
    include_symmetric_unary: bool = True
    include_raw_mixed_pairs: bool = True
    include_stf_shortcuts: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("stage name must be a valid identifier")
        if self.name in _RAW_SOURCES:
            raise ValueError("stage name is reserved")
        if self.output_stream not in ("A", "B"):
            raise ValueError("output_stream must be A or B")
        object.__setattr__(
            self, "source_names", _names(self.source_names, "source_names")
        )
        if self.invariant_source_names is not None:
            object.__setattr__(
                self,
                "invariant_source_names",
                _names(self.invariant_source_names, "invariant_source_names"),
            )
        if self.skip_source_names is not None:
            object.__setattr__(
                self,
                "skip_source_names",
                _names(self.skip_source_names, "skip_source_names"),
            )
        object.__setattr__(self, "channels", _positive_int(self.channels, "channels"))
        object.__setattr__(
            self, "trunk_width", _positive_int(self.trunk_width, "trunk_width")
        )
        if self.activation not in ("silu", "tanh", "relu", "gelu"):
            raise ValueError("unsupported activation")
        for field_name in (
            "include_symmetric_unary",
            "include_raw_mixed_pairs",
            "include_stf_shortcuts",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvariantGateStageV2Config":
        return cls(
            name=value["name"],
            output_stream=value["output_stream"],
            source_names=tuple(value["source_names"]),
            channels=value["channels"],
            invariant_source_names=None
            if value.get("invariant_source_names") is None
            else tuple(value["invariant_source_names"]),
            skip_source_names=None
            if value.get("skip_source_names") is None
            else tuple(value["skip_source_names"]),
            trunk_width=value.get("trunk_width", 32),
            activation=value.get("activation", "silu"),
            include_symmetric_unary=value.get("include_symmetric_unary", True),
            include_raw_mixed_pairs=value.get("include_raw_mixed_pairs", True),
            include_stf_shortcuts=value.get("include_stf_shortcuts", True),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_stream": self.output_stream,
            "source_names": list(self.source_names),
            "channels": self.channels,
            "invariant_source_names": None
            if self.invariant_source_names is None
            else list(self.invariant_source_names),
            "skip_source_names": None
            if self.skip_source_names is None
            else list(self.skip_source_names),
            "trunk_width": self.trunk_width,
            "activation": self.activation,
            "include_symmetric_unary": self.include_symmetric_unary,
            "include_raw_mixed_pairs": self.include_raw_mixed_pairs,
            "include_stf_shortcuts": self.include_stf_shortcuts,
        }


@dataclass(frozen=True, slots=True)
class InvariantGatePipelineV2Config:
    """Configure a complete information preserving typed pipeline."""

    stages: tuple[InvariantGateStageV2Config, ...]
    output_stage: str
    architecture_id: str = "invariant_gate_pipeline_v2"
    anchor_ranks: tuple[int, ...] = tuple(range(1, 7))
    max_constraint_entries: int = 10_000_000
    max_gate_coefficients: int = 2_000_000
    max_invariant_channels: int = 20_000
    radial: RadialFeaturesV2Config = field(default_factory=RadialFeaturesV2Config)

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages or any(
            not isinstance(item, InvariantGateStageV2Config) for item in stages
        ):
            raise ValueError("stages must contain stage configurations")
        object.__setattr__(self, "stages", stages)
        if (
            not isinstance(self.architecture_id, str)
            or not self.architecture_id.isidentifier()
        ):
            raise ValueError("architecture_id must be a valid identifier")
        known = dict(_RESERVED_STREAMS)
        for stage in stages:
            if stage.name in known:
                raise ValueError(f"duplicate source name {stage.name}")
            invariant_names = (
                stage.source_names
                if stage.invariant_source_names is None
                else stage.invariant_source_names
            )
            skip_names = (
                stage.source_names
                if stage.skip_source_names is None
                else stage.skip_source_names
            )
            unknown = tuple(
                name
                for name in stage.source_names + invariant_names + skip_names
                if name not in known
            )
            if unknown:
                raise ValueError(
                    f"stage {stage.name} references unavailable sources {unknown}"
                )
            known[stage.name] = stage.output_stream
        if self.output_stage != stages[-1].name:
            raise ValueError("output_stage must name the final stage")
        output = stages[-1]
        if output.output_stream != "A" or output.channels != 1:
            raise ValueError("output stage must be a one channel A stage")
        if not _RAW_SOURCES.issubset(output.source_names):
            raise ValueError("output stage must explicitly read raw x and r sources")
        ranks = tuple(_positive_int(rank, "anchor rank") for rank in self.anchor_ranks)
        if not ranks or len(set(ranks)) != len(ranks):
            raise ValueError("anchor_ranks must be nonempty and unique")
        object.__setattr__(self, "anchor_ranks", ranks)
        for name in (
            "max_constraint_entries",
            "max_gate_coefficients",
            "max_invariant_channels",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if not isinstance(self.radial, RadialFeaturesV2Config):
            raise TypeError("radial must be RadialFeaturesV2Config")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvariantGatePipelineV2Config":
        return cls(
            stages=tuple(
                InvariantGateStageV2Config.from_dict(item) for item in value["stages"]
            ),
            output_stage=value["output_stage"],
            architecture_id=value.get("architecture_id", "invariant_gate_pipeline_v2"),
            anchor_ranks=tuple(value.get("anchor_ranks", range(1, 7))),
            max_constraint_entries=value.get("max_constraint_entries", 10_000_000),
            max_gate_coefficients=value.get("max_gate_coefficients", 2_000_000),
            max_invariant_channels=value.get("max_invariant_channels", 20_000),
            radial=RadialFeaturesV2Config.from_dict(value.get("radial", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [item.as_dict() for item in self.stages],
            "output_stage": self.output_stage,
            "architecture_id": self.architecture_id,
            "anchor_ranks": list(self.anchor_ranks),
            "max_constraint_entries": self.max_constraint_entries,
            "max_gate_coefficients": self.max_gate_coefficients,
            "max_invariant_channels": self.max_invariant_channels,
            "radial": self.radial.as_dict(),
        }


def default_invariant_gate_pipeline_v2_config() -> InvariantGatePipelineV2Config:
    """Return a compact alternating graph with an explicit mixed readout."""
    return InvariantGatePipelineV2Config(
        stages=(
            InvariantGateStageV2Config("a1", "A", ("x", "r"), 4),
            InvariantGateStageV2Config("a2", "A", ("x", "r", "a1"), 4),
            InvariantGateStageV2Config("b2", "B", ("x", "r", "a1", "a2"), 4),
            InvariantGateStageV2Config("a3", "A", ("x", "r", "a1", "a2", "b2"), 1),
        ),
        output_stage="a3",
    )


@dataclass(frozen=True, slots=True)
class CandidateAuditV2:
    stage: str
    bank: Literal["scalar", "covariant"]
    role: str
    status: CandidateStatus
    signature: str
    basis_dimension: int | None = None
    estimated_parameter_count: int | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class PipelineV2CompilationError(RuntimeError):
    """Report a complete candidate audit when offline construction fails."""

    def __init__(self, message: str, audits: Sequence[CandidateAuditV2]) -> None:
        super().__init__(message)
        self.candidate_manifest = tuple(item.as_dict() for item in audits)


@dataclass(frozen=True, slots=True)
class _Endpoint:
    source: str
    key: TypeKey

    @property
    def is_raw(self) -> bool:
        return self.source in _RAW_SOURCES


@dataclass(frozen=True, slots=True)
class _Candidate:
    stage: str
    bank: Literal["scalar", "covariant"]
    role: str
    target: TypeKey | None
    endpoints: tuple[_Endpoint, ...]
    signature: CSignature | None
    shortcut_rank: int | None = None


@dataclass(frozen=True, slots=True)
class _ActivePath:
    candidate: _Candidate
    artifact_name: str | None
    primitive_channels: int
    coefficient_channels: int


@dataclass(frozen=True)
class PipelineV2Debug:
    output_local: Tensor
    output_world: Tensor
    state: Mapping[str, Mapping[TypeKey, Tensor]]
    invariants: Mapping[str, Tensor]
    branches: Mapping[str, Mapping[TypeKey, tuple[Tensor, ...]]]
    concats: Mapping[str, Mapping[TypeKey, Tensor]]
    direct_paths: Mapping[str, Tensor]


def _type_label(key: TypeKey) -> str:
    return "a" if key.stream == "A" else f"b{key.component}"


def _signature_label(signature: CSignature | None, shortcut_rank: int | None) -> str:
    if shortcut_rank is not None:
        return f"stf_shortcut_rank_{shortcut_rank}"
    assert signature is not None
    output = (
        "scalar"
        if signature.output == TRIVIAL_SCALAR
        else _type_label(signature.output)
    )
    slots = ",".join(
        f"{slot.name}:{_type_label(slot.type_key)}:{slot.mode}:{slot.power}"
        for slot in signature.inputs
    )
    return f"{output}<-{slots}"


def _keys(stream: Stream, b_keys: tuple[TypeKey, ...]) -> tuple[TypeKey, ...]:
    return (A,) if stream == "A" else b_keys


def _endpoints(
    names: Sequence[str], streams: Mapping[str, Stream], b_keys: tuple[TypeKey, ...]
) -> tuple[_Endpoint, ...]:
    return tuple(
        _Endpoint(name, key) for name in names for key in _keys(streams[name], b_keys)
    )


def _enumerate_candidates(
    config: InvariantGatePipelineV2Config,
    b_keys: tuple[TypeKey, ...],
    manifest: Sequence[BBlockManifest],
) -> tuple[_Candidate, ...]:
    streams: dict[str, Stream] = dict(_RESERVED_STREAMS)
    result: list[_Candidate] = []
    for stage in config.stages:
        invariant_names = (
            stage.source_names
            if stage.invariant_source_names is None
            else stage.invariant_source_names
        )
        scalar_endpoints = _endpoints(invariant_names, streams, b_keys)
        direct_endpoints = _endpoints(stage.source_names, streams, b_keys)
        for endpoint in scalar_endpoints:
            role = f"{stage.name}.scalar.unary.{endpoint.source}.{_type_label(endpoint.key)}"
            result.append(
                _Candidate(
                    stage.name,
                    "scalar",
                    role,
                    None,
                    (endpoint,),
                    CSignature(TRIVIAL_SCALAR, (CSlot("value", endpoint.key),)),
                )
            )
            if stage.include_symmetric_unary:
                role = f"{stage.name}.scalar.symmetric2.{endpoint.source}.{_type_label(endpoint.key)}"
                result.append(
                    _Candidate(
                        stage.name,
                        "scalar",
                        role,
                        None,
                        (endpoint,),
                        CSignature(
                            TRIVIAL_SCALAR,
                            (CSlot("value", endpoint.key, 2, "symmetric_power"),),
                        ),
                    )
                )
        if stage.include_raw_mixed_pairs:
            for left, right in combinations(scalar_endpoints, 2):
                if not (left.is_raw or right.is_raw):
                    continue
                role = f"{stage.name}.scalar.pair.{left.source}.{_type_label(left.key)}.{right.source}.{_type_label(right.key)}"
                result.append(
                    _Candidate(
                        stage.name,
                        "scalar",
                        role,
                        None,
                        (left, right),
                        CSignature(
                            TRIVIAL_SCALAR,
                            (CSlot("left", left.key), CSlot("right", right.key)),
                        ),
                    )
                )
        if (
            stage.include_stf_shortcuts
            and "x" in invariant_names
            and "r" in invariant_names
        ):
            for item in manifest:
                result.append(
                    _Candidate(
                        stage.name,
                        "scalar",
                        f"{stage.name}.scalar.stf.{item.stable_component_id}",
                        None,
                        (_Endpoint("x", A), _Endpoint("r", item.key)),
                        None,
                        item.stf_rank,
                    )
                )
        for target in _keys(stage.output_stream, b_keys):
            for endpoint in direct_endpoints:
                role = f"{stage.name}.{_type_label(target)}.unary.{endpoint.source}.{_type_label(endpoint.key)}"
                result.append(
                    _Candidate(
                        stage.name,
                        "covariant",
                        role,
                        target,
                        (endpoint,),
                        CSignature(target, (CSlot("value", endpoint.key),)),
                    )
                )
                if stage.include_symmetric_unary:
                    role = f"{stage.name}.{_type_label(target)}.symmetric2.{endpoint.source}.{_type_label(endpoint.key)}"
                    result.append(
                        _Candidate(
                            stage.name,
                            "covariant",
                            role,
                            target,
                            (endpoint,),
                            CSignature(
                                target,
                                (CSlot("value", endpoint.key, 2, "symmetric_power"),),
                            ),
                        )
                    )
            if stage.include_raw_mixed_pairs:
                for left, right in combinations(direct_endpoints, 2):
                    if not (left.is_raw or right.is_raw):
                        continue
                    role = f"{stage.name}.{_type_label(target)}.pair.{left.source}.{_type_label(left.key)}.{right.source}.{_type_label(right.key)}"
                    result.append(
                        _Candidate(
                            stage.name,
                            "covariant",
                            role,
                            target,
                            (left, right),
                            CSignature(
                                target,
                                (CSlot("left", left.key), CSlot("right", right.key)),
                            ),
                        )
                    )
            if (
                target == A
                and stage.include_stf_shortcuts
                and "x" in stage.source_names
                and "r" in stage.source_names
            ):
                for item in manifest:
                    result.append(
                        _Candidate(
                            stage.name,
                            "covariant",
                            f"{stage.name}.a.stf.{item.stable_component_id}",
                            A,
                            (_Endpoint("x", A), _Endpoint("r", item.key)),
                            None,
                            item.stf_rank,
                        )
                    )
        streams[stage.name] = stage.output_stream
    return tuple(result)


def _activation(name: str) -> nn.Module:
    return {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[name]()


def _head_parameter_upper_bound(
    candidate: _Candidate,
    catalog: Any,
    stage: InvariantGateStageV2Config,
    channels: Mapping[tuple[str, TypeKey], int],
) -> int:
    """Estimate a safe dense head bound before a Hom basis is available."""
    if candidate.bank == "scalar" or candidate.signature is None:
        return 0
    signature = candidate.signature
    output_dimension = catalog.resolve(signature.output).representation_dim
    effective_input = 1
    for slot in signature.inputs:
        dimension = catalog.resolve(slot.type_key).representation_dim
        effective_input *= math.comb(dimension + slot.power - 1, slot.power)
    input_channels = math.prod(
        channels[(endpoint.source, endpoint.key)] for endpoint in candidate.endpoints
    )
    coefficient_count = (
        stage.channels * input_channels * output_dimension * effective_input
    )
    return (stage.trunk_width + 1) * coefficient_count


class _ChannelProjection(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, *, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_channels, input_channels, dtype=dtype)
        )
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(input_channels))

    def forward(self, value: Tensor) -> Tensor:
        return torch.einsum("oi,...id->...od", self.weight, value)


class InvariantGatePipelineV2(nn.Module):
    """Evaluate a versioned typed pipeline without discarding prior sources."""

    def __init__(
        self,
        config: InvariantGatePipelineV2Config,
        pose_encoder: PoseEncoder,
        anchor_compilation: AnchorCompilation,
        manifest: Sequence[BBlockManifest],
        artifacts: Mapping[CSignature, CovariantCompilation],
        candidates: Sequence[_Candidate],
        audits: Sequence[CandidateAuditV2],
    ) -> None:
        super().__init__()
        self.config = config
        self.pose_encoder = pose_encoder
        self._anchor_compilation = anchor_compilation
        self._manifest = tuple(manifest)
        self._b_keys = tuple(item.key for item in self._manifest)
        self._candidate_audits = tuple(audits)
        self.register_buffer(
            "radial_centers",
            torch.tensor(config.radial.rbf_centers, dtype=torch.float64),
        )
        self.register_buffer(
            "_runtime_reference", torch.empty(0, dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "_manifest_components",
            torch.tensor(
                [item.component for item in self._manifest], dtype=torch.int64
            ),
        )
        self.register_buffer(
            "_manifest_ranks",
            torch.tensor([item.stf_rank for item in self._manifest], dtype=torch.int64),
        )
        counts = [len(item.anchor_columns) for item in self._manifest]
        self.register_buffer(
            "_manifest_anchor_counts", torch.tensor(counts, dtype=torch.int64)
        )
        self.register_buffer(
            "_manifest_anchor_columns",
            torch.tensor(
                [column for item in self._manifest for column in item.anchor_columns],
                dtype=torch.int64,
            ),
        )

        self.covariants = nn.ModuleDict()
        artifact_names: dict[CSignature, str] = {}
        for signature, artifact in artifacts.items():
            name = f"c_{len(artifact_names):04d}"
            artifact_names[signature] = name
            self.covariants[name] = RegisteredCovariant(artifact)

        streams: dict[str, Stream] = dict(_RESERVED_STREAMS)
        channels: dict[tuple[str, TypeKey], int] = {("x", A): 1}
        for item in self._manifest:
            channels[("r", item.key)] = len(item.anchor_columns)
        for stage in config.stages:
            for key in _keys(stage.output_stream, self._b_keys):
                channels[(stage.name, key)] = stage.channels
            streams[stage.name] = stage.output_stream
        stage_scalars: dict[str, list[_ActivePath]] = {
            stage.name: [] for stage in config.stages
        }
        stage_paths: dict[str, dict[TypeKey, list[_ActivePath]]] = {
            stage.name: {key: [] for key in _keys(stage.output_stream, self._b_keys)}
            for stage in config.stages
        }
        stage_lookup = {stage.name: stage for stage in config.stages}
        audit_by_role = {item.role: item for item in audits}
        for candidate in candidates:
            audit = audit_by_role[candidate.role]
            if audit.status != "compiled":
                continue
            if candidate.shortcut_rank is not None:
                endpoint = candidate.endpoints[1]
                primitive_channels = channels[(endpoint.source, endpoint.key)]
                artifact_name = None
                coefficient_channels = (
                    stage_lookup[candidate.stage].channels * primitive_channels
                )
            else:
                assert candidate.signature is not None
                primitive_channels = artifacts[
                    candidate.signature
                ].basis_dimension * math.prod(
                    channels[(item.source, item.key)] for item in candidate.endpoints
                )
                artifact_name = artifact_names[candidate.signature]
                coefficient_channels = (
                    stage_lookup[candidate.stage].channels * primitive_channels
                )
            active = _ActivePath(
                candidate, artifact_name, primitive_channels, coefficient_channels
            )
            if candidate.bank == "scalar":
                stage_scalars[candidate.stage].append(active)
            else:
                assert candidate.target is not None
                stage_paths[candidate.stage][candidate.target].append(active)

        self.stage_trunks = nn.ModuleDict()
        self.path_heads = nn.ModuleDict()
        self.channel_projections = nn.ModuleDict()
        self._head_names: dict[str, str] = {}
        self._projection_names: dict[tuple[str, TypeKey], str] = {}
        invariant_counts: dict[str, int] = {}
        for stage in config.stages:
            radial_count = (
                2 + len(config.radial.rbf_centers) + len(config.radial.inverse_powers)
            )
            invariant_count = radial_count + sum(
                path.primitive_channels for path in stage_scalars[stage.name]
            )
            if invariant_count > config.max_invariant_channels:
                raise PipelineV2CompilationError(
                    f"stage {stage.name} invariant count {invariant_count} exceeds limit {config.max_invariant_channels}",
                    audits,
                )
            invariant_counts[stage.name] = invariant_count
            self.stage_trunks[stage.name] = nn.Sequential(
                nn.Linear(invariant_count, stage.trunk_width, dtype=torch.float64),
                _activation(stage.activation),
            )
            for target, paths in stage_paths[stage.name].items():
                if not paths:
                    raise PipelineV2CompilationError(
                        f"stage {stage.name} has no path for {_type_label(target)}",
                        audits,
                    )
                for path in paths:
                    head_name = f"head_{len(self._head_names):04d}"
                    self._head_names[path.candidate.role] = head_name
                    self.path_heads[head_name] = nn.Linear(
                        stage.trunk_width,
                        path.coefficient_channels,
                        dtype=torch.float64,
                    )
                skip_names = (
                    stage.source_names
                    if stage.skip_source_names is None
                    else stage.skip_source_names
                )
                skip_channels = sum(
                    channels[(name, target)]
                    for name in skip_names
                    if (name, target) in channels
                )
                concat_channels = skip_channels + stage.channels * len(paths)
                projection_name = f"projection_{len(self._projection_names):04d}"
                self._projection_names[(stage.name, target)] = projection_name
                self.channel_projections[projection_name] = _ChannelProjection(
                    concat_channels, stage.channels, dtype=torch.float64
                )
                channels[(stage.name, target)] = stage.channels
        self._streams = streams
        self._channels = channels
        self._stage_scalars = {
            name: tuple(items) for name, items in stage_scalars.items()
        }
        self._stage_paths = {
            name: {key: tuple(items) for key, items in targets.items()}
            for name, targets in stage_paths.items()
        }
        self._invariant_counts = invariant_counts
        output_config = config.stages[-1]
        self._readout_sources = tuple(output_config.source_names)
        self._state_metadata = {
            "schema_version": 2,
            "config": config.as_dict(),
            "candidate_manifest": self.candidate_manifest,
            "readout_sources": self._readout_sources,
        }

    @property
    def anchor_compilation(self) -> AnchorCompilation:
        return self._anchor_compilation

    @property
    def manifest(self) -> tuple[BBlockManifest, ...]:
        return self._manifest

    @property
    def candidate_manifest(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.as_dict() for item in self._candidate_audits)

    @property
    def readout_source_manifest(self) -> tuple[str, ...]:
        return self._readout_sources

    @property
    def trunks(self) -> nn.ModuleDict:
        return self.stage_trunks

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def offline_compilation_summary(self) -> dict[str, Any]:
        statuses = {
            status: sum(item.status == status for item in self._candidate_audits)
            for status in ("compiled", "empty_hom", "over_budget", "failed")
        }
        return {
            "candidate_status_counts": statuses,
            "c_artifact_count": len(self.covariants),
            "trainable_parameter_count": self.trainable_parameter_count,
            "forward_compilation": False,
            "runtime_group_expansion": False,
        }

    def get_extra_state(self) -> dict[str, Any]:
        return dict(self._state_metadata)

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping) or dict(state) != self._state_metadata:
            raise RuntimeError("pipeline checkpoint configuration does not match")

    def zero_output_heads(self) -> int:
        """Zero the final typed channel projection and its scalar heads."""
        count = 0
        with torch.no_grad():
            for path in self._stage_paths[self.config.output_stage][A]:
                head = self.path_heads[self._head_names[path.candidate.role]]
                head.weight.zero_()
                if head.bias is not None:
                    head.bias.zero_()
                count += 1
            projection = self.channel_projections[
                self._projection_names[(self.config.output_stage, A)]
            ]
            projection.weight.zero_()
            count += 1
        return count

    def _root_geometry(self, centers: Tensor, frames: Tensor) -> tuple[Tensor, Tensor]:
        if centers.shape[-2:] != (2, 3) or frames.shape != centers.shape[:-2] + (
            2,
            3,
            3,
        ):
            raise ValueError(
                "centers and frames must describe one ordered endpoint pair"
            )
        if centers.dtype != frames.dtype or centers.device != frames.device:
            raise ValueError("centers and frames must share dtype and device")
        if (
            centers.dtype != self._runtime_reference.dtype
            or centers.device != self._runtime_reference.device
        ):
            raise ValueError("inputs must match network dtype and device")
        root = frames[..., 0, :, :]
        displacement = centers[..., 1, :] - centers[..., 0, :]
        local = torch.einsum("...ji,...j->...i", root, displacement)
        return local, root.mT @ frames[..., 1, :, :]

    def _radial(self, displacement: Tensor) -> Tensor:
        config = self.config.radial
        distance = torch.sqrt(
            displacement.square().sum(dim=-1) + config.distance_epsilon
        )
        scaled = distance / config.distance_scale
        rbf = torch.exp(
            -((scaled.unsqueeze(-1) - self.radial_centers) / config.rbf_width).square()
        )
        inverse = (1.0 + scaled).reciprocal()
        powers = (
            torch.stack(
                tuple(inverse.pow(power) for power in config.inverse_powers), dim=-1
            )
            if config.inverse_powers
            else scaled.new_empty(scaled.shape + (0,))
        )
        return torch.cat(
            (torch.ones_like(scaled).unsqueeze(-1), scaled.unsqueeze(-1), rbf, powers),
            dim=-1,
        )

    def _normalize_schema(self, value: Tensor) -> Tensor:
        return value / torch.sqrt(
            value.square().mean(dim=-1, keepdim=True) + self.config.radial.rms_epsilon
        )

    def _generic_primitive(self, path: _ActivePath, state: TypedStateV2) -> Tensor:
        candidate = path.candidate
        assert path.artifact_name is not None and candidate.signature is not None
        module = self.covariants[path.artifact_name]
        if len(candidate.endpoints) == 1:
            endpoint = candidate.endpoints[0]
            value = self._path_value(endpoint, state)
            result = module.evaluate_basis({"value": value})
            prefix = value.shape[:-2]
        else:
            left_endpoint, right_endpoint = candidate.endpoints
            left = self._path_value(left_endpoint, state)
            right = self._path_value(right_endpoint, state)
            result = module.evaluate_basis(
                {"left": left.unsqueeze(-2), "right": right.unsqueeze(-3)}
            )
            prefix = left.shape[:-2]
        return result.reshape(prefix + (path.primitive_channels, result.shape[-1]))

    def _path_value(self, endpoint: _Endpoint, state: TypedStateV2) -> Tensor:
        """Return a numerically scaled view while retaining raw state exactly."""
        value = state[endpoint.source][endpoint.key]
        if endpoint.source == "x":
            return value / self.config.radial.distance_scale
        return value

    def _shortcut_primitive(self, path: _ActivePath, state: TypedStateV2) -> Tensor:
        rank = path.candidate.shortcut_rank
        assert rank is not None
        x = state["x"][A][..., 0, :] / self.config.radial.distance_scale
        endpoint = path.candidate.endpoints[1]
        block = state[endpoint.source][endpoint.key].movedim(-2, -1)
        if path.candidate.bank == "scalar":
            return scalar_contraction(x, block, rank).unsqueeze(-1)
        return vector_covariant(x, block, rank)

    def _primitive(self, path: _ActivePath, state: TypedStateV2) -> Tensor:
        return (
            self._shortcut_primitive(path, state)
            if path.candidate.shortcut_rank is not None
            else self._generic_primitive(path, state)
        )

    def _stage_invariants(
        self, stage: InvariantGateStageV2Config, radial: Tensor, state: TypedStateV2
    ) -> Tensor:
        schemas = [self._normalize_schema(radial)]
        for path in self._stage_scalars[stage.name]:
            primitive = self._primitive(path, state).squeeze(-1)
            schemas.append(self._normalize_schema(primitive))
        return torch.cat(schemas, dim=-1)

    def _run_local(
        self, displacement: Tensor, relative_frame: Tensor, *, collect_debug: bool
    ) -> tuple[Tensor, dict[str, Any]]:
        state: TypedStateV2 = {
            "x": {A: displacement.unsqueeze(-2)},
            "r": dict(
                encode_typed_blocks(self.pose_encoder, relative_frame, self._manifest)
            ),
        }
        radial = self._radial(displacement)
        invariant_debug: dict[str, Tensor] | None = {} if collect_debug else None
        branch_debug: dict[str, dict[TypeKey, tuple[Tensor, ...]]] | None = (
            {} if collect_debug else None
        )
        concat_debug: dict[str, dict[TypeKey, Tensor]] | None = (
            {} if collect_debug else None
        )
        direct_debug: dict[str, Tensor] | None = {} if collect_debug else None
        for stage in self.config.stages:
            invariants = self._stage_invariants(stage, radial, state)
            trunk = self.stage_trunks[stage.name](invariants)
            if invariant_debug is not None:
                invariant_debug[stage.name] = invariants
            outputs: dict[TypeKey, Tensor] = {}
            if branch_debug is not None and concat_debug is not None:
                branch_debug[stage.name] = {}
                concat_debug[stage.name] = {}
            for target, paths in self._stage_paths[stage.name].items():
                branches: list[Tensor] = []
                for path in paths:
                    primitive = self._primitive(path, state)
                    head = self.path_heads[self._head_names[path.candidate.role]](trunk)
                    coefficients = head.reshape(
                        head.shape[:-1] + (stage.channels, path.primitive_channels)
                    )
                    branch = torch.einsum(
                        "...op,...pd->...od",
                        coefficients,
                        primitive,
                    )
                    branches.append(branch)
                    if direct_debug is not None:
                        direct_debug[path.candidate.role] = branch
                skip_names = (
                    stage.source_names
                    if stage.skip_source_names is None
                    else stage.skip_source_names
                )
                skip = [
                    state[name][target] for name in skip_names if target in state[name]
                ]
                concat = torch.cat((*skip, *branches), dim=-2)
                output = self.channel_projections[
                    self._projection_names[(stage.name, target)]
                ](concat)
                outputs[target] = output
                if branch_debug is not None and concat_debug is not None:
                    branch_debug[stage.name][target] = tuple(branches)
                    concat_debug[stage.name][target] = concat
            state[stage.name] = outputs
        local = state[self.config.output_stage][A][..., 0, :]
        debug = (
            {}
            if not collect_debug
            else {
                "state": state,
                "invariants": invariant_debug,
                "branches": branch_debug,
                "concats": concat_debug,
                "direct_paths": direct_debug,
            }
        )
        return local, debug

    def forward_local(self, centers: Tensor, frames: Tensor) -> Tensor:
        displacement, relative_frame = self._root_geometry(centers, frames)
        return self._run_local(displacement, relative_frame, collect_debug=False)[0]

    def forward(self, centers: Tensor, frames: Tensor) -> Tensor:
        local = self.forward_local(centers, frames)
        return torch.einsum("...ij,...j->...i", frames[..., 0, :, :], local)

    def debug_forward(self, centers: Tensor, frames: Tensor) -> PipelineV2Debug:
        displacement, relative_frame = self._root_geometry(centers, frames)
        local, values = self._run_local(
            displacement, relative_frame, collect_debug=True
        )
        world = torch.einsum("...ij,...j->...i", frames[..., 0, :, :], local)
        state = MappingProxyType(
            {
                name: MappingProxyType(dict(blocks))
                for name, blocks in values["state"].items()
            }
        )
        return PipelineV2Debug(
            local,
            world,
            state,
            MappingProxyType(values["invariants"]),
            MappingProxyType(
                {
                    name: MappingProxyType(items)
                    for name, items in values["branches"].items()
                }
            ),
            MappingProxyType(
                {
                    name: MappingProxyType(items)
                    for name, items in values["concats"].items()
                }
            ),
            MappingProxyType(values["direct_paths"]),
        )


def build_invariant_gate_pipeline_v2(
    generators: Tensor,
    config: InvariantGatePipelineV2Config | Mapping[str, Any] | None = None,
    *,
    generator_names: Sequence[str] | None = None,
) -> InvariantGatePipelineV2:
    """Compile all candidate paths offline and construct the runtime module."""
    if (
        not isinstance(generators, Tensor)
        or generators.ndim != 3
        or generators.shape[-2:] != (3, 3)
    ):
        raise ValueError("generators must have shape count by three by three")
    resolved = (
        default_invariant_gate_pipeline_v2_config()
        if config is None
        else InvariantGatePipelineV2Config.from_dict(config)
        if isinstance(config, Mapping)
        else config
    )
    if not isinstance(resolved, InvariantGatePipelineV2Config):
        raise TypeError("config must be an InvariantGatePipelineV2Config or mapping")
    names = (
        tuple(generator_names)
        if generator_names is not None
        else tuple(f"generator_{index}" for index in range(generators.shape[0]))
    )
    anchors = compile_anchors(generators, output_ranks=resolved.anchor_ranks)
    manifest = build_primitive_b_manifest(anchors)
    if not manifest:
        raise ValueError("generator system produced no primitive pose block")
    catalog = build_type_catalog(GeneratorSystem(names, generators), manifest)
    candidates = _enumerate_candidates(
        resolved, tuple(item.key for item in manifest), manifest
    )
    stage_lookup = {stage.name: stage for stage in resolved.stages}
    channel_schedule: dict[tuple[str, TypeKey], int] = {("x", A): 1}
    for item in manifest:
        channel_schedule[("r", item.key)] = len(item.anchor_columns)
    b_keys = tuple(item.key for item in manifest)
    for stage in resolved.stages:
        for key in _keys(stage.output_stream, b_keys):
            channel_schedule[(stage.name, key)] = stage.channels
    artifacts: dict[CSignature, CovariantCompilation] = {}
    audits: list[CandidateAuditV2] = []
    failures = False
    for candidate in candidates:
        signature_label = _signature_label(candidate.signature, candidate.shortcut_rank)
        if candidate.shortcut_rank is not None:
            stage = stage_lookup[candidate.stage]
            primitive_channels = channel_schedule[
                (candidate.endpoints[1].source, candidate.endpoints[1].key)
            ]
            coefficient_count = stage.channels * primitive_channels
            estimated = (
                (stage.trunk_width + 1) * coefficient_count
                if candidate.bank == "covariant"
                else 0
            )
            status: CandidateStatus = "compiled"
            reason = None
            if (
                candidate.bank == "covariant"
                and coefficient_count > resolved.max_gate_coefficients
            ):
                status = "over_budget"
                failures = True
                reason = (
                    f"coefficient head width {coefficient_count} exceeds "
                    f"max_gate_coefficients {resolved.max_gate_coefficients}"
                )
            audits.append(
                CandidateAuditV2(
                    candidate.stage,
                    candidate.bank,
                    candidate.role,
                    status,
                    signature_label,
                    1,
                    estimated,
                    reason,
                )
            )
            continue
        assert candidate.signature is not None
        try:
            artifact = artifacts.get(candidate.signature)
            if artifact is None:
                artifact = compile_covariant_basis(
                    catalog,
                    candidate.signature,
                    max_constraint_entries=resolved.max_constraint_entries,
                )
                artifacts[candidate.signature] = artifact
            status: CandidateStatus = (
                "compiled" if artifact.basis_dimension else "empty_hom"
            )
            stage = stage_lookup[candidate.stage]
            input_channels = math.prod(
                channel_schedule[(endpoint.source, endpoint.key)]
                for endpoint in candidate.endpoints
            )
            coefficient_count = (
                stage.channels * input_channels * artifact.basis_dimension
            )
            estimated = (
                0
                if candidate.bank == "scalar"
                else (stage.trunk_width + 1) * coefficient_count
            )
            reason = None
            if (
                candidate.bank == "covariant"
                and coefficient_count > resolved.max_gate_coefficients
            ):
                status = "over_budget"
                failures = True
                reason = (
                    f"coefficient head width {coefficient_count} exceeds "
                    f"max_gate_coefficients {resolved.max_gate_coefficients}"
                )
            audits.append(
                CandidateAuditV2(
                    candidate.stage,
                    candidate.bank,
                    candidate.role,
                    status,
                    signature_label,
                    artifact.basis_dimension,
                    estimated,
                    reason,
                )
            )
        except Exception as error:
            reason = str(error)
            status = "over_budget" if "exceeds allocation guard" in reason else "failed"
            failures = True
            estimate = _head_parameter_upper_bound(
                candidate,
                catalog,
                stage_lookup[candidate.stage],
                channel_schedule,
            )
            audits.append(
                CandidateAuditV2(
                    candidate.stage,
                    candidate.bank,
                    candidate.role,
                    status,
                    signature_label,
                    None,
                    estimate,
                    reason,
                )
            )
    if failures:
        raise PipelineV2CompilationError(
            "one or more candidate paths could not be compiled", audits
        )
    return InvariantGatePipelineV2(
        resolved, PoseEncoder(anchors), anchors, manifest, artifacts, candidates, audits
    )
