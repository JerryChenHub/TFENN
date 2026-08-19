"""Information preserving invariant gate pipeline, version two.

All representation dependent tensors are compiled before construction and are
registered as buffers.  Runtime evaluation only performs fixed contractions,
scalar normalization, scalar neural maps, and channel axis projections.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement
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
    TypeCatalog,
    TypeKey,
    build_primitive_b_manifest,
    build_type_catalog,
    compile_anchors,
    compile_covariant_basis,
    encode_typed_blocks,
    normalized_symmetric_power_basis,
    normalized_symmetric_power_representation,
    scalar_contraction,
    tensor_product_representation,
    vector_covariant,
)


__all__ = [
    "CandidateAuditV2",
    "ChannelProjection",
    "CoefficientActivation",
    "CoefficientHead",
    "Degree3Policy",
    "DescriptorMask",
    "InvariantGatePipelineV2",
    "InvariantGatePipelineV2Config",
    "InvariantGateStageV2Config",
    "PipelineV2CompilationError",
    "PipelineV2Debug",
    "RadialFeaturesV2Config",
    "MetricGate",
    "PathAggregation",
    "SkipPolicy",
    "TypedStateV2",
    "build_invariant_gate_pipeline_v2",
    "default_invariant_gate_pipeline_v2_config",
]


A = TypeKey("A")
Stream = Literal["A", "B"]
CandidateStatus = Literal[
    "compiled", "empty_hom", "over_budget", "failed", "unsupported"
]
SkipPolicy = Literal["legacy", "none", "id", "local_proj", "dense_proj"]
Degree3Policy = Literal["none", "sym3", "a2b", "ab2", "union", "all"]
CoefficientActivation = Literal["identity", "sigmoid", "tanh", "silu"]
CoefficientHead = Literal[
    "dense",
    "factorized",
    "orthogonal",
    "static_mixing",
    "context_lora",
    "axis_cp",
]
DescriptorMask = Literal["full", "raw_only", "unary", "mixed"]
MetricGate = Literal["none", "norm", "multiply", "skip_identity"]
ChannelProjection = Literal[
    "dense",
    "factorized",
    "tucker",
    "tensor_train",
    "toeplitz",
    "cayley",
]
PathAggregation = Literal["linear", "attention", "soft_moe"]
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


def _names(
    value: Sequence[str], name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    if (not result and not allow_empty) or any(
        not isinstance(item, str) or not item for item in result
    ):
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
    skip_policy: SkipPolicy = "legacy"
    covariant_include_symmetric_unary: bool | None = None
    covariant_include_raw_mixed_pairs: bool | None = None
    covariant_include_stf_shortcuts: bool | None = None
    invariant_include_symmetric_unary: bool | None = None
    invariant_include_raw_mixed_pairs: bool | None = None
    invariant_include_stf_shortcuts: bool | None = None
    degree3_policy: Degree3Policy = "none"
    coefficient_activation: CoefficientActivation = "identity"
    coefficient_head: CoefficientHead = "dense"
    coefficient_rank: int | None = None
    descriptor_mask: DescriptorMask = "full"
    trunk_depth: int = 1
    trunk_linearized: bool = False
    trunk_residual: bool = False
    metric_gate: MetricGate = "none"
    execution_level: int | None = None
    covariant_live_mixed_only: bool = False
    covariant_path_quota: int | None = None
    covariant_required_source_names: tuple[str, ...] | None = None
    parameter_share_group: str | None = None
    channel_projection: ChannelProjection = "dense"
    channel_projection_rank: int | None = None
    path_aggregation: PathAggregation = "linear"
    path_temperature: float = 1.0
    type_channel_overrides: tuple[tuple[int, int], ...] = ()
    reversible_coupling: bool = False

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
                _names(
                    self.skip_source_names,
                    "skip_source_names",
                    allow_empty=True,
                ),
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
        if self.skip_policy not in (
            "legacy",
            "none",
            "id",
            "local_proj",
            "dense_proj",
        ):
            raise ValueError("unsupported skip_policy")
        for field_name in (
            "covariant_include_symmetric_unary",
            "covariant_include_raw_mixed_pairs",
            "covariant_include_stf_shortcuts",
            "invariant_include_symmetric_unary",
            "invariant_include_raw_mixed_pairs",
            "invariant_include_stf_shortcuts",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be bool or None")
        if self.degree3_policy not in (
            "none",
            "sym3",
            "a2b",
            "ab2",
            "union",
            "all",
        ):
            raise ValueError("unsupported degree3_policy")
        if self.coefficient_activation not in (
            "identity",
            "sigmoid",
            "tanh",
            "silu",
        ):
            raise ValueError("unsupported coefficient_activation")
        if self.coefficient_head not in (
            "dense",
            "factorized",
            "orthogonal",
            "static_mixing",
            "context_lora",
            "axis_cp",
        ):
            raise ValueError("unsupported coefficient_head")
        if self.coefficient_head in ("dense", "static_mixing"):
            if self.coefficient_rank is not None:
                raise ValueError(
                    "dense and static coefficient heads do not use coefficient_rank"
                )
        elif self.coefficient_rank is None:
            raise ValueError("factorized coefficient heads require coefficient_rank")
        else:
            object.__setattr__(
                self,
                "coefficient_rank",
                _positive_int(self.coefficient_rank, "coefficient_rank"),
            )
            if self.coefficient_rank > 4:
                raise ValueError("coefficient_rank cannot exceed four")
        if self.descriptor_mask not in ("full", "raw_only", "unary", "mixed"):
            raise ValueError("unsupported descriptor_mask")
        object.__setattr__(
            self, "trunk_depth", _positive_int(self.trunk_depth, "trunk_depth")
        )
        if self.trunk_depth > 3:
            raise ValueError("trunk_depth cannot exceed three")
        if not isinstance(self.trunk_linearized, bool):
            raise TypeError("trunk_linearized must be bool")
        if not isinstance(self.trunk_residual, bool):
            raise TypeError("trunk_residual must be bool")
        if self.trunk_residual and self.trunk_depth != 3:
            raise ValueError("residual trunk requires trunk_depth equal to three")
        if self.trunk_residual and self.trunk_linearized:
            raise ValueError("residual trunk cannot be linearized")
        if self.metric_gate not in ("none", "norm", "multiply", "skip_identity"):
            raise ValueError("unsupported metric_gate")
        if self.metric_gate == "skip_identity" and self.skip_policy in (
            "legacy",
            "none",
        ):
            raise ValueError("skip_identity metric gate requires an explicit skip")
        if self.execution_level is not None:
            if (
                isinstance(self.execution_level, bool)
                or not isinstance(self.execution_level, int)
                or self.execution_level < 0
            ):
                raise ValueError(
                    "execution_level must be a nonnegative integer or None"
                )
        if not isinstance(self.covariant_live_mixed_only, bool):
            raise TypeError("covariant_live_mixed_only must be bool")
        if self.covariant_path_quota is not None:
            object.__setattr__(
                self,
                "covariant_path_quota",
                _positive_int(self.covariant_path_quota, "covariant_path_quota"),
            )
        if self.covariant_required_source_names is not None:
            required_sources = _names(
                self.covariant_required_source_names,
                "covariant_required_source_names",
                allow_empty=True,
            )
            unknown_required = tuple(
                name for name in required_sources if name not in self.source_names
            )
            if unknown_required:
                raise ValueError(
                    "covariant required sources must be stage source names: "
                    f"{unknown_required}"
                )
            object.__setattr__(
                self,
                "covariant_required_source_names",
                required_sources,
            )
        if self.parameter_share_group is not None and (
            not isinstance(self.parameter_share_group, str)
            or not self.parameter_share_group
        ):
            raise ValueError("parameter_share_group must be a nonempty string or None")
        if self.channel_projection not in (
            "dense",
            "factorized",
            "tucker",
            "tensor_train",
            "toeplitz",
            "cayley",
        ):
            raise ValueError("unsupported channel_projection")
        if self.channel_projection in ("dense", "toeplitz", "cayley"):
            if self.channel_projection_rank is not None:
                raise ValueError(
                    "dense, toeplitz, and cayley channel projections do not use a rank"
                )
        elif self.channel_projection_rank is None:
            raise ValueError("structured channel projection requires a rank")
        else:
            object.__setattr__(
                self,
                "channel_projection_rank",
                _positive_int(self.channel_projection_rank, "channel_projection_rank"),
            )
        if self.path_aggregation not in ("linear", "attention", "soft_moe"):
            raise ValueError("unsupported path_aggregation")
        object.__setattr__(
            self,
            "path_temperature",
            _positive_float(self.path_temperature, "path_temperature"),
        )
        overrides = tuple(
            (int(component), int(width))
            for component, width in self.type_channel_overrides
        )
        if any(component < 0 or width < 1 for component, width in overrides):
            raise ValueError("type channel overrides must be nonnegative and positive")
        if len({component for component, _width in overrides}) != len(overrides):
            raise ValueError("type channel override components must be unique")
        if overrides and self.output_stream != "B":
            raise ValueError("type channel overrides are only valid for B stages")
        object.__setattr__(self, "type_channel_overrides", overrides)
        if not isinstance(self.reversible_coupling, bool):
            raise TypeError("reversible_coupling must be bool")
        if self.reversible_coupling and self.channels < 2:
            raise ValueError("reversible coupling requires at least two channels")

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
            skip_policy=value.get("skip_policy", "legacy"),
            covariant_include_symmetric_unary=value.get(
                "covariant_include_symmetric_unary"
            ),
            covariant_include_raw_mixed_pairs=value.get(
                "covariant_include_raw_mixed_pairs"
            ),
            covariant_include_stf_shortcuts=value.get(
                "covariant_include_stf_shortcuts"
            ),
            invariant_include_symmetric_unary=value.get(
                "invariant_include_symmetric_unary"
            ),
            invariant_include_raw_mixed_pairs=value.get(
                "invariant_include_raw_mixed_pairs"
            ),
            invariant_include_stf_shortcuts=value.get(
                "invariant_include_stf_shortcuts"
            ),
            degree3_policy=value.get("degree3_policy", "none"),
            coefficient_activation=value.get("coefficient_activation", "identity"),
            coefficient_head=value.get("coefficient_head", "dense"),
            coefficient_rank=value.get("coefficient_rank"),
            descriptor_mask=value.get("descriptor_mask", "full"),
            trunk_depth=value.get("trunk_depth", 1),
            trunk_linearized=value.get("trunk_linearized", False),
            trunk_residual=value.get("trunk_residual", False),
            metric_gate=value.get("metric_gate", "none"),
            execution_level=value.get("execution_level"),
            covariant_live_mixed_only=value.get("covariant_live_mixed_only", False),
            covariant_path_quota=value.get("covariant_path_quota"),
            covariant_required_source_names=None
            if value.get("covariant_required_source_names") is None
            else tuple(value["covariant_required_source_names"]),
            parameter_share_group=value.get("parameter_share_group"),
            channel_projection=value.get("channel_projection", "dense"),
            channel_projection_rank=value.get("channel_projection_rank"),
            path_aggregation=value.get("path_aggregation", "linear"),
            path_temperature=value.get("path_temperature", 1.0),
            type_channel_overrides=tuple(
                tuple(item) for item in value.get("type_channel_overrides", ())
            ),
            reversible_coupling=value.get("reversible_coupling", False),
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
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
            "skip_policy": self.skip_policy,
            "covariant_include_symmetric_unary": self.covariant_include_symmetric_unary,
            "covariant_include_raw_mixed_pairs": self.covariant_include_raw_mixed_pairs,
            "covariant_include_stf_shortcuts": self.covariant_include_stf_shortcuts,
            "invariant_include_symmetric_unary": self.invariant_include_symmetric_unary,
            "invariant_include_raw_mixed_pairs": self.invariant_include_raw_mixed_pairs,
            "invariant_include_stf_shortcuts": self.invariant_include_stf_shortcuts,
            "degree3_policy": self.degree3_policy,
            "coefficient_activation": self.coefficient_activation,
            "coefficient_head": self.coefficient_head,
            "coefficient_rank": self.coefficient_rank,
            "descriptor_mask": self.descriptor_mask,
            "trunk_depth": self.trunk_depth,
            "trunk_linearized": self.trunk_linearized,
            "trunk_residual": self.trunk_residual,
            "metric_gate": self.metric_gate,
        }
        if self.execution_level is not None:
            result["execution_level"] = self.execution_level
        if self.covariant_live_mixed_only:
            result["covariant_live_mixed_only"] = True
        if self.covariant_path_quota is not None:
            result["covariant_path_quota"] = self.covariant_path_quota
        if self.covariant_required_source_names is not None:
            result["covariant_required_source_names"] = list(
                self.covariant_required_source_names
            )
        if self.parameter_share_group is not None:
            result["parameter_share_group"] = self.parameter_share_group
        if self.channel_projection != "dense":
            result["channel_projection"] = self.channel_projection
            result["channel_projection_rank"] = self.channel_projection_rank
        if self.path_aggregation != "linear":
            result["path_aggregation"] = self.path_aggregation
        if self.path_temperature != 1.0:
            result["path_temperature"] = self.path_temperature
        if self.type_channel_overrides:
            result["type_channel_overrides"] = [
                list(item) for item in self.type_channel_overrides
            ]
        if self.reversible_coupling:
            result["reversible_coupling"] = True
        return result


def _resolved_execution_levels(
    stages: Sequence[InvariantGateStageV2Config],
) -> tuple[int, ...]:
    """Resolve default sequential stages and explicit synchronous levels."""
    result: list[int] = []
    previous = -1
    for stage in stages:
        level = previous + 1 if stage.execution_level is None else stage.execution_level
        if level < previous:
            raise ValueError("execution levels must be nondecreasing")
        result.append(level)
        previous = level
    return tuple(result)


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
    degree3_overflow_policy: Literal["raise", "audit_skip"] = "raise"
    radial: RadialFeaturesV2Config = field(default_factory=RadialFeaturesV2Config)
    implemented_mechanism: str | None = None

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
        levels = _resolved_execution_levels(stages)
        names = tuple(stage.name for stage in stages)
        if len(set(names)) != len(names) or any(name in _RAW_SOURCES for name in names):
            raise ValueError("stage source names must be unique and nonreserved")
        known = dict(_RESERVED_STREAMS)
        cursor = 0
        while cursor < len(stages):
            level = levels[cursor]
            end = cursor + 1
            while end < len(stages) and levels[end] == level:
                end += 1
            group = stages[cursor:end]
            for stage in group:
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
                        f"stage {stage.name} references unavailable sources {unknown}; "
                        "same level stages use a frozen prelevel snapshot"
                    )
            known.update((stage.name, stage.output_stream) for stage in group)
            cursor = end
        if self.output_stage != stages[-1].name:
            raise ValueError("output_stage must name the final stage")
        output = stages[-1]
        if output.output_stream != "A" or output.channels != 1:
            raise ValueError("output stage must be a one channel A stage")
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
        if self.degree3_overflow_policy not in ("raise", "audit_skip"):
            raise ValueError("unsupported degree3_overflow_policy")
        if not isinstance(self.radial, RadialFeaturesV2Config):
            raise TypeError("radial must be RadialFeaturesV2Config")
        if self.implemented_mechanism is not None and (
            not isinstance(self.implemented_mechanism, str)
            or not self.implemented_mechanism
        ):
            raise ValueError("implemented_mechanism must be a nonempty string or None")

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
            degree3_overflow_policy=value.get("degree3_overflow_policy", "raise"),
            radial=RadialFeaturesV2Config.from_dict(value.get("radial", {})),
            implemented_mechanism=value.get("implemented_mechanism"),
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "stages": [item.as_dict() for item in self.stages],
            "output_stage": self.output_stage,
            "architecture_id": self.architecture_id,
            "anchor_ranks": list(self.anchor_ranks),
            "max_constraint_entries": self.max_constraint_entries,
            "max_gate_coefficients": self.max_gate_coefficients,
            "max_invariant_channels": self.max_invariant_channels,
            "degree3_overflow_policy": self.degree3_overflow_policy,
            "radial": self.radial.as_dict(),
        }
        if self.implemented_mechanism is not None:
            result["implemented_mechanism"] = self.implemented_mechanism
        return result


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
    unsupported_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _ActivePath:
    candidate: _Candidate
    artifact_name: str | None
    primitive_channels: int
    coefficient_channels: int


def _path_selection_key(path: _ActivePath) -> tuple[int, str]:
    role = path.candidate.role
    family = (
        0
        if ".unary." in role
        else 1
        if ".pair." in role
        else 2
        if ".symmetric2." in role
        else 3
        if ".stf." in role
        else 4
    )
    return family, role


def _candidate_family(candidate: _Candidate) -> str:
    if candidate.shortcut_rank is not None or ".stf." in candidate.role:
        return "stf"
    if ".degree3." in candidate.role:
        return "degree3"
    if ".symmetric2." in candidate.role:
        return "symmetric_unary"
    if ".pair." in candidate.role:
        return "pair"
    if ".unary." in candidate.role:
        return "unary"
    return "other"


def _select_covariant_paths(
    stage: InvariantGateStageV2Config,
    target: TypeKey,
    paths: Sequence[_ActivePath],
    required_sources: Sequence[str] = (),
    audits: Sequence[CandidateAuditV2] = (),
) -> list[_ActivePath]:
    """Apply a stable quota where cross messages replace self messages."""
    quota = stage.covariant_path_quota
    ordered = sorted(paths, key=_path_selection_key)
    uncovered = set(required_sources)
    coverage: list[_ActivePath] = []
    while uncovered:
        candidates = tuple(
            (
                len(
                    uncovered.intersection(
                        endpoint.source for endpoint in path.candidate.endpoints
                    )
                ),
                path,
            )
            for path in ordered
            if path not in coverage
        )
        gain, selected = max(
            candidates,
            key=lambda item: item[0],
            default=(0, None),
        )
        if gain == 0 or selected is None:
            raise PipelineV2CompilationError(
                f"stage {stage.name} cannot cover required sources "
                f"{tuple(sorted(uncovered))} for {_type_label(target)}",
                audits,
            )
        coverage.append(selected)
        uncovered.difference_update(
            endpoint.source for endpoint in selected.candidate.endpoints
        )
    if quota is None or len(paths) <= quota:
        return ordered
    if len(coverage) > quota:
        raise PipelineV2CompilationError(
            f"stage {stage.name} quota {quota} cannot cover required sources "
            f"{tuple(required_sources)} for {_type_label(target)}",
            audits,
        )
    remaining = [path for path in ordered if path not in coverage]
    remaining_quota = quota - len(coverage)
    if remaining_quota == 0:
        return sorted(coverage, key=_path_selection_key)
    cross = [
        path
        for path in remaining
        if any(
            not endpoint.is_raw and endpoint.key.stream != target.stream
            for endpoint in path.candidate.endpoints
        )
    ]
    self_paths = [path for path in remaining if path not in cross]
    if not cross:
        selected_paths = [*coverage, *self_paths[:remaining_quota]]
        return sorted(selected_paths, key=_path_selection_key)
    cross_quota = min(len(cross), max(1, remaining_quota // 2))
    self_quota = min(len(self_paths), remaining_quota - cross_quota)
    selected_paths = [
        *coverage,
        *self_paths[:self_quota],
        *cross[: remaining_quota - self_quota],
    ]
    if len(selected_paths) < quota:
        selected_paths.extend(
            self_paths[self_quota : self_quota + quota - len(selected_paths)]
        )
    return sorted(selected_paths, key=_path_selection_key)


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


def _stage_channel_count(stage: InvariantGateStageV2Config, target: TypeKey) -> int:
    if target.stream == "B":
        overrides = dict(stage.type_channel_overrides)
        return overrides.get(target.component, stage.channels)
    return stage.channels


def _resolved_skip_sources(
    stage: InvariantGateStageV2Config,
    target: TypeKey,
    channels: Mapping[tuple[str, TypeKey], int],
) -> tuple[str, ...]:
    skip_names = (
        stage.source_names
        if stage.skip_source_names is None
        else stage.skip_source_names
    )
    available = tuple(name for name in skip_names if (name, target) in channels)
    if stage.skip_policy in ("legacy", "dense_proj"):
        return available
    if stage.skip_policy == "id":
        output_channels = _stage_channel_count(stage, target)
        return tuple(
            name for name in available if channels[(name, target)] == output_channels
        )[-1:]
    if stage.skip_policy == "local_proj":
        return available[-1:]
    return ()


def _endpoints(
    names: Sequence[str], streams: Mapping[str, Stream], b_keys: tuple[TypeKey, ...]
) -> tuple[_Endpoint, ...]:
    return tuple(
        _Endpoint(name, key) for name in names for key in _keys(streams[name], b_keys)
    )


def _bank_flag(
    stage: InvariantGateStageV2Config,
    bank: Literal["covariant", "invariant"],
    feature: Literal["symmetric_unary", "raw_mixed_pairs", "stf_shortcuts"],
) -> bool:
    specific = getattr(stage, f"{bank}_include_{feature}")
    return getattr(stage, f"include_{feature}") if specific is None else specific


def _degree3_candidate(
    stage: InvariantGateStageV2Config,
    target: TypeKey,
    family: str,
    endpoints: tuple[_Endpoint, ...],
    slots: tuple[CSlot, ...] | None,
    *,
    unsupported_reason: str | None = None,
) -> _Candidate:
    endpoint_label = ".".join(
        f"{item.source}.{_type_label(item.key)}" for item in endpoints
    )
    return _Candidate(
        stage.name,
        "covariant",
        f"{stage.name}.{_type_label(target)}.degree3.{family}.{endpoint_label}",
        target,
        endpoints,
        None if slots is None else CSignature(target, slots),
        unsupported_reason=unsupported_reason,
    )


def _degree3_candidates(
    stage: InvariantGateStageV2Config,
    target: TypeKey,
    endpoints: tuple[_Endpoint, ...],
) -> tuple[_Candidate, ...]:
    policy = stage.degree3_policy
    if policy == "none":
        return ()
    result: list[_Candidate] = []
    if policy in ("sym3", "all"):
        for endpoint in endpoints:
            result.append(
                _degree3_candidate(
                    stage,
                    target,
                    "sym3",
                    (endpoint,),
                    (CSlot("value", endpoint.key, 3, "symmetric_power"),),
                )
            )
    x_endpoints = tuple(
        endpoint
        for endpoint in endpoints
        if endpoint.source == "x" and endpoint.key == A
    )
    b_endpoints = tuple(
        endpoint for endpoint in endpoints if endpoint.key.stream == "B"
    )
    if policy in ("a2b", "union"):
        for x_endpoint in x_endpoints:
            for b_endpoint in b_endpoints:
                result.append(
                    _degree3_candidate(
                        stage,
                        target,
                        "a2b",
                        (x_endpoint, b_endpoint),
                        (
                            CSlot("left", x_endpoint.key, 2, "symmetric_power"),
                            CSlot("right", b_endpoint.key),
                        ),
                    )
                )
    if policy in ("ab2", "union"):
        for x_endpoint in x_endpoints:
            for left, right in combinations_with_replacement(b_endpoints, 2):
                if left == right:
                    result.append(
                        _degree3_candidate(
                            stage,
                            target,
                            "ab2",
                            (x_endpoint, left),
                            (
                                CSlot("left", x_endpoint.key),
                                CSlot("right", left.key, 2, "symmetric_power"),
                            ),
                        )
                    )
                else:
                    result.append(
                        _degree3_candidate(
                            stage,
                            target,
                            "ab2_independent",
                            (x_endpoint, left, right),
                            (
                                CSlot("first", x_endpoint.key),
                                CSlot("second", left.key),
                                CSlot("third", right.key),
                            ),
                        )
                    )
    if policy == "all":
        for left, right in combinations(endpoints, 2):
            result.append(
                _degree3_candidate(
                    stage,
                    target,
                    "left2_right",
                    (left, right),
                    (
                        CSlot("left", left.key, 2, "symmetric_power"),
                        CSlot("right", right.key),
                    ),
                )
            )
            result.append(
                _degree3_candidate(
                    stage,
                    target,
                    "left_right2",
                    (left, right),
                    (
                        CSlot("left", left.key),
                        CSlot("right", right.key, 2, "symmetric_power"),
                    ),
                )
            )
        for first, second, third in combinations(endpoints, 3):
            result.append(
                _degree3_candidate(
                    stage,
                    target,
                    "independent3",
                    (first, second, third),
                    (
                        CSlot("first", first.key),
                        CSlot("second", second.key),
                        CSlot("third", third.key),
                    ),
                )
            )
    unique: dict[str, _Candidate] = {}
    for candidate in result:
        unique.setdefault(candidate.role, candidate)
    return tuple(unique.values())


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
            if _bank_flag(stage, "invariant", "symmetric_unary"):
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
        if _bank_flag(stage, "invariant", "raw_mixed_pairs"):
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
            _bank_flag(stage, "invariant", "stf_shortcuts")
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
                if stage.covariant_live_mixed_only and endpoint.is_raw:
                    continue
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
                if _bank_flag(stage, "covariant", "symmetric_unary"):
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
            if _bank_flag(stage, "covariant", "raw_mixed_pairs"):
                for left, right in combinations(direct_endpoints, 2):
                    if stage.covariant_live_mixed_only:
                        if left.is_raw and right.is_raw:
                            continue
                    elif not (left.is_raw or right.is_raw):
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
                and _bank_flag(stage, "covariant", "stf_shortcuts")
                and "x" in stage.source_names
                and "r" in stage.source_names
                and not stage.covariant_live_mixed_only
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
            result.extend(_degree3_candidates(stage, target, direct_endpoints))
        streams[stage.name] = stage.output_stream
    return tuple(result)


def _finite_group_words(
    generators: Tensor, *, maximum_order: int = 4096
) -> tuple[tuple[int, ...], ...]:
    identity = torch.eye(generators.shape[-1], dtype=generators.dtype)
    elements = [identity]
    words: list[tuple[int, ...]] = [()]
    cursor = 0
    while cursor < len(elements):
        current = elements[cursor]
        word = words[cursor]
        cursor += 1
        for index, generator in enumerate(generators):
            candidate = current @ generator
            if any(
                torch.allclose(candidate, known, atol=2.0e-10, rtol=2.0e-10)
                for known in elements
            ):
                continue
            if len(elements) >= maximum_order:
                raise ValueError("generator closure exceeds maximum_order")
            elements.append(candidate)
            words.append((*word, index))
    return tuple(words)


def _actions_from_words(
    generator_actions: Tensor, words: Sequence[tuple[int, ...]]
) -> Tensor:
    dimension = generator_actions.shape[-1]
    identity = torch.eye(dimension, dtype=generator_actions.dtype)
    result = []
    for word in words:
        action = identity
        for index in word:
            action = action @ generator_actions[index]
        result.append(action)
    return torch.stack(result)


def _group_inverse_indices(actions: Tensor) -> tuple[int, ...]:
    identity = torch.eye(actions.shape[-1], dtype=actions.dtype)
    result = []
    for action in actions:
        residuals = torch.linalg.vector_norm(
            action.unsqueeze(0) @ actions - identity,
            dim=(-2, -1),
        )
        index = int(torch.argmin(residuals))
        if float(residuals[index]) > 2.0e-8:
            raise RuntimeError("finite group closure has no numerical inverse")
        reverse_residual = torch.linalg.vector_norm(actions[index] @ action - identity)
        if float(reverse_residual) > 2.0e-8:
            raise RuntimeError("finite group inverse is not two sided")
        result.append(index)
    return tuple(result)


def _invariant_metric_from_actions(actions: Tensor) -> Tensor:
    metric = torch.einsum("gji,gjk->ik", actions, actions) / actions.shape[0]
    metric = 0.5 * (metric + metric.mT)
    metric = metric * (metric.shape[0] / torch.trace(metric))
    eigenvalues = torch.linalg.eigvalsh(metric)
    if float(eigenvalues[0]) <= 0.0:
        raise RuntimeError("compiled invariant metric is not positive definite")
    residual = max(
        float(torch.linalg.vector_norm(action.mT @ metric @ action - metric))
        for action in actions
    )
    if residual > 2.0e-8:
        raise RuntimeError(
            f"compiled invariant metric residual {residual} is too large"
        )
    return metric


@dataclass(frozen=True, slots=True)
class _ReynoldsCovariantArtifact:
    signature: CSignature
    basis: Tensor
    lifts: tuple[Tensor | None, ...]
    input_dimensions: tuple[int, ...]
    effective_input_dimension: int
    output_dimension: int
    basis_dimension: int
    residual: float


class _ReynoldsContext:
    def __init__(self, catalog: TypeCatalog) -> None:
        self.catalog = catalog
        self.words = _finite_group_words(catalog.generator_system.matrices)
        defining_actions = _actions_from_words(
            catalog.generator_system.matrices, self.words
        )
        self.inverse_indices = _group_inverse_indices(defining_actions)
        self.actions = {
            key: _actions_from_words(block.actions, self.words)
            for key, block in catalog.blocks.items()
        }
        for key, actions in self.actions.items():
            generators = catalog.resolve(key).actions
            for action in actions:
                for generator in generators:
                    product = action @ generator
                    if not any(
                        torch.allclose(product, known, atol=2.0e-9, rtol=2.0e-9)
                        for known in actions
                    ):
                        raise RuntimeError(
                            f"compiled action closure failed for {_type_label(key)}"
                        )
        self.slot_actions: dict[tuple[TypeKey, int, str], Tensor] = {}
        self.source_action_cache: dict[CSignature, Tensor] = {}

    def actions_for_slot(self, slot: CSlot) -> Tensor:
        cache_key = (slot.type_key, slot.power, slot.mode)
        cached = self.slot_actions.get(cache_key)
        if cached is not None:
            return cached
        base = self.actions[slot.type_key]
        value = (
            base
            if slot.mode == "distinct"
            else normalized_symmetric_power_representation(base, slot.power)
        )
        self.slot_actions[cache_key] = value
        return value

    def metric(self, key: TypeKey) -> Tensor:
        return _invariant_metric_from_actions(self.actions[key])

    def source_actions(self, signature: CSignature) -> Tensor:
        cached = self.source_action_cache.get(signature)
        if cached is not None:
            return cached
        slot_actions = tuple(self.actions_for_slot(slot) for slot in signature.inputs)
        source = slot_actions[0]
        for actions in slot_actions[1:]:
            source = tensor_product_representation(source, actions)
        self.source_action_cache[signature] = source
        return source

    def hom_dimension(self, signature: CSignature) -> int:
        if not isinstance(signature.output, TypeKey):
            raise TypeError("Reynolds compilation requires a typed output")
        source = self.source_actions(signature)
        output = self.actions[signature.output]
        characters = torch.einsum("gii->g", output) * torch.einsum("gii->g", source)
        return int(round(float(characters.mean())))


def _canonical_projector_basis(orthonormal_range: Tensor, dimension: int) -> Tensor:
    coefficient_basis: list[Tensor] = []
    residual_diagonal = orthonormal_range.square().sum(dim=1)
    threshold = 1.0e-10
    for _index in range(dimension):
        pivot = int(torch.argmax(residual_diagonal))
        residual = orthonormal_range[pivot].clone()
        for _pass in range(2):
            if coefficient_basis:
                known = torch.stack(coefficient_basis)
                residual = residual - (residual @ known.mT) @ known
        norm = torch.linalg.vector_norm(residual)
        if float(norm) <= threshold:
            raise RuntimeError("canonical Reynolds pivot lost numerical rank")
        normalized = residual / norm
        coefficient_basis.append(normalized)
        projected = orthonormal_range @ normalized
        residual_diagonal = (residual_diagonal - projected.square()).clamp_min(0.0)
        residual_diagonal[pivot] = 0.0
    if len(coefficient_basis) != dimension:
        raise RuntimeError("canonical Reynolds basis did not reach the Hom dimension")
    coefficients = torch.stack(coefficient_basis)
    basis = coefficients @ orthonormal_range.mT
    return torch.round(basis * 1.0e12) / 1.0e12


def _compile_reynolds_covariant(
    context: _ReynoldsContext,
    signature: CSignature,
) -> _ReynoldsCovariantArtifact:
    source_actions = context.source_actions(signature)
    if not isinstance(signature.output, TypeKey):
        raise TypeError("Reynolds degree three compilation requires a typed output")
    output_actions = context.actions[signature.output]
    expected_dimension = context.hom_dimension(signature)
    output_dimension = output_actions.shape[-1]
    input_dimension = source_actions.shape[-1]
    variable_dimension = output_dimension * input_dimension
    if expected_dimension < 0 or expected_dimension > variable_dimension:
        raise RuntimeError("invalid Reynolds Hom dimension")
    if expected_dimension == 0:
        basis = torch.empty((0, output_dimension, input_dimension), dtype=torch.float64)
        residual = 0.0
    else:
        oversample = min(8, variable_dimension - expected_dimension)
        sample_count = expected_dimension + oversample
        signature_digest = hashlib.sha256(
            (
                context.catalog.fingerprint + ":" + _signature_label(signature, None)
            ).encode("utf8")
        ).digest()
        seed = int.from_bytes(signature_digest[:8], "little") % (2**63 - 1)
        random = torch.Generator(device="cpu")
        random.manual_seed(seed)
        samples = torch.randn(
            sample_count,
            output_dimension,
            input_dimension,
            dtype=torch.float64,
            generator=random,
        )
        projected = torch.zeros_like(samples)
        inverse_source = source_actions[
            torch.tensor(context.inverse_indices, dtype=torch.int64)
        ]
        for output_action, source_inverse in zip(output_actions, inverse_source):
            left = torch.einsum("oi,kij->koj", output_action, samples)
            projected.add_(torch.einsum("koj,ji->koi", left, source_inverse))
        projected.div_(len(context.words))
        matrix = projected.reshape(sample_count, variable_dimension).mT
        left, singular, _right = torch.linalg.svd(matrix, full_matrices=False)
        threshold = (
            torch.finfo(matrix.dtype).eps
            * max(matrix.shape)
            * float(singular[0])
            * 100.0
        )
        numerical_rank = int((singular > threshold).sum())
        if numerical_rank != expected_dimension:
            raise RuntimeError(
                f"Reynolds range rank {numerical_rank} does not match character dimension {expected_dimension}"
            )
        vectors = _canonical_projector_basis(
            left[:, :expected_dimension], expected_dimension
        )
        basis = vectors.reshape(expected_dimension, output_dimension, input_dimension)
        residual = 0.0
        for output_action, source_action in zip(output_actions, source_actions):
            value = torch.einsum("oi,bij->boj", output_action, basis) - torch.einsum(
                "boi,ij->boj", basis, source_action
            )
            residual = max(residual, float(torch.linalg.vector_norm(value)))
        if residual > 2.0e-8:
            raise RuntimeError(f"Reynolds covariant residual {residual} is too large")
    lifts = tuple(
        None
        if slot.mode == "distinct"
        else normalized_symmetric_power_basis(
            context.catalog.resolve(slot.type_key).representation_dim,
            slot.power,
        )
        for slot in signature.inputs
    )
    input_dimensions = tuple(
        context.catalog.resolve(slot.type_key).representation_dim
        for slot in signature.inputs
    )
    return _ReynoldsCovariantArtifact(
        signature,
        basis,
        lifts,
        input_dimensions,
        input_dimension,
        output_dimension,
        expected_dimension,
        residual,
    )


class _RegisteredReynoldsCovariant(nn.Module):
    def __init__(self, artifact: _ReynoldsCovariantArtifact) -> None:
        super().__init__()
        self.signature = artifact.signature
        self.input_dimensions = artifact.input_dimensions
        self.register_buffer("basis", artifact.basis.detach().clone())
        self._lift_names: list[str | None] = []
        for index, lift in enumerate(artifact.lifts):
            if lift is None:
                self._lift_names.append(None)
                continue
            name = f"lift_{index}"
            self._lift_names.append(name)
            self.register_buffer(name, lift.detach().clone())

    def evaluate_basis(self, inputs: Mapping[str, Tensor]) -> Tensor:
        lifted = []
        for slot, dimension, lift_name in zip(
            self.signature.inputs, self.input_dimensions, self._lift_names
        ):
            value = inputs[slot.name]
            if lift_name is None:
                lifted.append(value)
                continue
            tensor_power = value
            for degree in range(2, slot.power + 1):
                tensor_power = torch.einsum(
                    "...i,...j->...ij", tensor_power, value
                ).reshape(value.shape[:-1] + (dimension**degree,))
            lifted.append(
                torch.einsum("...i,ij->...j", tensor_power, getattr(self, lift_name))
            )
        source = lifted[0]
        for value in lifted[1:]:
            leading = torch.broadcast_shapes(source.shape[:-1], value.shape[:-1])
            source = torch.einsum(
                "...i,...j->...ij",
                source.expand(leading + (source.shape[-1],)),
                value.expand(leading + (value.shape[-1],)),
            ).reshape(leading + (source.shape[-1] * value.shape[-1],))
        return torch.einsum("boi,...i->...bo", self.basis, source)


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
        _stage_channel_count(stage, candidate.target)
        * input_channels
        * output_dimension
        * effective_input
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


class _FactorizedChannelProjection(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        rank: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = min(rank, input_channels, output_channels)
        self.left = nn.Parameter(torch.empty(output_channels, self.rank, dtype=dtype))
        self.right = nn.Parameter(torch.empty(self.rank, input_channels, dtype=dtype))
        nn.init.normal_(self.left, std=1.0 / math.sqrt(max(1, self.rank)))
        nn.init.normal_(self.right, std=1.0 / math.sqrt(input_channels))

    def forward(self, value: Tensor) -> Tensor:
        hidden = torch.einsum("ri,...id->...rd", self.right, value)
        return torch.einsum("or,...rd->...od", self.left, hidden)


class _TuckerChannelProjection(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        rank: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = min(rank, input_channels, output_channels)
        self.left = nn.Parameter(torch.empty(output_channels, self.rank, dtype=dtype))
        self.core = nn.Parameter(torch.eye(self.rank, dtype=dtype))
        self.right = nn.Parameter(torch.empty(self.rank, input_channels, dtype=dtype))
        nn.init.normal_(self.left, std=1.0 / math.sqrt(max(1, self.rank)))
        nn.init.normal_(self.right, std=1.0 / math.sqrt(input_channels))

    def forward(self, value: Tensor) -> Tensor:
        hidden = torch.einsum("ri,...id->...rd", self.right, value)
        hidden = torch.einsum("qr,...rd->...qd", self.core, hidden)
        return torch.einsum("oq,...qd->...od", self.left, hidden)


class _TensorTrainChannelProjection(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        rank: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = min(rank, input_channels, output_channels)
        self.left = nn.Parameter(torch.empty(output_channels, self.rank, dtype=dtype))
        self.scale = nn.Parameter(torch.ones(self.rank, dtype=dtype))
        self.right = nn.Parameter(torch.empty(self.rank, input_channels, dtype=dtype))
        nn.init.normal_(self.left, std=1.0 / math.sqrt(max(1, self.rank)))
        nn.init.normal_(self.right, std=1.0 / math.sqrt(input_channels))

    def forward(self, value: Tensor) -> Tensor:
        hidden = torch.einsum("ri,...id->...rd", self.right, value)
        hidden = hidden * self.scale.reshape((-1,) + (1,))
        return torch.einsum("or,...rd->...od", self.left, hidden)


class _ToeplitzChannelProjection(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, *, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.diagonals = nn.Parameter(
            torch.empty(input_channels + output_channels - 1, dtype=dtype)
        )
        nn.init.normal_(self.diagonals, std=1.0 / math.sqrt(input_channels))
        row = torch.arange(output_channels).unsqueeze(1)
        column = torch.arange(input_channels).unsqueeze(0)
        self.register_buffer(
            "indices", column - row + output_channels - 1, persistent=False
        )

    def forward(self, value: Tensor) -> Tensor:
        weight = self.diagonals[self.indices]
        return torch.einsum("oi,...id->...od", weight, value)


class _CayleyChannelProjection(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, *, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.base = nn.Parameter(
            torch.empty(output_channels, input_channels, dtype=dtype)
        )
        nn.init.normal_(self.base, std=1.0 / math.sqrt(input_channels))
        indices = torch.triu_indices(output_channels, output_channels, offset=1)
        self.register_buffer("upper_indices", indices, persistent=False)
        self.skew_values = nn.Parameter(torch.zeros(indices.shape[1], dtype=dtype))

    def forward(self, value: Tensor) -> Tensor:
        projected = torch.einsum("oi,...id->...od", self.base, value)
        skew = self.base.new_zeros((self.output_channels, self.output_channels))
        row, column = self.upper_indices.unbind(dim=0)
        skew[row, column] = self.skew_values
        skew[column, row] = -self.skew_values
        identity = torch.eye(
            self.output_channels, dtype=self.base.dtype, device=self.base.device
        )
        rotation = torch.linalg.solve(identity + skew, identity - skew)
        return torch.einsum("oi,...id->...od", rotation, projected)


def _build_channel_projection(
    stage: InvariantGateStageV2Config,
    input_channels: int,
    output_channels: int,
    *,
    dtype: torch.dtype,
) -> nn.Module:
    kind = stage.channel_projection
    rank = stage.channel_projection_rank
    if kind == "dense":
        return _ChannelProjection(input_channels, output_channels, dtype=dtype)
    if kind == "toeplitz":
        return _ToeplitzChannelProjection(input_channels, output_channels, dtype=dtype)
    if kind == "cayley":
        return _CayleyChannelProjection(input_channels, output_channels, dtype=dtype)
    assert rank is not None
    if kind == "factorized":
        return _FactorizedChannelProjection(
            input_channels, output_channels, rank, dtype=dtype
        )
    if kind == "tucker":
        return _TuckerChannelProjection(
            input_channels, output_channels, rank, dtype=dtype
        )
    if kind == "tensor_train":
        return _TensorTrainChannelProjection(
            input_channels, output_channels, rank, dtype=dtype
        )
    raise RuntimeError(f"unsupported channel projection {kind}")


class _ReversibleChannelCoupling(nn.Module):
    """Apply two block triangular updates on a multiplicity axis."""

    def __init__(
        self, channels: int, context_width: int, *, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.left_channels = channels // 2
        self.right_channels = channels - self.left_channels
        self.left_head = nn.Linear(
            context_width,
            self.left_channels * self.right_channels,
            dtype=dtype,
        )
        self.right_head = nn.Linear(
            context_width,
            self.right_channels * self.left_channels,
            dtype=dtype,
        )

    def forward(self, value: Tensor, context: Tensor) -> Tensor:
        left, right = value.split((self.left_channels, self.right_channels), dim=-2)
        left_matrix = self.left_head(context).reshape(
            context.shape[:-1] + (self.left_channels, self.right_channels)
        )
        left = left + torch.einsum("...ij,...jd->...id", left_matrix, right)
        right_matrix = self.right_head(context).reshape(
            context.shape[:-1] + (self.right_channels, self.left_channels)
        )
        right = right + torch.einsum("...ij,...jd->...id", right_matrix, left)
        return torch.cat((left, right), dim=-2)


class _FactorizedCoefficientHead(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        rank: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.requested_rank = rank
        self.rank = min(rank, input_features, output_features)
        self.left = nn.Parameter(torch.empty(output_features, self.rank, dtype=dtype))
        self.right = nn.Parameter(torch.empty(self.rank, input_features, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(output_features, dtype=dtype))
        nn.init.normal_(self.left, std=1.0 / math.sqrt(max(1, self.rank)))
        nn.init.normal_(self.right, std=1.0 / math.sqrt(input_features))

    def forward(self, value: Tensor) -> Tensor:
        hidden = torch.einsum("ri,...i->...r", self.right, value)
        return torch.einsum("or,...r->...o", self.left, hidden) + self.bias

    def initialize_from_dense(self, weight: Tensor, bias: Tensor | None) -> None:
        if weight.shape != (self.output_features, self.input_features):
            raise ValueError("dense head weight shape does not match")
        with torch.no_grad():
            left, singular, right = torch.linalg.svd(weight, full_matrices=False)
            root = singular[: self.rank].clamp_min(0.0).sqrt()
            self.left.copy_(left[:, : self.rank] * root)
            self.right.copy_(root.unsqueeze(-1) * right[: self.rank])
            self.bias.copy_(torch.zeros_like(self.bias) if bias is None else bias)


class _OrthogonalCoefficientHead(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        rank: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.requested_rank = rank
        self.rank = min(rank, input_features, output_features)
        self.left_raw = nn.Parameter(
            torch.randn(output_features, self.rank, dtype=dtype)
        )
        self.right_raw = nn.Parameter(
            torch.randn(input_features, self.rank, dtype=dtype)
        )
        self.scale = nn.Parameter(torch.ones(self.rank, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(output_features, dtype=dtype))

    def _factors(self) -> tuple[Tensor, Tensor]:
        left = torch.linalg.qr(self.left_raw, mode="reduced").Q
        right = torch.linalg.qr(self.right_raw, mode="reduced").Q
        return left, right

    def forward(self, value: Tensor) -> Tensor:
        left, right = self._factors()
        hidden = torch.einsum("ir,...i->...r", right, value) * self.scale
        return torch.einsum("or,...r->...o", left, hidden) + self.bias

    def initialize_from_dense(self, weight: Tensor, bias: Tensor | None) -> None:
        if weight.shape != (self.output_features, self.input_features):
            raise ValueError("dense head weight shape does not match")
        with torch.no_grad():
            left, singular, right = torch.linalg.svd(weight, full_matrices=False)
            self.left_raw.copy_(left[:, : self.rank])
            self.right_raw.copy_(right[: self.rank].mT)
            self.scale.copy_(singular[: self.rank])
            self.bias.copy_(torch.zeros_like(self.bias) if bias is None else bias)


class _StaticCoefficientHead(nn.Module):
    def __init__(self, output_features: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_features, dtype=dtype))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(max(1, output_features)))

    def forward(self, value: Tensor) -> Tensor:
        return self.weight.expand(value.shape[:-1] + self.weight.shape)


class _ContextLoRACoefficientHead(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        rank: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = min(rank, input_features, output_features)
        self.base = nn.Linear(input_features, output_features, dtype=dtype)
        self.output_factor = nn.Parameter(
            torch.empty(output_features, self.rank, dtype=dtype)
        )
        self.context_factor = nn.Parameter(
            torch.empty(self.rank, input_features, dtype=dtype)
        )
        self.value_factor = nn.Parameter(
            torch.empty(self.rank, input_features, dtype=dtype)
        )
        nn.init.normal_(self.output_factor, std=1.0 / math.sqrt(max(1, self.rank)))
        nn.init.normal_(self.context_factor, std=1.0 / math.sqrt(input_features))
        nn.init.normal_(self.value_factor, std=1.0 / math.sqrt(input_features))

    def forward(self, value: Tensor) -> Tensor:
        context = torch.einsum("ri,...i->...r", self.context_factor, value)
        routed = torch.einsum("ri,...i->...r", self.value_factor, value)
        update = torch.einsum(
            "or,...r->...o", self.output_factor, torch.tanh(context) * routed
        )
        return self.base(value) + update


class _AxisCPCoefficientHead(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        rank: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = min(rank, input_features, output_features)
        self.output_factor = nn.Parameter(
            torch.empty(output_features, self.rank, dtype=dtype)
        )
        self.context_factor = nn.Parameter(
            torch.empty(self.rank, input_features, dtype=dtype)
        )
        self.path_factor = nn.Parameter(
            torch.empty(self.rank, input_features, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.zeros(output_features, dtype=dtype))
        nn.init.normal_(self.output_factor, std=1.0 / math.sqrt(max(1, self.rank)))
        nn.init.normal_(self.context_factor, std=1.0 / math.sqrt(input_features))
        nn.init.normal_(self.path_factor, std=1.0 / math.sqrt(input_features))

    def forward(self, value: Tensor) -> Tensor:
        context = torch.einsum("ri,...i->...r", self.context_factor, value)
        path = torch.einsum("ri,...i->...r", self.path_factor, value)
        return (
            torch.einsum("or,...r->...o", self.output_factor, context * path)
            + self.bias
        )


class _ResidualTrunk(nn.Module):
    def __init__(
        self,
        input_features: int,
        width: int,
        activation: str,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_features, width, dtype=dtype)
        self.middle_layer = nn.Linear(width, width, dtype=dtype)
        self.output_layer = nn.Linear(width, width, dtype=dtype)
        self.activation = _activation(activation)

    def forward(self, value: Tensor) -> Tensor:
        first = self.activation(self.input_layer(value))
        update = self.output_layer(self.activation(self.middle_layer(first)))
        return first + update


class _DescriptorTransform(nn.Module):
    def __init__(self, mask: Tensor, *, dtype: torch.dtype) -> None:
        super().__init__()
        dimension = int(mask.numel())
        self.dimension = dimension
        self.register_buffer("mask", mask.to(dtype=dtype), persistent=False)
        self.register_buffer(
            "components", torch.empty((0, dimension), dtype=dtype), persistent=False
        )
        self.register_buffer(
            "mean", torch.zeros(dimension, dtype=dtype), persistent=False
        )

    def set_projection(self, components: Tensor, mean: Tensor | None = None) -> None:
        if (
            not isinstance(components, Tensor)
            or components.ndim != 2
            or components.shape[1] != self.dimension
            or components.shape[0] <= 0
        ):
            raise ValueError("projection components must have shape rank by dimension")
        if not torch.is_floating_point(components) or not bool(
            torch.isfinite(components).all()
        ):
            raise ValueError("projection components must be finite floating point")
        resolved_mean = (
            torch.zeros(
                self.dimension, dtype=components.dtype, device=components.device
            )
            if mean is None
            else mean
        )
        if (
            not isinstance(resolved_mean, Tensor)
            or resolved_mean.shape != (self.dimension,)
            or not torch.is_floating_point(resolved_mean)
            or not bool(torch.isfinite(resolved_mean).all())
        ):
            raise ValueError("projection mean must be a finite vector")
        self.components = (
            components.detach()
            .to(device=self.mask.device, dtype=self.mask.dtype)
            .clone()
        )
        self.mean = (
            resolved_mean.detach()
            .to(device=self.mask.device, dtype=self.mask.dtype)
            .clone()
        )

    def clear(self) -> None:
        self.components = self.mask.new_empty((0, self.dimension))
        self.mean = self.mask.new_zeros((self.dimension,))

    def forward(self, value: Tensor) -> Tensor:
        masked = value * self.mask
        if self.components.shape[0] == 0:
            return masked
        centered = masked - self.mean
        reduced = torch.einsum("...i,ri->...r", centered, self.components)
        return torch.einsum("...r,ri->...i", reduced, self.components)


def _build_trunk(
    stage: InvariantGateStageV2Config,
    input_features: int,
    *,
    dtype: torch.dtype,
) -> nn.Module:
    if stage.metric_gate == "norm" or stage.coefficient_head == "static_mixing":
        return nn.Identity()
    if stage.trunk_residual:
        return _ResidualTrunk(
            input_features,
            stage.trunk_width,
            stage.activation,
            dtype=dtype,
        )
    modules: list[nn.Module] = []
    current = input_features
    for _index in range(stage.trunk_depth):
        modules.append(nn.Linear(current, stage.trunk_width, dtype=dtype))
        if not stage.trunk_linearized:
            modules.append(_activation(stage.activation))
        current = stage.trunk_width
    return nn.Sequential(*modules)


def _build_coefficient_head(
    stage: InvariantGateStageV2Config,
    output_features: int,
    *,
    dtype: torch.dtype,
) -> nn.Module:
    if stage.metric_gate == "norm" or stage.coefficient_head == "static_mixing":
        return _StaticCoefficientHead(output_features, dtype=dtype)
    if stage.coefficient_head == "dense":
        return nn.Linear(stage.trunk_width, output_features, dtype=dtype)
    assert stage.coefficient_rank is not None
    if stage.coefficient_head == "factorized":
        return _FactorizedCoefficientHead(
            stage.trunk_width,
            output_features,
            stage.coefficient_rank,
            dtype=dtype,
        )
    if stage.coefficient_head == "orthogonal":
        return _OrthogonalCoefficientHead(
            stage.trunk_width,
            output_features,
            stage.coefficient_rank,
            dtype=dtype,
        )
    if stage.coefficient_head == "context_lora":
        return _ContextLoRACoefficientHead(
            stage.trunk_width,
            output_features,
            stage.coefficient_rank,
            dtype=dtype,
        )
    return _AxisCPCoefficientHead(
        stage.trunk_width,
        output_features,
        stage.coefficient_rank,
        dtype=dtype,
    )


def _coefficient_activation(value: Tensor, name: CoefficientActivation) -> Tensor:
    if name == "identity":
        return value
    if name == "sigmoid":
        return torch.sigmoid(value)
    if name == "tanh":
        return torch.tanh(value)
    return torch.nn.functional.silu(value)


class _RunningRMS(nn.Module):
    """Normalize one schema with a cumulative training distribution RMS."""

    def __init__(self, epsilon: float, *, dtype: torch.dtype) -> None:
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
                raise ValueError("cannot normalize an empty schema")
            with torch.no_grad():
                total_count = self.sample_count + value_count
                total = total_count.to(dtype=self.mean_square.dtype)
                old_weight = self.sample_count.to(dtype=self.mean_square.dtype) / total
                new_weight = value_count / total
                batch_mean_square = detached.square().mean()
                self.mean_square.mul_(old_weight).add_(batch_mean_square * new_weight)
                self.sample_count.copy_(total_count)
        ready = self.sample_count > 0
        scale = torch.where(
            ready,
            torch.sqrt(self.mean_square.clamp_min(0.0) + self.epsilon),
            torch.ones_like(self.mean_square),
        )
        return value / scale


def _descriptor_mask_tensor(
    stage: InvariantGateStageV2Config,
    radial_count: int,
    paths: Sequence[_ActivePath],
    *,
    dtype: torch.dtype,
) -> Tensor:
    selected = [torch.ones(radial_count, dtype=dtype)]
    for path in paths:
        sources = frozenset(endpoint.source for endpoint in path.candidate.endpoints)
        keep = (
            stage.descriptor_mask == "full"
            or stage.descriptor_mask == "raw_only"
            and sources.issubset(_RAW_SOURCES)
            or stage.descriptor_mask == "unary"
            and len(sources) == 1
            or stage.descriptor_mask == "mixed"
            and len(sources) >= 2
        )
        selected.append(
            torch.full((path.primitive_channels,), float(keep), dtype=dtype)
        )
    return torch.cat(selected)


class InvariantGatePipelineV2(nn.Module):
    """Evaluate a versioned typed pipeline without discarding prior sources."""

    def __init__(
        self,
        config: InvariantGatePipelineV2Config,
        pose_encoder: PoseEncoder,
        anchor_compilation: AnchorCompilation,
        manifest: Sequence[BBlockManifest],
        artifacts: Mapping[
            CSignature, CovariantCompilation | _ReynoldsCovariantArtifact
        ],
        invariant_metrics: Mapping[TypeKey, Tensor],
        candidates: Sequence[_Candidate],
        audits: Sequence[CandidateAuditV2],
    ) -> None:
        super().__init__()
        self.config = config
        self.strict_flow_manifest: Mapping[str, Any] | None = None
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
            self.covariants[name] = (
                RegisteredCovariant(artifact)
                if isinstance(artifact, CovariantCompilation)
                else _RegisteredReynoldsCovariant(artifact)
            )
        self._metric_names: dict[TypeKey, str] = {}
        for key, metric in invariant_metrics.items():
            name = f"_invariant_metric_{len(self._metric_names):04d}"
            self._metric_names[key] = name
            self.register_buffer(name, metric.detach().clone())

        streams: dict[str, Stream] = dict(_RESERVED_STREAMS)
        channels: dict[tuple[str, TypeKey], int] = {("x", A): 1}
        for item in self._manifest:
            channels[("r", item.key)] = len(item.anchor_columns)
        for stage in config.stages:
            for key in _keys(stage.output_stream, self._b_keys):
                channels[(stage.name, key)] = _stage_channel_count(stage, key)
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
            target_channels = (
                0
                if candidate.target is None
                else _stage_channel_count(
                    stage_lookup[candidate.stage], candidate.target
                )
            )
            if candidate.shortcut_rank is not None:
                endpoint = candidate.endpoints[1]
                primitive_channels = channels[(endpoint.source, endpoint.key)]
                artifact_name = None
                coefficient_channels = target_channels * primitive_channels
            else:
                assert candidate.signature is not None
                primitive_channels = artifacts[
                    candidate.signature
                ].basis_dimension * math.prod(
                    channels[(item.source, item.key)] for item in candidate.endpoints
                )
                artifact_name = artifact_names[candidate.signature]
                coefficient_channels = target_channels * primitive_channels
            active = _ActivePath(
                candidate, artifact_name, primitive_channels, coefficient_channels
            )
            if candidate.bank == "scalar":
                stage_scalars[candidate.stage].append(active)
            else:
                assert candidate.target is not None
                stage_paths[candidate.stage][candidate.target].append(active)

        for stage in config.stages:
            for target, paths in stage_paths[stage.name].items():
                represented_by_skip = set(
                    _resolved_skip_sources(stage, target, channels)
                )
                required_sources = tuple(
                    source
                    for source in (
                        ()
                        if stage.covariant_required_source_names is None
                        else stage.covariant_required_source_names
                    )
                    if source not in represented_by_skip
                )
                stage_paths[stage.name][target] = _select_covariant_paths(
                    stage,
                    target,
                    paths,
                    required_sources,
                    audits,
                )

        self.stage_trunks = nn.ModuleDict()
        self.path_heads = nn.ModuleDict()
        self.channel_projections = nn.ModuleDict()
        self.skip_projections = nn.ModuleDict()
        self.skip_gates = nn.ModuleDict()
        self.reversible_couplings = nn.ModuleDict()
        self.descriptor_transforms = nn.ModuleDict()
        self.normalization = nn.ModuleDict()
        self._head_names: dict[str, str] = {}
        self._projection_names: dict[tuple[str, TypeKey], str] = {}
        self._skip_projection_names: dict[tuple[str, TypeKey], str] = {}
        self._skip_gate_names: dict[tuple[str, TypeKey], str] = {}
        self._reversible_names: dict[tuple[str, TypeKey], str] = {}
        self._skip_sources: dict[tuple[str, TypeKey], tuple[str, ...]] = {}
        shared_modules: dict[tuple[Any, ...], nn.Module] = {}

        def resolve_shared(
            stage: InvariantGateStageV2Config,
            key: tuple[Any, ...],
            factory: Any,
        ) -> nn.Module:
            group = stage.parameter_share_group
            if group is None:
                return factory()
            shared_key = (group, *key)
            module = shared_modules.get(shared_key)
            if module is None:
                module = factory()
                shared_modules[shared_key] = module
            return module

        radial_count = (
            2 + len(config.radial.rbf_centers) + len(config.radial.inverse_powers)
        )
        invariant_counts = {
            stage.name: radial_count
            + sum(path.primitive_channels for path in stage_scalars[stage.name])
            for stage in config.stages
        }
        for stage_name, invariant_count in invariant_counts.items():
            if invariant_count > config.max_invariant_channels:
                raise PipelineV2CompilationError(
                    f"stage {stage_name} invariant count {invariant_count} exceeds limit {config.max_invariant_channels}",
                    audits,
                )
        shared_context_widths: dict[str, int] = {}
        for stage in config.stages:
            if stage.parameter_share_group is not None:
                shared_context_widths[stage.parameter_share_group] = max(
                    shared_context_widths.get(stage.parameter_share_group, 0),
                    invariant_counts[stage.name],
                )
        self._trunk_input_counts = {
            stage.name: invariant_counts[stage.name]
            if stage.parameter_share_group is None
            else shared_context_widths[stage.parameter_share_group]
            for stage in config.stages
        }
        for stage in config.stages:
            invariant_count = invariant_counts[stage.name]
            trunk_input_count = self._trunk_input_counts[stage.name]
            self.normalization[stage.name] = nn.ModuleList(
                _RunningRMS(config.radial.rms_epsilon, dtype=torch.float64)
                for _index in range(1 + len(stage_scalars[stage.name]))
            )
            self.descriptor_transforms[stage.name] = _DescriptorTransform(
                _descriptor_mask_tensor(
                    stage,
                    radial_count,
                    stage_scalars[stage.name],
                    dtype=torch.float64,
                ),
                dtype=torch.float64,
            )
            self.stage_trunks[stage.name] = resolve_shared(
                stage,
                (
                    "trunk",
                    trunk_input_count,
                    stage.trunk_width,
                    stage.activation,
                    stage.trunk_depth,
                    stage.trunk_linearized,
                    stage.trunk_residual,
                    stage.metric_gate,
                    stage.coefficient_head,
                ),
                lambda: _build_trunk(
                    stage,
                    trunk_input_count,
                    dtype=torch.float64,
                ),
            )
            for target, paths in stage_paths[stage.name].items():
                output_channels = _stage_channel_count(stage, target)
                if stage.reversible_coupling and output_channels < 2:
                    raise ValueError(
                        f"stage {stage.name} reversible target {_type_label(target)} has fewer than two channels"
                    )
                if not paths:
                    raise PipelineV2CompilationError(
                        f"stage {stage.name} has no path for {_type_label(target)}",
                        audits,
                    )
                for path_index, path in enumerate(paths):
                    head_name = f"head_{len(self._head_names):04d}"
                    self._head_names[path.candidate.role] = head_name
                    self.path_heads[head_name] = resolve_shared(
                        stage,
                        (
                            "head",
                            _type_label(target),
                            path_index,
                            path.coefficient_channels,
                            stage.trunk_width,
                            stage.coefficient_head,
                            stage.coefficient_rank,
                        ),
                        lambda path=path: _build_coefficient_head(
                            stage,
                            path.coefficient_channels,
                            dtype=torch.float64,
                        ),
                    )
                skip_names = (
                    stage.source_names
                    if stage.skip_source_names is None
                    else stage.skip_source_names
                )
                available_skip = tuple(
                    name for name in skip_names if (name, target) in channels
                )
                path_channels = output_channels * len(paths)
                if stage.skip_policy == "legacy":
                    skip_channels = sum(
                        channels[(name, target)] for name in available_skip
                    )
                    concat_channels = skip_channels + path_channels
                    self._skip_sources[(stage.name, target)] = available_skip
                else:
                    concat_channels = path_channels
                    selected_skip: tuple[str, ...] = ()
                    if stage.skip_policy == "id":
                        selected_skip = tuple(
                            name
                            for name in available_skip
                            if channels[(name, target)] == output_channels
                        )[-1:]
                        if not selected_skip:
                            raise ValueError(
                                f"stage {stage.name} has no matching identity skip for {_type_label(target)}"
                            )
                    elif stage.skip_policy == "local_proj":
                        selected_skip = available_skip[-1:]
                        if not selected_skip:
                            raise ValueError(
                                f"stage {stage.name} has no local skip for {_type_label(target)}"
                            )
                    elif stage.skip_policy == "dense_proj":
                        selected_skip = available_skip
                        if not selected_skip:
                            raise ValueError(
                                f"stage {stage.name} has no dense skip for {_type_label(target)}"
                            )
                    self._skip_sources[(stage.name, target)] = selected_skip
                    if stage.skip_policy in ("local_proj", "dense_proj"):
                        skip_input_channels = sum(
                            channels[(name, target)] for name in selected_skip
                        )
                        skip_projection_name = (
                            f"skip_projection_{len(self._skip_projection_names):04d}"
                        )
                        self._skip_projection_names[(stage.name, target)] = (
                            skip_projection_name
                        )
                        self.skip_projections[skip_projection_name] = resolve_shared(
                            stage,
                            (
                                "skip_projection",
                                _type_label(target),
                                skip_input_channels,
                                output_channels,
                                stage.channel_projection,
                                stage.channel_projection_rank,
                            ),
                            lambda: _build_channel_projection(
                                stage,
                                skip_input_channels,
                                output_channels,
                                dtype=torch.float64,
                            ),
                        )
                    if stage.metric_gate == "skip_identity":
                        gate_name = f"skip_gate_{len(self._skip_gate_names):04d}"
                        self._skip_gate_names[(stage.name, target)] = gate_name
                        self.skip_gates[gate_name] = nn.Linear(
                            stage.trunk_width,
                            output_channels,
                            dtype=torch.float64,
                        )
                projection_name = f"projection_{len(self._projection_names):04d}"
                self._projection_names[(stage.name, target)] = projection_name
                self.channel_projections[projection_name] = resolve_shared(
                    stage,
                    (
                        "projection",
                        _type_label(target),
                        concat_channels,
                        output_channels,
                        stage.channel_projection,
                        stage.channel_projection_rank,
                    ),
                    lambda: _build_channel_projection(
                        stage,
                        concat_channels,
                        output_channels,
                        dtype=torch.float64,
                    ),
                )
                if stage.reversible_coupling:
                    reversible_name = f"reversible_{len(self._reversible_names):04d}"
                    self._reversible_names[(stage.name, target)] = reversible_name
                    self.reversible_couplings[reversible_name] = resolve_shared(
                        stage,
                        (
                            "reversible",
                            _type_label(target),
                            output_channels,
                            stage.trunk_width,
                        ),
                        lambda: _ReversibleChannelCoupling(
                            output_channels,
                            stage.trunk_width,
                            dtype=torch.float64,
                        ),
                    )
                channels[(stage.name, target)] = output_channels
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
        all_parameterized_modules = (
            *self.stage_trunks.values(),
            *self.path_heads.values(),
            *self.channel_projections.values(),
            *self.skip_projections.values(),
            *self.reversible_couplings.values(),
        )
        self._shared_module_reference_count = len(all_parameterized_modules) - len(
            {id(module) for module in all_parameterized_modules}
        )
        resolved_levels = _resolved_execution_levels(config.stages)
        self._execution_levels = {
            stage.name: level for stage, level in zip(config.stages, resolved_levels)
        }
        self._execution_group_count = len(set(resolved_levels))
        output_config = config.stages[-1]
        self._readout_sources = tuple(output_config.source_names)
        self._state_metadata = {
            "schema_version": 2,
            "normalization_schema_version": 1,
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
    def descriptor_role_manifest(self) -> tuple[dict[str, Any], ...]:
        """Map normalized Gate descriptor columns to stable scalar roles."""
        radial_count = (
            2
            + len(self.config.radial.rbf_centers)
            + len(self.config.radial.inverse_powers)
        )
        result: list[dict[str, Any]] = []
        for stage in self.config.stages:
            transform = self.descriptor_transforms[stage.name]
            assert isinstance(transform, _DescriptorTransform)
            start = 0

            def append_role(
                role: str,
                kind: str,
                source_names: tuple[str, ...],
                stop: int,
                basis_role: str,
            ) -> None:
                nonlocal start
                active_mask = transform.mask[start:stop]
                result.append(
                    {
                        "stage": stage.name,
                        "role": role,
                        "kind": kind,
                        "source_names": source_names,
                        "start": start,
                        "stop": stop,
                        "active": bool(active_mask.bool().all().item()),
                        "active_column_count": int(
                            active_mask.bool().sum().item()
                        ),
                        "basis_role": basis_role,
                    }
                )
                start = stop

            append_role(
                f"{stage.name}.scalar.radial",
                "radial",
                ("x",),
                radial_count,
                "radial",
            )
            for path in self._stage_scalars[stage.name]:
                candidate = path.candidate
                append_role(
                    candidate.role,
                    _candidate_family(candidate),
                    tuple(
                        dict.fromkeys(
                            endpoint.source for endpoint in candidate.endpoints
                        )
                    ),
                    start + path.primitive_channels,
                    _signature_label(
                        candidate.signature,
                        candidate.shortcut_rank,
                    ),
                )
            if start != self._invariant_counts[stage.name]:
                raise RuntimeError(
                    f"descriptor role manifest for {stage.name} has "
                    f"{start} columns, expected {self._invariant_counts[stage.name]}"
                )
        return tuple(result)

    @property
    def coefficient_head_role_manifest(self) -> tuple[dict[str, Any], ...]:
        """Describe every dense or structured coefficient head by path role."""
        result: list[dict[str, Any]] = []
        for stage in self.config.stages:
            for target, paths in self._stage_paths[stage.name].items():
                target_channels = self._channels[(stage.name, target)]
                for path in paths:
                    candidate = path.candidate
                    module_name = self._head_names[candidate.role]
                    result.append(
                        {
                            "role": candidate.role,
                            "stage": stage.name,
                            "target": _type_label(target),
                            "source_names": tuple(
                                dict.fromkeys(
                                    endpoint.source
                                    for endpoint in candidate.endpoints
                                )
                            ),
                            "source_types": tuple(
                                _type_label(endpoint.key)
                                for endpoint in candidate.endpoints
                            ),
                            "path_family": _candidate_family(candidate),
                            "basis_role": _signature_label(
                                candidate.signature,
                                candidate.shortcut_rank,
                            ),
                            "primitive_channels": path.primitive_channels,
                            "target_channels": target_channels,
                            "coefficient_channels": path.coefficient_channels,
                            "module_name": f"path_heads.{module_name}",
                        }
                    )
        return tuple(result)

    def coefficient_head_modules_by_role(self) -> Mapping[str, nn.Module]:
        """Return a stable read only role to coefficient head mapping."""
        return MappingProxyType(
            {
                role: self.path_heads[module_name]
                for role, module_name in self._head_names.items()
            }
        )

    def first_trunk_linear_modules_by_stage(self) -> Mapping[str, nn.Linear]:
        """Return the first learned Gate trunk layer for each stage."""
        result: dict[str, nn.Linear] = {}
        for stage_name, trunk in self.stage_trunks.items():
            first = next(
                (module for module in trunk.modules() if isinstance(module, nn.Linear)),
                None,
            )
            if first is not None:
                result[stage_name] = first
        return MappingProxyType(result)

    @property
    def selected_covariant_roles(self) -> tuple[str, ...]:
        return tuple(
            path.candidate.role
            for targets in self._stage_paths.values()
            for paths in targets.values()
            for path in paths
        )

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
    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "model_family": "invariant_gate_pipeline_v2",
            "architecture_id": self.config.architecture_id,
            "implemented_mechanism": self.config.implemented_mechanism,
            "trainable_parameter_count": self.trainable_parameter_count,
            "execution_group_count": self._execution_group_count,
            "shared_module_reference_count": self._shared_module_reference_count,
            "channel_projection_kinds": tuple(
                dict.fromkeys(stage.channel_projection for stage in self.config.stages)
            ),
            "coefficient_head_kinds": tuple(
                dict.fromkeys(stage.coefficient_head for stage in self.config.stages)
            ),
            "path_aggregation_kinds": tuple(
                dict.fromkeys(stage.path_aggregation for stage in self.config.stages)
            ),
            "reversible_coupling_stage_count": sum(
                stage.reversible_coupling for stage in self.config.stages
            ),
            "selected_covariant_path_count": len(self.selected_covariant_roles),
        }

    @property
    def offline_compilation_summary(self) -> dict[str, Any]:
        statuses = {
            status: sum(item.status == status for item in self._candidate_audits)
            for status in (
                "compiled",
                "empty_hom",
                "over_budget",
                "failed",
                "unsupported",
            )
        }
        return {
            "candidate_status_counts": statuses,
            "c_artifact_count": len(self.covariants),
            "reynolds_artifact_count": sum(
                isinstance(module, _RegisteredReynoldsCovariant)
                for module in self.covariants.values()
            ),
            "invariant_metric_count": len(self._metric_names),
            "trainable_parameter_count": self.trainable_parameter_count,
            "forward_compilation": False,
            "runtime_group_expansion": False,
            "selected_covariant_path_count": len(self.selected_covariant_roles),
            "execution_group_count": self._execution_group_count,
            "shared_module_reference_count": self._shared_module_reference_count,
            "synchronous_stage_count": sum(
                sum(item == level for item in self._execution_levels.values())
                for level in set(self._execution_levels.values())
                if sum(item == level for item in self._execution_levels.values()) > 1
            ),
            "normalization_path_count": sum(
                len(items) for items in self.normalization.values()
            ),
        }

    def get_extra_state(self) -> dict[str, Any]:
        return dict(self._state_metadata)

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping) or dict(state) != self._state_metadata:
            raise RuntimeError("pipeline checkpoint configuration does not match")

    def reset_normalization_stats(self) -> None:
        """Reset every schema RMS to its unobserved state."""
        for stage in self.normalization.values():
            for normalizer in stage:
                normalizer.reset()

    def normalization_state_dict(self) -> dict[str, Tensor]:
        """Return only the normalization buffers for a compact checkpoint."""
        result: dict[str, Tensor] = {}
        for stage_name, stage in self.normalization.items():
            for index, normalizer in enumerate(stage):
                prefix = f"{stage_name}.{index}"
                result[f"{prefix}.mean_square"] = (
                    normalizer.mean_square.detach().cpu().clone()
                )
                result[f"{prefix}.sample_count"] = (
                    normalizer.sample_count.detach().cpu().clone()
                )
        return result

    def load_normalization_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore normalization buffers without loading compiled tensors."""
        if not isinstance(state, Mapping):
            raise TypeError("normalization state must be a mapping")
        expected = set(self.normalization_state_dict())
        if set(state) != expected:
            missing = tuple(sorted(expected.difference(state)))
            unexpected = tuple(sorted(set(state).difference(expected)))
            raise ValueError(
                "normalization state keys do not match the pipeline: "
                f"missing={missing}, unexpected={unexpected}"
            )
        with torch.no_grad():
            for stage_name, stage in self.normalization.items():
                for index, normalizer in enumerate(stage):
                    prefix = f"{stage_name}.{index}"
                    mean_square = state[f"{prefix}.mean_square"]
                    sample_count = state[f"{prefix}.sample_count"]
                    if (
                        not isinstance(mean_square, Tensor)
                        or mean_square.shape != torch.Size()
                        or not torch.is_floating_point(mean_square)
                        or not bool(torch.isfinite(mean_square).item())
                        or float(mean_square.item()) < 0.0
                    ):
                        raise ValueError(f"{prefix}.mean_square is invalid")
                    if (
                        not isinstance(sample_count, Tensor)
                        or sample_count.shape != torch.Size()
                        or torch.is_floating_point(sample_count)
                        or torch.is_complex(sample_count)
                        or sample_count.dtype == torch.bool
                        or int(sample_count.item()) < 0
                    ):
                        raise ValueError(f"{prefix}.sample_count is invalid")
                    normalizer.mean_square.copy_(
                        mean_square.to(
                            device=normalizer.mean_square.device,
                            dtype=normalizer.mean_square.dtype,
                        )
                    )
                    normalizer.sample_count.copy_(
                        sample_count.to(
                            device=normalizer.sample_count.device,
                            dtype=normalizer.sample_count.dtype,
                        )
                    )

    def set_descriptor_projection(
        self,
        stage_name: str,
        components: Tensor,
        mean: Tensor | None = None,
    ) -> None:
        """Install one fixed training split descriptor projection."""
        if stage_name not in self.descriptor_transforms:
            raise KeyError(f"unknown stage {stage_name}")
        transform = self.descriptor_transforms[stage_name]
        assert isinstance(transform, _DescriptorTransform)
        transform.set_projection(components, mean)

    def clear_descriptor_projections(self) -> None:
        """Remove every calibrated descriptor projection."""
        for transform in self.descriptor_transforms.values():
            assert isinstance(transform, _DescriptorTransform)
            transform.clear()

    def descriptor_projection_state_dict(self) -> dict[str, Tensor]:
        """Return fixed descriptor transforms for checkpoint storage."""
        result: dict[str, Tensor] = {}
        for stage_name, transform in self.descriptor_transforms.items():
            assert isinstance(transform, _DescriptorTransform)
            result[f"{stage_name}.components"] = (
                transform.components.detach().cpu().clone()
            )
            result[f"{stage_name}.mean"] = transform.mean.detach().cpu().clone()
        return result

    def load_descriptor_projection_state_dict(
        self, state: Mapping[str, Tensor]
    ) -> None:
        """Restore fixed descriptor transforms from a checkpoint."""
        if not isinstance(state, Mapping):
            raise TypeError("descriptor projection state must be a mapping")
        expected = set(self.descriptor_projection_state_dict())
        if set(state) != expected:
            missing = tuple(sorted(expected.difference(state)))
            unexpected = tuple(sorted(set(state).difference(expected)))
            raise ValueError(
                "descriptor projection keys do not match the pipeline: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for stage_name, transform in self.descriptor_transforms.items():
            assert isinstance(transform, _DescriptorTransform)
            components = state[f"{stage_name}.components"]
            mean = state[f"{stage_name}.mean"]
            if not isinstance(components, Tensor):
                raise TypeError(f"{stage_name}.components must be a tensor")
            if components.shape[0] == 0:
                transform.clear()
                continue
            transform.set_projection(components, mean)

    def initialize_coefficient_heads_from(
        self, dense_model: "InvariantGatePipelineV2"
    ) -> int:
        """Initialize factorized heads from matching trained dense heads."""
        if not isinstance(dense_model, InvariantGatePipelineV2):
            raise TypeError("dense_model must be an InvariantGatePipelineV2")
        if set(self._head_names) != set(dense_model._head_names):
            raise ValueError("coefficient head roles do not match")
        count = 0
        for role, target_name in self._head_names.items():
            source = dense_model.path_heads[dense_model._head_names[role]]
            target = self.path_heads[target_name]
            if not isinstance(source, nn.Linear):
                raise TypeError("source coefficient head must be dense")
            initializer = getattr(target, "initialize_from_dense", None)
            if initializer is None:
                raise TypeError("target coefficient head is not factorized")
            initializer(source.weight.detach(), source.bias.detach())
            count += 1
        return count

    def zero_output_heads(self) -> int:
        """Zero the final typed channel projection and its scalar heads."""
        count = 0
        with torch.no_grad():
            for path in self._stage_paths[self.config.output_stage][A]:
                head = self.path_heads[self._head_names[path.candidate.role]]
                for parameter in head.parameters():
                    parameter.zero_()
                count += 1
            projection = self.channel_projections[
                self._projection_names[(self.config.output_stage, A)]
            ]
            for parameter in projection.parameters():
                parameter.zero_()
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

    def _normalize_schema(
        self, value: Tensor, stage_name: str, schema_index: int
    ) -> Tensor:
        return self.normalization[stage_name][schema_index](value)

    def _generic_primitive(self, path: _ActivePath, state: TypedStateV2) -> Tensor:
        candidate = path.candidate
        assert path.artifact_name is not None and candidate.signature is not None
        module = self.covariants[path.artifact_name]
        values = tuple(
            self._path_value(endpoint, state) for endpoint in candidate.endpoints
        )
        prefix = values[0].shape[:-2]
        slot_count = len(values)
        inputs: dict[str, Tensor] = {}
        for index, (slot, value) in enumerate(zip(candidate.signature.inputs, values)):
            if value.shape[:-2] != prefix:
                raise ValueError("candidate endpoint batch shapes do not match")
            channel_shape = (
                (1,) * index + (value.shape[-2],) + (1,) * (slot_count - index - 1)
            )
            inputs[slot.name] = value.reshape(
                prefix + channel_shape + (value.shape[-1],)
            )
        result = module.evaluate_basis(inputs)
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
    ) -> tuple[Tensor, Tensor]:
        schemas = [self._normalize_schema(radial, stage.name, 0)]
        for index, path in enumerate(self._stage_scalars[stage.name], start=1):
            primitive = self._primitive(path, state).squeeze(-1)
            schemas.append(self._normalize_schema(primitive, stage.name, index))
        normalized = torch.cat(schemas, dim=-1)
        invariants = self.descriptor_transforms[stage.name](normalized)
        target_width = self._trunk_input_counts[stage.name]
        if invariants.shape[-1] < target_width:
            invariants = torch.nn.functional.pad(
                invariants, (0, target_width - invariants.shape[-1])
            )
        return normalized, invariants

    def _metric_gate_factor(self, primitive: Tensor, target: TypeKey) -> Tensor:
        metric = getattr(self, self._metric_names[target])
        norm = torch.sqrt(
            torch.einsum("...i,ij,...j->...", primitive, metric, primitive)
            .clamp_min(0.0)
            .add(self.config.radial.rms_epsilon)
        )
        return norm / (1.0 + norm)

    def _explicit_skip(
        self,
        stage: InvariantGateStageV2Config,
        target: TypeKey,
        state: TypedStateV2,
        trunk: Tensor,
    ) -> Tensor | None:
        key = (stage.name, target)
        names = self._skip_sources[key]
        if not names:
            return None
        values = tuple(state[name][target] for name in names)
        if stage.skip_policy == "id":
            skip = values[0]
        else:
            projection = self.skip_projections[self._skip_projection_names[key]]
            skip = projection(torch.cat(values, dim=-2))
        gate_name = self._skip_gate_names.get(key)
        if gate_name is not None:
            gate = 1.0 + torch.tanh(self.skip_gates[gate_name](trunk))
            skip = skip * gate.unsqueeze(-1)
        return skip

    def _run_local(
        self,
        displacement: Tensor,
        relative_frame: Tensor,
        *,
        collect_debug: bool,
        collect_coefficients: bool = False,
    ) -> tuple[Tensor, dict[str, Any]]:
        state: TypedStateV2 = {
            "x": {A: displacement.unsqueeze(-2)},
            "r": dict(
                encode_typed_blocks(self.pose_encoder, relative_frame, self._manifest)
            ),
        }
        radial = self._radial(displacement)
        invariant_debug: dict[str, Tensor] | None = {} if collect_debug else None
        normalized_debug: dict[str, Tensor] | None = {} if collect_debug else None
        branch_debug: dict[str, dict[TypeKey, tuple[Tensor, ...]]] | None = (
            {} if collect_debug else None
        )
        concat_debug: dict[str, dict[TypeKey, Tensor]] | None = (
            {} if collect_debug else None
        )
        direct_debug: dict[str, Tensor] | None = {} if collect_debug else None
        coefficient_debug: dict[str, Tensor] | None = (
            {} if collect_coefficients else None
        )
        active_level: int | None = None
        pending: dict[str, dict[TypeKey, Tensor]] = {}
        for stage in self.config.stages:
            level = self._execution_levels[stage.name]
            if active_level is None:
                active_level = level
            elif level != active_level:
                state.update(pending)
                pending = {}
                active_level = level
            normalized, invariants = self._stage_invariants(stage, radial, state)
            trunk = self.stage_trunks[stage.name](invariants)
            if invariant_debug is not None:
                invariant_debug[stage.name] = invariants
            if normalized_debug is not None:
                normalized_debug[stage.name] = normalized
            outputs: dict[TypeKey, Tensor] = {}
            if branch_debug is not None and concat_debug is not None:
                branch_debug[stage.name] = {}
                concat_debug[stage.name] = {}
            for target, paths in self._stage_paths[stage.name].items():
                branches: list[Tensor] = []
                routing_logits: list[Tensor] = []
                for path in paths:
                    primitive = self._primitive(path, state)
                    head_value = self.path_heads[self._head_names[path.candidate.role]](
                        trunk
                    )
                    output_channels = self._channels[(stage.name, target)]
                    if coefficient_debug is not None:
                        coefficient_debug[path.candidate.role] = head_value.reshape(
                            head_value.shape[:-1]
                            + (output_channels, path.primitive_channels)
                        )
                    if stage.path_aggregation != "linear":
                        routing_logits.append(head_value.mean(dim=-1))
                    activation_name = (
                        "tanh"
                        if stage.metric_gate == "skip_identity"
                        else stage.coefficient_activation
                    )
                    activated = _coefficient_activation(head_value, activation_name)
                    coefficients = activated.reshape(
                        activated.shape[:-1]
                        + (output_channels, path.primitive_channels)
                    )
                    if stage.metric_gate in ("norm", "multiply"):
                        coefficients = coefficients * self._metric_gate_factor(
                            primitive, target
                        ).unsqueeze(-2)
                    branch = torch.einsum(
                        "...op,...pd->...od",
                        coefficients,
                        primitive,
                    )
                    branches.append(branch)
                if stage.path_aggregation != "linear":
                    scores = (
                        torch.stack(routing_logits, dim=-1) / stage.path_temperature
                    )
                    weights = torch.softmax(scores, dim=-1)
                    if stage.path_aggregation == "soft_moe" and len(paths) > 2:
                        selected = torch.topk(weights, k=2, dim=-1).indices
                        mask = torch.zeros_like(weights).scatter(-1, selected, 1.0)
                        weights = weights * mask
                        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                            torch.finfo(weights.dtype).eps
                        )
                    branches = [
                        branch * weights[..., index, None, None]
                        for index, branch in enumerate(branches)
                    ]
                if direct_debug is not None:
                    for path, branch in zip(paths, branches):
                        direct_debug[path.candidate.role] = branch
                if stage.skip_policy == "legacy":
                    skip = tuple(
                        state[name][target]
                        for name in self._skip_sources[(stage.name, target)]
                    )
                    concat = torch.cat((*skip, *branches), dim=-2)
                    output = self.channel_projections[
                        self._projection_names[(stage.name, target)]
                    ](concat)
                else:
                    concat = torch.cat(tuple(branches), dim=-2)
                    output = self.channel_projections[
                        self._projection_names[(stage.name, target)]
                    ](concat)
                    skip_value = self._explicit_skip(stage, target, state, trunk)
                    if skip_value is not None:
                        output = output + skip_value
                        if direct_debug is not None:
                            direct_debug[f"{stage.name}.{_type_label(target)}.skip"] = (
                                skip_value
                            )
                reversible_name = self._reversible_names.get((stage.name, target))
                if reversible_name is not None:
                    coupling = self.reversible_couplings[reversible_name]
                    assert isinstance(coupling, _ReversibleChannelCoupling)
                    output = coupling(output, trunk)
                outputs[target] = output
                if branch_debug is not None and concat_debug is not None:
                    branch_debug[stage.name][target] = tuple(branches)
                    concat_debug[stage.name][target] = concat
            pending[stage.name] = outputs
        state.update(pending)
        local = state[self.config.output_stage][A][..., 0, :]
        debug: dict[str, Any] = {}
        if collect_debug:
            debug.update(
                {
                    "state": state,
                    "invariants": invariant_debug,
                    "normalized_descriptors": normalized_debug,
                    "branches": branch_debug,
                    "concats": concat_debug,
                    "direct_paths": direct_debug,
                }
            )
        if coefficient_debug is not None:
            debug["coefficient_activations"] = coefficient_debug
        return local, debug

    def forward_local(self, centers: Tensor, frames: Tensor) -> Tensor:
        displacement, relative_frame = self._root_geometry(centers, frames)
        return self._run_local(displacement, relative_frame, collect_debug=False)[0]

    def forward(self, centers: Tensor, frames: Tensor) -> Tensor:
        local = self.forward_local(centers, frames)
        return torch.einsum("...ij,...j->...i", frames[..., 0, :, :], local)

    def collect_normalized_descriptors(
        self, centers: Tensor, frames: Tensor
    ) -> dict[str, Tensor]:
        """Collect normalized descriptors before masks and projections."""
        if self.training:
            raise RuntimeError("descriptor collection requires evaluation mode")
        with torch.no_grad():
            displacement, relative_frame = self._root_geometry(centers, frames)
            _local, values = self._run_local(
                displacement, relative_frame, collect_debug=True
            )
        return {
            name: value.detach()
            for name, value in values["normalized_descriptors"].items()
        }

    def collect_coefficient_activations(
        self, centers: Tensor, frames: Tensor
    ) -> dict[str, Tensor]:
        """Collect unactivated gamma values without updating normalization."""
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                displacement, relative_frame = self._root_geometry(
                    centers, frames
                )
                _local, values = self._run_local(
                    displacement,
                    relative_frame,
                    collect_debug=False,
                    collect_coefficients=True,
                )
        finally:
            self.train(was_training)
        return {
            role: value.detach()
            for role, value in values["coefficient_activations"].items()
        }

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
    reynolds_context = _ReynoldsContext(catalog)
    invariant_metrics = {key: reynolds_context.metric(key) for key in catalog.blocks}
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
            channel_schedule[(stage.name, key)] = _stage_channel_count(stage, key)
    artifacts: dict[CSignature, CovariantCompilation | _ReynoldsCovariantArtifact] = {}
    audits: list[CandidateAuditV2] = []
    failures = False
    for candidate in candidates:
        degree3_can_skip = (
            resolved.degree3_overflow_policy == "audit_skip"
            and ".degree3." in candidate.role
        )
        if candidate.unsupported_reason is not None:
            audits.append(
                CandidateAuditV2(
                    candidate.stage,
                    candidate.bank,
                    candidate.role,
                    "unsupported",
                    "unsupported_three_independent_slots",
                    None,
                    0,
                    candidate.unsupported_reason,
                )
            )
            continue
        signature_label = _signature_label(candidate.signature, candidate.shortcut_rank)
        if candidate.shortcut_rank is not None:
            stage = stage_lookup[candidate.stage]
            assert candidate.target is not None or candidate.bank == "scalar"
            primitive_channels = channel_schedule[
                (candidate.endpoints[1].source, candidate.endpoints[1].key)
            ]
            coefficient_count = (
                0
                if candidate.target is None
                else _stage_channel_count(stage, candidate.target) * primitive_channels
            )
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
                failures = failures or not degree3_can_skip
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
        if ".degree3." in candidate.role:
            predicted_basis_dimension = reynolds_context.hom_dimension(
                candidate.signature
            )
            predicted_input_channels = math.prod(
                channel_schedule[(endpoint.source, endpoint.key)]
                for endpoint in candidate.endpoints
            )
            predicted_coefficient_count = (
                _stage_channel_count(stage_lookup[candidate.stage], candidate.target)
                * predicted_input_channels
                * predicted_basis_dimension
            )
            if predicted_coefficient_count > resolved.max_gate_coefficients:
                failures = failures or not degree3_can_skip
                audits.append(
                    CandidateAuditV2(
                        candidate.stage,
                        candidate.bank,
                        candidate.role,
                        "over_budget",
                        signature_label,
                        predicted_basis_dimension,
                        (stage_lookup[candidate.stage].trunk_width + 1)
                        * predicted_coefficient_count,
                        (
                            f"coefficient head width {predicted_coefficient_count} exceeds "
                            f"max_gate_coefficients {resolved.max_gate_coefficients}"
                        ),
                    )
                )
                continue
        try:
            artifact = artifacts.get(candidate.signature)
            if artifact is None:
                artifact = (
                    _compile_reynolds_covariant(reynolds_context, candidate.signature)
                    if ".degree3." in candidate.role
                    else compile_covariant_basis(
                        catalog,
                        candidate.signature,
                        max_constraint_entries=resolved.max_constraint_entries,
                    )
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
                0
                if candidate.target is None
                else _stage_channel_count(stage, candidate.target)
                * input_channels
                * artifact.basis_dimension
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
                failures = failures or not degree3_can_skip
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
            failures = failures or not degree3_can_skip
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
        resolved,
        PoseEncoder(anchors),
        anchors,
        manifest,
        artifacts,
        invariant_metrics,
        candidates,
        audits,
    )
