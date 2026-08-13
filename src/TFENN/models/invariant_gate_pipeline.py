"""Configurable invariant controlled pipeline for one benzene pair.

The builder compiles anchors and complete C bases from the supplied generators.
The resulting module contains only fixed tensor contractions and scalar MLPs in
forward.  No orbit, group table, or compiled artifact is read from disk.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

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
    TypeKey,
    build_primitive_b_manifest,
    build_type_catalog,
    compile_anchors,
    compile_covariant_basis,
    encode_typed_blocks,
)

from .transformation import InvariantGate


__all__ = [
    "InvariantGatePipeline",
    "MLPConfig",
    "PairPipelineConfig",
    "RadialConfig",
    "StageConfig",
    "build_invariant_gate_pipeline",
    "default_pair_pipeline_config",
]


A = TypeKey("A")
Stream = Literal["A", "B"]
_RESERVED_INPUTS = {"x": "A", "r": "B"}


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


def _string_tuple(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Define every scalar coefficient MLP used by one stage."""

    hidden_widths: tuple[int, ...] = (32,)
    activation: str = "silu"
    output_activation: str = "identity"
    use_bias: bool = True

    def __post_init__(self) -> None:
        widths = tuple(
            _positive_int(width, f"hidden_widths[{index}]")
            for index, width in enumerate(self.hidden_widths)
        )
        object.__setattr__(self, "hidden_widths", widths)
        if not isinstance(self.activation, str):
            raise TypeError("activation must be a string")
        if not isinstance(self.output_activation, str):
            raise TypeError("output_activation must be a string")
        if not isinstance(self.use_bias, bool):
            raise TypeError("use_bias must be bool")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MLPConfig:
        if not isinstance(value, Mapping):
            raise TypeError("MLP config must be a mapping")
        return cls(
            hidden_widths=tuple(value.get("hidden_widths", (32,))),
            activation=value.get("activation", "silu"),
            output_activation=value.get("output_activation", "identity"),
            use_bias=value.get("use_bias", True),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_widths": list(self.hidden_widths),
            "activation": self.activation,
            "output_activation": self.output_activation,
            "use_bias": self.use_bias,
        }


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Describe one learned A or B stage in the ordered stage graph."""

    name: str
    output_stream: Stream
    inputs: tuple[str, ...]
    channels: int
    lift_orders: tuple[int, ...] = (1, 2)
    mix_inputs: bool = True
    mix_components: bool = True
    mix_channels: bool = False
    cross_grams: bool = True
    invariant_inputs: tuple[str, ...] | None = None
    mlp: MLPConfig = field(default_factory=MLPConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("stage name must be a valid identifier")
        if self.name in _RESERVED_INPUTS:
            raise ValueError("stage name is reserved for a geometric input")
        if self.output_stream not in {"A", "B"}:
            raise ValueError("output_stream must be A or B")
        object.__setattr__(self, "inputs", _string_tuple(self.inputs, "inputs"))
        if not self.inputs:
            raise ValueError("stage inputs cannot be empty")
        object.__setattr__(self, "channels", _positive_int(self.channels, "channels"))
        orders = tuple(
            _positive_int(order, f"lift_orders[{index}]")
            for index, order in enumerate(self.lift_orders)
        )
        if not orders or len(set(orders)) != len(orders):
            raise ValueError("lift_orders must be nonempty and unique")
        if tuple(sorted(orders)) != orders or any(order > 3 for order in orders):
            raise ValueError("lift_orders must be increasing and no greater than three")
        object.__setattr__(self, "lift_orders", orders)
        for name in (
            "mix_inputs",
            "mix_components",
            "mix_channels",
            "cross_grams",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.invariant_inputs is not None:
            object.__setattr__(
                self,
                "invariant_inputs",
                _string_tuple(self.invariant_inputs, "invariant_inputs"),
            )
        if not isinstance(self.mlp, MLPConfig):
            raise TypeError("mlp must be an MLPConfig")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StageConfig:
        if not isinstance(value, Mapping):
            raise TypeError("stage config must be a mapping")
        return cls(
            name=value["name"],
            output_stream=value["output_stream"],
            inputs=tuple(value["inputs"]),
            channels=value["channels"],
            lift_orders=tuple(value.get("lift_orders", (1, 2))),
            mix_inputs=value.get("mix_inputs", True),
            mix_components=value.get("mix_components", True),
            mix_channels=value.get("mix_channels", False),
            cross_grams=value.get("cross_grams", True),
            invariant_inputs=(
                None
                if value.get("invariant_inputs") is None
                else tuple(value["invariant_inputs"])
            ),
            mlp=MLPConfig.from_dict(value.get("mlp", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_stream": self.output_stream,
            "inputs": list(self.inputs),
            "channels": self.channels,
            "lift_orders": list(self.lift_orders),
            "mix_inputs": self.mix_inputs,
            "mix_components": self.mix_components,
            "mix_channels": self.mix_channels,
            "cross_grams": self.cross_grams,
            "invariant_inputs": (
                None if self.invariant_inputs is None else list(self.invariant_inputs)
            ),
            "mlp": self.mlp.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RadialConfig:
    """Define invariant distance features shared by all stages."""

    distance_scale: float = 10.0
    rbf_centers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    rbf_width: float = 0.4
    inverse_powers: tuple[int, ...] = (1, 2, 3)
    epsilon: float = 1.0e-12
    gram_activation: Literal["identity", "tanh"] = "tanh"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distance_scale",
            _positive_float(self.distance_scale, "distance_scale"),
        )
        centers = tuple(float(center) for center in self.rbf_centers)
        if any(not math.isfinite(center) for center in centers):
            raise ValueError("rbf_centers must be finite")
        object.__setattr__(self, "rbf_centers", centers)
        object.__setattr__(
            self,
            "rbf_width",
            _positive_float(self.rbf_width, "rbf_width"),
        )
        powers = tuple(
            _positive_int(power, f"inverse_powers[{index}]")
            for index, power in enumerate(self.inverse_powers)
        )
        if len(set(powers)) != len(powers):
            raise ValueError("inverse_powers cannot contain duplicates")
        object.__setattr__(self, "inverse_powers", powers)
        object.__setattr__(
            self,
            "epsilon",
            _positive_float(self.epsilon, "epsilon"),
        )
        if self.gram_activation not in {"identity", "tanh"}:
            raise ValueError("gram_activation must be identity or tanh")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RadialConfig:
        if not isinstance(value, Mapping):
            raise TypeError("radial config must be a mapping")
        return cls(
            distance_scale=value.get("distance_scale", 10.0),
            rbf_centers=tuple(value.get("rbf_centers", (0.0, 0.5, 1.0, 1.5, 2.0))),
            rbf_width=value.get("rbf_width", 0.4),
            inverse_powers=tuple(value.get("inverse_powers", (1, 2, 3))),
            epsilon=value.get("epsilon", 1.0e-12),
            gram_activation=value.get("gram_activation", "tanh"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "distance_scale": self.distance_scale,
            "rbf_centers": list(self.rbf_centers),
            "rbf_width": self.rbf_width,
            "inverse_powers": list(self.inverse_powers),
            "epsilon": self.epsilon,
            "gram_activation": self.gram_activation,
        }


@dataclass(frozen=True, slots=True)
class PairPipelineConfig:
    """Define the full ordered pair network and offline compiler limits."""

    stages: tuple[StageConfig, ...]
    output_stage: str
    architecture_id: str = "pair_invariant_gate_pipeline_v1"
    anchor_ranks: tuple[int, ...] = tuple(range(1, 7))
    max_constraint_entries: int = 10_000_000
    max_gate_coefficients: int = 2_000_000
    max_invariant_channels: int = 20_000
    radial: RadialConfig = field(default_factory=RadialConfig)

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages or any(not isinstance(stage, StageConfig) for stage in stages):
            raise ValueError("stages must contain StageConfig values")
        object.__setattr__(self, "stages", stages)
        if (
            not isinstance(self.architecture_id, str)
            or not self.architecture_id.isidentifier()
        ):
            raise ValueError("architecture_id must be a valid identifier")
        names = tuple(stage.name for stage in stages)
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        known = dict(_RESERVED_INPUTS)
        for stage in stages:
            references = stage.inputs + (
                stage.inputs
                if stage.invariant_inputs is None
                else stage.invariant_inputs
            )
            unknown = tuple(name for name in references if name not in known)
            if unknown:
                raise ValueError(
                    f"stage {stage.name} references unavailable inputs {unknown}"
                )
            known[stage.name] = stage.output_stream
        if self.output_stage not in known or self.output_stage in _RESERVED_INPUTS:
            raise ValueError("output_stage must name one configured stage")
        if self.output_stage != stages[-1].name:
            raise ValueError("output_stage must be the final configured stage")
        output = next(stage for stage in stages if stage.name == self.output_stage)
        if output.output_stream != "A" or output.channels != 1:
            raise ValueError("output stage must be a one channel A stage")
        ranks = tuple(
            _positive_int(rank, f"anchor_ranks[{index}]")
            for index, rank in enumerate(self.anchor_ranks)
        )
        if not ranks or len(set(ranks)) != len(ranks):
            raise ValueError("anchor_ranks must be nonempty and unique")
        object.__setattr__(self, "anchor_ranks", ranks)
        object.__setattr__(
            self,
            "max_constraint_entries",
            _positive_int(self.max_constraint_entries, "max_constraint_entries"),
        )
        object.__setattr__(
            self,
            "max_gate_coefficients",
            _positive_int(self.max_gate_coefficients, "max_gate_coefficients"),
        )
        object.__setattr__(
            self,
            "max_invariant_channels",
            _positive_int(self.max_invariant_channels, "max_invariant_channels"),
        )
        if not isinstance(self.radial, RadialConfig):
            raise TypeError("radial must be a RadialConfig")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PairPipelineConfig:
        if not isinstance(value, Mapping):
            raise TypeError("pipeline config must be a mapping")
        return cls(
            stages=tuple(StageConfig.from_dict(item) for item in value["stages"]),
            output_stage=value["output_stage"],
            architecture_id=value.get(
                "architecture_id",
                "pair_invariant_gate_pipeline_v1",
            ),
            anchor_ranks=tuple(value.get("anchor_ranks", range(1, 7))),
            max_constraint_entries=value.get("max_constraint_entries", 10_000_000),
            max_gate_coefficients=value.get("max_gate_coefficients", 2_000_000),
            max_invariant_channels=value.get("max_invariant_channels", 20_000),
            radial=RadialConfig.from_dict(value.get("radial", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.as_dict() for stage in self.stages],
            "output_stage": self.output_stage,
            "architecture_id": self.architecture_id,
            "anchor_ranks": list(self.anchor_ranks),
            "max_constraint_entries": self.max_constraint_entries,
            "max_gate_coefficients": self.max_gate_coefficients,
            "max_invariant_channels": self.max_invariant_channels,
            "radial": self.radial.as_dict(),
        }


def default_pair_pipeline_config() -> PairPipelineConfig:
    """Return the A1 A2 B2 A3 graph used for the first pair experiment."""
    return PairPipelineConfig(
        stages=(
            StageConfig("a1", "A", ("x",), 4),
            StageConfig("a2", "A", ("a1",), 4),
            StageConfig("b2", "B", ("a2", "r"), 4),
            StageConfig("a3", "A", ("b2",), 1),
        ),
        output_stage="a3",
    )


@dataclass(frozen=True, slots=True)
class _InputReference:
    slot: str
    source: str
    key: TypeKey


@dataclass(frozen=True, slots=True)
class _PathPlan:
    stage: str
    target: TypeKey
    role: str
    degree: int
    signature: CSignature
    inputs: tuple[_InputReference, ...]


def _type_label(key: TypeKey) -> str:
    return "a" if key.stream == "A" else f"b{key.component}"


def _component_keys(stream: Stream, b_keys: tuple[TypeKey, ...]) -> tuple[TypeKey, ...]:
    return (A,) if stream == "A" else b_keys


def _unary_slot(key: TypeKey, order: int) -> CSlot:
    if order == 1:
        return CSlot("value", key)
    return CSlot("value", key, power=order, mode="symmetric_power")


def _pair_signature(target: TypeKey, left: TypeKey, right: TypeKey) -> CSignature:
    return CSignature(target, (CSlot("left", left), CSlot("right", right)))


def _build_path_plans(
    config: PairPipelineConfig,
    b_keys: tuple[TypeKey, ...],
) -> tuple[_PathPlan, ...]:
    streams: dict[str, Stream] = dict(_RESERVED_INPUTS)  # type: ignore[arg-type]
    plans: list[_PathPlan] = []
    for stage in config.stages:
        targets = _component_keys(stage.output_stream, b_keys)
        for target in targets:
            for source in stage.inputs:
                source_keys = _component_keys(streams[source], b_keys)
                for source_key in source_keys:
                    for order in stage.lift_orders:
                        slot = _unary_slot(source_key, order)
                        role = (
                            f"{stage.name}.{_type_label(target)}."
                            f"from_{source}_{_type_label(source_key)}.lift{order}"
                        )
                        plans.append(
                            _PathPlan(
                                stage.name,
                                target,
                                role,
                                order,
                                CSignature(target, (slot,)),
                                (_InputReference("value", source, source_key),),
                            )
                        )
                if stage.mix_components and len(source_keys) > 1:
                    for left_key, right_key in combinations(source_keys, 2):
                        role = (
                            f"{stage.name}.{_type_label(target)}."
                            f"within_{source}_{_type_label(left_key)}_"
                            f"{_type_label(right_key)}.product2"
                        )
                        plans.append(
                            _PathPlan(
                                stage.name,
                                target,
                                role,
                                2,
                                _pair_signature(target, left_key, right_key),
                                (
                                    _InputReference("left", source, left_key),
                                    _InputReference("right", source, right_key),
                                ),
                            )
                        )
                if stage.mix_channels:
                    for source_key in source_keys:
                        role = (
                            f"{stage.name}.{_type_label(target)}."
                            f"within_{source}_{_type_label(source_key)}.channel_product2"
                        )
                        plans.append(
                            _PathPlan(
                                stage.name,
                                target,
                                role,
                                2,
                                _pair_signature(target, source_key, source_key),
                                (
                                    _InputReference("left", source, source_key),
                                    _InputReference("right", source, source_key),
                                ),
                            )
                        )
            if stage.mix_inputs:
                for left_source, right_source in combinations(stage.inputs, 2):
                    left_keys = _component_keys(streams[left_source], b_keys)
                    right_keys = _component_keys(streams[right_source], b_keys)
                    for left_key in left_keys:
                        for right_key in right_keys:
                            role = (
                                f"{stage.name}.{_type_label(target)}."
                                f"mix_{left_source}_{_type_label(left_key)}_"
                                f"{right_source}_{_type_label(right_key)}.product2"
                            )
                            plans.append(
                                _PathPlan(
                                    stage.name,
                                    target,
                                    role,
                                    2,
                                    _pair_signature(target, left_key, right_key),
                                    (
                                        _InputReference("left", left_source, left_key),
                                        _InputReference(
                                            "right", right_source, right_key
                                        ),
                                    ),
                                )
                            )
        streams[stage.name] = stage.output_stream
    return tuple(plans)


def _gram_channel_count(channels: int) -> int:
    return channels * (channels + 1) // 2


def _sum_tensors(values: Sequence[Tensor]) -> Tensor:
    items = tuple(values)
    if not items:
        raise RuntimeError("typed path sum cannot be empty")
    result = items[0]
    for value in items[1:]:
        result = result + value
    return result


class InvariantGatePipeline(nn.Module):
    """Evaluate one configured pair graph in the first benzene frame."""

    def __init__(
        self,
        config: PairPipelineConfig,
        pose_encoder: PoseEncoder,
        anchor_compilation: AnchorCompilation,
        manifest: Sequence[BBlockManifest],
        plans: Sequence[_PathPlan],
        artifacts: Mapping[CSignature, CovariantCompilation],
    ) -> None:
        super().__init__()
        if not isinstance(config, PairPipelineConfig):
            raise TypeError("config must be a PairPipelineConfig")
        if not isinstance(pose_encoder, PoseEncoder):
            raise TypeError("pose_encoder must be a PoseEncoder")
        if not isinstance(anchor_compilation, AnchorCompilation):
            raise TypeError("anchor_compilation must be an AnchorCompilation")
        manifest_items = tuple(manifest)
        if not manifest_items:
            raise ValueError("benzene pose manifest cannot be empty")
        self.config = config
        self.pose_encoder = pose_encoder
        self._anchor_compilation = anchor_compilation
        self._manifest = manifest_items
        self._b_keys = tuple(item.key for item in manifest_items)
        self.register_buffer(
            "radial_centers",
            torch.tensor(config.radial.rbf_centers, dtype=torch.float64),
        )
        self.register_buffer(
            "_runtime_reference",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )

        streams: dict[str, Stream] = dict(_RESERVED_INPUTS)  # type: ignore[arg-type]
        channels: dict[tuple[str, TypeKey], int] = {("x", A): 1}
        for item in manifest_items:
            channels[("r", item.key)] = len(item.anchor_columns)
        for stage in config.stages:
            for key in _component_keys(stage.output_stream, self._b_keys):
                channels[(stage.name, key)] = stage.channels
            streams[stage.name] = stage.output_stream

        self.gates = nn.ModuleDict()
        self._gate_records: list[tuple[str, str, str, int]] = []
        stage_paths: dict[str, dict[TypeKey, list[tuple[str, _PathPlan]]]] = {
            stage.name: {
                key: [] for key in _component_keys(stage.output_stream, self._b_keys)
            }
            for stage in config.stages
        }
        stage_lookup = {stage.name: stage for stage in config.stages}
        invariant_channels = {
            stage.name: self._invariant_channel_count(stage, streams, channels)
            for stage in config.stages
        }
        excessive_invariants = {
            name: count
            for name, count in invariant_channels.items()
            if count > config.max_invariant_channels
        }
        if excessive_invariants:
            raise ValueError(
                "invariant channel count exceeds configured guard: "
                f"{excessive_invariants}"
            )
        for plan in plans:
            artifact = artifacts.get(plan.signature)
            if artifact is None or artifact.basis_dimension == 0:
                continue
            stage = stage_lookup[plan.stage]
            input_channels = {
                item.slot: channels[(item.source, item.key)] for item in plan.inputs
            }
            coefficient_count = (
                stage.channels
                * math.prod(input_channels.values())
                * artifact.basis_dimension
            )
            if coefficient_count > config.max_gate_coefficients:
                raise ValueError(
                    f"gate {plan.role} has {coefficient_count} coefficients per "
                    "sample and exceeds max_gate_coefficients"
                )
            internal_name = f"gate_{len(self._gate_records):04d}"
            self.gates[internal_name] = InvariantGate(
                plan.signature,
                artifact,
                input_channels,
                stage.channels,
                invariant_channels[stage.name],
                hidden_channels=stage.mlp.hidden_widths,
                activation=stage.mlp.activation,
                output_activation=stage.mlp.output_activation,
                use_bias=stage.mlp.use_bias,
            )
            stage_paths[plan.stage][plan.target].append((internal_name, plan))
            self._gate_records.append(
                (plan.role, internal_name, plan.stage, plan.degree)
            )
        for stage_name, targets in stage_paths.items():
            empty = tuple(
                _type_label(key) for key, paths in targets.items() if not paths
            )
            if empty:
                raise ValueError(
                    f"stage {stage_name} has no covariant path for targets {empty}"
                )
        self._stage_paths = {
            stage: {key: tuple(paths) for key, paths in targets.items()}
            for stage, targets in stage_paths.items()
        }
        self._streams = streams
        self._channels = channels
        self._invariant_channels = invariant_channels
        self._state_metadata = {
            "schema_version": 1,
            "config": config.as_dict(),
            "b_ranks": self.b_ranks,
            "gates": tuple(
                (
                    role,
                    self.gates[internal_name].covariant.artifact_fingerprint,
                )
                for role, internal_name, _stage, _degree in self._gate_records
            ),
        }

    def _invariant_channel_count(
        self,
        stage: StageConfig,
        streams: Mapping[str, Stream],
        channels: Mapping[tuple[str, TypeKey], int],
    ) -> int:
        names = (
            stage.inputs if stage.invariant_inputs is None else stage.invariant_inputs
        )
        endpoints: list[tuple[str, TypeKey]] = []
        for name in names:
            for key in _component_keys(streams[name], self._b_keys):
                endpoints.append((name, key))
        gram_count = sum(
            _gram_channel_count(channels[endpoint]) for endpoint in endpoints
        )
        if stage.cross_grams:
            gram_count += sum(
                channels[left] * channels[right]
                for left, right in combinations(endpoints, 2)
                if left[1] == right[1]
            )
        radial_count = (
            2
            + len(self.config.radial.rbf_centers)
            + len(self.config.radial.inverse_powers)
        )
        return radial_count + gram_count

    @property
    def anchor_compilation(self) -> AnchorCompilation:
        return self._anchor_compilation

    @property
    def manifest(self) -> tuple[BBlockManifest, ...]:
        return self._manifest

    @property
    def b_keys(self) -> tuple[TypeKey, ...]:
        return self._b_keys

    @property
    def b_ranks(self) -> tuple[int, ...]:
        return tuple(item.stf_rank for item in self._manifest)

    @property
    def typed_channel_schedule(self) -> dict[str, int | tuple[int, ...]]:
        result: dict[str, int | tuple[int, ...]] = {"x": 1}
        result["r"] = tuple(self._channels[("r", key)] for key in self._b_keys)
        for stage in self.config.stages:
            result[stage.name] = (
                stage.channels
                if stage.output_stream == "A"
                else tuple(stage.channels for _ in self._b_keys)
            )
        return result

    @property
    def offline_compilation_summary(self) -> dict[str, Any]:
        fingerprints = {
            gate.covariant.artifact_fingerprint for gate in self.gates.values()
        }
        return {
            "anchor_ranks": list(self.config.anchor_ranks),
            "primitive_b_ranks": list(self.b_ranks),
            "unique_c_artifact_count": len(fingerprints),
            "gate_count": len(self.gates),
            "forward_compilation": False,
            "disk_artifact_cache": False,
            "orbit_storage": False,
        }

    def named_gates(self) -> tuple[tuple[str, InvariantGate], ...]:
        return tuple(
            (role, self.gates[internal_name])
            for role, internal_name, _stage, _degree in self._gate_records
        )

    def get_extra_state(self) -> dict[str, Any]:
        return dict(self._state_metadata)

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping) or dict(state) != self._state_metadata:
            raise RuntimeError("pipeline checkpoint configuration does not match")

    @property
    def gate_manifest(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "role": role,
                "stage": stage,
                "degree": degree,
                "basis_dimension": self.gates[internal_name].basis_dimension,
                "hidden_widths": list(self.gates[internal_name].hidden_widths),
            }
            for role, internal_name, stage, degree in self._gate_records
        )

    def zero_output_heads(self) -> int:
        """Set every coefficient head in the configured output stage to zero."""
        count = 0
        with torch.no_grad():
            for _role, internal_name, stage, _degree in self._gate_records:
                if stage != self.config.output_stage:
                    continue
                gate = self.gates[internal_name]
                head = next(
                    layer
                    for layer in reversed(gate.invariant_mlp)
                    if isinstance(layer, nn.Linear)
                )
                head.weight.zero_()
                if head.bias is not None:
                    head.bias.zero_()
                count += 1
        return count

    def _root_geometry(self, centers: Tensor, frames: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(centers, Tensor) or not isinstance(frames, Tensor):
            raise TypeError("centers and frames must be tensors")
        if centers.shape[-2:] != (2, 3):
            raise ValueError("centers must end with shape two by three")
        if frames.shape != centers.shape[:-2] + (2, 3, 3):
            raise ValueError("frames must match centers and contain two matrices")
        if centers.dtype != frames.dtype or centers.device != frames.device:
            raise ValueError("centers and frames must share dtype and device")
        if centers.dtype != self._runtime_reference.dtype:
            raise TypeError("inputs must match the network dtype")
        if centers.device != self._runtime_reference.device:
            raise ValueError("inputs must match the network device")
        root = frames[..., 0, :, :]
        displacement = centers[..., 1, :] - centers[..., 0, :]
        local_displacement = torch.einsum("...ji,...j->...i", root, displacement)
        relative_frame = root.mT @ frames[..., 1, :, :]
        return local_displacement, relative_frame

    def _radial_invariants(self, displacement: Tensor) -> Tensor:
        radial = self.config.radial
        distance = torch.sqrt(displacement.square().sum(dim=-1) + radial.epsilon)
        scaled = distance / radial.distance_scale
        rbf = torch.exp(
            -((scaled.unsqueeze(-1) - self.radial_centers) / radial.rbf_width).square()
        )
        inverse = (1.0 + scaled).reciprocal()
        inverse_powers = (
            torch.stack(
                tuple(inverse.pow(power) for power in radial.inverse_powers),
                dim=-1,
            )
            if radial.inverse_powers
            else scaled.new_empty(scaled.shape + (0,))
        )
        return torch.cat(
            (
                torch.ones_like(scaled).unsqueeze(-1),
                scaled.unsqueeze(-1),
                rbf,
                inverse_powers,
            ),
            dim=-1,
        )

    def _gram_entries(self, value: Tensor) -> Tensor:
        gram = torch.einsum("...ci,...di->...cd", value, value)
        entries = torch.stack(
            tuple(
                gram[..., row, column]
                for row in range(value.shape[-2])
                for column in range(row, value.shape[-2])
            ),
            dim=-1,
        )
        if self.config.radial.gram_activation == "tanh":
            return torch.tanh(entries)
        return entries

    def _cross_gram_entries(self, left: Tensor, right: Tensor) -> Tensor:
        gram = torch.einsum("...ci,...di->...cd", left, right)
        entries = gram.flatten(start_dim=gram.ndim - 2)
        if self.config.radial.gram_activation == "tanh":
            return torch.tanh(entries)
        return entries

    def _stage_invariants(
        self,
        stage: StageConfig,
        radial: Tensor,
        values: Mapping[str, Mapping[TypeKey, Tensor]],
    ) -> Tensor:
        names = (
            stage.inputs if stage.invariant_inputs is None else stage.invariant_inputs
        )
        entries = [radial]
        endpoints: list[tuple[str, TypeKey]] = []
        for name in names:
            for key in _component_keys(self._streams[name], self._b_keys):
                endpoints.append((name, key))
                entries.append(self._gram_entries(values[name][key]))
        if stage.cross_grams:
            for left, right in combinations(endpoints, 2):
                if left[1] == right[1]:
                    entries.append(
                        self._cross_gram_entries(
                            values[left[0]][left[1]],
                            values[right[0]][right[1]],
                        )
                    )
        return torch.cat(entries, dim=-1)

    def _typed_path(self, displacement: Tensor, relative_frame: Tensor) -> Tensor:
        radial = self._radial_invariants(displacement)
        values: dict[str, dict[TypeKey, Tensor]] = {
            "x": {A: (displacement / self.config.radial.distance_scale).unsqueeze(-2)},
            "r": encode_typed_blocks(
                self.pose_encoder,
                relative_frame,
                self._manifest,
            ),
        }
        for stage in self.config.stages:
            invariants = self._stage_invariants(stage, radial, values)
            targets: dict[TypeKey, Tensor] = {}
            for target, paths in self._stage_paths[stage.name].items():
                outputs = []
                for internal_name, plan in paths:
                    inputs = {
                        item.slot: values[item.source][item.key] for item in plan.inputs
                    }
                    outputs.append(self.gates[internal_name](inputs, invariants))
                targets[target] = _sum_tensors(outputs)
            values[stage.name] = targets
        return values[self.config.output_stage][A][..., 0, :]

    def forward_local(self, centers: Tensor, frames: Tensor) -> Tensor:
        displacement, relative_frame = self._root_geometry(centers, frames)
        return self._typed_path(displacement, relative_frame)

    def forward(self, centers: Tensor, frames: Tensor) -> Tensor:
        local = self.forward_local(centers, frames)
        root = frames[..., 0, :, :]
        return torch.einsum("...ij,...j->...i", root, local)


def build_invariant_gate_pipeline(
    generators: Tensor,
    config: PairPipelineConfig | Mapping[str, Any] | None = None,
    *,
    generator_names: Sequence[str] | None = None,
) -> InvariantGatePipeline:
    """Compile fixed tensor assets once and construct one configured pipeline."""
    if not isinstance(generators, Tensor):
        raise TypeError("generators must be a tensor")
    if generators.ndim != 3 or generators.shape[-2:] != (3, 3):
        raise ValueError("generators must have shape count by three by three")
    pipeline_config = (
        default_pair_pipeline_config()
        if config is None
        else PairPipelineConfig.from_dict(config)
        if isinstance(config, Mapping)
        else config
    )
    if not isinstance(pipeline_config, PairPipelineConfig):
        raise TypeError("config must be a PairPipelineConfig or mapping")
    names = (
        tuple(generator_names)
        if generator_names is not None
        else tuple(f"generator_{index}" for index in range(generators.shape[0]))
    )
    if len(names) != generators.shape[0] or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("generator names must match generator count")
    anchors = compile_anchors(
        generators,
        output_ranks=pipeline_config.anchor_ranks,
    )
    manifest = build_primitive_b_manifest(anchors)
    if not manifest:
        raise ValueError("generators did not produce a primitive B component")
    system = GeneratorSystem(names, generators)
    catalog = build_type_catalog(system, manifest)
    b_keys = tuple(item.key for item in manifest)
    plans = _build_path_plans(pipeline_config, b_keys)
    signatures = tuple(dict.fromkeys(plan.signature for plan in plans))
    artifacts = {
        signature: compile_covariant_basis(
            catalog,
            signature,
            max_constraint_entries=pipeline_config.max_constraint_entries,
        )
        for signature in signatures
    }
    return InvariantGatePipeline(
        pipeline_config,
        PoseEncoder(anchors),
        anchors,
        manifest,
        plans,
        artifacts,
    )
