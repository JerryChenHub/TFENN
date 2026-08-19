"""Compile strict declared edge dual stream covariant flows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from torch import Tensor

from .invariant_gate_pipeline_v2 import (
    A,
    InvariantGatePipelineV2,
    InvariantGatePipelineV2Config,
    InvariantGateStageV2Config,
    PipelineV2CompilationError,
    _type_label,
    build_invariant_gate_pipeline_v2,
)


__all__ = [
    "StrictDualStreamFlowConfig",
    "StrictFlowCompilationError",
    "StrictFlowStageConfig",
    "build_strict_dual_stream_flow",
    "compile_strict_dual_stream_config",
]


StrictStream = Literal["A", "B"]
StrictDescriptorMask = Literal["full", "raw_only"]
_RAW_SOURCES = frozenset(("x", "r"))


def _source_names(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    result = tuple(value)
    if not result or any(
        not isinstance(name, str) or not name.isidentifier() for name in result
    ):
        raise ValueError(f"{field_name} must contain valid identifiers")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


class StrictFlowCompilationError(RuntimeError):
    """Report a strict flow declaration that cannot be compiled exactly."""

    def __init__(
        self,
        message: str,
        *,
        edge_manifest: Sequence[Mapping[str, Any]] = (),
        candidate_manifest: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.edge_manifest = tuple(dict(item) for item in edge_manifest)
        self.candidate_manifest = tuple(
            dict(item) for item in candidate_manifest
        )


@dataclass(frozen=True, slots=True)
class StrictFlowStageConfig:
    """Declare one strict covariant node and its independent Gate context."""

    name: str
    output_stream: StrictStream
    source_names: tuple[str, ...]
    channels: int
    invariant_source_names: tuple[str, ...]
    execution_level: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("stage name must be a valid identifier")
        if self.name in _RAW_SOURCES:
            raise ValueError("stage name is reserved")
        if self.output_stream not in ("A", "B"):
            raise ValueError("output_stream must be A or B")
        object.__setattr__(
            self,
            "source_names",
            _source_names(self.source_names, "source_names"),
        )
        object.__setattr__(
            self,
            "invariant_source_names",
            _source_names(
                self.invariant_source_names,
                "invariant_source_names",
            ),
        )
        if (
            isinstance(self.channels, bool)
            or not isinstance(self.channels, int)
            or self.channels < 1
        ):
            raise ValueError("channels must be a positive integer")
        if (
            isinstance(self.execution_level, bool)
            or not isinstance(self.execution_level, int)
            or self.execution_level < 0
        ):
            raise ValueError("execution_level must be a nonnegative integer")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrictFlowStageConfig":
        return cls(
            name=value["name"],
            output_stream=value["output_stream"],
            source_names=tuple(value["source_names"]),
            channels=value["channels"],
            invariant_source_names=tuple(value["invariant_source_names"]),
            execution_level=value["execution_level"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_stream": self.output_stream,
            "source_names": list(self.source_names),
            "channels": self.channels,
            "invariant_source_names": list(self.invariant_source_names),
            "execution_level": self.execution_level,
        }


@dataclass(frozen=True, slots=True)
class StrictDualStreamFlowConfig:
    """Configure an exact typed DAG with no implicit covariant history."""

    stages: tuple[StrictFlowStageConfig, ...]
    output_stage: str = "out"
    architecture_id: str = "strict_dual_stream_covariant_flow"
    descriptor_mask: StrictDescriptorMask = "full"
    gate_width: int = 8
    anchor_ranks: tuple[int, ...] = (2, 6)
    max_constraint_entries: int = 10_000_000
    max_gate_coefficients: int = 2_000_000
    max_invariant_channels: int = 20_000

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages or any(
            not isinstance(stage, StrictFlowStageConfig) for stage in stages
        ):
            raise ValueError("stages must contain strict stage configurations")
        object.__setattr__(self, "stages", stages)
        if (
            not isinstance(self.architecture_id, str)
            or not self.architecture_id.isidentifier()
        ):
            raise ValueError("architecture_id must be a valid identifier")
        if self.descriptor_mask not in ("full", "raw_only"):
            raise ValueError("descriptor_mask must be full or raw_only")
        for field_name in (
            "gate_width",
            "max_constraint_entries",
            "max_gate_coefficients",
            "max_invariant_channels",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        ranks = tuple(self.anchor_ranks)
        if (
            not ranks
            or len(set(ranks)) != len(ranks)
            or any(
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 1
                for rank in ranks
            )
        ):
            raise ValueError("anchor_ranks must be unique positive integers")
        object.__setattr__(self, "anchor_ranks", ranks)
        names = tuple(stage.name for stage in stages)
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        if self.output_stage != stages[-1].name:
            raise ValueError("output_stage must name the final stage")
        output = stages[-1]
        if output.output_stream != "A" or output.channels != 1:
            raise ValueError("output stage must be one channel A")
        self._validate_declared_dag()

    def _validate_declared_dag(self) -> None:
        stages = self.stages
        previous_level = -1
        known = set(_RAW_SOURCES)
        cursor = 0
        while cursor < len(stages):
            level = stages[cursor].execution_level
            if level < previous_level:
                raise ValueError("execution levels must be nondecreasing")
            end = cursor + 1
            while end < len(stages) and stages[end].execution_level == level:
                end += 1
            for stage in stages[cursor:end]:
                unknown_covariant = tuple(
                    source for source in stage.source_names if source not in known
                )
                unknown_invariant = tuple(
                    source
                    for source in stage.invariant_source_names
                    if source not in known
                )
                if unknown_covariant or unknown_invariant:
                    raise ValueError(
                        f"stage {stage.name} reads unavailable or same level "
                        f"sources covariant={unknown_covariant}, "
                        f"invariant={unknown_invariant}"
                    )
            known.update(stage.name for stage in stages[cursor:end])
            previous_level = level
            cursor = end

        for stage in stages:
            hidden_parents = tuple(
                source
                for source in stage.source_names
                if source not in _RAW_SOURCES
            )
            expected_invariants = ("x", "r", *hidden_parents)
            if stage.invariant_source_names != expected_invariants:
                raise ValueError(
                    f"stage {stage.name} invariant sources must be raw x,r plus "
                    f"its declared hidden parents {hidden_parents}"
                )

        x_ingress = tuple(stage for stage in stages if "x" in stage.source_names)
        r_ingress = tuple(stage for stage in stages if "r" in stage.source_names)
        if len(x_ingress) != 1 or (
            x_ingress[0].name,
            x_ingress[0].output_stream,
            x_ingress[0].source_names,
        ) != ("a1", "A", ("x",)):
            raise ValueError("raw x must enter covariant flow only through a1")
        if len(r_ingress) != 1 or (
            r_ingress[0].name,
            r_ingress[0].output_stream,
            r_ingress[0].source_names,
        ) != ("b1", "B", ("r",)):
            raise ValueError("raw r must enter covariant flow only through b1")
        if x_ingress[0].execution_level != r_ingress[0].execution_level:
            raise ValueError("a1 and b1 must share the raw ingress level")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrictDualStreamFlowConfig":
        return cls(
            stages=tuple(
                StrictFlowStageConfig.from_dict(stage)
                for stage in value["stages"]
            ),
            output_stage=value.get("output_stage", "out"),
            architecture_id=value.get(
                "architecture_id", "strict_dual_stream_covariant_flow"
            ),
            descriptor_mask=value.get("descriptor_mask", "full"),
            gate_width=value.get("gate_width", 8),
            anchor_ranks=tuple(value.get("anchor_ranks", (2, 6))),
            max_constraint_entries=value.get(
                "max_constraint_entries", 10_000_000
            ),
            max_gate_coefficients=value.get("max_gate_coefficients", 2_000_000),
            max_invariant_channels=value.get("max_invariant_channels", 20_000),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.as_dict() for stage in self.stages],
            "output_stage": self.output_stage,
            "architecture_id": self.architecture_id,
            "descriptor_mask": self.descriptor_mask,
            "gate_width": self.gate_width,
            "anchor_ranks": list(self.anchor_ranks),
            "max_constraint_entries": self.max_constraint_entries,
            "max_gate_coefficients": self.max_gate_coefficients,
            "max_invariant_channels": self.max_invariant_channels,
        }


def compile_strict_dual_stream_config(
    config: StrictDualStreamFlowConfig | Mapping[str, Any],
) -> InvariantGatePipelineV2Config:
    """Lower one exact strict declaration to the existing V2 compiler."""
    resolved = (
        StrictDualStreamFlowConfig.from_dict(config)
        if isinstance(config, Mapping)
        else config
    )
    if not isinstance(resolved, StrictDualStreamFlowConfig):
        raise TypeError("config must be StrictDualStreamFlowConfig or mapping")
    stages = tuple(
        InvariantGateStageV2Config(
            name=stage.name,
            output_stream=stage.output_stream,
            source_names=stage.source_names,
            channels=stage.channels,
            invariant_source_names=stage.invariant_source_names,
            trunk_width=resolved.gate_width,
            activation="silu",
            include_symmetric_unary=True,
            include_raw_mixed_pairs=True,
            include_stf_shortcuts=True,
            skip_policy="legacy",
            covariant_include_symmetric_unary=True,
            covariant_include_raw_mixed_pairs=False,
            # Match E311 literally.  The strict graph never presents x and r
            # together as covariant sources, so this family remains dormant.
            covariant_include_stf_shortcuts=True,
            invariant_include_symmetric_unary=True,
            invariant_include_raw_mixed_pairs=True,
            invariant_include_stf_shortcuts=True,
            degree3_policy="none",
            coefficient_activation="identity",
            coefficient_head="dense",
            descriptor_mask=resolved.descriptor_mask,
            trunk_depth=1,
            execution_level=stage.execution_level,
            covariant_required_source_names=stage.source_names,
            channel_projection="dense",
            path_aggregation="linear",
        )
        for stage in resolved.stages
    )
    return InvariantGatePipelineV2Config(
        stages=stages,
        output_stage=resolved.output_stage,
        architecture_id=resolved.architecture_id,
        anchor_ranks=resolved.anchor_ranks,
        max_constraint_entries=resolved.max_constraint_entries,
        max_gate_coefficients=resolved.max_gate_coefficients,
        max_invariant_channels=resolved.max_invariant_channels,
        degree3_overflow_policy="raise",
        implemented_mechanism="strict_dual_stream_covariant_flow",
    )


def _strict_edge_manifest(
    model: InvariantGatePipelineV2,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    failures: list[str] = []
    for stage in model.config.stages:
        declared = set(stage.source_names)
        for target, paths in model._stage_paths[stage.name].items():
            skip_sources = model._skip_sources[(stage.name, target)]
            endpoint_sources = {
                endpoint.source
                for path in paths
                for endpoint in path.candidate.endpoints
            }
            undeclared_sources = tuple(
                sorted(endpoint_sources.difference(declared))
            )
            cross_edges: list[dict[str, Any]] = []
            same_type_edges: list[dict[str, Any]] = []
            missing_edges: list[dict[str, Any]] = []
            for source in stage.source_names:
                source_keys = (
                    (A,)
                    if model._streams[source] == "A"
                    else model._b_keys
                )
                for source_key in source_keys:
                    edge = {
                        "source": source,
                        "source_type": _type_label(source_key),
                        "target_type": _type_label(target),
                    }
                    if source_key == target:
                        covered = source in skip_sources
                        same_type_edges.append({**edge, "covered": covered})
                    else:
                        roles = tuple(
                            path.candidate.role
                            for path in paths
                            if any(
                                endpoint.source == source
                                and endpoint.key == source_key
                                for endpoint in path.candidate.endpoints
                            )
                        )
                        covered = bool(roles)
                        cross_edges.append(
                            {**edge, "covered": covered, "path_roles": roles}
                        )
                    if not covered:
                        missing_edges.append(edge)
            live_live_roles = tuple(
                path.candidate.role
                for path in paths
                if len(
                    {
                        endpoint.source
                        for endpoint in path.candidate.endpoints
                        if endpoint.source not in _RAW_SOURCES
                    }
                )
                > 1
            )
            entry = {
                "stage": stage.name,
                "target": _type_label(target),
                "execution_level": model._execution_levels[stage.name],
                "declared_source_names": stage.source_names,
                "skip_source_names": skip_sources,
                "compiled_path_source_names": tuple(sorted(endpoint_sources)),
                "same_type_edges": tuple(same_type_edges),
                "cross_type_edges": tuple(cross_edges),
                "selected_path_roles": tuple(
                    path.candidate.role for path in paths
                ),
                "missing_edges": tuple(missing_edges),
                "undeclared_sources": undeclared_sources,
                "live_live_path_roles": live_live_roles,
            }
            result.append(entry)
            if missing_edges:
                failures.append(
                    f"{stage.name}.{_type_label(target)} missing {missing_edges}"
                )
            if undeclared_sources:
                failures.append(
                    f"{stage.name}.{_type_label(target)} has undeclared "
                    f"sources {undeclared_sources}"
                )
            if live_live_roles:
                failures.append(
                    f"{stage.name}.{_type_label(target)} has live live paths "
                    f"{live_live_roles}"
                )
    if failures:
        raise StrictFlowCompilationError(
            "strict covariant edge audit failed: " + "; ".join(failures),
            edge_manifest=result,
            candidate_manifest=model.candidate_manifest,
        )
    return tuple(result)


def build_strict_dual_stream_flow(
    generators: Tensor,
    config: StrictDualStreamFlowConfig | Mapping[str, Any],
    *,
    generator_names: Sequence[str] | None = None,
) -> InvariantGatePipelineV2:
    """Compile a strict flow and prove every declared typed edge is active."""
    resolved = (
        StrictDualStreamFlowConfig.from_dict(config)
        if isinstance(config, Mapping)
        else config
    )
    if not isinstance(resolved, StrictDualStreamFlowConfig):
        raise TypeError("config must be StrictDualStreamFlowConfig or mapping")
    lowered = compile_strict_dual_stream_config(resolved)
    try:
        model = build_invariant_gate_pipeline_v2(
            generators,
            lowered,
            generator_names=generator_names,
        )
    except PipelineV2CompilationError as error:
        raise StrictFlowCompilationError(
            "strict flow path compilation failed",
            candidate_manifest=error.candidate_manifest,
        ) from error
    edge_manifest = _strict_edge_manifest(model)
    model.strict_flow_manifest = MappingProxyType(
        {
            "schema_version": 1,
            "mathematical_contract": {
                "raw_covariant_ingress": {"x": "a1:A", "R_via_pose_r": "b1:B"},
                "covariant_visibility": "declared parent edges only",
                "invariant_visibility": "raw x,r plus declared hidden parents",
                "same_type_flow": "legacy bypass plus gated unary and symmetric2 paths",
                "cross_type_flow": "compiled unary and symmetric2 intertwiners",
                "joint_hidden_tensor_products": False,
                "b_channel_semantics": "channels per registered B TypeKey",
                "group_convolution": False,
            },
            "config": resolved.as_dict(),
            "edge_audit": edge_manifest,
            "candidate_manifest": model.candidate_manifest,
            "selected_covariant_roles": model.selected_covariant_roles,
            "descriptor_role_manifest": model.descriptor_role_manifest,
            "coefficient_head_role_manifest": (
                model.coefficient_head_role_manifest
            ),
            "trainable_parameter_count": model.trainable_parameter_count,
        }
    )
    return model
