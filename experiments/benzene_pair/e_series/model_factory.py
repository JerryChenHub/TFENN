"""Compile E series catalog entries into trainable models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from TFENN.models import (
    InvariantGatePipelineV2Config,
    InvariantGateStageV2Config,
    OrdinaryRawMLPConfig,
    build_ordinary_raw_mlp,
    build_invariant_gate_pipeline_v2,
)
from TFENN.models.model_level_group_conv_mlp import (
    ModelLevelGroupConvMLPConfig,
    build_model_level_group_conv_mlp,
)

from .catalog import EModelSpec, get_model_spec


__all__ = [
    "build_e_series_model",
    "pipeline_config_from_spec",
]


def _skip_sources(
    stage: Mapping[str, Any],
    stream_by_name: Mapping[str, str],
) -> tuple[str, ...] | None:
    policy = str(stage["skip_policy"])
    if policy == "legacy":
        return None
    if policy == "none":
        return ()
    matching = tuple(
        source
        for source in stage["source_names"]
        if stream_by_name.get(str(source)) == stage["output_stream"]
    )
    if policy in {"id", "local_proj"}:
        return matching[-1:]
    if policy == "dense_proj":
        return matching
    raise ValueError(f"unknown skip policy {policy}")


def pipeline_config_from_spec(spec: EModelSpec) -> InvariantGatePipelineV2Config:
    """Translate a sequential or synchronous catalog row into a V2 config."""
    if not isinstance(spec, EModelSpec):
        raise TypeError("spec must be an EModelSpec")
    if spec.family not in {"sequential_gate", "synchronous_dual_stream"}:
        raise TypeError("this E model family does not use a direct pipeline config")
    path_policy = str(spec.options.get("path_policy", "FULL"))
    raw_mixed = path_policy != "NO_RAW_MIXED"
    stages: list[InvariantGateStageV2Config] = []
    stream_by_name: dict[str, str] = {"x": "A", "r": "B"}
    for value in spec.options["stages"]:
        stage = dict(value)
        config = InvariantGateStageV2Config(
            name=str(stage["name"]),
            output_stream=str(stage["output_stream"]),
            source_names=tuple(stage["source_names"]),
            channels=int(stage["channels"]),
            invariant_source_names=tuple(stage["invariant_source_names"]),
            skip_source_names=_skip_sources(stage, stream_by_name),
            trunk_width=int(stage["trunk_width"]),
            activation="silu",
            include_symmetric_unary=True,
            include_raw_mixed_pairs=True,
            include_stf_shortcuts=True,
            skip_policy=str(stage["skip_policy"]),
            covariant_include_symmetric_unary=True,
            covariant_include_raw_mixed_pairs=raw_mixed,
            covariant_include_stf_shortcuts=True,
            invariant_include_symmetric_unary=True,
            invariant_include_raw_mixed_pairs=True,
            invariant_include_stf_shortcuts=True,
            coefficient_activation="identity",
            coefficient_head="dense",
            descriptor_mask="full",
            execution_level=stage.get("execution_level"),
            covariant_live_mixed_only=bool(
                stage.get("covariant_live_mixed_only", False)
            ),
            covariant_path_quota=stage.get("path_head_quota"),
            parameter_share_group=stage.get("parameter_share_group"),
            channel_projection=str(stage.get("channel_projection", "dense")),
            channel_projection_rank=stage.get("channel_projection_rank"),
            path_aggregation=str(stage.get("path_aggregation", "linear")),
            path_temperature=float(stage.get("path_temperature", 1.0)),
        )
        stages.append(config)
        stream_by_name[config.name] = config.output_stream
    return InvariantGatePipelineV2Config(
        stages=tuple(stages),
        output_stage="out",
        architecture_id=f"benzene_pair_e_series_{spec.model_id.lower()}",
        anchor_ranks=(2, 6),
        max_constraint_entries=10_000_000,
        degree3_overflow_policy="raise",
        implemented_mechanism=spec.architecture_name,
    )


def _build_reference(
    reference: str,
    generators: Tensor,
    generator_names: Sequence[str] | None,
) -> nn.Module:
    if reference.startswith("C"):
        from experiments.benzene_pair.invariant_gate_v2_20k_sweep import (
            build_sweep_model,
        )

        return build_sweep_model(
            reference,
            generators,
            generator_names=generator_names,
        )
    if reference.startswith("D"):
        from experiments.benzene_pair.d_series.model_factory import (
            build_d_series_model,
        )

        return build_d_series_model(
            reference,
            generators,
            generator_names=generator_names,
        )
    raise ValueError(f"unknown reference model {reference}")


def _build_budget_compiled(
    spec: EModelSpec,
    generators: Tensor,
    generator_names: Sequence[str] | None,
) -> nn.Module:
    from TFENN.models import build_budget_compiled_invariant_gate

    return build_budget_compiled_invariant_gate(
        generators,
        model_id=spec.model_id,
        architecture_name=spec.architecture_name,
        mechanism={
            key: value
            for key, value in spec.options.items()
            if key != "compiled_preflight"
        },
        target_range=spec.target_parameter_range,
        generator_names=generator_names,
    )


def build_e_series_model(
    model: str | EModelSpec,
    generators: Tensor,
    *,
    generator_names: Sequence[str] | None = None,
) -> nn.Module:
    """Compile one E series model and validate its required budget range."""
    spec = get_model_spec(model) if isinstance(model, str) else model
    if not isinstance(spec, EModelSpec):
        raise TypeError("model must be an identifier or EModelSpec")
    if spec.family == "raw_mlp":
        result: nn.Module = build_ordinary_raw_mlp(
            OrdinaryRawMLPConfig(
                hidden_widths=tuple(spec.options["hidden_widths"]),
                activation="silu",
                distance_scale=6.0,
                seed=int(spec.options["seed"]),
            ),
        )
    elif spec.family == "group_mlp":
        result = build_model_level_group_conv_mlp(
            generators,
            ModelLevelGroupConvMLPConfig(
                hidden_widths=tuple(spec.options["hidden_widths"]),
                activation="silu",
                distance_scale=6.0,
                seed=int(spec.options["seed"]),
            ),
        )
    elif spec.family == "reference":
        result = _build_reference(
            str(spec.options["reference_model_id"]),
            generators,
            generator_names,
        )
    elif spec.family in {"sequential_gate", "synchronous_dual_stream"}:
        result = build_invariant_gate_pipeline_v2(
            generators,
            pipeline_config_from_spec(spec),
            generator_names=generator_names,
        )
    elif spec.family == "budget_compiled_gate":
        result = _build_budget_compiled(spec, generators, generator_names)
    else:
        raise ValueError(f"unknown E model family {spec.family}")
    actual = sum(
        parameter.numel()
        for parameter in result.parameters()
        if parameter.requires_grad
    )
    if spec.family in {"raw_mlp", "group_mlp", "reference"}:
        if actual != spec.planned_parameter_count:
            raise RuntimeError(
                f"{spec.model_id} compiled with {actual} parameters instead of "
                f"its exact reference count {spec.planned_parameter_count}"
            )
    if spec.target_parameter_range is not None:
        lower, upper = spec.target_parameter_range
        if not lower <= actual <= upper:
            raise RuntimeError(
                f"{spec.model_id} compiled with {actual} parameters outside "
                f"the required range {lower} through {upper}"
            )
    return result
