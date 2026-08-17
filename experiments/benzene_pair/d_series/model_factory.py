"""Translate D series catalog entries into version two pipeline models."""

from __future__ import annotations

from typing import Sequence

from torch import Tensor

from TFENN.models import (
    InvariantGatePipelineV2,
    InvariantGatePipelineV2Config,
    InvariantGateStageV2Config,
    build_invariant_gate_pipeline_v2,
)

from .catalog import DModelSpec, get_model_spec


__all__ = ["build_d_series_model", "pipeline_config_from_spec"]


_STAGE_OPTION_NAMES = (
    "skip_policy",
    "covariant_include_symmetric_unary",
    "covariant_include_raw_mixed_pairs",
    "covariant_include_stf_shortcuts",
    "invariant_include_symmetric_unary",
    "invariant_include_raw_mixed_pairs",
    "invariant_include_stf_shortcuts",
    "degree3_policy",
    "coefficient_activation",
    "coefficient_head",
    "coefficient_rank",
    "descriptor_mask",
    "trunk_depth",
    "trunk_linearized",
    "trunk_residual",
    "metric_gate",
)


def pipeline_config_from_spec(spec: DModelSpec) -> InvariantGatePipelineV2Config:
    """Create the executable pipeline configuration for one catalog entry."""
    if not isinstance(spec, DModelSpec):
        raise TypeError("spec must be a DModelSpec")
    stage_options = {name: spec.options[name] for name in _STAGE_OPTION_NAMES}
    stages = tuple(
        InvariantGateStageV2Config(
            name=stage.name,
            output_stream=stage.output_stream,
            source_names=stage.source_names,
            channels=stage.channels,
            invariant_source_names=stage.invariant_source_names,
            skip_source_names=stage.skip_source_names,
            trunk_width=stage.trunk_width,
            activation="silu",
            include_symmetric_unary=True,
            include_raw_mixed_pairs=True,
            include_stf_shortcuts=True,
            **stage_options,
        )
        for stage in spec.stages
    )
    return InvariantGatePipelineV2Config(
        stages=stages,
        output_stage="out",
        architecture_id=f"benzene_pair_d_series_{spec.model_id.lower()}",
        anchor_ranks=(2, 6),
        max_constraint_entries=int(spec.options["max_constraint_entries"]),
        degree3_overflow_policy=str(spec.options["degree3_overflow_policy"]),
    )


def build_d_series_model(
    model: str | DModelSpec,
    generators: Tensor,
    *,
    generator_names: Sequence[str] | None = None,
) -> InvariantGatePipelineV2:
    """Compile one D series model and verify any planned parameter count."""
    spec = get_model_spec(model) if isinstance(model, str) else model
    if not isinstance(spec, DModelSpec):
        raise TypeError("model must be an identifier or DModelSpec")
    result = build_invariant_gate_pipeline_v2(
        generators,
        pipeline_config_from_spec(spec),
        generator_names=generator_names,
    )
    expected = spec.expected_parameter_count
    if expected is not None and result.trainable_parameter_count != expected:
        raise RuntimeError(
            f"{spec.model_id} compiled with {result.trainable_parameter_count} "
            f"parameters instead of {expected}"
        )
    return result
