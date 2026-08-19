"""Compile F series catalog entries into trainable models."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from TFENN.models import (
    StrictDualStreamFlowConfig,
    StrictFlowStageConfig,
    build_strict_dual_stream_flow,
)

from .catalog import FModelSpec, get_model_spec


__all__ = [
    "build_f_series_model",
    "strict_config_from_spec",
]


def strict_config_from_spec(spec: FModelSpec) -> StrictDualStreamFlowConfig:
    """Translate one strict catalog row to the public compiler contract."""
    if not isinstance(spec, FModelSpec):
        raise TypeError("spec must be an FModelSpec")
    if spec.family != "strict_flow":
        raise TypeError("only strict flow models have a strict config")
    stages = tuple(
        StrictFlowStageConfig.from_dict(dict(value)) for value in spec.options["stages"]
    )
    return StrictDualStreamFlowConfig(
        stages=stages,
        output_stage="out",
        architecture_id=f"benzene_pair_f_series_{spec.model_id.lower()}",
        descriptor_mask=spec.descriptor_mask,
        gate_width=int(spec.options["gate_width"]),
        anchor_ranks=(2, 6),
        max_constraint_entries=10_000_000,
        max_gate_coefficients=2_000_000,
        max_invariant_channels=20_000,
    )


def _build_reference(
    spec: FModelSpec,
    generators: Tensor,
    generator_names: Sequence[str] | None,
) -> nn.Module:
    from experiments.benzene_pair.e_series.model_factory import build_e_series_model

    return build_e_series_model(
        str(spec.options["reference_model_id"]),
        generators,
        generator_names=generator_names,
    )


def build_f_series_model(
    model: str | FModelSpec,
    generators: Tensor,
    *,
    generator_names: Sequence[str] | None = None,
) -> nn.Module:
    """Compile one historical or strict F series model."""
    spec = get_model_spec(model) if isinstance(model, str) else model
    if not isinstance(spec, FModelSpec):
        raise TypeError("model must be an identifier or FModelSpec")
    if spec.family == "reference":
        result = _build_reference(spec, generators, generator_names)
    elif spec.family == "strict_flow":
        result = build_strict_dual_stream_flow(
            generators,
            strict_config_from_spec(spec),
            generator_names=generator_names,
        )
    else:
        raise ValueError(f"unknown F model family {spec.family}")
    actual = sum(
        parameter.numel()
        for parameter in result.parameters()
        if parameter.requires_grad
    )
    expected = spec.expected_parameter_count
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"{spec.model_id} compiled with {actual} parameters instead of {expected}"
        )
    return result
