"""Compile fixed-shape G-series mechanism cells around E311."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from TFENN.models import build_invariant_gate_pipeline_v2

from experiments.benzene_pair.e_series.model_factory import (
    build_e_series_model,
    pipeline_config_from_spec,
)
from experiments.benzene_pair.e_series.catalog import get_model_spec as get_e_spec

from .catalog import GModelSpec, get_model_spec


__all__ = ["build_g_series_model"]


def _build_union(
    generators: Tensor,
    generator_names: Sequence[str] | None,
) -> nn.Module:
    """Compile one FULL C17 union while retaining E311 scalar descriptors."""
    reference_config = pipeline_config_from_spec(get_e_spec("E311"))
    union = reference_config.as_dict()
    union["architecture_id"] = "benzene_pair_g_series_e311_fixed_shape_union"
    union["implemented_mechanism"] = "E311 fixed-shape causal audit union"
    for stage in union["stages"]:
        stage["covariant_include_raw_mixed_pairs"] = True
        stage["covariant_include_stf_shortcuts"] = True
        # These are scalar Gate descriptors, not the covariant treatment.
        stage["invariant_include_raw_mixed_pairs"] = True
        stage["invariant_include_stf_shortcuts"] = True
    return build_invariant_gate_pipeline_v2(
        generators,
        union,
        generator_names=generator_names,
    )


def _copy_module(source: nn.Module, target: nn.Module, *, role: str) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    if set(source_state) != set(target_state):
        raise RuntimeError(f"G transplant state keys differ for {role}")
    for name in source_state:
        if source_state[name].shape != target_state[name].shape:
            raise RuntimeError(f"G transplant tensor shape differs for {role}.{name}")
    target.load_state_dict(source_state, strict=True)


def _projection_rows(model: nn.Module) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in model.projection_input_role_manifest:
        row = dict(item)
        key = (str(row["projection_role"]), str(row["kind"]), str(row["role"]))
        if key in rows:
            raise RuntimeError(f"duplicate projection input role {key}")
        rows[key] = row
    return rows


def _transplant_e311_initialization(
    reference: nn.Module, union: nn.Module
) -> dict[str, int]:
    """Copy every common E311 role without reusing union fan-in scaling."""
    if set(reference.stage_trunks) != set(union.stage_trunks):
        raise RuntimeError("G union stage trunks do not match E311")
    for stage_name in reference.stage_trunks:
        _copy_module(
            reference.stage_trunks[stage_name],
            union.stage_trunks[stage_name],
            role=f"trunk.{stage_name}",
        )

    reference_heads = reference.coefficient_head_modules_by_role()
    union_heads = union.coefficient_head_modules_by_role()
    missing_heads = set(reference_heads).difference(union_heads)
    if missing_heads:
        raise RuntimeError(
            f"G union is missing E311 heads {tuple(sorted(missing_heads))}"
        )
    for role, source in reference_heads.items():
        _copy_module(source, union_heads[role], role=f"head.{role}")

    reference_projections = reference.channel_projection_modules_by_role()
    union_projections = union.channel_projection_modules_by_role()
    if set(reference_projections) != set(union_projections):
        raise RuntimeError("G union projection targets do not match E311")
    reference_rows = _projection_rows(reference)
    union_rows = _projection_rows(union)
    missing_rows = set(reference_rows).difference(union_rows)
    if missing_rows:
        raise RuntimeError(
            f"G union is missing E311 projection roles {tuple(sorted(missing_rows))}"
        )
    copied_columns = 0
    with torch.no_grad():
        for key, source_row in reference_rows.items():
            target_row = union_rows[key]
            source = reference_projections[str(source_row["projection_role"])]
            target = union_projections[str(target_row["projection_role"])]
            source_weight = getattr(source, "weight", None)
            target_weight = getattr(target, "weight", None)
            if not isinstance(source_weight, Tensor) or not isinstance(
                target_weight, Tensor
            ):
                raise TypeError("G role transplant requires dense channel projections")
            source_start, source_stop = int(source_row["start"]), int(
                source_row["stop"]
            )
            target_start, target_stop = int(target_row["start"]), int(
                target_row["stop"]
            )
            if source_stop - source_start != target_stop - target_start:
                raise RuntimeError(f"G projection role width differs for {key}")
            if source_weight.shape[0] != target_weight.shape[0]:
                raise RuntimeError(f"G projection output width differs for {key}")
            target_weight[:, target_start:target_stop].copy_(
                source_weight[:, source_start:source_stop]
            )
            copied_columns += target_stop - target_start
    return {
        "stage_trunk_count": len(reference.stage_trunks),
        "coefficient_head_count": len(reference_heads),
        "projection_role_count": len(reference_rows),
        "projection_column_count": copied_columns,
    }


def _carrier_group(stage: str, source: str) -> str:
    if (stage, source) in {("a1", "x"), ("b1", "r")}:
        return "stem"
    if (stage, source) == ("b2", "b1"):
        return "adjacent"
    if (stage, source) in {("b2", "r"), ("out", "x")}:
        return "raw_deep"
    if (stage, source) == ("out", "a1"):
        return "hidden_deep"
    raise RuntimeError(f"unexpected E311 carrier edge {stage}:{source}")


def _apply_treatment(model: nn.Module, spec: GModelSpec) -> None:
    options = spec.options
    generic_pair_enabled = bool(options["generic_pair_enabled"])
    stf_by_stage = {
        "a1": bool(options["stf_a1_enabled"]),
        "out": bool(options["stf_out_enabled"]),
    }
    for item in model.coefficient_head_role_manifest:
        role = str(item["role"])
        family = str(item["path_family"])
        active = True
        if family == "pair":
            active = generic_pair_enabled
        elif family == "stf":
            stage = str(item["stage"])
            if stage not in stf_by_stage:
                raise RuntimeError(f"unexpected covariant STF stage {stage}")
            active = stf_by_stage[stage]
        model.set_covariant_path_activity(role, active)

    carrier_mode = str(options["carrier_mode"])
    gate_mode = "direct"
    if carrier_mode == "gated_identity":
        gate_mode = str(options["gated_identity_initialization"])
    model.configure_legacy_carrier_gates(gate_mode)
    carrier_mask = dict(options["carrier_group_mask"])
    seen_groups: set[str] = set()
    for item in model.projection_input_role_manifest:
        if item["kind"] != "legacy_carrier":
            continue
        group = _carrier_group(str(item["stage"]), str(item["source_names"][0]))
        seen_groups.add(group)
        active = carrier_mode != "none" and bool(carrier_mask[group])
        model.set_legacy_carrier_activity(str(item["role"]), active)
    if seen_groups != set(carrier_mask):
        raise RuntimeError("G carrier treatment did not cover every registered group")


def _module_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _active_parameter_audit(model: nn.Module) -> dict[str, int]:
    """Count nominal tensors separately from causally active scalar entries."""
    trunk_nominal = sum(
        _module_parameter_count(module) for module in model.stage_trunks.values()
    )
    head_modules = model.coefficient_head_modules_by_role()
    head_rows = {
        str(item["role"]): dict(item)
        for item in model.coefficient_head_role_manifest
    }
    head_nominal = sum(
        _module_parameter_count(module) for module in head_modules.values()
    )
    head_active = sum(
        _module_parameter_count(head_modules[role])
        for role, row in head_rows.items()
        if bool(row["active"])
    )

    projection_modules = model.channel_projection_modules_by_role()
    projection_rows: dict[str, list[dict[str, Any]]] = {}
    for item in model.projection_input_role_manifest:
        row = dict(item)
        projection_rows.setdefault(str(row["projection_role"]), []).append(row)
    projection_nominal = 0
    projection_active = 0
    for role, module in projection_modules.items():
        weight = getattr(module, "weight", None)
        parameters = tuple(module.parameters())
        if (
            not isinstance(weight, Tensor)
            or len(parameters) != 1
            or parameters[0] is not weight
        ):
            raise TypeError(
                "G active-parameter audit requires bias-free dense projections"
            )
        projection_nominal += int(weight.numel())
        active_columns = sum(
            int(row["stop"]) - int(row["start"])
            for row in projection_rows[role]
            if bool(row["active"])
        )
        projection_active += int(weight.shape[0]) * active_columns

    carrier_modules = model.legacy_carrier_gate_modules_by_role()
    carrier_rows = {
        str(item["role"]): dict(item)
        for item in model.projection_input_role_manifest
        if item["kind"] == "legacy_carrier"
    }
    carrier_nominal = sum(
        _module_parameter_count(module) for module in carrier_modules.values()
    )
    carrier_active = sum(
        _module_parameter_count(carrier_modules[role])
        for role, row in carrier_rows.items()
        if bool(row["active"])
        and row["gate_mode"] in {"residual_zero", "default"}
    )

    nominal = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    component_nominal = (
        trunk_nominal + head_nominal + projection_nominal + carrier_nominal
    )
    if component_nominal != nominal:
        raise RuntimeError("G active-parameter audit did not cover every parameter")
    active = trunk_nominal + head_active + projection_active + carrier_active
    if not 0 < active <= nominal:
        raise RuntimeError("G active-parameter audit produced an invalid count")
    return {
        "nominal_trainable_parameter_count": nominal,
        "causally_active_parameter_scalar_count": active,
        "inactive_parameter_scalar_count": nominal - active,
        "active_trunk_parameter_count": trunk_nominal,
        "active_coefficient_head_parameter_count": head_active,
        "active_projection_parameter_scalar_count": projection_active,
        "active_carrier_gate_parameter_count": carrier_active,
    }


def build_g_series_model(
    model: str | GModelSpec,
    generators: Tensor,
    *,
    generator_names: Sequence[str] | None = None,
) -> nn.Module:
    """Build one G cell with common shape and role-transplanted E311 weights."""
    spec = get_model_spec(model) if isinstance(model, str) else model
    if not isinstance(spec, GModelSpec):
        raise TypeError("model must be an identifier or GModelSpec")

    initial_rng = torch.random.get_rng_state()
    union = _build_union(generators, generator_names)
    extension_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(initial_rng)
    try:
        reference = build_e_series_model(
            "E311",
            generators,
            generator_names=generator_names,
        )
    finally:
        torch.random.set_rng_state(extension_rng)
    transplant = _transplant_e311_initialization(reference, union)
    del reference
    _apply_treatment(union, spec)

    coefficient_manifest = tuple(union.coefficient_head_role_manifest)
    projection_manifest = tuple(union.projection_input_role_manifest)
    parameter_audit = _active_parameter_audit(union)
    union.g_series_manifest = {
        "schema_name": "tfenn_g_series_model",
        "schema_version": 1,
        "model_id": spec.model_id,
        "variant_id": spec.variant_id,
        "seed_index": spec.seed_index,
        "reference_model_id": "E311",
        "fixed_shape_supernet": True,
        "native_e311_initialization_transplant": transplant,
        **parameter_audit,
        "active_covariant_path_count": sum(
            bool(item["active"]) for item in coefficient_manifest
        ),
        "inactive_covariant_path_count": sum(
            not bool(item["active"]) for item in coefficient_manifest
        ),
        "active_carrier_role_count": sum(
            bool(item["active"])
            for item in projection_manifest
            if item["kind"] == "legacy_carrier"
        ),
        "inactive_carrier_role_count": sum(
            not bool(item["active"])
            for item in projection_manifest
            if item["kind"] == "legacy_carrier"
        ),
        "causal_masks": union.causal_mask_manifest,
    }
    return union
