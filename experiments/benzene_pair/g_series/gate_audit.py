"""Export role-labelled Invariant Gate diagnostics for selected G checkpoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from experiments.benzene_pair import sweep30 as common


VALIDATION_PROBE_COUNT = 10_000
VALIDATION_PROBE_BATCH_SIZE = 1_000
VALIDATION_PROBE_SEED = 20260822


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _magnitude_stats(value: Tensor) -> dict[str, float]:
    flat = value.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise RuntimeError("Gate audit received empty or nonfinite values")
    return {
        "mean_abs": float(flat.abs().mean()),
        "rms": float(torch.sqrt(flat.square().mean())),
        "max_abs": float(flat.abs().max()),
    }


def _trunk_column_statistics(model: nn.Module) -> list[dict[str, Any]]:
    manifest = tuple(getattr(model, "descriptor_role_manifest"))
    modules = model.first_trunk_linear_modules_by_stage()
    result = []
    for item in manifest:
        row = dict(item)
        stage = str(row["stage"])
        start = int(row["start"])
        stop = int(row["stop"])
        linear = modules.get(stage)
        if not isinstance(linear, nn.Linear):
            raise RuntimeError(f"stage {stage} has no public first trunk linear layer")
        if start < 0 or stop <= start or stop > linear.weight.shape[1]:
            raise RuntimeError(f"stage {stage} descriptor slice is invalid")
        result.append(
            {
                **_json_value(row),
                "column_count": stop - start,
                "ranking_eligible": bool(row["active"]),
                "weight": _magnitude_stats(linear.weight[:, start:stop]),
            }
        )
    return result


def _coefficient_parameter_statistics(model: nn.Module) -> list[dict[str, Any]]:
    manifest = tuple(getattr(model, "coefficient_head_role_manifest"))
    modules = model.coefficient_head_modules_by_role()
    result = []
    for item in manifest:
        row = dict(item)
        role = str(row["role"])
        module = modules.get(role)
        if not isinstance(module, nn.Linear) or module.bias is None:
            raise RuntimeError(f"G dense coefficient head {role} is unavailable")
        target_channels = int(row["target_channels"])
        primitive_channels = int(row["primitive_channels"])
        expected = target_channels * primitive_channels
        if module.weight.shape[0] != expected or module.bias.shape != (expected,):
            raise RuntimeError(
                f"G coefficient head {role} shape does not match manifest"
            )
        weight = module.weight.reshape(
            target_channels,
            primitive_channels,
            module.weight.shape[1],
        )
        bias = module.bias.reshape(target_channels, primitive_channels)
        for primitive_index in range(primitive_channels):
            result.append(
                {
                    **_json_value(row),
                    "primitive_index": primitive_index,
                    "primitive_index_semantics": (
                        "flattened intertwiner-basis and source-channel index"
                    ),
                    "ranking_eligible": bool(row.get("active", True)),
                    "weight": _magnitude_stats(weight[:, primitive_index, :]),
                    "bias": _magnitude_stats(bias[:, primitive_index]),
                }
            )
    return result


def _legacy_carrier_parameter_statistics(model: nn.Module) -> list[dict[str, Any]]:
    method = getattr(model, "legacy_carrier_gate_modules_by_role", None)
    projection_manifest = getattr(model, "projection_input_role_manifest", ())
    if not callable(method):
        return []
    modules = method()
    rows = {
        str(item["role"]): dict(item)
        for item in projection_manifest
        if item.get("kind") == "legacy_carrier"
    }
    result = []
    for role, module in modules.items():
        if not isinstance(module, nn.Linear) or module.bias is None:
            raise RuntimeError(f"G legacy carrier Gate head {role} is unavailable")
        row = rows.get(str(role), {"role": str(role)})
        result.append(
            {
                **_json_value(row),
                "ranking_eligible": bool(row.get("active", True))
                and row.get("gate_mode") in {"residual_zero", "default"},
                "weight": _magnitude_stats(module.weight),
                "bias": _magnitude_stats(module.bias),
            }
        )
    return result


def _head_composed_weight_statistics(model: nn.Module) -> list[dict[str, Any]]:
    """Describe ``V @ W`` while explicitly leaving out sample SiLU derivatives."""
    descriptor_manifest = tuple(getattr(model, "descriptor_role_manifest"))
    head_manifest = tuple(getattr(model, "coefficient_head_role_manifest"))
    trunks = model.first_trunk_linear_modules_by_stage()
    heads = model.coefficient_head_modules_by_role()
    descriptors_by_stage: dict[str, list[dict[str, Any]]] = {}
    for item in descriptor_manifest:
        descriptors_by_stage.setdefault(str(item["stage"]), []).append(dict(item))
    result = []
    for item in head_manifest:
        head_row = dict(item)
        stage = str(head_row["stage"])
        role = str(head_row["role"])
        trunk = trunks.get(stage)
        head = heads.get(role)
        if not isinstance(trunk, nn.Linear) or not isinstance(head, nn.Linear):
            raise RuntimeError(f"Gate parameter chain for {role} is unavailable")
        if head.weight.shape[1] != trunk.weight.shape[0]:
            raise RuntimeError(
                f"Gate parameter chain for {role} has incompatible shapes"
            )
        for descriptor in descriptors_by_stage.get(stage, ()):
            start = int(descriptor["start"])
            stop = int(descriptor["stop"])
            composed = head.weight @ trunk.weight[:, start:stop]
            result.append(
                {
                    "stage": stage,
                    "head_role": role,
                    "head_path_family": str(head_row["path_family"]),
                    "head_target": str(head_row["target"]),
                    "descriptor_role": str(descriptor["role"]),
                    "descriptor_kind": str(descriptor["kind"]),
                    "descriptor_source_names": _json_value(
                        descriptor["source_names"]
                    ),
                    "descriptor_column_count": stop - start,
                    "head_active": bool(head_row.get("active", True)),
                    "ranking_eligible": bool(descriptor["active"])
                    and bool(head_row.get("active", True)),
                    "composed_weight": _magnitude_stats(composed),
                    "interpretation": (
                        "parameter-chain V@W; descriptive only because the "
                        "sample-dependent SiLU derivative is not included"
                    ),
                }
            )
    carrier_method = getattr(model, "legacy_carrier_gate_modules_by_role", None)
    projection_manifest = getattr(model, "projection_input_role_manifest", ())
    if callable(carrier_method):
        carrier_rows = {
            str(item["role"]): dict(item)
            for item in projection_manifest
            if item.get("kind") == "legacy_carrier"
        }
        for role, head in carrier_method().items():
            row = carrier_rows[str(role)]
            stage = str(row["stage"])
            trunk = trunks.get(stage)
            if not isinstance(trunk, nn.Linear) or not isinstance(head, nn.Linear):
                raise RuntimeError(
                    f"G carrier Gate parameter chain for {role} is unavailable"
                )
            if head.weight.shape[1] != trunk.weight.shape[0]:
                raise RuntimeError(
                    f"G carrier Gate parameter chain for {role} has incompatible shapes"
                )
            for descriptor in descriptors_by_stage.get(stage, ()):
                start = int(descriptor["start"])
                stop = int(descriptor["stop"])
                composed = head.weight @ trunk.weight[:, start:stop]
                head_active = bool(row.get("active", True)) and row.get(
                    "gate_mode"
                ) in {"residual_zero", "default"}
                result.append(
                    {
                        "stage": stage,
                        "head_role": str(role),
                        "head_path_family": "legacy_carrier_gate",
                        "head_target": str(row["target"]),
                        "descriptor_role": str(descriptor["role"]),
                        "descriptor_kind": str(descriptor["kind"]),
                        "descriptor_source_names": _json_value(
                            descriptor["source_names"]
                        ),
                        "descriptor_column_count": stop - start,
                        "head_active": head_active,
                        "ranking_eligible": bool(descriptor["active"])
                        and head_active,
                        "composed_weight": _magnitude_stats(composed),
                        "interpretation": (
                            "carrier-head parameter-chain V@W; descriptive only "
                            "because the sample-dependent SiLU derivative and Gate "
                            "activation are not included"
                        ),
                    }
                )
    return result


def _gate_parameter_snapshot(model: nn.Module, spec: Any) -> dict[str, Any]:
    trunks = {
        stage: {
            name: value.detach().cpu()
            for name, value in module.state_dict().items()
        }
        for stage, module in model.stage_trunks.items()
    }
    heads = {
        role: {
            name: value.detach().cpu()
            for name, value in module.state_dict().items()
        }
        for role, module in model.coefficient_head_modules_by_role().items()
    }
    carrier_method = getattr(model, "legacy_carrier_gate_modules_by_role", None)
    carrier_heads = (
        {}
        if not callable(carrier_method)
        else {
            role: {
                name: value.detach().cpu()
                for name, value in module.state_dict().items()
            }
            for role, module in carrier_method().items()
        }
    )
    return {
        "schema_name": "tfenn_g_series_invariant_gate_parameters",
        "schema_version": 1,
        "model_id": str(spec.model_id),
        "variant_id": int(getattr(spec, "variant_id", 0)),
        "seed_index": int(getattr(spec, "seed_index", 0)),
        "descriptor_role_manifest": _json_value(model.descriptor_role_manifest),
        "coefficient_head_role_manifest": _json_value(
            model.coefficient_head_role_manifest
        ),
        "stage_trunks": trunks,
        "coefficient_heads_by_role": heads,
        "legacy_carrier_gate_heads_by_role": carrier_heads,
        "typed_channel_projections": {
            name: value.detach().cpu()
            for name, value in model.channel_projections.state_dict().items()
        },
        "g_series_manifest": _json_value(getattr(model, "g_series_manifest", {})),
        "causal_mask_manifest": _json_value(
            getattr(model, "causal_mask_manifest", {})
        ),
        "projection_input_role_manifest": _json_value(
            getattr(model, "projection_input_role_manifest", ())
        ),
    }


def _validation_gamma_statistics(
    model: nn.Module,
    data: Any,
    split: common.SplitIndices,
    *,
    device: str,
) -> tuple[int, list[dict[str, Any]]]:
    was_training = model.training
    model.eval()
    manifest = {
        str(item["role"]): dict(item)
        for item in getattr(model, "coefficient_head_role_manifest")
    }
    generator = torch.Generator(device="cpu").manual_seed(VALIDATION_PROBE_SEED)
    order = torch.randperm(int(split.validation.numel()), generator=generator)
    probe = split.validation[
        order[: min(VALIDATION_PROBE_COUNT, int(split.validation.numel()))]
    ]
    if int(probe.numel()) < 1:
        raise RuntimeError("G Gate audit validation probe is empty")
    accumulators = {
        role: {
            "count": 0,
            "sum": 0.0,
            "sum_abs": 0.0,
            "sum_square": 0.0,
            "branch_count": 0,
            "branch_sum_square": 0.0,
        }
        for role in manifest
    }
    try:
        for start in range(0, int(probe.numel()), VALIDATION_PROBE_BATCH_SIZE):
            selection = probe[start : start + VALIDATION_PROBE_BATCH_SIZE]
            centers, frames, _target = common._batch_inputs(data, selection, device)
            with torch.no_grad():
                values = model.collect_coefficient_activations(centers, frames)
                branch_values = model.debug_forward(centers, frames).direct_paths
            if set(values) != set(manifest):
                raise RuntimeError(
                    "G gamma roles do not match the coefficient manifest"
                )
            active_roles = {
                role
                for role, row in manifest.items()
                if bool(row.get("active", True))
            }
            if not active_roles.issubset(branch_values):
                raise RuntimeError(
                    "G active branch roles do not match the coefficient manifest"
                )
            for role, value in values.items():
                flat = value.detach().to(
                    dtype=torch.float64,
                    device="cpu",
                ).reshape(-1)
                if not bool(torch.isfinite(flat).all()):
                    raise RuntimeError(
                        f"G gamma role {role} contains nonfinite values"
                    )
                accumulator = accumulators[role]
                accumulator["count"] += int(flat.numel())
                accumulator["sum"] += float(flat.sum())
                accumulator["sum_abs"] += float(flat.abs().sum())
                accumulator["sum_square"] += float(flat.square().sum())
                branch_value = branch_values.get(role)
                if branch_value is None:
                    if bool(manifest[role].get("active", True)):
                        raise RuntimeError(
                            f"G active branch role {role} is unavailable"
                        )
                    accumulator["branch_count"] += 1
                    continue
                branch = branch_value.detach().to(
                    dtype=torch.float64, device="cpu"
                ).reshape(-1)
                if not bool(torch.isfinite(branch).all()):
                    raise RuntimeError(
                        f"G branch role {role} contains nonfinite values"
                    )
                accumulator["branch_count"] += int(branch.numel())
                accumulator["branch_sum_square"] += float(branch.square().sum())
    finally:
        model.train(was_training)
    result = []
    for role, accumulator in accumulators.items():
        count = int(accumulator["count"])
        branch_count = int(accumulator["branch_count"])
        if count < 1 or branch_count < 1:
            raise RuntimeError(f"G gamma role {role} has no observations")
        mean = float(accumulator["sum"]) / count
        mean_square = float(accumulator["sum_square"]) / count
        result.append(
            {
                **_json_value(manifest[role]),
                "observation_count": count,
                "mean": mean,
                "mean_abs": float(accumulator["sum_abs"]) / count,
                "rms": math.sqrt(max(0.0, mean_square)),
                "standard_deviation": math.sqrt(
                    max(0.0, mean_square - mean * mean)
                ),
                "pre_projection_branch_rms": math.sqrt(
                    max(
                        0.0,
                        float(accumulator["branch_sum_square"]) / branch_count,
                    )
                ),
            }
        )
    return int(probe.numel()), result


def _type_label(value: Any) -> str:
    return "a" if value.stream == "A" else f"b{value.component}"


def _validation_carrier_gate_statistics(
    model: nn.Module,
    data: Any,
    split: common.SplitIndices,
    *,
    device: str,
) -> list[dict[str, Any]]:
    """Measure the realized G10/G11 carrier gates and gated carrier branches."""
    method = getattr(model, "legacy_carrier_gate_modules_by_role", None)
    if not callable(method):
        return []
    modules = method()
    rows = {
        str(item["role"]): dict(item)
        for item in getattr(model, "projection_input_role_manifest", ())
        if item.get("kind") == "legacy_carrier"
        and bool(item.get("active", True))
        and item.get("gate_mode") in {"residual_zero", "default"}
    }
    if not rows:
        return []
    if set(rows).difference(modules):
        raise RuntimeError("G active carrier Gate roles have no registered heads")
    generator = torch.Generator(device="cpu").manual_seed(VALIDATION_PROBE_SEED)
    order = torch.randperm(int(split.validation.numel()), generator=generator)
    probe = split.validation[
        order[: min(VALIDATION_PROBE_COUNT, int(split.validation.numel()))]
    ]
    accumulators = {
        role: {
            "count": 0,
            "sum": 0.0,
            "sum_square": 0.0,
            "sum_abs_delta_one": 0.0,
            "near_zero": 0,
            "near_one": 0,
            "near_two": 0,
            "branch_count": 0,
            "branch_sum_square": 0.0,
        }
        for role in rows
    }
    captured: dict[str, Tensor] = {}

    def capture(role: str):
        def hook(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
            captured[role] = 1.0 + torch.tanh(output.detach())

        return hook

    handles = [
        modules[role].register_forward_hook(capture(role)) for role in rows
    ]
    was_training = model.training
    model.eval()
    try:
        for start in range(0, int(probe.numel()), VALIDATION_PROBE_BATCH_SIZE):
            captured = {}
            selection = probe[start : start + VALIDATION_PROBE_BATCH_SIZE]
            centers, frames, _target = common._batch_inputs(data, selection, device)
            with torch.no_grad():
                debug = model.debug_forward(centers, frames)
            if set(captured) != set(rows):
                raise RuntimeError("G carrier Gate runtime roles do not match manifest")
            for role, gate_value in captured.items():
                gate = gate_value.to(dtype=torch.float64, device="cpu").reshape(-1)
                if not bool(torch.isfinite(gate).all()):
                    raise RuntimeError(f"G carrier Gate role {role} is nonfinite")
                accumulator = accumulators[role]
                accumulator["count"] += int(gate.numel())
                accumulator["sum"] += float(gate.sum())
                accumulator["sum_square"] += float(gate.square().sum())
                accumulator["sum_abs_delta_one"] += float(
                    (gate - 1.0).abs().sum()
                )
                accumulator["near_zero"] += int((gate < 0.1).sum())
                accumulator["near_one"] += int(((gate - 1.0).abs() < 0.1).sum())
                accumulator["near_two"] += int((gate > 1.9).sum())

                row = rows[role]
                stage_concats = debug.concats[str(row["stage"])]
                matching = tuple(
                    value
                    for key, value in stage_concats.items()
                    if _type_label(key) == str(row["target"])
                )
                if len(matching) != 1:
                    raise RuntimeError(
                        f"G carrier Gate role {role} has no unique concat target"
                    )
                branch = matching[0][
                    ..., int(row["start"]) : int(row["stop"]), :
                ].to(dtype=torch.float64, device="cpu")
                if not bool(torch.isfinite(branch).all()):
                    raise RuntimeError(f"G carrier branch role {role} is nonfinite")
                accumulator["branch_count"] += int(branch.numel())
                accumulator["branch_sum_square"] += float(branch.square().sum())
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
    result = []
    for role, accumulator in accumulators.items():
        count = int(accumulator["count"])
        branch_count = int(accumulator["branch_count"])
        if count < 1 or branch_count < 1:
            raise RuntimeError(f"G carrier Gate role {role} has no observations")
        mean_value = float(accumulator["sum"]) / count
        mean_square = float(accumulator["sum_square"]) / count
        result.append(
            {
                **_json_value(rows[role]),
                "observation_count": count,
                "mean": mean_value,
                "standard_deviation": math.sqrt(
                    max(0.0, mean_square - mean_value * mean_value)
                ),
                "mean_absolute_deviation_from_one": float(
                    accumulator["sum_abs_delta_one"]
                )
                / count,
                "fraction_below_0_1": int(accumulator["near_zero"]) / count,
                "fraction_within_0_1_of_one": int(accumulator["near_one"])
                / count,
                "fraction_above_1_9": int(accumulator["near_two"]) / count,
                "gated_carrier_branch_rms": math.sqrt(
                    max(
                        0.0,
                        float(accumulator["branch_sum_square"]) / branch_count,
                    )
                ),
            }
        )
    return result


def export_selected_gate_audit(
    *,
    model: nn.Module,
    spec: Any,
    data: Any,
    split: common.SplitIndices,
    config: common.SweepConfig,
    paths: common.TrialPaths,
    device: str,
    comet_logger: Any,
    **_values: Any,
) -> Mapping[str, Any]:
    """Write and upload one selected-checkpoint Gate parameter audit."""
    del config
    required = (
        "descriptor_role_manifest",
        "coefficient_head_role_manifest",
        "coefficient_head_modules_by_role",
        "first_trunk_linear_modules_by_stage",
        "collect_coefficient_activations",
        "debug_forward",
        "stage_trunks",
        "channel_projections",
    )
    missing = tuple(name for name in required if not hasattr(model, name))
    if missing:
        raise RuntimeError(f"G selected model lacks Gate audit interfaces {missing}")
    descriptor_manifest = tuple(getattr(model, "descriptor_role_manifest"))
    coefficient_manifest = tuple(getattr(model, "coefficient_head_role_manifest"))
    g_series_manifest = dict(getattr(model, "g_series_manifest", {}))
    masked_columns = sum(
        int(item["stop"]) - int(item["start"])
        for item in descriptor_manifest
        if not bool(item["active"])
    )
    probe_count, gamma_statistics = _validation_gamma_statistics(
        model,
        data,
        split,
        device=device,
    )
    carrier_gate_statistics = _validation_carrier_gate_statistics(
        model,
        data,
        split,
        device=device,
    )
    parameter_artifact = paths.directory / "invariant_gate_parameters.pt"
    parameter_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_gate_parameter_snapshot(model, spec), parameter_artifact)
    parameter_sha256 = common.sha256_file(parameter_artifact)
    report = {
        "schema_name": "tfenn_g_series_gate_audit",
        "schema_version": 1,
        "model_id": str(spec.model_id),
        "variant_id": int(getattr(spec, "variant_id", 0)),
        "seed_index": int(getattr(spec, "seed_index", 0)),
        "model_seed": int(getattr(spec, "model_seed", 0)),
        "shuffle_seed": int(getattr(spec, "shuffle_seed", 0)),
        "selected_checkpoint_rule": "minimum validation normalized MSE",
        "validation_probe_sample_count": probe_count,
        "validation_probe_batch_size": VALIDATION_PROBE_BATCH_SIZE,
        "validation_probe_seed": VALIDATION_PROBE_SEED,
        "validation_probe_selection": "fixed seed permutation of validation split",
        "masked_descriptor_column_count": masked_columns,
        "inactive_columns_excluded_from_rankings": True,
        "exchangeable_hidden_channels_ranked_individually": False,
        "importance_interpretation": (
            "weight magnitude and V@W are descriptive parameter allocation; "
            "gamma and branch RMS are functional activity, not causal ablations"
        ),
        "descriptor_role_manifest": _json_value(descriptor_manifest),
        "candidate_manifest": _json_value(getattr(model, "candidate_manifest", ())),
        "coefficient_head_role_manifest": _json_value(coefficient_manifest),
        "projection_input_role_manifest": _json_value(
            getattr(model, "projection_input_role_manifest", ())
        ),
        "causal_mask_manifest": _json_value(
            getattr(model, "causal_mask_manifest", {})
        ),
        "g_series_manifest": _json_value(g_series_manifest),
        "trunk_input_column_statistics": _trunk_column_statistics(model),
        "coefficient_parameter_statistics": _coefficient_parameter_statistics(model),
        "legacy_carrier_gate_parameter_statistics": (
            _legacy_carrier_parameter_statistics(model)
        ),
        "head_composed_weight_statistics": _head_composed_weight_statistics(model),
        "validation_gamma_statistics": gamma_statistics,
        "validation_carrier_gate_statistics": carrier_gate_statistics,
        "invariant_gate_parameter_artifact": {
            "path": str(parameter_artifact.resolve()),
            "sha256": parameter_sha256,
            "contents": (
                "raw stage-trunk and role-labelled coefficient-head weight "
                "matrices, typed channel projections, and role manifests"
            ),
        },
    }
    artifact = paths.directory / "gate_audit.json"
    common._atomic_json(artifact, report)
    sha256 = common.sha256_file(artifact)
    comet_logger.log_asset(
        artifact,
        name=f"{spec.model_id}_gate_audit.json",
        metadata={
            "model_id": spec.model_id,
            "variant_id": getattr(spec, "variant_id", ""),
            "seed_index": getattr(spec, "seed_index", 0),
            "sha256": sha256,
            "validation_probe_sample_count": probe_count,
            "masked_descriptor_column_count": masked_columns,
        },
    )
    comet_logger.log_asset(
        parameter_artifact,
        name=f"{spec.model_id}_invariant_gate_parameters.pt",
        metadata={
            "model_id": spec.model_id,
            "variant_id": getattr(spec, "variant_id", ""),
            "seed_index": getattr(spec, "seed_index", 0),
            "sha256": parameter_sha256,
        },
    )
    return {
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": sha256,
        "descriptor_role_count": len(descriptor_manifest),
        "coefficient_head_role_count": len(coefficient_manifest),
        "validation_probe_sample_count": probe_count,
        "validation_probe_seed": VALIDATION_PROBE_SEED,
        "validation_carrier_gate_role_count": len(carrier_gate_statistics),
        "masked_descriptor_column_count": masked_columns,
        "inactive_columns_excluded_from_rankings": True,
        "nominal_trainable_parameter_count": int(
            g_series_manifest.get("nominal_trainable_parameter_count", 0)
        ),
        "causally_active_parameter_scalar_count": int(
            g_series_manifest.get("causally_active_parameter_scalar_count", 0)
        ),
        "invariant_gate_parameter_path": str(parameter_artifact.resolve()),
        "invariant_gate_parameter_sha256": parameter_sha256,
    }
