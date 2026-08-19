"""Export role labelled Gate diagnostics for selected F checkpoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from experiments.benzene_pair import sweep30 as common


VALIDATION_PROBE_COUNT = 10_000
VALIDATION_PROBE_BATCH_SIZE = 2_000
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
        raise RuntimeError("Gate parameter audit received empty or nonfinite values")
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
            raise RuntimeError(f"F dense coefficient head {role} is unavailable")
        target_channels = int(row["target_channels"])
        primitive_channels = int(row["primitive_channels"])
        expected = target_channels * primitive_channels
        if module.weight.shape[0] != expected or module.bias.shape != (expected,):
            raise RuntimeError(
                f"F coefficient head {role} shape does not match manifest"
            )
        weight = module.weight.reshape(
            target_channels,
            primitive_channels,
            module.weight.shape[1],
        )
        bias = module.bias.reshape(target_channels, primitive_channels)
        for basis_index in range(primitive_channels):
            result.append(
                {
                    **_json_value(row),
                    "basis_index": basis_index,
                    "weight": _magnitude_stats(weight[:, basis_index, :]),
                    "bias": _magnitude_stats(bias[:, basis_index]),
                }
            )
    return result


def _validation_gamma_statistics(
    model: nn.Module,
    data: Any,
    split: common.SplitIndices,
    *,
    device: str,
) -> tuple[int, list[dict[str, Any]]]:
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
        raise RuntimeError("F Gate audit validation probe is empty")
    accumulators = {
        role: {"count": 0, "sum": 0.0, "sum_abs": 0.0, "sum_square": 0.0}
        for role in manifest
    }
    for start in range(0, int(probe.numel()), VALIDATION_PROBE_BATCH_SIZE):
        selection = probe[start : start + VALIDATION_PROBE_BATCH_SIZE]
        centers, frames, _target = common._batch_inputs(data, selection, device)
        values = model.collect_coefficient_activations(centers, frames)
        if set(values) != set(manifest):
            raise RuntimeError("F gamma roles do not match the coefficient manifest")
        for role, value in values.items():
            flat = value.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
            if not bool(torch.isfinite(flat).all()):
                raise RuntimeError(f"F gamma role {role} contains nonfinite values")
            accumulator = accumulators[role]
            accumulator["count"] += int(flat.numel())
            accumulator["sum"] += float(flat.sum())
            accumulator["sum_abs"] += float(flat.abs().sum())
            accumulator["sum_square"] += float(flat.square().sum())
    result = []
    for role, accumulator in accumulators.items():
        count = int(accumulator["count"])
        if count < 1:
            raise RuntimeError(f"F gamma role {role} has no observations")
        mean = float(accumulator["sum"]) / count
        mean_square = float(accumulator["sum_square"]) / count
        result.append(
            {
                **_json_value(manifest[role]),
                "observation_count": count,
                "mean": mean,
                "mean_abs": float(accumulator["sum_abs"]) / count,
                "rms": math.sqrt(max(0.0, mean_square)),
                "standard_deviation": math.sqrt(max(0.0, mean_square - mean * mean)),
            }
        )
    return int(probe.numel()), result


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
    """Write and upload one selected checkpoint Gate audit."""
    required = (
        "descriptor_role_manifest",
        "coefficient_head_role_manifest",
        "coefficient_head_modules_by_role",
        "first_trunk_linear_modules_by_stage",
        "collect_coefficient_activations",
    )
    missing = tuple(name for name in required if not hasattr(model, name))
    if missing:
        raise RuntimeError(f"F selected model lacks Gate audit interfaces {missing}")
    descriptor_manifest = tuple(getattr(model, "descriptor_role_manifest"))
    coefficient_manifest = tuple(getattr(model, "coefficient_head_role_manifest"))
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
    strict_manifest = getattr(model, "strict_flow_manifest", None) or {}
    report = {
        "schema_name": "tfenn_f_series_gate_audit",
        "schema_version": 1,
        "model_id": str(spec.model_id),
        "descriptor_mask": str(getattr(spec, "descriptor_mask", "full")),
        "selected_checkpoint_rule": "minimum validation normalized MSE",
        "validation_probe_sample_count": probe_count,
        "validation_probe_batch_size": VALIDATION_PROBE_BATCH_SIZE,
        "validation_probe_seed": VALIDATION_PROBE_SEED,
        "validation_probe_selection": "fixed seed permutation of validation split",
        "masked_descriptor_column_count": masked_columns,
        "inactive_columns_excluded_from_rankings": True,
        "exchangeable_hidden_channels_ranked_individually": False,
        "descriptor_role_manifest": _json_value(descriptor_manifest),
        "candidate_manifest": _json_value(getattr(model, "candidate_manifest", ())),
        "strict_edge_audit": _json_value(strict_manifest.get("edge_audit", ())),
        "coefficient_head_role_manifest": _json_value(coefficient_manifest),
        "trunk_input_column_statistics": _trunk_column_statistics(model),
        "coefficient_parameter_statistics": _coefficient_parameter_statistics(model),
        "validation_gamma_statistics": gamma_statistics,
    }
    artifact = paths.directory / "gate_audit.json"
    common._atomic_json(artifact, report)
    sha256 = common.sha256_file(artifact)
    comet_logger.log_asset(
        artifact,
        name=f"{spec.model_id}_gate_audit.json",
        metadata={
            "model_id": spec.model_id,
            "sha256": sha256,
            "validation_probe_sample_count": probe_count,
            "masked_descriptor_column_count": masked_columns,
        },
    )
    return {
        "artifact_path": str(Path(artifact).resolve()),
        "artifact_sha256": sha256,
        "descriptor_role_count": len(descriptor_manifest),
        "coefficient_head_role_count": len(coefficient_manifest),
        "validation_probe_sample_count": probe_count,
        "validation_probe_seed": VALIDATION_PROBE_SEED,
        "masked_descriptor_column_count": masked_columns,
        "inactive_columns_excluded_from_rankings": True,
    }
