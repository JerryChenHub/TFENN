"""Run two paired F projects through four independent execution shards."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from experiments.benzene_pair import sweep30 as common
from experiments.benzene_pair.comet_logging import NullCometTrialLogger
from experiments.benzene_pair.f_series.catalog import (
    FModelSpec,
    F_SERIES_SPECS,
    get_execution_shard_specs,
    get_model_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "experiments" / "benzene_pair" / "runs" / "f_series_400k_v2"
)
E311_SPLIT_INDICES_SHA256 = (
    "50e6bf0e32c1bb9b0bddb689097a4a38a5d74a5bcf12b0fc8471f6b1f4cf50b1"
)
E311_SPLIT_MANIFEST_HASH = (
    "3a64eb6ac96805aad4fe41ef1fd44a0cdc2417d193f8c0852244780a3563ec98"
)
F_STUDY_METADATA = {
    "concurrent_run": True,
    "tmux_session_count": 4,
    "execution_shard_counts": [26, 25, 25, 25],
    "comet_project_count": 2,
    "plan_document_name": "EXPERIMENT_PLAN.md",
    "planned_model_count": 101,
    "executed_model_count": 101,
    "primary_metric": "Final Test MAE",
    "historical_control": "E311",
    "reference_split_model": "E311",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExecutionShardDefinition:
    shard_id: int
    key: str
    project_id: int
    tmux_session_name: str
    directory_name: str
    comet_project: str
    purpose: str


EXECUTION_SHARDS = {
    0: ExecutionShardDefinition(
        0,
        "f1a",
        1,
        "tfenn_f1_a",
        "shard_0_f1_a",
        "tfenn_f_series_f1_full_local",
        "F100 control and F1 models F101 through F125",
    ),
    1: ExecutionShardDefinition(
        1,
        "f1b",
        1,
        "tfenn_f1_b",
        "shard_1_f1_b",
        "tfenn_f_series_f1_full_local",
        "F1 models F126 through F150",
    ),
    2: ExecutionShardDefinition(
        2,
        "f2a",
        2,
        "tfenn_f2_a",
        "shard_2_f2_a",
        "tfenn_f_series_f2_raw_only",
        "F2 models F201 through F225",
    ),
    3: ExecutionShardDefinition(
        3,
        "f2b",
        2,
        "tfenn_f2_b",
        "shard_3_f2_b",
        "tfenn_f_series_f2_raw_only",
        "F2 models F226 through F250",
    ),
}
DEFAULT_CONFIG_PATHS = {
    shard_id: Path(__file__).resolve().parent / f"shard_{shard_id}.json"
    for shard_id in EXECUTION_SHARDS
}
EXPECTED_SHARD_COUNTS = {0: 26, 1: 25, 2: 25, 3: 25}


RESULT_FIELDS = (
    "model_id",
    "experiment_id",
    "project_id",
    "execution_shard_id",
    "topology",
    "invariant_policy",
    "pair_model_id",
    "channels",
    "changed_node",
    "description",
    "purpose",
    "comparison_role",
    "status",
    "planned_parameter_count",
    "actual_parameter_count",
    "parameter_count_delta",
    "masked_descriptor_column_count",
    "best_epoch",
    "best_validation_normalized_mse",
    "train_normalized_mse",
    "validation_normalized_mse",
    "test_normalized_mse",
    "train_mae",
    "validation_mae",
    "test_mae",
    "train_rmse",
    "validation_rmse",
    "test_rmse",
    "train_relative_rmse_percent",
    "validation_relative_rmse_percent",
    "test_relative_rmse_percent",
    "test_relative_force_norm_min",
    "test_relative_force_norm_median",
    "test_relative_force_norm_max",
    "d6_status",
    "gate_audit_path",
    "invariant_gate_parameter_path",
    "duration_seconds",
    "error_type",
    "error_message",
)


def _shard(value: int | str) -> ExecutionShardDefinition:
    if isinstance(value, str):
        normalized = value.lower()
        aliases = {item.key: key for key, item in EXECUTION_SHARDS.items()}
        if normalized in aliases:
            value = aliases[normalized]
    try:
        return EXECUTION_SHARDS[int(value)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("execution shard must be f1a, f1b, f2a, or f2b") from error


def _shared_split_directory(study_root: str | Path) -> Path:
    return Path(study_root).resolve() / "shared_split"


def _execution_directory(
    study_root: str | Path,
    shard: ExecutionShardDefinition,
) -> Path:
    return Path(study_root).resolve() / "execution_shards" / shard.directory_name


def _project_directory(study_root: str | Path, project_id: int) -> Path:
    return Path(study_root).resolve() / "projects" / f"f{project_id}"


def _model_directory(study_root: str | Path, spec: FModelSpec) -> Path:
    return _project_directory(study_root, spec.project_id) / "models" / spec.model_id


def make_config(
    shard_id: int | str,
    *,
    study_root: str | Path = DEFAULT_STUDY_ROOT,
) -> common.SweepConfig:
    """Build one fixed four hundred thousand sample F protocol."""
    shard = _shard(shard_id)
    value = json.loads(DEFAULT_CONFIG_PATHS[shard.shard_id].read_text(encoding="utf_8"))
    if value.get("schema_name") != "tfenn_benzene_pair_f_series":
        raise ValueError("unexpected F series config schema")
    if value.get("schema_version") != 1:
        raise ValueError("unexpected F series config version")
    if int(value.get("execution_shard_id", -1)) != shard.shard_id:
        raise ValueError("F config execution shard does not match")
    if value.get("execution_directory_name") != shard.directory_name:
        raise ValueError("F execution directory does not match")
    expected_models = tuple(
        item.model_id for item in get_execution_shard_specs(shard.shard_id)
    )
    if tuple(value.get("model_ids", ())) != expected_models:
        raise ValueError("F config model ids do not match the catalog")
    if any(
        item.project_id != shard.project_id
        for item in get_execution_shard_specs(shard.shard_id)
    ):
        raise ValueError("F execution shard crosses scientific projects")
    for name in ("concurrent_run", "tmux_session_count"):
        if value.get(name) != F_STUDY_METADATA[name]:
            raise ValueError(f"F config {name} does not match")
    config = common.SweepConfig(
        shard_paths=tuple(
            (REPOSITORY_ROOT / str(item)).resolve() for item in value["shard_paths"]
        ),
        study_directory=Path(study_root).resolve(),
        epochs=int(value["epochs"]),
        effective_batch_size=int(value["effective_batch_size"]),
        micro_batch_size=int(value["micro_batch_size"]),
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        scheduler_step_size=int(value["scheduler_step_size"]),
        scheduler_gamma=float(value["scheduler_gamma"]),
        validation_every=int(value["validation_every"]),
        split_seed=int(value["split_seed"]),
        model_seed=int(value["model_seed"]),
        shuffle_seed=int(value["shuffle_seed"]),
        split_fractions=tuple(float(item) for item in value["split_fractions"]),
        device=str(value["device"]),
        dtype=str(value["dtype"]),
        threads=int(value["threads"]),
        symmetry_tolerance=float(value["symmetry_tolerance"]),
        symmetry_probe_count=int(value["symmetry_probe_count"]),
        expected_sample_count=int(value["expected_sample_count"]),
        expected_dataset_revision=int(value["expected_dataset_revision"]),
        expected_opls_version=str(value["expected_opls_version"]),
        enable_tf32=bool(value["enable_tf32"]),
        relative_force_norm_sample_count=int(value["relative_force_norm_sample_count"]),
        relative_force_norm_seed=int(value["relative_force_norm_seed"]),
        comet=common.CometConfig.from_mapping(value["comet"]),
        schema_name=str(value["schema_name"]),
        schema_version=int(value["schema_version"]),
    )
    config.validate()
    if config.epochs != 500:
        raise ValueError("F series requires five hundred epochs")
    if config.effective_batch_size != 10_000 or config.micro_batch_size != 10_000:
        raise ValueError("F series requires batch size ten thousand")
    if config.expected_sample_count != 400_000:
        raise ValueError("F series requires four hundred thousand samples")
    if config.comet.project_name != shard.comet_project:
        raise ValueError("F Comet project does not match")
    return config


def _source_sha256() -> str:
    fixed_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "catalog.py",
        Path(__file__).resolve().parent / "model_factory.py",
        Path(__file__).resolve().parent / "gate_audit.py",
        *tuple(DEFAULT_CONFIG_PATHS.values()),
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "__init__.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "sweep30.py",
        REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "invariant_gate_v2_20k_sweep.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "e_series" / "catalog.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "e_series" / "__init__.py",
        REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "e_series"
        / "model_factory.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "comet_logging.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "metrics.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "train.py",
    )
    dependency_paths = tuple(
        sorted(
            (
                *Path(__file__).resolve().parent.rglob("*.py"),
                *(REPOSITORY_ROOT / "src" / "TFENN" / "models").rglob("*.py"),
                *(REPOSITORY_ROOT / "src" / "TFENN" / "tensor_math").rglob("*.py"),
                *(
                    REPOSITORY_ROOT / "experiments" / "benzene_pair" / "data"
                ).rglob("*.py"),
                *(
                    REPOSITORY_ROOT
                    / "experiments"
                    / "benzene_pair"
                    / "e_series"
                ).rglob("*.py"),
                REPOSITORY_ROOT
                / "experiments"
                / "benzene_pair"
                / "group_conv_baseline.py",
            ),
            key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix(),
        )
    )
    paths = tuple(dict.fromkeys((*fixed_paths, *dependency_paths)))
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf_8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _catalog_sha256() -> str:
    return common._canonical_sha256([spec.as_dict() for spec in F_SERIES_SPECS])


def _preflight_path(study_root: str | Path) -> Path:
    return Path(study_root).resolve() / "preflight_manifest.json"


def _trainable_initialization_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        label = name.encode("utf_8")
        value = parameter.detach().cpu().contiguous()
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _covariant_unit_check(model: nn.Module) -> dict[str, Any]:
    reference = next(model.parameters())
    dtype = reference.dtype
    device = reference.device
    tolerance = 1.0e-10 if dtype == torch.float64 else 1.0e-4
    angle = torch.tensor(0.37, dtype=dtype, device=device)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine, angle.new_zeros(()))),
            torch.stack((sine, cosine, angle.new_zeros(()))),
            torch.stack((angle.new_zeros(()), angle.new_zeros(()), angle.new_ones(()))),
        )
    )
    centers = torch.tensor(
        (
            ((0.0, 0.0, 0.0), (5.2, 0.4, -0.2)),
            ((0.0, 0.0, 0.0), (5.8, -0.3, 0.5)),
        ),
        dtype=dtype,
        device=device,
    )
    identity = torch.eye(3, dtype=dtype, device=device)
    frames = torch.stack(
        (
            torch.stack((identity, rotation)),
            torch.stack((rotation, identity)),
        )
    )
    model.eval()
    model.zero_grad(set_to_none=True)
    prediction = model(centers, frames)
    if prediction.shape != (2, 3) or not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("F forward check failed")
    prediction.square().mean().backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    gradients = [parameter.grad for parameter in trainable]
    if (
        not gradients
        or any(item is None for item in gradients)
        or not all(
            bool(torch.isfinite(item).all()) for item in gradients if item is not None
        )
    ):
        raise RuntimeError("F gradient connectivity check failed")
    symmetry = common.symmetry_metrics(model, centers, frames, tolerance=tolerance)
    if not symmetry["passed"]:
        raise RuntimeError("F covariance check failed")
    return {
        "forward_finite": True,
        "gradient_finite": True,
        "all_trainable_parameter_tensors_have_gradients": True,
        "trainable_tensor_count": len(trainable),
        "gradient_tensor_count": len(gradients),
        "dtype": str(dtype),
        "symmetry": symmetry,
    }


def _f100_reference_parity(model: nn.Module) -> dict[str, Any]:
    from experiments.benzene_pair.e_series.model_factory import build_e_series_model

    torch.manual_seed(20260822)
    reference = build_e_series_model(
        "E311",
        common._proper_d6_generators(),
        generator_names=("sixfold", "twofold"),
    ).to(dtype=next(model.parameters()).dtype)
    model_config = getattr(model, "config").as_dict()
    reference_config = getattr(reference, "config").as_dict()
    initialization_equal = _trainable_initialization_sha256(
        model
    ) == _trainable_initialization_sha256(reference)
    centers = torch.tensor(
        (((0.0, 0.0, 0.0), (5.4, 0.2, -0.3)),),
        dtype=next(model.parameters()).dtype,
    )
    frames = torch.eye(3, dtype=centers.dtype).expand(1, 2, 3, 3).clone()
    model.eval()
    reference.eval()
    with torch.no_grad():
        forward_equal = torch.equal(model(centers, frames), reference(centers, frames))
    result = {
        "builder_reference_model_id": "E311",
        "compiled_config_equal": model_config == reference_config,
        "candidate_manifest_equal": getattr(model, "candidate_manifest", ())
        == getattr(reference, "candidate_manifest", ()),
        "initialization_equal": initialization_equal,
        "forward_bitwise_equal": bool(forward_equal),
        "parameter_count": sum(
            parameter.numel()
            for parameter in reference.parameters()
            if parameter.requires_grad
        ),
    }
    if not all(
        result[name]
        for name in (
            "compiled_config_equal",
            "candidate_manifest_equal",
            "initialization_equal",
            "forward_bitwise_equal",
        )
    ):
        raise RuntimeError("F100 does not reproduce exact E311 model behavior")
    del reference
    gc.collect()
    return result


def _build_model(
    spec: FModelSpec,
    device: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    from experiments.benzene_pair.f_series.model_factory import build_f_series_model

    model = build_f_series_model(
        spec,
        common._proper_d6_generators(),
        generator_names=("sixfold", "twofold"),
    )
    if not isinstance(model, nn.Module):
        raise TypeError("F series builder must return a torch module")
    return model.to(device=torch.device(device), dtype=dtype)


def _preflight_model(spec: FModelSpec) -> dict[str, Any]:
    torch.manual_seed(20260822)
    model = _build_model(spec, "cpu", dtype=torch.float64)
    actual = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    configuration = getattr(model, "config", None)
    configuration_dict = (
        configuration.as_dict()
        if configuration is not None
        and callable(getattr(configuration, "as_dict", None))
        else None
    )
    candidate_manifest = tuple(getattr(model, "candidate_manifest", ()))
    candidate_status_counts: dict[str, int] = {}
    for item in candidate_manifest:
        status = str(item.get("status", "unknown"))
        candidate_status_counts[status] = candidate_status_counts.get(status, 0) + 1
    descriptor_manifest = tuple(getattr(model, "descriptor_role_manifest", ()))
    coefficient_manifest = tuple(getattr(model, "coefficient_head_role_manifest", ()))
    strict_manifest = getattr(model, "strict_flow_manifest", None)
    if spec.family == "strict_flow":
        if strict_manifest is None:
            raise RuntimeError("strict F model is missing its strict flow manifest")
        edge_audit = tuple(strict_manifest.get("edge_audit", ()))
        if not edge_audit:
            raise RuntimeError("strict F model has no compiled edge audit")
        if any(
            item.get("missing_edges")
            or item.get("undeclared_sources")
            or item.get("live_live_path_roles")
            for item in edge_audit
        ):
            raise RuntimeError("strict F model failed its explicit edge audit")
    record = {
        "status": "passed",
        "model_id": spec.model_id,
        "pair_model_id": spec.pair_model_id,
        "planned_parameter_count": spec.planned_parameter_count,
        "actual_parameter_count": actual,
        "parameter_count_delta": actual - spec.planned_parameter_count,
        "planned_parameter_count_matches": actual == spec.planned_parameter_count,
        "initial_parameter_sha256": _trainable_initialization_sha256(model),
        "architecture_metadata": _json_value(
            getattr(model, "architecture_metadata", {})
        ),
        "compiled_config": configuration_dict,
        "strict_flow_manifest": _json_value(strict_manifest),
        "descriptor_role_manifest": _json_value(descriptor_manifest),
        "coefficient_head_role_manifest": _json_value(coefficient_manifest),
        "offline_compilation_summary": _json_value(
            getattr(model, "offline_compilation_summary", {})
        ),
        "candidate_status_counts": candidate_status_counts,
        "selected_covariant_roles": _json_value(
            getattr(model, "selected_covariant_roles", ())
        ),
        "covariant_unit_check": _covariant_unit_check(model),
        **(
            {}
            if spec.model_id != "F100"
            else {"e311_reference_parity": _f100_reference_parity(model)}
        ),
    }
    if actual != spec.planned_parameter_count:
        record.update(
            {
                "status": "failed",
                "exception_type": "RuntimeError",
                "message": (
                    f"{spec.model_id} compiled with {actual} parameters instead of "
                    f"the planned {spec.planned_parameter_count}"
                ),
            }
        )
    del model
    gc.collect()
    return record


def _descriptor_schema_signature(record: Mapping[str, Any]) -> str:
    rows = []
    for item in record.get("descriptor_role_manifest", ()):
        value = dict(item)
        value.pop("active", None)
        value.pop("active_column_count", None)
        value.pop("descriptor_mask", None)
        rows.append(value)
    return common._canonical_sha256(rows)


def _apply_pair_gates(records: list[dict[str, Any]]) -> None:
    by_id = {str(item["model_id"]): item for item in records}
    for spec in F_SERIES_SPECS:
        pair_id = spec.pair_model_id
        if pair_id is None or spec.model_id > pair_id:
            continue
        left = by_id[spec.model_id]
        right = by_id[pair_id]
        if left.get("status") != "passed" or right.get("status") != "passed":
            continue
        gates = {
            "pair_parameter_count_equal": left["actual_parameter_count"]
            == right["actual_parameter_count"],
            "pair_initialization_equal": left["initial_parameter_sha256"]
            == right["initial_parameter_sha256"],
            "pair_descriptor_schema_equal": _descriptor_schema_signature(left)
            == _descriptor_schema_signature(right),
            "pair_selected_roles_equal": left["selected_covariant_roles"]
            == right["selected_covariant_roles"],
            "pair_candidate_manifest_equal": left.get("strict_flow_manifest", {}).get(
                "candidate_manifest"
            )
            == right.get("strict_flow_manifest", {}).get("candidate_manifest"),
            "pair_coefficient_manifest_equal": left["coefficient_head_role_manifest"]
            == right["coefficient_head_role_manifest"],
        }
        for record, paired in ((left, pair_id), (right, spec.model_id)):
            record["paired_model_id"] = paired
            record.update(gates)
        failed = tuple(name for name, passed in gates.items() if not passed)
        if failed:
            for record in (left, right):
                record["status"] = "failed"
                record["exception_type"] = "RuntimeError"
                record["message"] = f"paired preflight gates failed {failed}"


def _compile_preflight(
    study_root: str | Path,
    *,
    force: bool = False,
) -> Mapping[str, Any]:
    path = _preflight_path(study_root)
    source_sha = _source_sha256()
    catalog_sha = _catalog_sha256()
    if path.is_file() and not force:
        existing = common._load_json(path)
        if (
            existing.get("status") == "passed"
            and existing.get("source_sha256") == source_sha
            and existing.get("catalog_sha256") == catalog_sha
            and existing.get("passed_model_count") == 101
        ):
            return existing
    records: list[dict[str, Any]] = []
    for spec in F_SERIES_SPECS:
        try:
            record = _preflight_model(spec)
        except BaseException as error:
            record = {
                "status": "failed",
                "model_id": spec.model_id,
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
        records.append(record)
        print(
            json.dumps(
                {
                    "model_id": spec.model_id,
                    "preflight_status": record["status"],
                    "actual_parameter_count": record.get("actual_parameter_count"),
                }
            ),
            flush=True,
        )
    _apply_pair_gates(records)
    passed = sum(item["status"] == "passed" for item in records)
    manifest = {
        "schema_name": "tfenn_f_series_preflight",
        "schema_version": 1,
        "status": "passed" if passed == 101 else "failed",
        "source_sha256": source_sha,
        "catalog_sha256": catalog_sha,
        "model_count": 101,
        "passed_model_count": passed,
        "failed_model_count": 101 - passed,
        "models": records,
        **F_STUDY_METADATA,
        "created_at_utc": common._utc_now(),
    }
    manifest["preflight_hash"] = common._canonical_sha256(manifest)
    common._atomic_json(path, manifest)
    if manifest["status"] != "passed":
        raise RuntimeError(
            f"F preflight failed for {manifest['failed_model_count']} models"
        )
    return manifest


def _require_preflight(study_root: str | Path) -> Mapping[str, Any]:
    path = _preflight_path(study_root)
    if not path.is_file():
        raise RuntimeError("F preflight manifest is missing")
    manifest = common._load_json(path)
    if manifest.get("status") != "passed" or manifest.get("passed_model_count") != 101:
        raise RuntimeError("F preflight did not pass all one hundred one models")
    if manifest.get("source_sha256") != _source_sha256():
        raise RuntimeError("F preflight source hash is stale")
    if manifest.get("catalog_sha256") != _catalog_sha256():
        raise RuntimeError("F preflight catalog hash is stale")
    expected_hash = manifest.get("preflight_hash")
    unsigned = dict(manifest)
    unsigned.pop("preflight_hash", None)
    if expected_hash != common._canonical_sha256(unsigned):
        raise RuntimeError("F preflight manifest hash is invalid")
    return manifest


def _preflight_record(preflight: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for record in preflight["models"]:
        if record.get("model_id") == model_id:
            return record
    raise KeyError(f"preflight has no record for {model_id}")


def _enriched_spec(spec: FModelSpec, preflight: Mapping[str, Any]) -> FModelSpec:
    options = dict(spec.options)
    options["compiled_preflight"] = dict(_preflight_record(preflight, spec.model_id))
    return replace(spec, options=options)


def _study_metadata(
    preflight: Mapping[str, Any] | None,
    *,
    project_id: int | None = None,
    execution_shard_id: int | None = None,
) -> dict[str, Any]:
    result = {
        **F_STUDY_METADATA,
        "preflight_hash": None if preflight is None else preflight["preflight_hash"],
        "reference_split_manifest_hash": E311_SPLIT_MANIFEST_HASH,
        "reference_split_indices_sha256": E311_SPLIT_INDICES_SHA256,
    }
    if project_id is not None:
        result["scientific_project_id"] = project_id
    if execution_shard_id is not None:
        result["execution_shard_id"] = execution_shard_id
    return result


def _require_e311_split(manifest: Mapping[str, Any]) -> None:
    if manifest.get("sample_count") != 400_000:
        raise RuntimeError("F split does not contain four hundred thousand samples")
    if manifest.get("partition_counts") != {
        "train": 320_000,
        "validation": 40_000,
        "test": 40_000,
    }:
        raise RuntimeError("F split partition counts do not match E311")
    if manifest.get("indices_sha256") != E311_SPLIT_INDICES_SHA256:
        raise RuntimeError("F split indices do not exactly match E311")
    if manifest.get("reference_manifest_hash") != E311_SPLIT_MANIFEST_HASH:
        raise RuntimeError("F split does not record the exact E311 manifest")
    if manifest.get("reference_indices_sha256") != E311_SPLIT_INDICES_SHA256:
        raise RuntimeError("F split does not record the exact E311 indices")


def _require_e311_reference_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("manifest_hash") != E311_SPLIT_MANIFEST_HASH:
        raise RuntimeError("reference split manifest does not match E311")
    if manifest.get("indices_sha256") != E311_SPLIT_INDICES_SHA256:
        raise RuntimeError("reference split indices do not match E311")
    if manifest.get("sample_count") != 400_000:
        raise RuntimeError("reference split sample count does not match E311")


def _prepare_reference_split(
    study_root: str | Path,
    reference_split_directory: str | Path,
) -> tuple[common.SplitIndices, dict[str, Any]]:
    reference = Path(reference_split_directory).resolve()
    split, reference_manifest = common._load_split(reference)
    _require_e311_reference_manifest(reference_manifest)
    directory = _shared_split_directory(study_root)
    manifest_path = directory / "split_manifest.json"
    if manifest_path.is_file():
        existing_split, existing_manifest = common._load_split(directory)
        _require_e311_split(existing_manifest)
        if existing_manifest.get("data_provenance") != reference_manifest.get(
            "data_provenance"
        ):
            raise RuntimeError("existing F split provenance does not match E311")
        return existing_split, existing_manifest
    directory.mkdir(parents=True, exist_ok=True)
    source_indices = Path(str(reference_manifest["indices_path"]))
    target_indices = directory / "split_indices.npz"
    partial = target_indices.with_name(f"{target_indices.name}.{os.getpid()}.partial")
    shutil.copyfile(source_indices, partial)
    os.replace(partial, target_indices)
    if common.sha256_file(target_indices) != E311_SPLIT_INDICES_SHA256:
        raise RuntimeError("copied F split indices hash does not match E311")
    local_manifest = {
        key: value
        for key, value in reference_manifest.items()
        if key not in {"manifest_hash", "indices_path"}
    }
    local_manifest.update(
        {
            "indices_path": str(target_indices),
            "indices_sha256": E311_SPLIT_INDICES_SHA256,
            "reference_split_directory": str(reference),
            "reference_manifest_hash": E311_SPLIT_MANIFEST_HASH,
            "reference_indices_sha256": E311_SPLIT_INDICES_SHA256,
        }
    )
    local_manifest["manifest_hash"] = common._canonical_sha256(local_manifest)
    common._atomic_json(manifest_path, local_manifest)
    copied_split, copied_manifest = common._load_split(directory)
    _require_e311_split(copied_manifest)
    if copied_split.counts() != split.counts():
        raise RuntimeError("copied F split counts do not match E311")
    return copied_split, copied_manifest


def _load_shared_split(
    study_root: str | Path,
) -> tuple[common.SplitIndices, dict[str, Any]]:
    directory = _shared_split_directory(study_root)
    if not (directory / "split_manifest.json").is_file():
        raise RuntimeError("F shared split is missing, run prepare first")
    split, manifest = common._load_split(directory)
    _require_e311_split(manifest)
    return split, manifest


def _select_specs(
    shard_id: int | str,
    values: Sequence[str],
) -> tuple[FModelSpec, ...]:
    shard = _shard(shard_id)
    available = tuple(get_execution_shard_specs(shard.shard_id))
    if len(available) != EXPECTED_SHARD_COUNTS[shard.shard_id]:
        raise RuntimeError("F execution shard size changed")
    if not values:
        return available
    allowed = {item.model_id: item for item in available}
    selected = []
    for value in values:
        key = str(value).upper()
        if key not in allowed:
            raise ValueError(f"model {value} is outside execution shard {shard.key}")
        selected.append(allowed[key])
    if len({item.model_id for item in selected}) != len(selected):
        raise ValueError("model selection contains duplicates")
    return tuple(selected)


def _result_row(config: common.SweepConfig, spec: FModelSpec) -> dict[str, Any]:
    paths = common.TrialPaths.create(_model_directory(config.study_directory, spec))
    summary = common._load_json(paths.summary) if paths.summary.is_file() else {}
    status = common._load_json(paths.status) if paths.status.is_file() else {}
    error = common._load_json(paths.error) if paths.error.is_file() else {}
    selection = summary.get("selection", {})
    metrics = selection.get("selected_metrics", {})
    norm = summary.get("relative_force_norm_difference", {}).get("test", {})
    audit = summary.get("selected_model_audit", {})

    def metric(partition: str, name: str) -> Any:
        value = metrics.get(partition, {})
        return value.get(name, "") if isinstance(value, Mapping) else ""

    actual = summary.get("model", {}).get("parameter_count", "")
    return {
        "model_id": spec.model_id,
        "experiment_id": spec.experiment_id,
        "project_id": spec.project_id,
        "execution_shard_id": spec.execution_shard_id,
        "topology": spec.options.get("topology", ""),
        "invariant_policy": spec.options.get("invariant_policy", ""),
        "pair_model_id": spec.pair_model_id or "",
        "channels": json.dumps(list(spec.options.get("channels", ()))),
        "changed_node": spec.options.get("changed_node", ""),
        "description": spec.description,
        "purpose": spec.purpose,
        "comparison_role": spec.comparison_role,
        "status": summary.get("status", status.get("status", "planned")),
        "planned_parameter_count": spec.planned_parameter_count,
        "actual_parameter_count": actual,
        "parameter_count_delta": ""
        if actual == ""
        else int(actual) - spec.planned_parameter_count,
        "masked_descriptor_column_count": audit.get(
            "masked_descriptor_column_count", ""
        ),
        "best_epoch": selection.get("best_epoch", ""),
        "best_validation_normalized_mse": selection.get(
            "best_validation_during_training", {}
        ).get("normalized_mse", ""),
        "train_normalized_mse": metric("train", "normalized_mse"),
        "validation_normalized_mse": metric("validation", "normalized_mse"),
        "test_normalized_mse": metric("test", "normalized_mse"),
        "train_mae": metric("train", "mae"),
        "validation_mae": metric("validation", "mae"),
        "test_mae": metric("test", "mae"),
        "train_rmse": metric("train", "rmse"),
        "validation_rmse": metric("validation", "rmse"),
        "test_rmse": metric("test", "rmse"),
        "train_relative_rmse_percent": metric("train", "relative_rmse_percent"),
        "validation_relative_rmse_percent": metric(
            "validation", "relative_rmse_percent"
        ),
        "test_relative_rmse_percent": metric("test", "relative_rmse_percent"),
        "test_relative_force_norm_min": norm.get("min", ""),
        "test_relative_force_norm_median": norm.get("median", ""),
        "test_relative_force_norm_max": norm.get("max", ""),
        "d6_status": (
            ""
            if not selection.get("symmetry")
            else "passed"
            if selection["symmetry"].get("passed")
            else "failed"
        ),
        "gate_audit_path": audit.get("artifact_path", ""),
        "invariant_gate_parameter_path": audit.get(
            "invariant_gate_parameter_path",
            "",
        ),
        "duration_seconds": summary.get("runtime", {}).get("duration_seconds", ""),
        "error_type": error.get("exception_type", ""),
        "error_message": error.get("message", ""),
    }


def _refresh_results(
    config: common.SweepConfig,
    specs: Sequence[FModelSpec],
    *,
    output_directory: str | Path | None = None,
) -> Path:
    directory = (
        config.study_directory
        if output_directory is None
        else Path(output_directory).resolve()
    )
    directory.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(config, spec) for spec in specs]
    path = directory / "results.csv"
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)
    completed = [row for row in rows if row["status"] == "complete"]
    ranking = sorted(completed, key=lambda row: float(row["test_mae"]))
    by_id = {str(row["model_id"]): row for row in completed}
    pairs = []
    for spec in specs:
        pair_id = spec.pair_model_id
        if pair_id is None or spec.model_id > pair_id:
            continue
        if spec.model_id not in by_id or pair_id not in by_id:
            continue
        f1_mae = float(by_id[spec.model_id]["test_mae"])
        f2_mae = float(by_id[pair_id]["test_mae"])
        pairs.append(
            {
                "f1_model_id": spec.model_id,
                "f2_model_id": pair_id,
                "f1_test_mae": f1_mae,
                "f2_test_mae": f2_mae,
                "absolute_test_mae_change": f2_mae - f1_mae,
                "percentage_test_mae_change": 100.0 * (f2_mae - f1_mae) / f1_mae,
            }
        )
    common._atomic_json(
        directory / "comparison.json",
        {
            "schema_name": "tfenn_f_series_comparison",
            "schema_version": 1,
            "primary_metric": "Final Test MAE",
            "completed_model_count": len(completed),
            "error_model_count": sum(row["status"] == "error" for row in rows),
            "ranking_by_test_mae": ranking,
            "paired_f1_f2_test_mae": pairs,
            "updated_at_utc": common._utc_now(),
        },
    )
    return path


def _selected_model_audit(**values: Any) -> Mapping[str, Any]:
    from experiments.benzene_pair.f_series.gate_audit import (
        export_selected_gate_audit,
    )

    return export_selected_gate_audit(**values)


def run_study(arguments: argparse.Namespace) -> int:
    shard = _shard(arguments.shard)
    config = make_config(shard.shard_id, study_root=arguments.study_root)
    if not os.environ.get("COMET_API_KEY", "").strip():
        raise RuntimeError("COMET_API_KEY must be set for a formal F series run")
    preflight = _require_preflight(arguments.study_root)
    study_metadata = _study_metadata(
        preflight,
        project_id=shard.project_id,
        execution_shard_id=shard.shard_id,
    )
    device = common._resolve_device(arguments.device or config.device)
    all_specs = tuple(
        _enriched_spec(spec, preflight)
        for spec in get_execution_shard_specs(shard.shard_id)
    )
    selected = tuple(
        _enriched_spec(spec, preflight)
        for spec in _select_specs(shard.shard_id, arguments.model)
    )
    _split, split_manifest = _load_shared_split(arguments.study_root)
    manifest = {
        "schema_name": "tfenn_f_series_study",
        "schema_version": 1,
        "execution_shard_id": shard.shard_id,
        "execution_shard_key": shard.key,
        "scientific_project_id": shard.project_id,
        "comet_project": shard.comet_project,
        "execution_purpose": shard.purpose,
        "model_count": len(all_specs),
        "models": [item.as_dict() for item in all_specs],
        "config": config.as_dict(device=device),
        "shared_split_directory": str(_shared_split_directory(arguments.study_root)),
        "reference_split_directory": split_manifest["reference_split_directory"],
        "reference_split_manifest_hash": split_manifest["reference_manifest_hash"],
        "reference_split_indices_sha256": split_manifest["reference_indices_sha256"],
        "split_manifest_hash": split_manifest["manifest_hash"],
        "split_indices_sha256": split_manifest["indices_sha256"],
        "source_sha256": _source_sha256(),
        **study_metadata,
    }
    manifest["study_hash"] = common._canonical_sha256(manifest)
    execution_directory = _execution_directory(arguments.study_root, shard)
    execution_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = execution_directory / "manifest.json"
    if manifest_path.is_file() and common._load_json(manifest_path) != manifest:
        raise RuntimeError("existing F series manifest does not match this run")
    common._atomic_json(manifest_path, manifest)
    _refresh_results(config, all_specs, output_directory=execution_directory)
    failed_models: list[str] = []
    for spec in selected:
        paths = common.TrialPaths.create(_model_directory(config.study_directory, spec))
        if paths.summary.is_file():
            completed = common._load_json(paths.summary)
            expected_hash = common._trial_hash(
                config,
                spec,
                split_manifest,
                device=device,
                epochs=config.epochs,
                source_sha256=_source_sha256(),
                study_metadata=study_metadata,
            )
            if completed.get("status") != "complete":
                raise RuntimeError("existing F trial summary is not complete")
            if completed.get("trial_hash") != expected_hash:
                raise RuntimeError("existing F trial summary hash does not match")
            if not paths.best.is_file() or not paths.final.is_file():
                raise RuntimeError("completed F trial is missing a checkpoint")
            continue
        paths.directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "experiments.benzene_pair.f_series.runner",
            "trial",
            "--study_root",
            str(Path(arguments.study_root).resolve()),
            "--model",
            spec.model_id,
            "--device",
            device,
        ]
        with (
            paths.stdout.open("a", encoding="utf_8") as stdout,
            paths.stderr.open("a", encoding="utf_8") as stderr,
        ):
            process = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        _refresh_results(config, all_specs, output_directory=execution_directory)
        if process.returncode == 130:
            return 130
        if process.returncode:
            failed_models.append(spec.model_id)
    _refresh_results(config, all_specs, output_directory=execution_directory)
    if failed_models:
        print(
            json.dumps(
                {"status": "error", "failed_models": failed_models},
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1
    return 0


def run_trial_command(arguments: argparse.Namespace) -> int:
    spec = get_model_spec(arguments.model)
    config = make_config(spec.execution_shard_id, study_root=arguments.study_root)
    preflight = (
        None
        if arguments.sample_limit is not None
        else _require_preflight(arguments.study_root)
    )
    if preflight is not None:
        spec = _enriched_spec(spec, preflight)
    study_metadata = _study_metadata(
        preflight,
        project_id=spec.project_id,
        execution_shard_id=spec.execution_shard_id,
    )
    device = common._resolve_device(arguments.device or config.device)
    epochs = config.epochs if arguments.epochs is None else int(arguments.epochs)
    if epochs < 1 or epochs > config.epochs:
        raise ValueError("epoch override is outside the F protocol")
    paths = common.TrialPaths.create(
        _model_directory(config.study_directory, spec)
        if arguments.output_directory is None
        else arguments.output_directory
    )
    logger: Any = NullCometTrialLogger()
    try:
        if arguments.disable_comet and arguments.sample_limit is None:
            raise ValueError("Comet can only be disabled for sampled smoke trials")
        logger = common._create_trial_comet_logger(
            config,
            spec,
            paths,
            disabled=arguments.disable_comet,
        )
        if arguments.sample_limit is None:
            split, split_manifest = _load_shared_split(arguments.study_root)
        else:
            data = common.load_data(config, sample_limit=arguments.sample_limit)
            split, report = common.create_group_aware_split(
                data.centers,
                data.frames,
                seed=config.split_seed,
                fractions=config.split_fractions,
            )
            split_manifest = common._write_split(
                paths.directory,
                split,
                report,
                data.provenance,
            )
        summary = common.run_trial(
            config,
            spec,
            paths,
            split,
            split_manifest,
            logger,
            device=device,
            epochs=epochs,
            sample_limit=arguments.sample_limit,
            model_builder=_build_model,
            selected_model_audit_hook=_selected_model_audit,
            source_sha256=_source_sha256(),
            study_metadata=study_metadata,
        )
        print(json.dumps({"status": "complete", "summary": str(paths.summary)}))
        return 0 if summary["status"] == "complete" else 1
    except KeyboardInterrupt:
        logger.finish("interrupted")
        return 130
    except BaseException as error:
        common._record_error(paths, spec, error)
        try:
            logger.log_error(error, stage="trial")
            logger.finish("error")
        except BaseException:
            traceback.print_exc(file=sys.stderr)
        traceback.print_exception(error, file=sys.stderr)
        return 1


def run_smoke(arguments: argparse.Namespace) -> int:
    shard = _shard(arguments.shard)
    config = make_config(shard.shard_id, study_root=arguments.study_root)
    device = common._resolve_device(arguments.device or config.device)
    defaults = {
        0: ("F100", "F101", "F125"),
        1: ("F126", "F134", "F150"),
        2: ("F201", "F208", "F225"),
        3: ("F226", "F234", "F250"),
    }
    selected = _select_specs(
        shard.shard_id,
        arguments.model or defaults[shard.shard_id],
    )
    smoke_root = (
        Path(arguments.output_directory).resolve()
        if arguments.output_directory is not None
        else config.study_directory / "smoke" / shard.key
    )
    for spec in selected:
        output = smoke_root / spec.model_id
        command = [
            sys.executable,
            "-m",
            "experiments.benzene_pair.f_series.runner",
            "trial",
            "--study_root",
            str(Path(arguments.study_root).resolve()),
            "--model",
            spec.model_id,
            "--device",
            device,
            "--epochs",
            str(arguments.epochs),
            "--sample_limit",
            str(arguments.sample_limit),
            "--output_directory",
            str(output),
            "--disable_comet",
        ]
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


def run_prepare(arguments: argparse.Namespace) -> int:
    configs = tuple(
        make_config(shard_id, study_root=arguments.study_root) for shard_id in range(4)
    )
    protocol_fields = (
        "epochs",
        "effective_batch_size",
        "micro_batch_size",
        "learning_rate",
        "weight_decay",
        "scheduler_step_size",
        "scheduler_gamma",
        "validation_every",
        "split_seed",
        "model_seed",
        "shuffle_seed",
        "split_fractions",
        "dtype",
        "threads",
        "symmetry_tolerance",
        "symmetry_probe_count",
        "expected_sample_count",
        "expected_dataset_revision",
        "expected_opls_version",
        "enable_tf32",
        "relative_force_norm_sample_count",
        "relative_force_norm_seed",
        "schema_name",
        "schema_version",
    )
    reference_protocol = tuple(getattr(configs[0], name) for name in protocol_fields)
    if any(
        tuple(getattr(config, name) for name in protocol_fields) != reference_protocol
        for config in configs[1:]
    ) or any(config.shard_paths != configs[0].shard_paths for config in configs[1:]):
        raise RuntimeError("F execution shards do not share one training protocol")
    if arguments.reference_split_directory is None:
        _split, manifest = _load_shared_split(arguments.study_root)
    else:
        _split, manifest = _prepare_reference_split(
            arguments.study_root,
            arguments.reference_split_directory,
        )
    preflight = _compile_preflight(
        arguments.study_root,
        force=bool(getattr(arguments, "force_preflight", False)),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "shared_split_directory": str(
                    _shared_split_directory(arguments.study_root)
                ),
                "reference_split_directory": manifest["reference_split_directory"],
                "reference_manifest_hash": manifest["reference_manifest_hash"],
                "reference_indices_sha256": manifest["reference_indices_sha256"],
                "manifest_hash": manifest["manifest_hash"],
                "indices_sha256": manifest["indices_sha256"],
                "partition_counts": manifest["partition_counts"],
                "preflight_hash": preflight["preflight_hash"],
                "preflight_passed_model_count": preflight["passed_model_count"],
            }
        )
    )
    return 0


def run_aggregate(arguments: argparse.Namespace) -> int:
    """Build one cross-shard table and the fifty paired F1/F2 contrasts."""
    config = make_config(0, study_root=arguments.study_root)
    path = _refresh_results(
        config,
        F_SERIES_SPECS,
        output_directory=Path(arguments.study_root).resolve(),
    )
    print(json.dumps({"status": "complete", "results": str(path)}))
    return 0


def _tmux_devices(values: Sequence[str]) -> tuple[str, ...]:
    devices = tuple(str(value) for value in values)
    if not devices:
        visible = torch.cuda.device_count()
        if visible < 4:
            raise ValueError(
                "automatic tmux launch requires four visible CUDA devices; "
                "provide one --device to share intentionally or four explicit "
                "--device mappings"
            )
        return tuple(f"cuda:{index}" for index in range(4))
    if len(devices) == 1:
        return devices * 4
    if len(devices) != 4:
        raise ValueError("provide either one device or exactly four devices")
    return devices


def tmux_launch_commands(
    *,
    study_root: str | Path,
    devices: Sequence[str] = (),
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the four tmux commands without changing experimental semantics."""
    resolved_devices = _tmux_devices(devices)
    root = str(Path(study_root).resolve())
    result = []
    for shard_id, device in zip(range(4), resolved_devices):
        shard = EXECUTION_SHARDS[shard_id]
        job = shlex.join(
            (
                sys.executable,
                "-m",
                "experiments.benzene_pair.f_series.runner",
                "run",
                "--shard",
                shard.key,
                "--study_root",
                root,
                "--device",
                device,
            )
        )
        pane_command = (
            'if [ -z "${COMET_API_KEY:-}" ]; then '
            "echo 'COMET_API_KEY is unavailable inside tmux'; status=1; "
            f"else {job}; status=$?; fi; "
            f"echo 'F execution shard {shard.key} exited' $status; "
            'exec "${SHELL:-/bin/bash}"'
        )
        command = (
            "tmux",
            "new-session",
            "-d",
            "-s",
            shard.tmux_session_name,
            "-c",
            str(REPOSITORY_ROOT),
            pane_command,
        )
        result.append((shard.tmux_session_name, command))
    return tuple(result)


def run_launch_tmux(arguments: argparse.Namespace) -> int:
    commands = tmux_launch_commands(
        study_root=arguments.study_root,
        devices=arguments.device,
    )
    if arguments.dry_run:
        print(
            json.dumps(
                [
                    {"session": session, "command": shlex.join(command)}
                    for session, command in commands
                ],
                indent=2,
            )
        )
        return 0
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is not installed")
    if not os.environ.get("COMET_API_KEY", "").strip():
        raise RuntimeError("COMET_API_KEY must be set before launching tmux")
    _require_preflight(arguments.study_root)
    _load_shared_split(arguments.study_root)
    existing = []
    for session, _command in commands:
        check = subprocess.run(
            ("tmux", "has-session", "-t", f"={session}"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if check.returncode == 0:
            existing.append(session)
    if existing:
        raise RuntimeError(f"tmux sessions already exist: {tuple(existing)}")
    for session, command in commands:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        print(json.dumps({"status": "started", "session": session}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    prepare.add_argument("--reference_split_directory", type=Path, default=None)
    prepare.add_argument("--force_preflight", action="store_true")
    prepare.set_defaults(handler=run_prepare)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    preflight.add_argument("--force", action="store_true")
    preflight.set_defaults(
        handler=lambda arguments: (
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "preflight_hash": _compile_preflight(
                            arguments.study_root,
                            force=arguments.force,
                        )["preflight_hash"],
                    }
                )
            )
            or 0
        )
    )
    run = commands.add_parser("run")
    run.add_argument(
        "--shard",
        choices=tuple(item.key for item in EXECUTION_SHARDS.values()),
        required=True,
    )
    run.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    run.add_argument("--device", default=None)
    run.add_argument("--model", action="append", default=[])
    run.set_defaults(handler=run_study)
    trial = commands.add_parser("trial")
    trial.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    trial.add_argument("--model", required=True)
    trial.add_argument("--device", default=None)
    trial.add_argument("--epochs", type=int, default=None)
    trial.add_argument("--sample_limit", type=int, default=None)
    trial.add_argument("--output_directory", type=Path, default=None)
    trial.add_argument("--disable_comet", action="store_true")
    trial.set_defaults(handler=run_trial_command)
    smoke = commands.add_parser("smoke")
    smoke.add_argument(
        "--shard",
        choices=tuple(item.key for item in EXECUTION_SHARDS.values()),
        required=True,
    )
    smoke.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    smoke.add_argument("--device", default=None)
    smoke.add_argument("--model", action="append", default=[])
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--sample_limit", type=int, default=16000)
    smoke.add_argument("--output_directory", type=Path, default=None)
    smoke.set_defaults(handler=run_smoke)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    aggregate.set_defaults(handler=run_aggregate)
    tmux = commands.add_parser("launch-tmux")
    tmux.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    tmux.add_argument(
        "--device",
        action="append",
        default=[],
        help="repeat four times for a per-session device mapping",
    )
    tmux.add_argument("--dry-run", action="store_true")
    tmux.set_defaults(handler=run_launch_tmux)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    return int(parsed.handler(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
