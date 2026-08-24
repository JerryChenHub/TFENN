"""Run the paired-seed G study on CPU and record every trial in Comet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from experiments.benzene_pair import sweep30 as common
from experiments.benzene_pair.comet_logging import (
    LOSS_TIME_TEST_ERROR_PROFILE,
    NullCometTrialLogger,
)
from experiments.benzene_pair.g_series.catalog import (
    GModelSpec,
    G_SERIES_SPECS,
    get_group_specs,
    get_model_spec,
)
from experiments.benzene_pair.g_series.model_factory import build_g_series_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "benzene_pair"
    / "runs"
    / "g_series_e311_mechanisms_cpu_v1"
)
E311_SPLIT_INDICES_SHA256 = (
    "50e6bf0e32c1bb9b0bddb689097a4a38a5d74a5bcf12b0fc8471f6b1f4cf50b1"
)
E311_SPLIT_MANIFEST_HASH = (
    "3a64eb6ac96805aad4fe41ef1fd44a0cdc2417d193f8c0852244780a3563ec98"
)
GROUP_NAMES = ("factorial", "legacy", "carrier")
EXPECTED_MODEL_COUNT = 70
MODEL_BUDGET = 100
PRACTICAL_MAE_MARGIN_FRACTION = 0.02
EXPECTED_COMET_PROJECT = "tfenn_g_series_e311_mechanisms_cpu"
EXPECTED_STUDY_DIRECTORY_NAME = "g_series_e311_mechanisms_cpu_v1"
G_STUDY_METADATA = {
    "series": "G",
    "planned_model_count": EXPECTED_MODEL_COUNT,
    "model_budget": MODEL_BUDGET,
    "execution_device": "cpu",
    "historical_control": "E311",
    "reference_split_model": "E311",
    "fresh_gx01_controls_required": True,
    "historical_e311_metric_is_direct_control": False,
    "optimizer_updates_per_epoch": 320,
    "historical_e311_optimizer_updates_per_epoch": 32,
    "batch_protocol_note": (
        "batch 1000 changes the optimizer-step count, so only same-seed Gx01 "
        "runs are primary controls"
    ),
    "origin_trigger": "preliminary F1/F2 underperformance",
    "strict_flow_in_scope": False,
    "gnn_in_scope": False,
    "f_results_used_as_g_evidence": False,
    "primary_checkpoint_rule": "minimum validation normalized MSE",
    "practical_mae_margin_percent": 100.0 * PRACTICAL_MAE_MARGIN_FRACTION,
    "necessity_language_rule": (
        "use necessary only when paired retraining effects exceed the practical "
        "margin consistently; otherwise report marginal, substitutable, or "
        "optimization effects"
    ),
}


class GSeriesConfig(common.SweepConfig):
    """The fixed CPU protocol for G, isolated from the older GPU protocol."""

    def validate(self) -> None:
        if len(self.shard_paths) != 4:
            raise ValueError("G requires all four four-hundred-thousand-sample shards")
        if self.epochs != 500:
            raise ValueError("G requires five hundred epochs")
        if self.effective_batch_size != 1_000:
            raise ValueError("G effective batch size must equal one thousand")
        if self.micro_batch_size != 1_000:
            raise ValueError("G micro batch size must equal one thousand")
        if self.validation_every != 1:
            raise ValueError("G validation must run after every epoch")
        if not math.isclose(self.learning_rate, 0.003):
            raise ValueError("G learning rate must equal 0.003")
        if not math.isclose(self.weight_decay, 0.0001):
            raise ValueError("G weight decay must equal 0.0001")
        if self.scheduler_step_size != 125:
            raise ValueError("G scheduler cadence must equal 125 epochs")
        if not math.isclose(self.scheduler_gamma, 0.5):
            raise ValueError("G scheduler gamma must equal 0.5")
        if self.split_seed != 20260821:
            raise ValueError("G must retain the E311 split seed")
        if len(self.split_fractions) != 3 or any(
            not math.isclose(value, expected)
            for value, expected in zip(self.split_fractions, (0.8, 0.1, 0.1))
        ):
            raise ValueError("G must retain the E311 80/10/10 split fractions")
        if self.device != "cpu":
            raise ValueError("G is a CPU-only study")
        if self.dtype != "float32":
            raise ValueError("G requires float32")
        if self.threads < 1:
            raise ValueError("G CPU thread count must be positive")
        if self.expected_sample_count != 400_000:
            raise ValueError("G requires four hundred thousand samples")
        if self.expected_dataset_revision != 3:
            raise ValueError("G requires benzene-pair dataset revision three")
        if self.expected_opls_version != "2.0.0":
            raise ValueError("G requires OPLS runtime version 2.0.0")
        if self.enable_tf32:
            raise ValueError("G CPU training must disable TF32")
        if self.symmetry_tolerance <= 0.0 or self.symmetry_probe_count < 1:
            raise ValueError("G symmetry settings must be positive")
        if self.relative_force_norm_sample_count < 1:
            raise ValueError("G force-norm sample count must be positive")
        if not self.comet.enabled or not self.comet.required_online:
            raise ValueError("formal G trials require online Comet recording")


RESULT_FIELDS = (
    "model_id",
    "variant_id",
    "seed_index",
    "model_seed",
    "shuffle_seed",
    "family",
    "architecture_name",
    "description",
    "purpose",
    "comparison_role",
    "options",
    "status",
    "parameter_count",
    "causally_active_parameter_scalar_count",
    "best_epoch",
    "best_validation_normalized_mse",
    "train_normalized_mse",
    "validation_normalized_mse",
    "test_normalized_mse",
    "train_mae",
    "validation_mae",
    "test_mae",
    "test_sae",
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


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def make_config(
    *,
    study_root: str | Path = DEFAULT_STUDY_ROOT,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> GSeriesConfig:
    """Read and strictly validate the single formal G protocol."""
    path = Path(config_path).resolve()
    value = common._load_json(path)
    if value.get("schema_name") != "tfenn_benzene_pair_g_series":
        raise ValueError("unexpected G series config schema")
    if value.get("schema_version") != 1:
        raise ValueError("unexpected G series config version")
    if value.get("study_directory_name") != EXPECTED_STUDY_DIRECTORY_NAME:
        raise ValueError("unexpected G study directory name")
    if int(value.get("model_count", -1)) != EXPECTED_MODEL_COUNT:
        raise ValueError("G config model count does not match the catalog")
    if int(value.get("variant_count", -1)) != 14:
        raise ValueError("G config variant count must equal fourteen")
    if int(value.get("paired_seed_count", -1)) != 5:
        raise ValueError("G config paired seed count must equal five")
    if int(value.get("batch_size", -1)) != 1_000:
        raise ValueError("G config batch size must equal one thousand")
    if int(value.get("batch_size", -1)) != 1_000:
        raise ValueError("G config batch_size must equal one thousand")
    if int(value.get("model_count", -1)) != EXPECTED_MODEL_COUNT:
        raise ValueError("G config model count does not match the registered study")
    if int(value.get("paired_seed_count", -1)) != 5:
        raise ValueError("G config must declare five paired seeds")
    model_ids = tuple(str(item) for item in value.get("model_ids", ()))
    expected_ids = tuple(spec.model_id for spec in G_SERIES_SPECS)
    if model_ids != expected_ids:
        raise ValueError("G config model ids do not match the catalog")
    if len(expected_ids) != EXPECTED_MODEL_COUNT or len(set(expected_ids)) != len(
        expected_ids
    ):
        raise RuntimeError("G catalog must contain seventy unique trials")
    if len(expected_ids) > MODEL_BUDGET:
        raise RuntimeError("G catalog exceeds the one-hundred-model budget")
    registered_seeds = tuple(
        (
            int(item["seed_index"]),
            int(item["model_seed"]),
            int(item["shuffle_seed"]),
        )
        for item in value.get("paired_seeds", ())
    )
    expected_seeds = tuple(
        (
            seed_index,
            next(
                spec.model_seed
                for spec in G_SERIES_SPECS
                if spec.seed_index == seed_index
            ),
            next(
                spec.shuffle_seed
                for spec in G_SERIES_SPECS
                if spec.seed_index == seed_index
            ),
        )
        for seed_index in range(1, 6)
    )
    if registered_seeds != expected_seeds:
        raise ValueError("G config paired seeds do not match the catalog")
    paired_seeds = {
        int(item["seed_index"]): (
            int(item["model_seed"]),
            int(item["shuffle_seed"]),
        )
        for item in value.get("paired_seeds", ())
    }
    if set(paired_seeds) != set(range(1, 6)) or any(
        paired_seeds[spec.seed_index] != (spec.model_seed, spec.shuffle_seed)
        for spec in G_SERIES_SPECS
    ):
        raise ValueError("G config paired seeds do not match the catalog")
    config = GSeriesConfig(
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
        relative_force_norm_sample_count=int(
            value["relative_force_norm_sample_count"]
        ),
        relative_force_norm_seed=int(value["relative_force_norm_seed"]),
        comet=common.CometConfig.from_mapping(value["comet"]),
        schema_name=str(value["schema_name"]),
        schema_version=int(value["schema_version"]),
    )
    config.validate()
    if config.comet.project_name != EXPECTED_COMET_PROJECT:
        raise ValueError("G Comet project does not match the registered protocol")
    return config


def _config_for_spec(config: GSeriesConfig, spec: GModelSpec) -> GSeriesConfig:
    result = replace(
        config,
        model_seed=int(spec.model_seed),
        shuffle_seed=int(spec.shuffle_seed),
    )
    if not isinstance(result, GSeriesConfig):
        raise TypeError("G seed replacement changed the config type")
    result.validate()
    return result


def _resolve_execution_device(requested: str | None, config: GSeriesConfig) -> str:
    device = common._resolve_device(config.device if requested is None else requested)
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid G execution device: {device}") from error
    if parsed.type not in {"cpu", "cuda"}:
        raise ValueError("G training supports only CPU and CUDA devices")
    return str(parsed)


def _config_for_device(config: GSeriesConfig, device: str) -> GSeriesConfig:
    if torch.device(device).type != "cuda":
        return config
    tags = tuple(item for item in config.comet.tags if item not in {"cpu", "cuda"})
    result = replace(
        config,
        comet=replace(config.comet, tags=(*tags, "cuda")),
    )
    if not isinstance(result, GSeriesConfig):
        raise TypeError("G device replacement changed the config type")
    result.validate()
    return result


def _source_sha256() -> str:
    g_directory = Path(__file__).resolve().parent
    fixed_paths = (
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "__init__.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "sweep30.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "comet_logging.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "metrics.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "train.py",
    )
    dependency_paths = tuple(
        sorted(
            (
                *g_directory.rglob("*.py"),
                *g_directory.rglob("*.json"),
                *(
                    REPOSITORY_ROOT
                    / "experiments"
                    / "benzene_pair"
                    / "e_series"
                ).rglob("*.py"),
                *(REPOSITORY_ROOT / "src" / "TFENN" / "models").rglob("*.py"),
                *(
                    REPOSITORY_ROOT / "src" / "TFENN" / "tensor_math"
                ).rglob("*.py"),
                *(
                    REPOSITORY_ROOT / "experiments" / "benzene_pair" / "data"
                ).rglob("*.py"),
                REPOSITORY_ROOT
                / "experiments"
                / "benzene_pair"
                / "group_conv_baseline.py",
            ),
            key=lambda item: item.relative_to(REPOSITORY_ROOT).as_posix(),
        )
    )
    paths = tuple(
        path
        for path in dict.fromkeys((*fixed_paths, *dependency_paths))
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf_8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _shared_split_directory(study_root: str | Path) -> Path:
    return Path(study_root).resolve() / "shared_split"


def _require_e311_split(manifest: Mapping[str, Any]) -> None:
    if manifest.get("sample_count") != 400_000:
        raise RuntimeError("G split does not contain four hundred thousand samples")
    if manifest.get("partition_counts") != {
        "train": 320_000,
        "validation": 40_000,
        "test": 40_000,
    }:
        raise RuntimeError("G split partition counts do not match E311")
    if manifest.get("indices_sha256") != E311_SPLIT_INDICES_SHA256:
        raise RuntimeError("G split indices do not exactly match E311")
    if manifest.get("reference_manifest_hash") != E311_SPLIT_MANIFEST_HASH:
        raise RuntimeError("G split does not record the exact E311 manifest")
    if manifest.get("reference_indices_sha256") != E311_SPLIT_INDICES_SHA256:
        raise RuntimeError("G split does not record the exact E311 indices")


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
            raise RuntimeError("existing G split provenance does not match E311")
        return existing_split, existing_manifest
    directory.mkdir(parents=True, exist_ok=True)
    source_indices = Path(str(reference_manifest["indices_path"]))
    target_indices = directory / "split_indices.npz"
    partial = target_indices.with_name(f"{target_indices.name}.{os.getpid()}.partial")
    shutil.copyfile(source_indices, partial)
    os.replace(partial, target_indices)
    if common.sha256_file(target_indices) != E311_SPLIT_INDICES_SHA256:
        raise RuntimeError("copied G split indices hash does not match E311")
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
        raise RuntimeError("copied G split counts do not match E311")
    return copied_split, copied_manifest


def _load_shared_split(
    study_root: str | Path,
) -> tuple[common.SplitIndices, dict[str, Any]]:
    directory = _shared_split_directory(study_root)
    if not (directory / "split_manifest.json").is_file():
        raise RuntimeError("G shared split is missing; run prepare first")
    split, manifest = common._load_split(directory)
    _require_e311_split(manifest)
    return split, manifest


def _study_metadata(device: str) -> dict[str, Any]:
    return {
        **G_STUDY_METADATA,
        "execution_device": device,
        "reference_split_manifest_hash": E311_SPLIT_MANIFEST_HASH,
        "reference_split_indices_sha256": E311_SPLIT_INDICES_SHA256,
    }


def _study_manifest(
    config: GSeriesConfig,
    split_manifest: Mapping[str, Any],
    *,
    device: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Build the selection-independent manifest for one immutable G study."""
    value = {
        "schema_name": "tfenn_g_series_study",
        "schema_version": 1,
        "model_count": len(G_SERIES_SPECS),
        "models": [spec.as_dict() for spec in G_SERIES_SPECS],
        "config": config.as_dict(device=device),
        "shared_split_directory": str(_shared_split_directory(config.study_directory)),
        "reference_split_manifest_hash": E311_SPLIT_MANIFEST_HASH,
        "reference_split_indices_sha256": E311_SPLIT_INDICES_SHA256,
        "split_manifest_hash": split_manifest["manifest_hash"],
        "split_indices_sha256": split_manifest["indices_sha256"],
        "source_sha256": source_sha256,
        **_study_metadata(device),
    }
    value["study_hash"] = common._canonical_sha256(value)
    return value


def _write_invocation_record(
    config: GSeriesConfig,
    manifest: Mapping[str, Any],
    selected: Sequence[GModelSpec],
    *,
    groups: Sequence[str],
    requested_models: Sequence[str],
) -> Path:
    """Record a mutable runner selection without changing the study identity."""
    value = {
        "schema_name": "tfenn_g_series_invocation",
        "schema_version": 1,
        "study_hash": manifest["study_hash"],
        "groups": [str(item) for item in groups],
        "requested_models": [str(item).upper() for item in requested_models],
        "selected_model_ids": [spec.model_id for spec in selected],
    }
    invocation_hash = common._canonical_sha256(value)
    value["invocation_hash"] = invocation_hash
    return common._atomic_json(
        config.study_directory / "invocations" / f"{invocation_hash}.json",
        value,
    )


def _select_specs(
    groups: Sequence[str],
    values: Sequence[str],
) -> tuple[GModelSpec, ...]:
    if groups:
        available_list: list[GModelSpec] = []
        for group in groups:
            available_list.extend(get_group_specs(str(group).lower()))
        available = tuple(
            {spec.model_id: spec for spec in available_list}.values()
        )
    else:
        available = tuple(G_SERIES_SPECS)
    if not values:
        return available
    allowed = {spec.model_id: spec for spec in available}
    selected = []
    for value in values:
        key = str(value).upper()
        if key not in allowed:
            scope = "selected G groups" if groups else "the G catalog"
            raise ValueError(f"model {value} is outside {scope}")
        selected.append(allowed[key])
    if len({spec.model_id for spec in selected}) != len(selected):
        raise ValueError("G model selection contains duplicates")
    return tuple(selected)


def _build_model(spec: GModelSpec, device: str) -> nn.Module:
    model = build_g_series_model(
        spec,
        common._proper_d6_generators(),
        generator_names=("sixfold", "twofold"),
    )
    return model.to(device=torch.device(device), dtype=torch.float32)


def _selected_model_audit(**values: Any) -> Mapping[str, Any]:
    from experiments.benzene_pair.g_series.gate_audit import (
        export_selected_gate_audit,
    )

    return export_selected_gate_audit(**values)


def _result_row(config: GSeriesConfig, spec: GModelSpec) -> dict[str, Any]:
    paths = common.TrialPaths.create(config.study_directory / "models" / spec.model_id)
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

    return {
        "model_id": spec.model_id,
        "variant_id": spec.variant_id,
        "seed_index": spec.seed_index,
        "model_seed": spec.model_seed,
        "shuffle_seed": spec.shuffle_seed,
        "family": spec.family,
        "architecture_name": spec.architecture_name,
        "description": spec.description,
        "purpose": spec.purpose,
        "comparison_role": spec.comparison_role,
        "options": json.dumps(
            _json_value(spec.options),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "status": summary.get("status", status.get("status", "planned")),
        "parameter_count": summary.get("model", {}).get("parameter_count", ""),
        "causally_active_parameter_scalar_count": audit.get(
            "causally_active_parameter_scalar_count", ""
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
        "test_sae": metric("test", "sae"),
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
            "invariant_gate_parameter_path", ""
        ),
        "duration_seconds": summary.get("runtime", {}).get("duration_seconds", ""),
        "error_type": error.get("exception_type", ""),
        "error_message": error.get("message", ""),
    }


def _control_gate_importance(
    completed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate role-level Gate parameter evidence across fresh Gx01 controls."""
    per_role: dict[tuple[str, str], list[dict[str, Any]]] = {}
    control_model_ids = []
    for row in completed:
        if int(row["variant_id"]) != 1 or not row.get("gate_audit_path"):
            continue
        path = Path(str(row["gate_audit_path"]))
        if not path.is_file():
            continue
        report = common._load_json(path)
        if int(report.get("variant_id", -1)) != 1:
            raise RuntimeError("G control Gate artifact has the wrong variant")
        control_model_ids.append(str(row["model_id"]))
        composed_by_role: dict[tuple[str, str], list[float]] = {}
        for item in report.get("head_composed_weight_statistics", ()):
            if not bool(item.get("ranking_eligible", False)):
                continue
            key = (str(item["stage"]), str(item["descriptor_role"]))
            composed_by_role.setdefault(key, []).append(
                float(item["composed_weight"]["rms"])
            )
        seed_rows: list[dict[str, Any]] = []
        for item in report.get("trunk_input_column_statistics", ()):
            if not bool(item.get("ranking_eligible", False)):
                continue
            key = (str(item["stage"]), str(item["role"]))
            composed = composed_by_role.get(key, ())
            if not composed:
                continue
            seed_rows.append(
                {
                    "key": key,
                    "stage": key[0],
                    "descriptor_role": key[1],
                    "descriptor_kind": str(item["kind"]),
                    "source_names": _json_value(item.get("source_names", ())),
                    "trunk_weight_rms": float(item["weight"]["rms"]),
                    "active_head_composed_weight_rms": mean(composed),
                }
            )
        by_stage: dict[str, list[dict[str, Any]]] = {}
        for item in seed_rows:
            by_stage.setdefault(str(item["stage"]), []).append(item)
        for values in by_stage.values():
            ordered = sorted(
                values,
                key=lambda item: float(
                    item["active_head_composed_weight_rms"]
                ),
                reverse=True,
            )
            for rank, item in enumerate(ordered, start=1):
                item["stage_rank"] = rank
                per_role.setdefault(item["key"], []).append(item)
    summary = []
    for (stage, role), values in per_role.items():
        trunk_values = [float(item["trunk_weight_rms"]) for item in values]
        composed_values = [
            float(item["active_head_composed_weight_rms"]) for item in values
        ]
        ranks = [int(item["stage_rank"]) for item in values]
        summary.append(
            {
                "stage": stage,
                "descriptor_role": role,
                "descriptor_kind": str(values[0]["descriptor_kind"]),
                "source_names": values[0]["source_names"],
                "completed_control_seed_count": len(values),
                "mean_trunk_weight_rms": mean(trunk_values),
                "median_trunk_weight_rms": median(trunk_values),
                "mean_active_head_composed_weight_rms": mean(composed_values),
                "median_active_head_composed_weight_rms": median(
                    composed_values
                ),
                "mean_within_stage_rank": mean(ranks),
                "median_within_stage_rank": median(ranks),
                "top_five_seed_frequency": sum(rank <= 5 for rank in ranks)
                / len(ranks),
            }
        )
    summary.sort(
        key=lambda item: (
            str(item["stage"]),
            float(item["mean_within_stage_rank"]),
            str(item["descriptor_role"]),
        )
    )
    return {
        "control_model_ids": sorted(control_model_ids),
        "completed_control_seed_count": len(control_model_ids),
        "ranking_metric": "mean active-head V@W RMS within each stage",
        "interpretation": (
            "descriptive parameter allocation only; V@W omits sample-dependent "
            "SiLU derivatives and is not a causal invariant ablation"
        ),
        "descriptor_role_summary": summary,
    }


def _contrast_summary(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in values:
        grouped.setdefault(str(item["contrast"]), []).append(item)
    margin = math.log1p(PRACTICAL_MAE_MARGIN_FRACTION)
    result = []
    for name, rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "contrast": name,
            "completed_seed_count": len(rows),
            "seed_indices": sorted(int(row["seed_index"]) for row in rows),
            "practical_equivalence_log_mae_margin": margin,
            "practical_equivalence_mae_percent": (
                100.0 * PRACTICAL_MAE_MARGIN_FRACTION
            ),
        }
        for partition in ("validation", "test"):
            key = f"{partition}_mae_log_contrast"
            observations = [float(row[key]) for row in rows]
            item[f"mean_{key}"] = mean(observations)
            item[f"median_{key}"] = median(observations)
            item[f"minimum_{key}"] = min(observations)
            item[f"maximum_{key}"] = max(observations)
            item[f"seed_count_above_positive_margin_{partition}"] = sum(
                value > margin for value in observations
            )
            item[f"seed_count_below_negative_margin_{partition}"] = sum(
                value < -margin for value in observations
            )
        result.append(item)
    return result


def _refresh_results(config: GSeriesConfig) -> Path:
    config.study_directory.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(config, spec) for spec in G_SERIES_SPECS]
    path = config.study_directory / "results.csv"
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)
    completed = [row for row in rows if row["status"] == "complete"]
    ranking = sorted(
        completed,
        key=lambda row: float(row["validation_mae"]),
    )
    by_variant: dict[int, list[dict[str, Any]]] = {}
    for row in completed:
        by_variant.setdefault(int(row["variant_id"]), []).append(row)
    variant_summary = []
    for variant_id, values in sorted(by_variant.items()):
        validation_mae = [float(item["validation_mae"]) for item in values]
        test_mae = [float(item["test_mae"]) for item in values]
        variant_summary.append(
            {
                "variant_id": variant_id,
                "completed_seed_count": len(values),
                "seed_indices": sorted(int(item["seed_index"]) for item in values),
                "mean_validation_mae": mean(validation_mae),
                "median_validation_mae": median(validation_mae),
                "mean_test_mae": mean(test_mae),
                "median_test_mae": median(test_mae),
            }
        )
    by_seed_variant = {
        (int(row["seed_index"]), int(row["variant_id"])): row
        for row in completed
    }
    paired_control_contrasts = []
    for (seed_index, variant_id), row in sorted(by_seed_variant.items()):
        if variant_id == 1:
            continue
        control = by_seed_variant.get((seed_index, 1))
        if control is None:
            continue
        validation_ratio = float(row["validation_mae"]) / float(
            control["validation_mae"]
        )
        test_ratio = float(row["test_mae"]) / float(control["test_mae"])
        if min(validation_ratio, test_ratio) <= 0.0:
            raise RuntimeError("G paired MAE ratios must be positive")
        paired_control_contrasts.append(
            {
                "seed_index": seed_index,
                "control_model_id": str(control["model_id"]),
                "candidate_model_id": str(row["model_id"]),
                "candidate_variant_id": variant_id,
                "validation_log_mae_ratio": math.log(validation_ratio),
                "validation_mae_percent_change": 100.0 * (validation_ratio - 1.0),
                "test_log_mae_ratio": math.log(test_ratio),
                "test_mae_percent_change": 100.0 * (test_ratio - 1.0),
            }
        )
    factorial_definitions = {
        "generic_addback_with_both_stf": {5: 1.0, 1: -1.0},
        "both_stf_at_generic_off": {1: 1.0, 4: -1.0},
        "a1_stf_at_generic_off_out_on": {1: 1.0, 3: -1.0},
        "out_stf_at_generic_off_a1_on": {1: 1.0, 2: -1.0},
        "generic_by_both_stf_interaction": {
            5: 1.0,
            1: -1.0,
            8: -1.0,
            4: 1.0,
        },
        "a1_by_out_stf_interaction_generic_off": {
            1: 1.0,
            2: -1.0,
            3: -1.0,
            4: 1.0,
        },
        "a1_by_out_stf_interaction_generic_on": {
            5: 1.0,
            6: -1.0,
            7: -1.0,
            8: 1.0,
        },
        "generic_by_a1_by_out_three_way_interaction": {
            5: 1.0,
            6: -1.0,
            7: -1.0,
            8: 1.0,
            1: -1.0,
            2: 1.0,
            3: 1.0,
            4: -1.0,
        },
    }
    factorial_log_mae_contrasts = []
    for seed_index in range(1, 6):
        if not all((seed_index, variant) in by_seed_variant for variant in range(1, 9)):
            continue
        for name, coefficients in factorial_definitions.items():
            value = {
                "seed_index": seed_index,
                "contrast": name,
                "coefficient_by_variant": {
                    str(key): coefficient
                    for key, coefficient in coefficients.items()
                },
            }
            for metric in ("validation_mae", "test_mae"):
                logs = {
                    variant: math.log(
                        float(by_seed_variant[(seed_index, variant)][metric])
                    )
                    for variant in coefficients
                }
                value[f"{metric}_log_contrast"] = sum(
                    coefficients[variant] * logs[variant]
                    for variant in coefficients
                )
            factorial_log_mae_contrasts.append(value)
    factorial_orthogonal_definitions = {
        "generic_pair_main_effect": {
            1: -0.25,
            2: -0.25,
            3: -0.25,
            4: -0.25,
            5: 0.25,
            6: 0.25,
            7: 0.25,
            8: 0.25,
        },
        "a1_stf_main_effect": {
            1: 0.25,
            2: 0.25,
            3: -0.25,
            4: -0.25,
            5: 0.25,
            6: 0.25,
            7: -0.25,
            8: -0.25,
        },
        "out_stf_main_effect": {
            1: 0.25,
            2: -0.25,
            3: 0.25,
            4: -0.25,
            5: 0.25,
            6: -0.25,
            7: 0.25,
            8: -0.25,
        },
        "generic_pair_by_a1_stf": {
            1: -0.25,
            2: -0.25,
            3: 0.25,
            4: 0.25,
            5: 0.25,
            6: 0.25,
            7: -0.25,
            8: -0.25,
        },
        "generic_pair_by_out_stf": {
            1: -0.25,
            2: 0.25,
            3: -0.25,
            4: 0.25,
            5: 0.25,
            6: -0.25,
            7: 0.25,
            8: -0.25,
        },
        "a1_stf_by_out_stf": {
            1: 0.25,
            2: -0.25,
            3: -0.25,
            4: 0.25,
            5: 0.25,
            6: -0.25,
            7: -0.25,
            8: 0.25,
        },
        "generic_pair_by_a1_stf_by_out_stf": {
            1: -0.25,
            2: 0.25,
            3: 0.25,
            4: -0.25,
            5: 0.25,
            6: -0.25,
            7: -0.25,
            8: 0.25,
        },
    }
    factorial_orthogonal_log_mae_contrasts = []
    for seed_index in range(1, 6):
        if not all(
            (seed_index, variant) in by_seed_variant
            for variant in range(1, 9)
        ):
            continue
        for name, coefficients in factorial_orthogonal_definitions.items():
            value = {
                "seed_index": seed_index,
                "contrast": name,
                "coefficient_by_variant": {
                    str(key): coefficient
                    for key, coefficient in coefficients.items()
                },
            }
            for metric in ("validation_mae", "test_mae"):
                value[f"{metric}_log_contrast"] = sum(
                    coefficient
                    * math.log(
                        float(by_seed_variant[(seed_index, variant)][metric])
                    )
                    for variant, coefficient in coefficients.items()
                )
            factorial_orthogonal_log_mae_contrasts.append(value)
    mechanism_definitions = {
        "no_carrier_vs_direct_legacy": {9: 1.0, 1: -1.0},
        "gated_residual_zero_vs_direct_legacy": {10: 1.0, 1: -1.0},
        "gated_default_vs_gated_residual_zero": {11: 1.0, 10: -1.0},
        "raw_deep_carrier_effect_hidden_on": {1: 1.0, 12: -1.0},
        "hidden_deep_carrier_effect_raw_on": {1: 1.0, 13: -1.0},
        "raw_deep_by_hidden_deep_carrier_interaction": {
            1: 1.0,
            12: -1.0,
            13: -1.0,
            14: 1.0,
        },
        "stem_and_adjacent_carriers_vs_none": {14: 1.0, 9: -1.0},
    }
    mechanism_log_mae_contrasts = []
    for seed_index in range(1, 6):
        for name, coefficients in mechanism_definitions.items():
            if not all(
                (seed_index, variant) in by_seed_variant
                for variant in coefficients
            ):
                continue
            value = {
                "seed_index": seed_index,
                "contrast": name,
                "coefficient_by_variant": {
                    str(key): coefficient
                    for key, coefficient in coefficients.items()
                },
            }
            for metric in ("validation_mae", "test_mae"):
                value[f"{metric}_log_contrast"] = sum(
                    coefficient
                    * math.log(
                        float(by_seed_variant[(seed_index, variant)][metric])
                    )
                    for variant, coefficient in coefficients.items()
                )
            mechanism_log_mae_contrasts.append(value)
    gate_importance = _control_gate_importance(completed)
    factorial_orthogonal_summary = _contrast_summary(
        factorial_orthogonal_log_mae_contrasts
    )
    mechanism_summary = _contrast_summary(mechanism_log_mae_contrasts)
    common._atomic_json(
        config.study_directory / "comparison.json",
        {
            "schema_name": "tfenn_g_series_comparison",
            "schema_version": 1,
            "primary_selection_metric": "Validation MAE",
            "test_is_not_used_for_model_selection": True,
            "planned_model_count": len(G_SERIES_SPECS),
            "completed_model_count": len(completed),
            "error_model_count": sum(row["status"] == "error" for row in rows),
            "ranking_by_validation_mae": ranking,
            "variant_summary": variant_summary,
            "paired_vs_same_seed_gx01_control": paired_control_contrasts,
            "factorial_log_mae_contrasts": factorial_log_mae_contrasts,
            "factorial_orthogonal_log_mae_contrasts": (
                factorial_orthogonal_log_mae_contrasts
            ),
            "factorial_orthogonal_contrast_summary": (
                factorial_orthogonal_summary
            ),
            "factorial_primary_analysis": (
                "seven standard orthogonal effects on log MAE with factors coded "
                "off=-1 and on=+1; coefficients are one quarter"
            ),
            "mechanism_log_mae_contrasts": mechanism_log_mae_contrasts,
            "mechanism_contrast_summary": mechanism_summary,
            "g5_control_gate_parameter_importance": gate_importance,
            "factor_interpretation": (
                "generic CSignature and analytic STF parameterizations can span "
                "overlapping directions; factorial contrasts measure marginal, "
                "substitution, and interaction effects rather than orthogonal "
                "information spaces"
            ),
            "updated_at_utc": common._utc_now(),
        },
    )
    return path


def run_study(arguments: argparse.Namespace) -> int:
    config = make_config(study_root=arguments.study_root)
    device = _resolve_execution_device(arguments.device, config)
    config = _config_for_device(config, device)
    if not os.environ.get("COMET_API_KEY", "").strip():
        raise RuntimeError("COMET_API_KEY must be set for a formal G series run")
    _split, split_manifest = _load_shared_split(arguments.study_root)
    selected = _select_specs(arguments.group, arguments.model)
    source_sha256 = _source_sha256()
    study_metadata = _study_metadata(device)
    manifest = _study_manifest(
        config,
        split_manifest,
        device=device,
        source_sha256=source_sha256,
    )
    config.study_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = config.study_directory / "manifest.json"
    if manifest_path.is_file() and common._load_json(manifest_path) != manifest:
        raise RuntimeError("existing G study manifest does not match this run")
    common._atomic_json(manifest_path, manifest)
    _write_invocation_record(
        config,
        manifest,
        selected,
        groups=arguments.group,
        requested_models=arguments.model,
    )
    _refresh_results(config)
    failed_models = []
    for spec in selected:
        trial_config = _config_for_spec(config, spec)
        paths = common.TrialPaths.create(
            trial_config.study_directory / "models" / spec.model_id
        )
        if paths.summary.is_file():
            completed = common._load_json(paths.summary)
            expected_hash = common._trial_hash(
                trial_config,
                spec,
                split_manifest,
                device=device,
                epochs=trial_config.epochs,
                source_sha256=source_sha256,
                study_metadata=study_metadata,
            )
            if completed.get("status") != "complete":
                raise RuntimeError("existing G trial summary is not complete")
            if completed.get("trial_hash") != expected_hash:
                raise RuntimeError("existing G trial summary hash does not match")
            if not paths.best.is_file() or not paths.final.is_file():
                raise RuntimeError("completed G trial is missing a checkpoint")
            continue
        paths.directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "experiments.benzene_pair.g_series.runner",
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
        _refresh_results(config)
        if process.returncode == 130:
            return 130
        if process.returncode:
            failed_models.append(spec.model_id)
    _refresh_results(config)
    if failed_models:
        print(
            json.dumps({"status": "error", "failed_models": failed_models}),
            file=sys.stderr,
        )
        return 1
    return 0


def run_trial_command(arguments: argparse.Namespace) -> int:
    spec = get_model_spec(arguments.model)
    base_config = make_config(study_root=arguments.study_root)
    device = _resolve_execution_device(arguments.device, base_config)
    base_config = _config_for_device(base_config, device)
    config = _config_for_spec(base_config, spec)
    epochs = config.epochs if arguments.epochs is None else int(arguments.epochs)
    if epochs < 1 or epochs > config.epochs:
        raise ValueError("epoch override is outside the G protocol")
    nonformal = arguments.sample_limit is not None or epochs != config.epochs
    if nonformal and arguments.output_directory is None:
        raise ValueError(
            "sampled or shortened G trials require an explicit output directory"
        )
    paths = common.TrialPaths.create(
        config.study_directory / "models" / spec.model_id
        if arguments.output_directory is None
        else arguments.output_directory
    )
    logger: Any = NullCometTrialLogger()
    try:
        if arguments.disable_comet and arguments.sample_limit is None:
            raise ValueError("Comet can only be disabled for sampled smoke trials")
        if (
            arguments.sample_limit is None
            and not os.environ.get("COMET_API_KEY", "").strip()
        ):
            raise RuntimeError("COMET_API_KEY must be set for a formal G trial")
        logger = common._create_trial_comet_logger(
            config,
            spec,
            paths,
            disabled=arguments.disable_comet,
            metric_profile=LOSS_TIME_TEST_ERROR_PROFILE,
            experiment_name=(
                f"{spec.model_id}_" if device.startswith("cuda") else spec.model_id
            ),
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
            study_metadata=_study_metadata(device),
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
    config = make_config(study_root=arguments.study_root)
    device = _resolve_execution_device(arguments.device, config)
    config = _config_for_device(config, device)
    available = _select_specs(arguments.group, ())
    selected = (
        _select_specs(arguments.group, arguments.model)
        if arguments.model
        else available[:1]
    )
    smoke_root = (
        Path(arguments.output_directory).resolve()
        if arguments.output_directory is not None
        else config.study_directory / "smoke"
    )
    for spec in selected:
        command = [
            sys.executable,
            "-m",
            "experiments.benzene_pair.g_series.runner",
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
            str(smoke_root / spec.model_id),
            "--disable_comet",
        ]
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
        if result.returncode:
            return int(result.returncode)
    return 0


def run_prepare(arguments: argparse.Namespace) -> int:
    config = make_config(study_root=arguments.study_root)
    _resolve_execution_device(None, config)
    if arguments.reference_split_directory is None:
        _split, manifest = _load_shared_split(arguments.study_root)
    else:
        _split, manifest = _prepare_reference_split(
            arguments.study_root,
            arguments.reference_split_directory,
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
                "model_count": len(G_SERIES_SPECS),
                "source_sha256": _source_sha256(),
            }
        )
    )
    return 0


def run_aggregate(arguments: argparse.Namespace) -> int:
    config = make_config(study_root=arguments.study_root)
    path = _refresh_results(config)
    print(json.dumps({"status": "complete", "results": str(path)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    prepare.add_argument("--reference_split_directory", type=Path, default=None)
    prepare.set_defaults(handler=run_prepare)
    run = commands.add_parser("run")
    run.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    run.add_argument("--device", default=None)
    run.add_argument(
        "--group",
        action="append",
        type=str.lower,
        choices=GROUP_NAMES,
        default=[],
    )
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
    smoke.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    smoke.add_argument("--device", default=None)
    smoke.add_argument(
        "--group",
        action="append",
        type=str.lower,
        choices=GROUP_NAMES,
        default=[],
    )
    smoke.add_argument("--model", action="append", default=[])
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--sample_limit", type=int, default=16_000)
    smoke.add_argument("--output_directory", type=Path, default=None)
    smoke.set_defaults(handler=run_smoke)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    aggregate.set_defaults(handler=run_aggregate)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    return int(parsed.handler(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
