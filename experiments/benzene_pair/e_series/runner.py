"""Run the five E series studies with one shared data split."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
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
from experiments.benzene_pair.e_series.catalog import (
    EModelSpec,
    get_experiment_specs,
    get_model_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "experiments" / "benzene_pair" / "runs" / "e_series_400k_v1"
)
E_STUDY_METADATA = {
    "concurrent_run": True,
    "shared_gpu_process_count": 5,
    "plan_document_name": "E_SERIES_125_MODEL_EXPERIMENT_PLAN (1).md",
    "plan_title_model_count": 125,
    "plan_body_model_count": 108,
    "executed_model_count": 108,
    "count_resolution": "The explicit scope and complete model tables define 108 runs",
    "primary_metric": "Test_MAE",
}


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    experiment_id: int
    directory_name: str
    comet_project: str
    purpose: str


EXPERIMENTS = {
    0: ExperimentDefinition(
        0,
        "e0_shared_controls",
        "tfenn_e_series_e0_controls",
        "shared controls and capacity references",
    ),
    1: ExperimentDefinition(
        1,
        "e1_raw_reuse_bypass",
        "tfenn_e_series_e1_raw_reuse_bypass",
        "raw input reuse and legacy bypass",
    ),
    2: ExperimentDefinition(
        2,
        "e2_dual_stream_exchange",
        "tfenn_e_series_e2_dual_stream_exchange",
        "synchronous parallel streams and typed exchange",
    ),
    3: ExperimentDefinition(
        3,
        "e3_path_gate_width",
        "tfenn_e_series_e3_path_gate_width",
        "path bank sparsity and Gate width",
    ),
    4: ExperimentDefinition(
        4,
        "e4_compact_8k",
        "tfenn_e_series_e4_compact_8k",
        "complete context compression near eight thousand parameters",
    ),
}
DEFAULT_CONFIG_PATHS = {
    experiment_id: Path(__file__).resolve().parent / f"experiment_{experiment_id}.json"
    for experiment_id in EXPERIMENTS
}


RESULT_FIELDS = (
    "model_id",
    "architecture_name",
    "description",
    "purpose",
    "comparison_role",
    "status",
    "planned_parameter_count",
    "actual_parameter_count",
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
    "duration_seconds",
    "error_type",
    "error_message",
)


def _experiment(value: int) -> ExperimentDefinition:
    try:
        return EXPERIMENTS[int(value)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("experiment must be zero through four") from error


def _shared_split_directory(study_root: str | Path) -> Path:
    return Path(study_root).resolve() / "shared_split"


def make_config(
    experiment_id: int,
    *,
    study_root: str | Path = DEFAULT_STUDY_ROOT,
) -> common.SweepConfig:
    """Build one fixed four hundred thousand sample E protocol."""
    experiment = _experiment(experiment_id)
    value = json.loads(DEFAULT_CONFIG_PATHS[experiment_id].read_text(encoding="utf_8"))
    if value.get("schema_name") != "tfenn_benzene_pair_e_series":
        raise ValueError("unexpected E series config schema")
    if value.get("schema_version") != 1:
        raise ValueError("unexpected E series config version")
    if int(value.get("experiment_id", -1)) != experiment_id:
        raise ValueError("E config experiment does not match")
    if value.get("study_directory_name") != experiment.directory_name:
        raise ValueError("E study directory does not match")
    expected_models = tuple(
        item.model_id for item in get_experiment_specs(experiment_id)
    )
    if tuple(value.get("model_ids", ())) != expected_models:
        raise ValueError("E config model ids do not match the catalog")
    for name in ("concurrent_run", "shared_gpu_process_count"):
        if value.get(name) != E_STUDY_METADATA[name]:
            raise ValueError(f"E config {name} does not match")
    config = common.SweepConfig(
        shard_paths=tuple(
            (REPOSITORY_ROOT / str(item)).resolve() for item in value["shard_paths"]
        ),
        study_directory=Path(study_root).resolve() / experiment.directory_name,
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
        raise ValueError("E series requires five hundred epochs")
    if config.effective_batch_size != 10_000 or config.micro_batch_size != 10_000:
        raise ValueError("E series requires batch size ten thousand")
    if config.expected_sample_count != 400_000:
        raise ValueError("E series requires four hundred thousand samples")
    if config.comet.project_name != experiment.comet_project:
        raise ValueError("E Comet project does not match")
    return config


def _source_sha256() -> str:
    fixed_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "catalog.py",
        Path(__file__).resolve().parent / "model_factory.py",
        *tuple(DEFAULT_CONFIG_PATHS.values()),
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "sweep30.py",
        REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "invariant_gate_v2_20k_sweep.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "d_series" / "catalog.py",
        REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "d_series"
        / "model_factory.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "comet_logging.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "metrics.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "train.py",
        REPOSITORY_ROOT / "src" / "TFENN" / "models" / "invariant_gate_pipeline_v2.py",
        REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "e_series"
        / "model_support.py",
        REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "group_conv_baseline.py",
        REPOSITORY_ROOT / "src" / "TFENN" / "models" / "__init__.py",
    )
    dependency_paths = tuple(
        sorted(
            (
                *(REPOSITORY_ROOT / "src" / "TFENN" / "tensor_math").rglob("*.py"),
                *(
                    REPOSITORY_ROOT / "experiments" / "benzene_pair" / "data"
                ).rglob("*.py"),
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


def _preflight_path(study_root: str | Path) -> Path:
    return Path(study_root).resolve() / "preflight_manifest.json"


def _catalog_sha256() -> str:
    models = [
        spec.as_dict()
        for experiment_id in EXPERIMENTS
        for spec in get_experiment_specs(experiment_id)
    ]
    return common._canonical_sha256(models)


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
        raise RuntimeError("Tier C forward check failed")
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
        raise RuntimeError("covariant gradient connectivity check failed")
    symmetry = common.symmetry_metrics(
        model,
        centers,
        frames,
        tolerance=tolerance,
    )
    if not symmetry["passed"]:
        raise RuntimeError("covariance check failed")
    return {
        "forward_finite": True,
        "gradient_finite": True,
        "all_trainable_parameters_connected": True,
        "trainable_tensor_count": len(trainable),
        "gradient_tensor_count": len(gradients),
        "dtype": str(dtype),
        "symmetry": symmetry,
    }


def _preflight_model(spec: EModelSpec) -> dict[str, Any]:
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
    architecture_metadata = getattr(model, "architecture_metadata", {})
    offline = getattr(model, "offline_compilation_summary", {})
    selected_roles = tuple(getattr(model, "selected_covariant_roles", ()))
    budget = getattr(model, "budget_compilation_manifest", None)
    candidate_manifest = tuple(getattr(model, "candidate_manifest", ()))
    candidate_status_counts: dict[str, int] = {}
    for item in candidate_manifest:
        status = str(item.get("status", "unknown"))
        candidate_status_counts[status] = candidate_status_counts.get(status, 0) + 1
    unit_check = None if spec.d6_covariance_exempt else _covariant_unit_check(model)
    tier_c = spec.experiment_id == 4 and spec.options.get("tier") == "C"
    record = {
        "status": "passed",
        "model_id": spec.model_id,
        "planned_parameter_count": spec.planned_parameter_count,
        "actual_parameter_count": actual,
        "target_parameter_range": None
        if spec.target_parameter_range is None
        else list(spec.target_parameter_range),
        "budget_passed": spec.target_parameter_range is None
        or spec.target_parameter_range[0] <= actual <= spec.target_parameter_range[1],
        "d6_covariance_exempt": spec.d6_covariance_exempt,
        "architecture_metadata": architecture_metadata,
        "compiled_config": configuration_dict,
        "budget_compilation_manifest": budget,
        "offline_compilation_summary": offline,
        "candidate_status_counts": candidate_status_counts,
        "selected_covariant_roles": list(selected_roles),
        "covariant_unit_check": unit_check,
        "tier_c_requirement": bool(tier_c),
    }
    del model
    gc.collect()
    return record


def _compile_preflight(
    study_root: str | Path, *, force: bool = False
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
            and existing.get("passed_model_count") == 108
        ):
            return existing
    records = []
    for experiment_id in EXPERIMENTS:
        for spec in get_experiment_specs(experiment_id):
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
    passed = sum(item["status"] == "passed" for item in records)
    manifest = {
        "schema_name": "tfenn_e_series_preflight",
        "schema_version": 1,
        "status": "passed" if passed == 108 else "failed",
        "source_sha256": source_sha,
        "catalog_sha256": catalog_sha,
        "model_count": 108,
        "passed_model_count": passed,
        "failed_model_count": 108 - passed,
        "models": records,
        **E_STUDY_METADATA,
        "created_at_utc": common._utc_now(),
    }
    manifest["preflight_hash"] = common._canonical_sha256(manifest)
    common._atomic_json(path, manifest)
    if manifest["status"] != "passed":
        raise RuntimeError(
            f"E preflight failed for {manifest['failed_model_count']} models"
        )
    return manifest


def _require_preflight(study_root: str | Path) -> Mapping[str, Any]:
    path = _preflight_path(study_root)
    if not path.is_file():
        raise RuntimeError("E preflight manifest is missing")
    manifest = common._load_json(path)
    if manifest.get("status") != "passed" or manifest.get("passed_model_count") != 108:
        raise RuntimeError("E preflight did not pass all one hundred eight models")
    if manifest.get("source_sha256") != _source_sha256():
        raise RuntimeError("E preflight source hash is stale")
    if manifest.get("catalog_sha256") != _catalog_sha256():
        raise RuntimeError("E preflight catalog hash is stale")
    expected_hash = manifest.get("preflight_hash")
    unsigned = dict(manifest)
    unsigned.pop("preflight_hash", None)
    if expected_hash != common._canonical_sha256(unsigned):
        raise RuntimeError("E preflight manifest hash is invalid")
    return manifest


def _study_metadata(preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        **E_STUDY_METADATA,
        "preflight_hash": None if preflight is None else preflight["preflight_hash"],
    }


def _preflight_record(preflight: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for record in preflight["models"]:
        if record.get("model_id") == model_id:
            return record
    raise KeyError(f"preflight has no record for {model_id}")


def _enriched_spec(spec: EModelSpec, preflight: Mapping[str, Any]) -> EModelSpec:
    options = dict(spec.options)
    options["compiled_preflight"] = dict(_preflight_record(preflight, spec.model_id))
    return replace(spec, options=options)


def _build_model(
    spec: EModelSpec,
    device: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    from experiments.benzene_pair.e_series.model_factory import build_e_series_model

    model = build_e_series_model(
        spec,
        common._proper_d6_generators(),
        generator_names=("sixfold", "twofold"),
    )
    if not isinstance(model, nn.Module):
        raise TypeError("E series builder must return a torch module")
    return model.to(device=torch.device(device), dtype=dtype)


def _select_specs(experiment_id: int, values: Sequence[str]) -> tuple[EModelSpec, ...]:
    available = tuple(get_experiment_specs(experiment_id))
    expected_count = 8 if experiment_id == 0 else 25
    if len(available) != expected_count:
        raise RuntimeError("E experiment size changed")
    if not values:
        return available
    allowed = {item.model_id: item for item in available}
    selected = []
    for value in values:
        key = str(value).upper()
        if key not in allowed:
            raise ValueError(f"model {value} is outside experiment {experiment_id}")
        selected.append(allowed[key])
    if len({item.model_id for item in selected}) != len(selected):
        raise ValueError("model selection contains duplicates")
    return tuple(selected)


def _result_row(config: common.SweepConfig, spec: EModelSpec) -> dict[str, Any]:
    paths = common.TrialPaths.create(config.study_directory / "models" / spec.model_id)
    summary = common._load_json(paths.summary) if paths.summary.is_file() else {}
    status = common._load_json(paths.status) if paths.status.is_file() else {}
    error = common._load_json(paths.error) if paths.error.is_file() else {}
    selection = summary.get("selection", {})
    metrics = selection.get("selected_metrics", {})
    norm = summary.get("relative_force_norm_difference", {}).get("test", {})

    def metric(partition: str, name: str) -> Any:
        value = metrics.get(partition, {})
        return value.get(name, "") if isinstance(value, Mapping) else ""

    symmetry = selection.get("symmetry", {})
    d6_status: Any
    if spec.d6_covariance_exempt:
        d6_status = "exempt"
    elif not symmetry:
        d6_status = ""
    else:
        d6_status = "passed" if symmetry.get("passed") else "failed"
    return {
        "model_id": spec.model_id,
        "architecture_name": spec.architecture_name,
        "description": spec.description,
        "purpose": spec.purpose,
        "comparison_role": spec.comparison_role,
        "status": summary.get("status", status.get("status", "planned")),
        "planned_parameter_count": spec.planned_parameter_count or "",
        "actual_parameter_count": summary.get("model", {}).get("parameter_count", ""),
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
        "d6_status": d6_status,
        "duration_seconds": summary.get("runtime", {}).get("duration_seconds", ""),
        "error_type": error.get("exception_type", ""),
        "error_message": error.get("message", ""),
    }


def _refresh_results(
    config: common.SweepConfig,
    specs: Sequence[EModelSpec],
) -> Path:
    config.study_directory.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(config, spec) for spec in specs]
    path = config.study_directory / "results.csv"
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)
    completed = [row for row in rows if row["status"] == "complete"]
    ranking = sorted(completed, key=lambda row: float(row["test_mae"]))
    common._atomic_json(
        config.study_directory / "comparison.json",
        {
            "schema_name": "tfenn_e_series_comparison",
            "schema_version": 1,
            "primary_metric": "Test_MAE",
            "completed_model_count": len(completed),
            "error_model_count": sum(row["status"] == "error" for row in rows),
            "ranking_by_test_mae": ranking,
            "updated_at_utc": common._utc_now(),
        },
    )
    return path


def run_study(arguments: argparse.Namespace) -> int:
    experiment = _experiment(arguments.experiment)
    config = make_config(experiment.experiment_id, study_root=arguments.study_root)
    if not os.environ.get("COMET_API_KEY", "").strip():
        raise RuntimeError("COMET_API_KEY must be set for a formal E series run")
    preflight = _require_preflight(arguments.study_root)
    study_metadata = _study_metadata(preflight)
    device = common._resolve_device(arguments.device or config.device)
    all_specs = tuple(
        _enriched_spec(spec, preflight)
        for spec in get_experiment_specs(experiment.experiment_id)
    )
    selected = tuple(
        _enriched_spec(spec, preflight)
        for spec in _select_specs(experiment.experiment_id, arguments.model)
    )
    split_config = replace(
        config,
        study_directory=_shared_split_directory(arguments.study_root),
    )
    split_manifest = common._prepare_split(split_config)
    manifest = {
        "schema_name": "tfenn_e_series_study",
        "schema_version": 1,
        "experiment_id": experiment.experiment_id,
        "experiment_purpose": experiment.purpose,
        "model_count": len(all_specs),
        "models": [item.as_dict() for item in all_specs],
        "config": config.as_dict(device=device),
        "shared_split_directory": str(_shared_split_directory(arguments.study_root)),
        "split_manifest_hash": split_manifest["manifest_hash"],
        "source_sha256": _source_sha256(),
        **study_metadata,
    }
    manifest["study_hash"] = common._canonical_sha256(manifest)
    manifest_path = config.study_directory / "manifest.json"
    if manifest_path.is_file() and common._load_json(manifest_path) != manifest:
        raise RuntimeError("existing E series manifest does not match this run")
    common._atomic_json(manifest_path, manifest)
    _refresh_results(config, all_specs)
    for spec in selected:
        paths = common.TrialPaths.create(
            config.study_directory / "models" / spec.model_id
        )
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
                raise RuntimeError("existing E trial summary is not complete")
            if completed.get("trial_hash") != expected_hash:
                raise RuntimeError("existing E trial summary hash does not match")
            if not paths.best.is_file() or not paths.final.is_file():
                raise RuntimeError("completed E trial is missing a checkpoint")
            continue
        paths.directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "experiments.benzene_pair.e_series.runner",
            "trial",
            "--experiment",
            str(experiment.experiment_id),
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
        _refresh_results(config, all_specs)
        if process.returncode == 130:
            return 130
    _refresh_results(config, all_specs)
    return 0


def run_trial_command(arguments: argparse.Namespace) -> int:
    experiment = _experiment(arguments.experiment)
    config = make_config(experiment.experiment_id, study_root=arguments.study_root)
    spec = get_model_spec(arguments.model)
    if spec not in tuple(get_experiment_specs(experiment.experiment_id)):
        raise ValueError("model does not belong to the selected experiment")
    preflight = (
        None
        if arguments.sample_limit is not None
        else _require_preflight(arguments.study_root)
    )
    if preflight is not None:
        spec = _enriched_spec(spec, preflight)
    study_metadata = _study_metadata(preflight)
    device = common._resolve_device(arguments.device or config.device)
    epochs = config.epochs if arguments.epochs is None else int(arguments.epochs)
    if epochs < 1 or epochs > config.epochs:
        raise ValueError("epoch override is outside the E protocol")
    paths = common.TrialPaths.create(
        config.study_directory / "models" / spec.model_id
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
            split, split_manifest = common._load_split(
                _shared_split_directory(arguments.study_root)
            )
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
    experiment = _experiment(arguments.experiment)
    config = make_config(experiment.experiment_id, study_root=arguments.study_root)
    device = common._resolve_device(arguments.device or config.device)
    defaults = {
        0: ("E001", "E003", "E008"),
        1: ("E101", "E118", "E125"),
        2: ("E201", "E213", "E225"),
        3: ("E301", "E312", "E325"),
        4: ("E401", "E423", "E424", "E425"),
    }
    selected = _select_specs(
        experiment.experiment_id,
        arguments.model or defaults[experiment.experiment_id],
    )
    smoke_root = (
        Path(arguments.output_directory).resolve()
        if arguments.output_directory is not None
        else config.study_directory / "smoke"
    )
    for spec in selected:
        output = smoke_root / spec.model_id
        command = [
            sys.executable,
            "-m",
            "experiments.benzene_pair.e_series.runner",
            "trial",
            "--experiment",
            str(experiment.experiment_id),
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
    config = make_config(0, study_root=arguments.study_root)
    shared = replace(
        config,
        study_directory=_shared_split_directory(arguments.study_root),
    )
    manifest = common._prepare_split(shared)
    preflight = _compile_preflight(
        arguments.study_root,
        force=bool(getattr(arguments, "force_preflight", False)),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "shared_split_directory": str(shared.study_directory),
                "manifest_hash": manifest["manifest_hash"],
                "partition_counts": manifest["partition_counts"],
                "preflight_hash": preflight["preflight_hash"],
                "preflight_passed_model_count": preflight["passed_model_count"],
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
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
        "--experiment", type=int, choices=tuple(EXPERIMENTS), required=True
    )
    run.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    run.add_argument("--device", default=None)
    run.add_argument("--model", action="append", default=[])
    run.set_defaults(handler=run_study)
    trial = commands.add_parser("trial")
    trial.add_argument(
        "--experiment", type=int, choices=tuple(EXPERIMENTS), required=True
    )
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
        "--experiment", type=int, choices=tuple(EXPERIMENTS), required=True
    )
    smoke.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    smoke.add_argument("--device", default=None)
    smoke.add_argument("--model", action="append", default=[])
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--sample_limit", type=int, default=16000)
    smoke.add_argument("--output_directory", type=Path, default=None)
    smoke.set_defaults(handler=run_smoke)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    return int(parsed.handler(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
