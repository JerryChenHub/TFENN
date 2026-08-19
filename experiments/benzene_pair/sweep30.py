"""Train one group averaged baseline and thirty invariant gate models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from experiments.benzene_pair.comet_logging import (
    CometConfig,
    CometTrialLogger,
    NullCometTrialLogger,
    create_comet_trial_logger,
)
from experiments.benzene_pair.invariant_gate_v2_20k_sweep import (
    MODEL_SPECS,
    ModelSpec,
    build_sweep_model,
    get_model_spec,
)
from experiments.benzene_pair.metrics import (
    summarize_relative_force_norm_difference,
)
from experiments.benzene_pair.train import (
    _proper_d6_generators,
    regression_metrics,
    sha256_file,
    symmetry_metrics,
)
from TFENN.data import load_benzene_cluster_csv, load_benzene_cluster_metadata
from TFENN.models.model_level_group_conv_mlp import (
    ModelLevelGroupConvMLPConfig,
    build_model_level_group_conv_mlp,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "sweep30_config.json"
HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "train_normalized_mse",
    "validation_normalized_mse",
    "validation_relative_rmse_percent",
    "normalization_minimum_count",
    "normalization_maximum_count",
    "epoch_duration_seconds",
)
RESULT_FIELDS = (
    "model_id",
    "description",
    "purpose",
    "comparison_role",
    "status",
    "parameter_count",
    "best_epoch",
    "best_validation_normalized_mse",
    "train_relative_rmse_percent",
    "validation_relative_rmse_percent",
    "test_relative_rmse_percent",
    "train_fit_accuracy_percent",
    "validation_fit_accuracy_percent",
    "test_fit_accuracy_percent",
    "test_r2",
    "test_r2_percent",
    "train_relative_force_norm_min",
    "train_relative_force_norm_median",
    "train_relative_force_norm_max",
    "validation_relative_force_norm_min",
    "validation_relative_force_norm_median",
    "validation_relative_force_norm_max",
    "test_relative_force_norm_min",
    "test_relative_force_norm_median",
    "test_relative_force_norm_max",
    "d6_passed",
    "duration_seconds",
    "error_type",
    "error_message",
)


@dataclass(frozen=True, slots=True)
class GroupConvSpec:
    """Describe the model level group averaged comparison baseline."""

    model_id: str
    description: str
    purpose: str
    expected_parameter_count: int
    architecture: ModelLevelGroupConvMLPConfig
    comparison_role: str = "baseline"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "description": self.description,
            "purpose": self.purpose,
            "expected_parameter_count": self.expected_parameter_count,
            "comparison_role": self.comparison_role,
            "model_family": "model_level_group_conv_mlp",
            "architecture": self.architecture.as_dict(),
        }


GROUP_CONV_SPEC = GroupConvSpec(
    model_id="G00",
    description="model level D6 by D6 Reynolds averaged MLP with widths 96 96 96",
    purpose="twenty thousand parameter model level group convolution baseline",
    expected_parameter_count=20_160,
    architecture=ModelLevelGroupConvMLPConfig(
        hidden_widths=(96, 96, 96),
        activation="silu",
        distance_scale=6.0,
        seed=20260822,
    ),
)
StudySpec = ModelSpec | GroupConvSpec
STUDY_SPECS: tuple[StudySpec, ...] = (GROUP_CONV_SPEC, *MODEL_SPECS)
CometLogger = CometTrialLogger | NullCometTrialLogger
ModelBuilder = Callable[[Any, str], nn.Module]
CalibrationHook = Callable[..., Mapping[str, Any] | None]
SelectedModelAuditHook = Callable[..., Mapping[str, Any]]


def get_study_spec(model_id: str) -> StudySpec:
    """Return the baseline or one invariant gate model specification."""
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string")
    if model_id.upper() == GROUP_CONV_SPEC.model_id:
        return GROUP_CONV_SPEC
    return get_model_spec(model_id)


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Store the common protocol for every model."""

    shard_paths: tuple[Path, ...]
    study_directory: Path
    epochs: int
    effective_batch_size: int
    micro_batch_size: int
    learning_rate: float
    weight_decay: float
    scheduler_step_size: int
    scheduler_gamma: float
    validation_every: int
    split_seed: int
    model_seed: int
    shuffle_seed: int
    split_fractions: tuple[float, float, float]
    device: str
    dtype: str
    threads: int
    symmetry_tolerance: float
    symmetry_probe_count: int
    expected_sample_count: int
    expected_dataset_revision: int
    expected_opls_version: str
    enable_tf32: bool
    relative_force_norm_sample_count: int
    relative_force_norm_seed: int
    comet: CometConfig
    schema_name: str = "tfenn_benzene_pair_sweep31"
    schema_version: int = 1

    @classmethod
    def from_path(cls, path: str | Path) -> SweepConfig:
        config_path = Path(path).resolve()
        value = _load_json(config_path)
        if value.get("schema_name") != "tfenn_benzene_pair_sweep31":
            raise ValueError("unexpected sweep config schema")
        if value.get("schema_version") != 1:
            raise ValueError("unexpected sweep config version")

        def resolve(value: str) -> Path:
            candidate = Path(value)
            if candidate.is_absolute():
                return candidate.resolve()
            return (REPOSITORY_ROOT / candidate).resolve()

        result = cls(
            shard_paths=tuple(resolve(str(item)) for item in value["shard_paths"]),
            study_directory=resolve(str(value["study_directory"])),
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
            comet=CometConfig.from_mapping(value.get("comet")),
            schema_name=str(value["schema_name"]),
            schema_version=int(value["schema_version"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if len(self.shard_paths) != 4:
            raise ValueError("the formal study requires four shards")
        if self.epochs != 500:
            raise ValueError("the formal study requires five hundred epochs")
        if self.effective_batch_size != 10000:
            raise ValueError("effective batch size must equal 10000")
        if self.micro_batch_size != 10000:
            raise ValueError("micro batch size must equal 10000")
        if not 1 <= self.micro_batch_size <= self.effective_batch_size:
            raise ValueError("micro batch size is outside the effective batch")
        if self.effective_batch_size % self.micro_batch_size:
            raise ValueError("micro batch size must divide effective batch size")
        if self.validation_every != 1:
            raise ValueError("validation must run after every epoch")
        if self.scheduler_step_size != 125:
            raise ValueError("scheduler cadence must equal 125 epochs")
        if not math.isclose(self.scheduler_gamma, 0.5):
            raise ValueError("scheduler gamma must equal 0.5")
        if self.dtype != "float32":
            raise ValueError("the GPU sweep uses float32")
        if self.expected_sample_count != 400000:
            raise ValueError("the formal study requires 400000 samples")
        if not self.enable_tf32:
            raise ValueError("the formal GPU protocol enables TF32")
        if len(self.split_fractions) != 3 or not math.isclose(
            sum(self.split_fractions), 1.0, abs_tol=1.0e-12
        ):
            raise ValueError("split fractions must contain three values summing to one")
        if min(self.split_fractions) <= 0.0:
            raise ValueError("split fractions must be positive")
        if min(self.learning_rate, self.scheduler_gamma) <= 0.0:
            raise ValueError("optimizer values must be positive")
        if self.weight_decay < 0.0 or self.threads < 1:
            raise ValueError("weight decay and threads are invalid")
        if self.symmetry_probe_count < 1 or self.symmetry_tolerance <= 0.0:
            raise ValueError("symmetry settings must be positive")
        if self.relative_force_norm_sample_count < 1:
            raise ValueError("force norm sample count must be positive")
        if not self.comet.enabled or not self.comet.required_online:
            raise ValueError("the formal study requires online Comet recording")

    def protocol(self, *, device: str, epochs: int | None = None) -> dict[str, Any]:
        return {
            "epochs": self.epochs if epochs is None else int(epochs),
            "effective_batch_size": self.effective_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation": "sample weighted within each effective batch",
            "optimizer": "AdamW",
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "scheduler": "StepLR",
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
            "validation_every": self.validation_every,
            "split_seed": self.split_seed,
            "model_seed": self.model_seed,
            "shuffle_seed": self.shuffle_seed,
            "split_fractions": list(self.split_fractions),
            "device": device,
            "dtype": self.dtype,
            "threads": self.threads,
            "enable_tf32": self.enable_tf32,
            "relative_force_norm": {
                "sample_count_per_partition": self.relative_force_norm_sample_count,
                "sampling_seed": self.relative_force_norm_seed,
                "summary_statistics": [
                    "min",
                    "max",
                    "median",
                    "mean",
                    "p90",
                    "p95",
                    "p99",
                ],
            },
            "comet": self.comet.as_dict(),
            "normalization": (
                "invariant gate models use training partition warmup then cumulative "
                "running RMS during training and freeze it during validation and test; "
                "the group averaged baseline has no normalization buffers"
            ),
            "early_stopping": False,
            "model_selection": "minimum validation normalized MSE",
            "test_evaluation": "once from the validation selected checkpoint",
        }

    def as_dict(self, *, device: str, epochs: int | None = None) -> dict[str, Any]:
        """Return the complete credential free experiment configuration."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "shard_paths": [str(path) for path in self.shard_paths],
            "study_directory": str(self.study_directory),
            "protocol": self.protocol(device=device, epochs=epochs),
            "symmetry_tolerance": self.symmetry_tolerance,
            "symmetry_probe_count": self.symmetry_probe_count,
            "expected_sample_count": self.expected_sample_count,
            "expected_dataset_revision": self.expected_dataset_revision,
            "expected_opls_version": self.expected_opls_version,
        }


@dataclass(frozen=True, slots=True)
class TrainingData:
    centers: Tensor
    frames: Tensor
    root_force: Tensor
    provenance: dict[str, Any]

    def __len__(self) -> int:
        return int(self.root_force.shape[0])


@dataclass(frozen=True, slots=True)
class SplitIndices:
    train: Tensor
    validation: Tensor
    test: Tensor

    def counts(self) -> dict[str, int]:
        return {
            "train": int(self.train.numel()),
            "validation": int(self.validation.numel()),
            "test": int(self.test.numel()),
        }


@dataclass(frozen=True, slots=True)
class TrialPaths:
    directory: Path
    definition: Path
    comet: Path
    status: Path
    history: Path
    best: Path
    final: Path
    resume: Path
    summary: Path
    error: Path
    stdout: Path
    stderr: Path

    @classmethod
    def create(cls, directory: str | Path) -> TrialPaths:
        root = Path(directory).resolve()
        return cls(
            root,
            root / "config.json",
            root / "comet.json",
            root / "status.json",
            root / "history.csv",
            root / "best.pt",
            root / "final.pt",
            root / "resume.pt",
            root / "summary.json",
            root / "error.json",
            root / "stdout.log",
            root / "stderr.log",
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf_8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    partial.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf_8",
    )
    os.replace(partial, path)
    return path


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    torch.save(dict(value), partial)
    os.replace(partial, path)
    return path


def _comet_resume_backend(experiment_key: str):
    def factory(
        *,
        config: CometConfig,
        experiment_name: str,
        api_key: str,
        tags: Sequence[str],
    ) -> Any:
        import comet_ml

        experiment_config = comet_ml.ExperimentConfig(
            name=experiment_name,
            tags=list(tags),
        )
        return comet_ml.start(
            api_key=api_key,
            workspace=config.workspace,
            project_name=config.project_name,
            experiment_key=experiment_key,
            mode="get",
            online=True,
            experiment_config=experiment_config,
        )

    return factory


def _create_trial_comet_logger(
    config: SweepConfig,
    spec: StudySpec,
    paths: TrialPaths,
    *,
    disabled: bool,
) -> CometLogger:
    comet_config = CometConfig.from_mapping(None) if disabled else config.comet
    previous: dict[str, Any] | None = None
    backend_factory = None
    if not disabled and paths.comet.is_file():
        previous = _load_json(paths.comet)
        if previous.get("schema_name") != "tfenn_sweep31_comet_trial":
            raise ValueError("unexpected Comet trial record schema")
        if previous.get("model_id") != spec.model_id:
            raise ValueError("Comet trial record model does not match")
        if previous.get("project_name") != comet_config.project_name:
            raise ValueError("Comet trial record project does not match")
        experiment_key = str(previous.get("experiment_key", ""))
        if not experiment_key:
            raise ValueError("Comet trial record has no experiment key")
        backend_factory = _comet_resume_backend(experiment_key)
    logger = create_comet_trial_logger(
        comet_config,
        experiment_name=spec.model_id,
        study_name=config.study_directory.name,
        tags=(spec.model_id, spec.comparison_role),
        backend_factory=backend_factory,
    )
    if not logger.enabled:
        return logger
    identity = logger.identity
    experiment_key = str(identity.get("experiment_key", ""))
    if not experiment_key:
        raise RuntimeError("Comet did not provide an experiment key")
    if previous is not None and experiment_key != previous["experiment_key"]:
        raise RuntimeError("resumed Comet experiment key changed")
    _atomic_json(
        paths.comet,
        {
            "schema_name": "tfenn_sweep31_comet_trial",
            "schema_version": 1,
            "model_id": spec.model_id,
            "project_name": comet_config.project_name,
            "experiment_key": experiment_key,
            "identity": identity,
            "last_logged_epoch": -1
            if previous is None
            else int(previous.get("last_logged_epoch", -1)),
            "updated_at_utc": _utc_now(),
        },
    )
    return logger


def _log_comet_epoch(
    logger: CometLogger,
    paths: TrialPaths,
    row: Mapping[str, Any],
) -> None:
    validation_loss = row["validation_normalized_mse"]
    if validation_loss == "":
        return
    epoch = int(row["epoch"])
    record = _load_json(paths.comet) if logger.enabled else None
    if record is not None and epoch <= int(record.get("last_logged_epoch", -1)):
        return
    logger.log_epoch(
        epoch=epoch,
        train_loss=float(row["train_normalized_mse"]),
        validation_loss=float(validation_loss),
        learning_rate=float(row["learning_rate"]),
        extra_metrics={
            "validation_relative_rmse_percent": float(
                row["validation_relative_rmse_percent"]
            ),
            "normalization_minimum_count": int(row["normalization_minimum_count"]),
            "normalization_maximum_count": int(row["normalization_maximum_count"]),
            "epoch_duration_seconds": float(row["epoch_duration_seconds"]),
        },
    )
    if record is not None:
        record["last_logged_epoch"] = epoch
        record["updated_at_utc"] = _utc_now()
        _atomic_json(paths.comet, record)


def _canonical_sha256(value: Any) -> str:
    content = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf_8")
    return hashlib.sha256(content).hexdigest()


def _source_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "comet_logging.py",
        Path(__file__).resolve().parent / "invariant_gate_v2_20k_sweep.py",
        Path(__file__).resolve().parent / "metrics.py",
        REPOSITORY_ROOT / "src" / "TFENN" / "models" / "invariant_gate_pipeline_v2.py",
        REPOSITORY_ROOT / "src" / "TFENN" / "models" / "model_level_group_conv_mlp.py",
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


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _validate_shard(path: Path, config: SweepConfig) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    validation_path = path.with_suffix(".validation.json")
    if (
        not path.is_file()
        or not metadata_path.is_file()
        or not validation_path.is_file()
    ):
        raise FileNotFoundError(path)
    metadata = load_benzene_cluster_metadata(metadata_path)
    validation = _load_json(validation_path)
    csv_digest = sha256_file(path)
    metadata_digest = sha256_file(metadata_path)
    if metadata.get("csv_sha256") != csv_digest:
        raise ValueError(f"CSV digest mismatch for {path}")
    if validation.get("csv_sha256") != csv_digest:
        raise ValueError(f"validation CSV digest mismatch for {path}")
    if validation.get("metadata_sha256") != metadata_digest:
        raise ValueError(f"validation metadata digest mismatch for {path}")
    if validation.get("passed") is not True:
        raise ValueError(f"validation did not pass for {path}")
    if metadata.get("schema_version") != 2:
        raise ValueError("data schema version must equal two")
    if metadata.get("dataset_revision") != config.expected_dataset_revision:
        raise ValueError("dataset revision does not match the protocol")
    if metadata.get("molecule_count") != 2:
        raise ValueError("each sample must contain two molecules")
    opls = metadata.get("opls")
    if not isinstance(opls, Mapping):
        raise ValueError("OPLS provenance is missing")
    if opls.get("runtime_version") != config.expected_opls_version:
        raise ValueError("OPLS runtime version does not match the protocol")
    return {
        "csv_path": str(path),
        "csv_sha256": csv_digest,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_digest,
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "sample_count": int(metadata["sample_count"]),
    }


def load_data(
    config: SweepConfig,
    *,
    sample_limit: int | None = None,
    validate_files: bool = True,
) -> TrainingData:
    centers = []
    frames = []
    root_force = []
    provenance = []
    remaining = config.expected_sample_count if sample_limit is None else sample_limit
    if remaining < 3 or remaining > config.expected_sample_count:
        raise ValueError("sample limit is outside the available dataset")
    for path in config.shard_paths:
        record = (
            _validate_shard(path, config)
            if validate_files
            else {
                "csv_path": str(path),
                "csv_sha256": sha256_file(path),
            }
        )
        arrays = load_benzene_cluster_csv(path, dtype=np.float32)
        take = min(len(arrays), remaining)
        if take:
            centers.append(arrays.centers[:take])
            frames.append(arrays.rotations[:take])
            root_force.append(arrays.forces[:take, 0])
        record["loaded_sample_count"] = take
        provenance.append(record)
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError("shards contain fewer samples than requested")
    result = TrainingData(
        centers=torch.from_numpy(np.concatenate(centers)),
        frames=torch.from_numpy(np.concatenate(frames)),
        root_force=torch.from_numpy(np.concatenate(root_force)),
        provenance={"shards": provenance},
    )
    if not bool(
        torch.isfinite(result.centers).all()
        and torch.isfinite(result.frames).all()
        and torch.isfinite(result.root_force).all()
    ):
        raise ValueError("training data contains nonfinite values")
    return result


def create_group_aware_split(
    centers: Tensor,
    frames: Tensor,
    *,
    seed: int,
    fractions: Sequence[float],
) -> tuple[SplitIndices, dict[str, Any]]:
    """Assign exact duplicate poses to one deterministic partition."""
    count = int(centers.shape[0])
    if frames.shape != (count, 2, 3, 3):
        raise ValueError("frames have an unexpected shape")
    if centers.shape != (count, 2, 3):
        raise ValueError("centers have an unexpected shape")
    if len(fractions) != 3 or not math.isclose(sum(fractions), 1.0, abs_tol=1.0e-12):
        raise ValueError("split fractions are invalid")
    pose = np.ascontiguousarray(
        np.concatenate(
            (
                centers.numpy().reshape(count, -1),
                frames.numpy().reshape(count, -1),
            ),
            axis=1,
        )
    )
    keys = pose.view(np.dtype((np.void, pose.dtype.itemsize * pose.shape[1]))).ravel()
    _unique, inverse, group_sizes = np.unique(
        keys,
        return_inverse=True,
        return_counts=True,
    )
    group_order = np.random.default_rng(seed).permutation(len(group_sizes))
    target_counts = np.asarray(fractions, dtype=np.float64) * count
    group_partition = np.empty(len(group_sizes), dtype=np.int8)
    actual = np.zeros(3, dtype=np.int64)
    partition = 0
    for group in group_order:
        if partition < 2 and actual[partition] >= target_counts[partition]:
            partition += 1
        group_partition[group] = partition
        actual[partition] += group_sizes[group]
    sample_partition = group_partition[inverse]
    arrays = tuple(np.flatnonzero(sample_partition == index) for index in range(3))
    split = SplitIndices(*(torch.from_numpy(item.copy()) for item in arrays))
    duplicate_sizes = group_sizes[group_sizes > 1]
    report = {
        "seed": int(seed),
        "fractions": list(fractions),
        "sample_count": count,
        "unique_pose_count": int(len(group_sizes)),
        "duplicate_group_count": int(len(duplicate_sizes)),
        "duplicate_extra_sample_count": int((duplicate_sizes - 1).sum()),
        "largest_duplicate_group": int(duplicate_sizes.max(initial=1)),
        "partition_counts": split.counts(),
        "duplicate_groups_cross_partitions": 0,
    }
    return split, report


def _write_split(
    directory: Path,
    split: SplitIndices,
    report: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    indices_path = directory / "split_indices.npz"
    partial = indices_path.with_name(f"{indices_path.name}.{os.getpid()}.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(
            stream,
            train=split.train.numpy(),
            validation=split.validation.numpy(),
            test=split.test.numpy(),
        )
    os.replace(partial, indices_path)
    manifest = {
        "schema_name": "tfenn_benzene_pair_group_aware_split",
        "schema_version": 1,
        **dict(report),
        "indices_path": str(indices_path),
        "indices_sha256": sha256_file(indices_path),
        "data_provenance": dict(data_provenance),
    }
    manifest["manifest_hash"] = _canonical_sha256(manifest)
    _atomic_json(directory / "split_manifest.json", manifest)
    return manifest


def _load_split(directory: Path) -> tuple[SplitIndices, dict[str, Any]]:
    manifest = _load_json(directory / "split_manifest.json")
    expected_hash = manifest.pop("manifest_hash")
    if _canonical_sha256(manifest) != expected_hash:
        raise ValueError("split manifest hash mismatch")
    manifest["manifest_hash"] = expected_hash
    path = Path(str(manifest["indices_path"]))
    if sha256_file(path) != manifest["indices_sha256"]:
        raise ValueError("split indices hash mismatch")
    with np.load(path) as arrays:
        split = SplitIndices(
            torch.from_numpy(arrays["train"].copy()),
            torch.from_numpy(arrays["validation"].copy()),
            torch.from_numpy(arrays["test"].copy()),
        )
    if split.counts() != manifest["partition_counts"]:
        raise ValueError("split counts do not match the manifest")
    return split, manifest


def _prepare_split(config: SweepConfig) -> dict[str, Any]:
    manifest_path = config.study_directory / "split_manifest.json"
    if manifest_path.is_file():
        _split, manifest = _load_split(config.study_directory)
        if manifest["sample_count"] != config.expected_sample_count:
            raise RuntimeError(
                "existing split sample count does not match the protocol"
            )
        if manifest["seed"] != config.split_seed:
            raise RuntimeError("existing split seed does not match the protocol")
        if manifest["fractions"] != list(config.split_fractions):
            raise RuntimeError("existing split fractions do not match the protocol")
        return manifest
    data = load_data(config)
    split, report = create_group_aware_split(
        data.centers,
        data.frames,
        seed=config.split_seed,
        fractions=config.split_fractions,
    )
    return _write_split(config.study_directory, split, report, data.provenance)


def _build_model(spec: StudySpec, device: str) -> nn.Module:
    generators = _proper_d6_generators()
    if isinstance(spec, GroupConvSpec):
        model = build_model_level_group_conv_mlp(generators, spec.architecture)
    else:
        model = build_sweep_model(
            spec,
            generators,
            generator_names=("sixfold", "twofold"),
        )
    return model.to(device=torch.device(device), dtype=torch.float32)


def _parameter_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in model.named_parameters()}


def _normalization_state(model: nn.Module) -> dict[str, Tensor]:
    method = getattr(model, "normalization_state_dict", None)
    if method is None:
        return {}
    if not callable(method):
        raise TypeError("model normalization state attribute is not callable")
    return {name: value.detach().cpu() for name, value in method().items()}


def _calibration_state(model: nn.Module) -> dict[str, Tensor]:
    """Return optional learned independent calibration buffers."""
    method = getattr(model, "calibration_state_dict", None)
    if method is None:
        method = getattr(model, "descriptor_projection_state_dict", None)
    if method is None:
        return {}
    if not callable(method):
        raise TypeError("model calibration state attribute is not callable")
    state = method()
    if not isinstance(state, Mapping):
        raise TypeError("model calibration state must be a mapping")
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("model calibration state must map names to tensors")
        result[name] = value.detach().cpu()
    return result


def _restore_model_state(
    model: nn.Module,
    parameters: Mapping[str, Tensor],
    normalization: Mapping[str, Tensor],
    calibration: Mapping[str, Tensor] | None = None,
) -> None:
    current = dict(model.named_parameters())
    if set(current) != set(parameters):
        raise ValueError("checkpoint parameter names do not match the model")
    with torch.no_grad():
        for name, value in current.items():
            source = parameters[name]
            if source.shape != value.shape:
                raise ValueError(f"checkpoint shape mismatch for {name}")
            value.copy_(source.to(device=value.device, dtype=value.dtype))
    normalization_method = getattr(model, "load_normalization_state_dict", None)
    if normalization_method is None:
        if normalization:
            raise ValueError("model does not accept normalization state")
    elif not callable(normalization_method):
        raise TypeError("model normalization loader is not callable")
    else:
        normalization_method(normalization)
    if calibration is None:
        return
    calibration_values = calibration
    calibration_method = getattr(model, "load_calibration_state_dict", None)
    if calibration_method is None:
        calibration_method = getattr(
            model, "load_descriptor_projection_state_dict", None
        )
    if calibration_method is None:
        if calibration_values:
            raise ValueError("model does not accept calibration state")
        return
    if not callable(calibration_method):
        raise TypeError("model calibration loader is not callable")
    calibration_method(calibration_values)


def _checkpoint_payload(
    model: nn.Module,
    *,
    spec: StudySpec,
    epoch: int,
    target_scale: Tensor,
    metrics: Mapping[str, float],
    trial_hash: str,
    calibration_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": "tfenn_sweep31_learned_checkpoint",
        "schema_version": 1,
        "model_id": spec.model_id,
        "epoch": int(epoch),
        "target_scale": float(target_scale.detach().cpu()),
        "metrics": dict(metrics),
        "trial_hash": trial_hash,
        "fixed_tensor_artifacts_stored": False,
        "parameter_state_dict": _parameter_state(model),
        "normalization_state_dict": _normalization_state(model),
        "calibration_state_dict": _calibration_state(model),
        "calibration_report": dict(calibration_report or {}),
    }


def _save_resume(
    path: Path,
    base: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    shuffle_generator: torch.Generator,
    history: Sequence[Mapping[str, Any]],
    best_epoch: int,
    best_validation: Mapping[str, float],
) -> None:
    payload = {
        **dict(base),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "shuffle_generator_state": shuffle_generator.get_state(),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_validation": dict(best_validation),
    }
    _atomic_torch_save(path, payload)


def _batch_inputs(
    data: TrainingData,
    indices: Tensor,
    device: str,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        data.centers[indices].to(device=device, non_blocking=True),
        data.frames[indices].to(device=device, non_blocking=True),
        data.root_force[indices].to(device=device, non_blocking=True),
    )


def _normalization_counts(model: nn.Module) -> dict[str, Any]:
    state = _normalization_state(model)
    counts = [
        int(value.item())
        for name, value in state.items()
        if name.endswith("sample_count")
    ]
    return {
        "path_count": len(counts),
        "minimum": min(counts, default=0),
        "maximum": max(counts, default=0),
        "values": {
            name: int(value.item())
            for name, value in state.items()
            if name.endswith("sample_count")
        },
    }


def _warm_normalization(
    model: nn.Module,
    data: TrainingData,
    train_indices: Tensor,
    *,
    batch_size: int,
    device: str,
) -> None:
    reset = getattr(model, "reset_normalization_stats", None)
    if reset is None:
        return
    if not callable(reset):
        raise TypeError("model normalization reset is not callable")
    reset()
    model.train()
    with torch.no_grad():
        for start in range(0, int(train_indices.numel()), batch_size):
            selection = train_indices[start : start + batch_size]
            centers, frames, _target = _batch_inputs(data, selection, device)
            model(centers, frames)


def _predict_partition(
    model: nn.Module,
    data: TrainingData,
    indices: Tensor,
    *,
    batch_size: int,
    device: str,
    target_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    was_training = model.training
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for start in range(0, int(indices.numel()), batch_size):
            selection = indices[start : start + batch_size]
            centers, frames, target = _batch_inputs(data, selection, device)
            prediction = model(centers, frames)
            predictions.append(prediction.detach().cpu())
            targets.append((target / target_scale).detach().cpu())
    model.train(was_training)
    return torch.cat(predictions), torch.cat(targets)


def _prediction_metrics(
    predictions: Tensor,
    targets: Tensor,
    target_scale: Tensor,
) -> dict[str, float]:
    result = regression_metrics(
        predictions,
        targets,
        float(target_scale.detach().cpu()),
    )
    result["relative_rmse_percent"] = 100.0 * result["relative_rmse"]
    result["fit_accuracy_percent"] = 100.0 * (1.0 - result["relative_rmse"])
    result["r2_percent"] = 100.0 * result["r2"]
    return result


def _evaluate(
    model: nn.Module,
    data: TrainingData,
    indices: Tensor,
    *,
    batch_size: int,
    device: str,
    target_scale: Tensor,
) -> dict[str, float]:
    predictions, targets = _predict_partition(
        model,
        data,
        indices,
        batch_size=batch_size,
        device=device,
        target_scale=target_scale,
    )
    return _prediction_metrics(predictions, targets, target_scale)


def _selected_partition_results(
    model: nn.Module,
    data: TrainingData,
    partitions: Mapping[str, Tensor],
    *,
    config: SweepConfig,
    device: str,
    target_scale: Tensor,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    metrics: dict[str, dict[str, float]] = {}
    norm_differences: dict[str, dict[str, Any]] = {}
    for name, indices in partitions.items():
        predictions, targets = _predict_partition(
            model,
            data,
            indices,
            batch_size=config.micro_batch_size,
            device=device,
            target_scale=target_scale,
        )
        metrics[name] = _prediction_metrics(predictions, targets, target_scale)
        norm_differences[name] = summarize_relative_force_norm_difference(
            predictions,
            targets,
            partition=name,
            maximum_samples=config.relative_force_norm_sample_count,
            seed=config.relative_force_norm_seed,
        )
    return metrics, norm_differences


def _history_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def _history_row(
    epoch: int,
    learning_rate: float,
    train_loss: float,
    validation: Mapping[str, float] | None,
    normalization: Mapping[str, Any],
    duration: float,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "learning_rate": float(learning_rate),
        "train_normalized_mse": float(train_loss),
        "validation_normalized_mse": ""
        if validation is None
        else float(validation["normalized_mse"]),
        "validation_relative_rmse_percent": ""
        if validation is None
        else float(validation["relative_rmse_percent"]),
        "normalization_minimum_count": int(normalization["minimum"]),
        "normalization_maximum_count": int(normalization["maximum"]),
        "epoch_duration_seconds": float(duration),
    }


def _trial_hash(
    config: SweepConfig,
    spec: StudySpec,
    split_manifest: Mapping[str, Any],
    *,
    device: str,
    epochs: int,
    source_sha256: str | None = None,
    study_metadata: Mapping[str, Any] | None = None,
) -> str:
    return _canonical_sha256(
        {
            "model": spec.as_dict(),
            "protocol": config.protocol(device=device, epochs=epochs),
            "split_manifest_hash": split_manifest["manifest_hash"],
            "source_sha256": _source_sha256()
            if source_sha256 is None
            else source_sha256,
            **(
                {}
                if study_metadata is None
                else {"study_metadata": dict(study_metadata)}
            ),
        }
    )


def run_trial(
    config: SweepConfig,
    spec: StudySpec,
    paths: TrialPaths,
    split: SplitIndices,
    split_manifest: Mapping[str, Any],
    comet_logger: CometLogger,
    *,
    device: str,
    epochs: int,
    sample_limit: int | None = None,
    model_builder: ModelBuilder = _build_model,
    calibration_hook: CalibrationHook | None = None,
    selected_model_audit_hook: SelectedModelAuditHook | None = None,
    source_sha256: str | None = None,
    study_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    paths.directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.model_seed)
        torch.backends.cuda.matmul.allow_tf32 = config.enable_tf32
        torch.backends.cudnn.allow_tf32 = config.enable_tf32
    data = load_data(config, sample_limit=sample_limit, validate_files=True)
    if data.provenance != split_manifest["data_provenance"]:
        raise RuntimeError("loaded data provenance does not match the shared split")
    if max(
        int(item.max()) for item in (split.train, split.validation, split.test)
    ) >= len(data):
        raise ValueError("split contains indices beyond the loaded data")
    model = model_builder(spec, device)
    parameter_count = sum(
        item.numel() for item in model.parameters() if item.requires_grad
    )
    candidate_manifest = tuple(getattr(model, "candidate_manifest", ()))
    candidate_status_counts: dict[str, int] = {}
    for item in candidate_manifest:
        status = str(item.get("status", "unknown"))
        candidate_status_counts[status] = candidate_status_counts.get(status, 0) + 1
    expected_parameter_count = getattr(spec, "expected_parameter_count", None)
    if (
        expected_parameter_count is not None
        and parameter_count != expected_parameter_count
    ):
        raise RuntimeError("compiled parameter count does not match the catalog")
    target_scale = torch.sqrt(data.root_force[split.train].square().mean()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )
    shuffle_generator = torch.Generator().manual_seed(config.shuffle_seed)
    trial_hash = _trial_hash(
        config,
        spec,
        split_manifest,
        device=device,
        epochs=epochs,
        source_sha256=source_sha256,
        study_metadata=study_metadata,
    )
    resolved_source_sha256 = (
        _source_sha256() if source_sha256 is None else source_sha256
    )
    definition = {
        "schema_name": "tfenn_sweep31_trial",
        "schema_version": 1,
        "trial_hash": trial_hash,
        "model": spec.as_dict(),
        "protocol": config.protocol(device=device, epochs=epochs),
        "split_manifest_hash": split_manifest["manifest_hash"],
        "source_sha256": resolved_source_sha256,
        **({} if study_metadata is None else {"study_metadata": dict(study_metadata)}),
    }
    if paths.definition.is_file() and _load_json(paths.definition) != definition:
        raise RuntimeError("existing trial definition does not match this run")
    _atomic_json(paths.definition, definition)
    comet_logger.log_config(
        study_config={
            "study_name": config.study_directory.name,
            "config": config.as_dict(device=device, epochs=epochs),
            "split": {
                "manifest_hash": split_manifest["manifest_hash"],
                "partition_counts": split.counts(),
            },
            "source_sha256": resolved_source_sha256,
            **({} if study_metadata is None else dict(study_metadata)),
        },
        trial_config=definition,
        parameters={
            **spec.as_dict(),
            "compiled_parameter_count": parameter_count,
            "candidate_status_counts": candidate_status_counts,
        },
    )

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_validation: dict[str, float]
    calibration_report: dict[str, Any] = {}
    if paths.resume.is_file():
        resume = torch.load(paths.resume, map_location="cpu", weights_only=True)
        if resume.get("trial_hash") != trial_hash:
            raise RuntimeError("resume checkpoint trial hash mismatch")
        _restore_model_state(
            model,
            resume["parameter_state_dict"],
            resume["normalization_state_dict"],
            resume.get("calibration_state_dict"),
        )
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        shuffle_generator.set_state(resume["shuffle_generator_state"])
        history = list(resume["history"])
        start_epoch = int(resume["epoch"]) + 1
        best_epoch = int(resume["best_epoch"])
        best_validation = dict(resume["best_validation"])
        calibration_report = dict(resume.get("calibration_report", {}))
    else:
        _warm_normalization(
            model,
            data,
            split.train,
            batch_size=config.micro_batch_size,
            device=device,
        )
        if calibration_hook is not None:
            report = calibration_hook(
                model=model,
                spec=spec,
                data=data,
                split=split,
                config=config,
                paths=paths,
                device=device,
            )
            if report is not None:
                if not isinstance(report, Mapping):
                    raise TypeError("calibration hook must return a mapping or None")
                calibration_report = dict(report)
        initial_train = _evaluate(
            model,
            data,
            split.train,
            batch_size=config.micro_batch_size,
            device=device,
            target_scale=target_scale,
        )
        best_validation = _evaluate(
            model,
            data,
            split.validation,
            batch_size=config.micro_batch_size,
            device=device,
            target_scale=target_scale,
        )
        counts = _normalization_counts(model)
        history.append(
            _history_row(
                0,
                config.learning_rate,
                initial_train["normalized_mse"],
                best_validation,
                counts,
                0.0,
            )
        )
        _history_write(paths.history, history)
        _atomic_torch_save(
            paths.best,
            _checkpoint_payload(
                model,
                spec=spec,
                epoch=0,
                target_scale=target_scale,
                metrics=best_validation,
                trial_hash=trial_hash,
                calibration_report=calibration_report,
            ),
        )

    for row in history:
        _log_comet_epoch(comet_logger, paths, row)

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.perf_counter()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            order = split.train[
                torch.randperm(int(split.train.numel()), generator=shuffle_generator)
            ]
            model.train()
            epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
            epoch_samples = 0
            for effective_start in range(
                0,
                int(order.numel()),
                config.effective_batch_size,
            ):
                effective = order[
                    effective_start : effective_start + config.effective_batch_size
                ]
                effective_count = int(effective.numel())
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(
                    0,
                    effective_count,
                    config.micro_batch_size,
                ):
                    selection = effective[
                        micro_start : micro_start + config.micro_batch_size
                    ]
                    centers, frames, target = _batch_inputs(data, selection, device)
                    prediction = model(centers, frames)
                    loss = torch.nn.functional.mse_loss(
                        prediction,
                        target / target_scale,
                    )
                    weight = int(selection.numel()) / effective_count
                    (loss * weight).backward()
                    epoch_loss += loss.detach().to(torch.float64) * int(
                        selection.numel()
                    )
                    epoch_samples += int(selection.numel())
                optimizer.step()
            scheduler.step()
            train_loss = float((epoch_loss / epoch_samples).cpu())
            if not math.isfinite(train_loss):
                raise RuntimeError("training loss became nonfinite")
            validation = None
            if epoch % config.validation_every == 0 or epoch == epochs:
                validation = _evaluate(
                    model,
                    data,
                    split.validation,
                    batch_size=config.micro_batch_size,
                    device=device,
                    target_scale=target_scale,
                )
                if validation["normalized_mse"] < best_validation["normalized_mse"]:
                    best_epoch = epoch
                    best_validation = validation
                    _atomic_torch_save(
                        paths.best,
                        _checkpoint_payload(
                            model,
                            spec=spec,
                            epoch=epoch,
                            target_scale=target_scale,
                            metrics=validation,
                            trial_hash=trial_hash,
                            calibration_report=calibration_report,
                        ),
                    )
            counts = _normalization_counts(model)
            history.append(
                _history_row(
                    epoch,
                    learning_rate,
                    train_loss,
                    validation,
                    counts,
                    time.perf_counter() - epoch_started,
                )
            )
            _log_comet_epoch(comet_logger, paths, history[-1])
            _history_write(paths.history, history)
            base = _checkpoint_payload(
                model,
                spec=spec,
                epoch=epoch,
                target_scale=target_scale,
                metrics={} if validation is None else validation,
                trial_hash=trial_hash,
                calibration_report=calibration_report,
            )
            _save_resume(
                paths.resume,
                base,
                optimizer,
                scheduler,
                shuffle_generator,
                history,
                best_epoch,
                best_validation,
            )
            _atomic_json(
                paths.status,
                {
                    "status": "running",
                    "model_id": spec.model_id,
                    "epoch": epoch,
                    "epochs": epochs,
                    "best_epoch": best_epoch,
                    "best_validation_normalized_mse": best_validation["normalized_mse"],
                    "updated_at_utc": _utc_now(),
                },
            )
            print(
                json.dumps(
                    {
                        "model_id": spec.model_id,
                        "epoch": epoch,
                        "train_normalized_mse": train_loss,
                        "validation_normalized_mse": None
                        if validation is None
                        else validation["normalized_mse"],
                    }
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        _atomic_json(
            paths.status,
            {
                "status": "interrupted",
                "model_id": spec.model_id,
                "epoch": history[-1]["epoch"],
                "updated_at_utc": _utc_now(),
            },
        )
        raise

    best_payload = torch.load(paths.best, map_location="cpu", weights_only=True)
    _restore_model_state(
        model,
        best_payload["parameter_state_dict"],
        best_payload["normalization_state_dict"],
        best_payload.get("calibration_state_dict"),
    )
    selected_metrics, relative_force_norm_difference = _selected_partition_results(
        model,
        data,
        {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
        },
        config=config,
        device=device,
        target_scale=target_scale,
    )
    probe = split.test[: config.symmetry_probe_count]
    centers, frames, _target = _batch_inputs(data, probe, device)
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        symmetry = symmetry_metrics(
            model,
            centers,
            frames,
            tolerance=config.symmetry_tolerance,
        )
    finally:
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    baseline_train_loss = float(history[0]["train_normalized_mse"])
    baseline_validation_loss = float(history[0]["validation_normalized_mse"])
    loss_change = {
        "baseline_train_normalized_mse": baseline_train_loss,
        "baseline_validation_normalized_mse": baseline_validation_loss,
        "best_validation_normalized_mse": float(best_validation["normalized_mse"]),
        "selected_train_normalized_mse": float(
            selected_metrics["train"]["normalized_mse"]
        ),
        "selected_validation_normalized_mse": float(
            selected_metrics["validation"]["normalized_mse"]
        ),
        "best_validation_to_baseline_ratio": float(
            best_validation["normalized_mse"] / baseline_validation_loss
        ),
        "selected_train_to_baseline_ratio": float(
            selected_metrics["train"]["normalized_mse"] / baseline_train_loss
        ),
        "selected_validation_to_baseline_ratio": float(
            selected_metrics["validation"]["normalized_mse"] / baseline_validation_loss
        ),
    }
    selected_model_audit: dict[str, Any] | None = None
    if selected_model_audit_hook is not None:
        audit = selected_model_audit_hook(
            model=model,
            spec=spec,
            data=data,
            split=split,
            config=config,
            paths=paths,
            device=device,
            target_scale=target_scale,
            selected_metrics=selected_metrics,
            comet_logger=comet_logger,
        )
        if not isinstance(audit, Mapping):
            raise TypeError("selected model audit hook must return a mapping")
        selected_model_audit = dict(audit)
        json.dumps(selected_model_audit, allow_nan=False)
    final_payload = _checkpoint_payload(
        model,
        spec=spec,
        epoch=best_epoch,
        target_scale=target_scale,
        metrics=selected_metrics["validation"],
        trial_hash=trial_hash,
        calibration_report=calibration_report,
    )
    _atomic_torch_save(paths.final, final_payload)
    summary = {
        "schema_name": "tfenn_sweep31_trial_result",
        "schema_version": 1,
        "status": "complete",
        "trial_hash": trial_hash,
        "model": {
            **spec.as_dict(),
            "parameter_count": parameter_count,
            "candidate_manifest": list(candidate_manifest),
            "candidate_status_counts": candidate_status_counts,
            "checkpoint_content": (
                "learned parameters, nontrainable running RMS state, and optional "
                "training calibrated descriptor projection state"
            ),
            "fixed_compiled_basis_stored": False,
        },
        "training": {
            "epochs_completed": epochs,
            "optimizer_updates": epochs
            * math.ceil(int(split.train.numel()) / config.effective_batch_size),
            "history_path": str(paths.history),
            "history_sha256": sha256_file(paths.history),
            "normalization_counts": _normalization_counts(model),
            "calibration": calibration_report,
        },
        "selection": {
            "rule": "minimum validation normalized MSE",
            "best_epoch": best_epoch,
            "best_validation_during_training": best_validation,
            "selected_metrics": selected_metrics,
            "test_evaluated_once": True,
            "symmetry_uses_tf32": False,
            "symmetry": symmetry,
        },
        "relative_force_norm_difference": relative_force_norm_difference,
        "loss_change": loss_change,
        **(
            {}
            if selected_model_audit is None
            else {"selected_model_audit": selected_model_audit}
        ),
        "split": {
            "manifest_hash": split_manifest["manifest_hash"],
            "partition_counts": split.counts(),
        },
        **({} if study_metadata is None else dict(study_metadata)),
        "runtime": {
            "duration_seconds": time.perf_counter() - started,
            "finished_at_utc": _utc_now(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device,
        },
        "checkpoints": {
            "best_path": str(paths.best),
            "best_sha256": sha256_file(paths.best),
            "final_path": str(paths.final),
            "final_sha256": sha256_file(paths.final),
        },
        "conclusion_context": {
            "purpose": spec.purpose,
            "comparison_role": spec.comparison_role,
        },
        "comet": comet_logger.identity,
    }
    comet_logger.log_final(
        metrics=selected_metrics,
        relative_force_norm_stats=relative_force_norm_difference,
        summary=summary,
    )
    comet_logger.log_asset(
        paths.history,
        name=f"{spec.model_id}_history.csv",
        metadata={"model_id": spec.model_id, "trial_hash": trial_hash},
    )
    comet_logger.log_checkpoint_reference(
        "best",
        paths.best,
        sha256=summary["checkpoints"]["best_sha256"],
        metadata={"model_id": spec.model_id, "best_epoch": best_epoch},
    )
    comet_logger.log_checkpoint_reference(
        "final",
        paths.final,
        sha256=summary["checkpoints"]["final_sha256"],
        metadata={"model_id": spec.model_id, "selected_epoch": best_epoch},
    )
    comet_logger.finish("complete")
    _atomic_json(paths.summary, summary)
    paths.resume.unlink(missing_ok=True)
    paths.error.unlink(missing_ok=True)
    _atomic_json(
        paths.status,
        {
            "status": "complete",
            "model_id": spec.model_id,
            "epoch": epochs,
            "best_epoch": best_epoch,
            "updated_at_utc": _utc_now(),
        },
    )
    return summary


def _result_row(config: SweepConfig, spec: StudySpec) -> dict[str, Any]:
    paths = TrialPaths.create(config.study_directory / "models" / spec.model_id)
    summary = _load_json(paths.summary) if paths.summary.is_file() else {}
    status = _load_json(paths.status) if paths.status.is_file() else {}
    error = _load_json(paths.error) if paths.error.is_file() else {}
    selection = summary.get("selection", {})
    metrics = selection.get("selected_metrics", {})
    norm_differences = summary.get("relative_force_norm_difference", {})

    def metric(partition: str, name: str) -> Any:
        value = metrics.get(partition, {})
        return value.get(name, "") if isinstance(value, Mapping) else ""

    def norm_metric(partition: str, name: str) -> Any:
        value = norm_differences.get(partition, {})
        return value.get(name, "") if isinstance(value, Mapping) else ""

    return {
        "model_id": spec.model_id,
        "description": spec.description,
        "purpose": spec.purpose,
        "comparison_role": spec.comparison_role,
        "status": summary.get("status", status.get("status", "planned")),
        "parameter_count": summary.get("model", {}).get(
            "parameter_count", spec.expected_parameter_count
        ),
        "best_epoch": selection.get("best_epoch", ""),
        "best_validation_normalized_mse": selection.get(
            "best_validation_during_training", {}
        ).get("normalized_mse", ""),
        "train_relative_rmse_percent": metric("train", "relative_rmse_percent"),
        "validation_relative_rmse_percent": metric(
            "validation", "relative_rmse_percent"
        ),
        "test_relative_rmse_percent": metric("test", "relative_rmse_percent"),
        "train_fit_accuracy_percent": metric("train", "fit_accuracy_percent"),
        "validation_fit_accuracy_percent": metric("validation", "fit_accuracy_percent"),
        "test_fit_accuracy_percent": metric("test", "fit_accuracy_percent"),
        "test_r2": metric("test", "r2"),
        "test_r2_percent": metric("test", "r2_percent"),
        "train_relative_force_norm_min": norm_metric("train", "min"),
        "train_relative_force_norm_median": norm_metric("train", "median"),
        "train_relative_force_norm_max": norm_metric("train", "max"),
        "validation_relative_force_norm_min": norm_metric("validation", "min"),
        "validation_relative_force_norm_median": norm_metric("validation", "median"),
        "validation_relative_force_norm_max": norm_metric("validation", "max"),
        "test_relative_force_norm_min": norm_metric("test", "min"),
        "test_relative_force_norm_median": norm_metric("test", "median"),
        "test_relative_force_norm_max": norm_metric("test", "max"),
        "d6_passed": selection.get("symmetry", {}).get("passed", ""),
        "duration_seconds": summary.get("runtime", {}).get("duration_seconds", ""),
        "error_type": error.get("exception_type", ""),
        "error_message": error.get("message", ""),
    }


def _refresh_results(config: SweepConfig) -> Path:
    config.study_directory.mkdir(parents=True, exist_ok=True)
    path = config.study_directory / "results.csv"
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    rows = [_result_row(config, spec) for spec in STUDY_SPECS]
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)
    completed = [row for row in rows if row["status"] == "complete"]
    ranking = sorted(
        completed,
        key=lambda row: float(row["best_validation_normalized_mse"]),
    )
    _atomic_json(
        config.study_directory / "comparison.json",
        {
            "schema_name": "tfenn_sweep31_comparison",
            "schema_version": 1,
            "completed_model_count": len(completed),
            "ranking_by_validation": ranking,
            "primary_model": "C15",
            "model_level_baseline": "G00",
            "lower_control": "C29",
            "upper_control": "C30",
            "updated_at_utc": _utc_now(),
        },
    )
    return path


def _record_error(paths: TrialPaths, spec: StudySpec, error: BaseException) -> None:
    value = {
        "status": "error",
        "model_id": spec.model_id,
        "exception_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "recorded_at_utc": _utc_now(),
    }
    _atomic_json(paths.error, value)
    _atomic_json(paths.status, value)


def _select_models(values: Sequence[str]) -> tuple[StudySpec, ...]:
    if not values:
        return STUDY_SPECS
    selected = tuple(get_study_spec(value) for value in values)
    if len({item.model_id for item in selected}) != len(selected):
        raise ValueError("model selection contains duplicates")
    return selected


def run_study(arguments: argparse.Namespace) -> int:
    config = SweepConfig.from_path(arguments.config)
    if config.comet.enabled and not os.environ.get("COMET_API_KEY", "").strip():
        raise RuntimeError(
            "COMET_API_KEY must be set before the formal online study starts"
        )
    device = _resolve_device(arguments.device or config.device)
    split_manifest = _prepare_split(config)
    study_manifest = {
        "schema_name": "tfenn_benzene_pair_sweep31_study",
        "schema_version": 1,
        "model_count": len(STUDY_SPECS),
        "models": [item.as_dict() for item in STUDY_SPECS],
        "config": config.as_dict(device=device),
        "split_manifest_hash": split_manifest["manifest_hash"],
        "source_sha256": _source_sha256(),
    }
    study_manifest["study_hash"] = _canonical_sha256(study_manifest)
    manifest_path = config.study_directory / "manifest.json"
    if manifest_path.is_file() and _load_json(manifest_path) != study_manifest:
        raise RuntimeError("existing study manifest does not match this run")
    _atomic_json(manifest_path, study_manifest)
    selected = _select_models(arguments.model)
    _refresh_results(config)
    for spec in selected:
        paths = TrialPaths.create(config.study_directory / "models" / spec.model_id)
        if paths.summary.is_file():
            completed = _load_json(paths.summary)
            expected_hash = _trial_hash(
                config,
                spec,
                split_manifest,
                device=device,
                epochs=config.epochs,
            )
            if completed.get("status") != "complete":
                raise RuntimeError("existing trial summary is not complete")
            if completed.get("trial_hash") != expected_hash:
                raise RuntimeError("existing trial summary hash does not match")
            if not paths.best.is_file() or not paths.final.is_file():
                raise RuntimeError("completed trial is missing a learned checkpoint")
            continue
        paths.directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "experiments.benzene_pair.sweep30",
            "trial",
            "--config",
            str(Path(arguments.config).resolve()),
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
    _refresh_results(config)
    return 0


def run_trial_command(arguments: argparse.Namespace) -> int:
    config = SweepConfig.from_path(arguments.config)
    spec = get_study_spec(arguments.model)
    device = _resolve_device(arguments.device or config.device)
    epochs = config.epochs if arguments.epochs is None else int(arguments.epochs)
    if epochs < 1 or epochs > config.epochs:
        raise ValueError("epoch override is outside the formal protocol")
    if arguments.output_directory is None:
        paths = TrialPaths.create(config.study_directory / "models" / spec.model_id)
    else:
        paths = TrialPaths.create(arguments.output_directory)
    comet_logger: CometLogger = NullCometTrialLogger()
    try:
        if arguments.disable_comet and arguments.sample_limit is None:
            raise ValueError("Comet can only be disabled for a sampled trial")
        comet_logger = _create_trial_comet_logger(
            config,
            spec,
            paths,
            disabled=arguments.disable_comet,
        )
        if arguments.sample_limit is None:
            split, manifest = _load_split(config.study_directory)
        else:
            data = load_data(config, sample_limit=arguments.sample_limit)
            split, report = create_group_aware_split(
                data.centers,
                data.frames,
                seed=config.split_seed,
                fractions=config.split_fractions,
            )
            manifest = _write_split(
                paths.directory,
                split,
                report,
                data.provenance,
            )
        run_trial(
            config,
            spec,
            paths,
            split,
            manifest,
            comet_logger,
            device=device,
            epochs=epochs,
            sample_limit=arguments.sample_limit,
        )
        print(json.dumps({"status": "complete", "summary": str(paths.summary)}))
        return 0
    except KeyboardInterrupt:
        comet_logger.finish("interrupted")
        return 130
    except BaseException as error:
        _record_error(paths, spec, error)
        try:
            comet_logger.log_error(error, stage="trial")
        except BaseException:
            traceback.print_exc(file=sys.stderr)
        try:
            comet_logger.finish("error")
        except BaseException:
            traceback.print_exc(file=sys.stderr)
        traceback.print_exception(error, file=sys.stderr)
        return 1


def run_smoke(arguments: argparse.Namespace) -> int:
    config = SweepConfig.from_path(arguments.config)
    device = _resolve_device(arguments.device or config.device)
    selected = _select_models(arguments.model or ("G00", "C15", "C30"))
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
            "experiments.benzene_pair.sweep30",
            "trial",
            "--config",
            str(Path(arguments.config).resolve()),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--device", default=None)
    run.add_argument("--model", action="append", default=[])
    run.set_defaults(handler=run_study)
    trial = commands.add_parser("trial")
    trial.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    trial.add_argument("--model", required=True)
    trial.add_argument("--device", default=None)
    trial.add_argument("--epochs", type=int, default=None)
    trial.add_argument("--sample_limit", type=int, default=None)
    trial.add_argument("--output_directory", type=Path, default=None)
    trial.add_argument("--disable_comet", action="store_true")
    trial.set_defaults(handler=run_trial_command)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke.add_argument("--device", default=None)
    smoke.add_argument("--model", action="append", default=[])
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--sample_limit", type=int, default=8000)
    smoke.add_argument("--output_directory", type=Path, default=None)
    smoke.set_defaults(handler=run_smoke)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    return int(parsed.handler(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
