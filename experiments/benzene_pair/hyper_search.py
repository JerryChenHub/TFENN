"""Run the isolated benzene pair hyperparameter study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from experiments.benzene_pair.train import (
    PairTrainingData,
    TARGET_DEFINITION,
    TrainingConfig,
    _build_model,
    _evaluate,
    _git_provenance,
    _optimizer_and_scheduler,
    _runtime_versions,
    create_split,
    load_trained_model,
    sha256_file,
    symmetry_metrics,
)
from TFENN.data import load_benzene_cluster_csv, load_benzene_cluster_metadata
from TFENN.models import PairPipelineConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_CONFIG = Path(__file__).resolve().parent / "hyper_config.json"
HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "train_objective_normalized_mse",
    "validation_normalized_mse",
    "validation_mse",
    "validation_rmse",
    "validation_mae",
    "validation_relative_rmse",
    "validation_r2",
    "epoch_duration_seconds",
)
MASTER_FIELDS = (
    "trial_id",
    "candidate_id",
    "status",
    "trial_hash",
    "epochs_completed",
    "duration_seconds",
    "parameter_count",
    "gate_count",
    "best_epoch",
    "best_validation_normalized_mse",
    "train_rmse",
    "train_mae",
    "train_r2",
    "train_relative_rmse",
    "validation_rmse",
    "validation_r2",
    "test_rmse",
    "test_mae",
    "test_r2",
    "test_relative_rmse",
    "symmetry_maximum_absolute",
    "best_checkpoint_path",
    "best_checkpoint_sha256",
    "final_checkpoint_path",
    "final_checkpoint_sha256",
    "error_type",
    "error_message",
    "error_path",
)
RANKING_FIELDS = ("rank", *MASTER_FIELDS)
CATALOG_HASH_KEYS = (
    "candidate_id",
    "topology_code",
    "topology_name",
    "profile_code",
    "pipeline",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "scheduler_step_size",
    "scheduler_gamma",
)
_STOP_REQUESTED = False


class TrialInterrupted(RuntimeError):
    """Indicate a requested stop after a complete epoch."""


@dataclass(frozen=True, slots=True)
class StudyConfig:
    catalog_path: Path
    csv_path: Path
    validation_path: Path
    study_directory: Path
    epochs: int
    resume_every: int
    split_seed: int
    model_seed: int
    shuffle_seed: int
    split_fractions: tuple[float, float, float]
    optimizer: str
    device: str
    dtype: str
    threads: int
    symmetry_tolerance: float
    symmetry_probe_count: int
    progress_every: int
    zero_output_heads: bool
    expected_sample_count: int
    expected_dataset_revision: int
    expected_opls_version: str
    deterministic_algorithms: bool

    @classmethod
    def from_path(cls, path: str | Path) -> StudyConfig:
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf_8"))
        if not isinstance(value, Mapping):
            raise TypeError("study config must contain one object")
        if value.get("schema_name") != "tfenn_benzene_pair_hyper_search":
            raise ValueError("unexpected study config schema name")
        if value.get("schema_version") != 1:
            raise ValueError("unexpected study config schema version")

        def resolved(name: str) -> Path:
            result = Path(str(value[name]))
            return (
                result.resolve()
                if result.is_absolute()
                else (REPOSITORY_ROOT / result).resolve()
            )

        result = cls(
            catalog_path=resolved("catalog_path"),
            csv_path=resolved("csv_path"),
            validation_path=resolved("validation_path"),
            study_directory=resolved("study_directory"),
            epochs=int(value.get("epochs", 1500)),
            resume_every=int(value.get("resume_every", 25)),
            split_seed=int(value.get("split_seed", 20260813)),
            model_seed=int(value.get("model_seed", 20260814)),
            shuffle_seed=int(value.get("shuffle_seed", 20260815)),
            split_fractions=tuple(
                float(item) for item in value.get("split_fractions", (0.8, 0.1, 0.1))
            ),
            optimizer=str(value.get("optimizer", "adamw")),
            device=str(value.get("device", "auto")),
            dtype=str(value.get("dtype", "float32")),
            threads=int(value.get("threads", 4)),
            symmetry_tolerance=float(value.get("symmetry_tolerance", 1.0e-4)),
            symmetry_probe_count=int(value.get("symmetry_probe_count", 8)),
            progress_every=int(value.get("progress_every", 25)),
            zero_output_heads=bool(value.get("zero_output_heads", True)),
            expected_sample_count=int(value.get("expected_sample_count", 5000)),
            expected_dataset_revision=int(value.get("expected_dataset_revision", 2)),
            expected_opls_version=str(value.get("expected_opls_version", "2.0.0")),
            deterministic_algorithms=bool(value.get("deterministic_algorithms", True)),
        )
        if result.epochs != 1500:
            raise ValueError("the full study must request exactly 1500 epochs")
        if result.resume_every != 25:
            raise ValueError("the full study must checkpoint every 25 epochs")
        if result.expected_sample_count != 5000:
            raise ValueError("the full study requires exactly 5000 samples")
        if result.expected_dataset_revision != 2:
            raise ValueError("the full study requires dataset revision two")
        if len(result.split_fractions) != 3 or not math.isclose(
            sum(result.split_fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("split fractions must contain three values summing to one")
        if min(result.split_fractions) <= 0.0:
            raise ValueError("split fractions must be positive")
        if result.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be adam or adamw")
        if result.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if result.threads < 1 or result.progress_every < 1:
            raise ValueError("threads and progress_every must be positive")
        if result.symmetry_probe_count < 1:
            raise ValueError("symmetry_probe_count must be positive")
        if len({result.split_seed, result.model_seed, result.shuffle_seed}) != 3:
            raise ValueError("split, model, and shuffle seeds must be distinct")
        return result

    def protocol_dict(
        self, *, device: str | None = None, epochs: int | None = None
    ) -> dict[str, Any]:
        return {
            "epochs": self.epochs if epochs is None else int(epochs),
            "resume_every": self.resume_every,
            "split_seed": self.split_seed,
            "model_seed": self.model_seed,
            "shuffle_seed": self.shuffle_seed,
            "split_fractions": list(self.split_fractions),
            "optimizer": self.optimizer,
            "device": self.device if device is None else device,
            "dtype": self.dtype,
            "threads": self.threads,
            "symmetry_tolerance": self.symmetry_tolerance,
            "symmetry_probe_count": self.symmetry_probe_count,
            "zero_output_heads": self.zero_output_heads,
            "expected_sample_count": self.expected_sample_count,
            "expected_dataset_revision": self.expected_dataset_revision,
            "expected_opls_version": self.expected_opls_version,
            "deterministic_algorithms": self.deterministic_algorithms,
        }


@dataclass(frozen=True, slots=True)
class TrialPaths:
    directory: Path
    definition: Path
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
    def from_directory(cls, directory: str | Path) -> TrialPaths:
        root = Path(directory).resolve()
        return cls(
            directory=root,
            definition=root / "config.json",
            status=root / "status.json",
            history=root / "history.csv",
            best=root / "best.pt",
            final=root / "final.pt",
            resume=root / "resume.pt",
            summary=root / "summary.json",
            error=root / "error.json",
            stdout=root / "stdout.log",
            stderr=root / "stderr.log",
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf_8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with partial.open("w", encoding="utf_8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    torch.save(dict(value), partial)
    with partial.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf_8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one object")
    return value


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            if tuple(row) != HISTORY_FIELDS:
                raise ValueError("history fields do not match the required schema")
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return path


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf_8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
            raise ValueError("history header does not match the required schema")
        rows = []
        for raw in reader:
            row: dict[str, Any] = {"epoch": int(raw["epoch"])}
            for name in HISTORY_FIELDS[1:]:
                row[name] = float(raw[name])
            rows.append(row)
    if [row["epoch"] for row in rows] != list(range(len(rows))):
        raise ValueError("history epochs must be contiguous from zero")
    return rows


def _append_journal(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": _utc_now(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        **dict(event),
    }
    encoded = (
        json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf_8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_sha256() -> str:
    files = sorted((REPOSITORY_ROOT / "src" / "TFENN").rglob("*.py"))
    files.extend(
        (
            Path(__file__).resolve(),
            Path(__file__).resolve().parent / "train.py",
            Path(__file__).resolve().parent / "hyper_catalog.py",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf_8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _load_catalog(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    catalog = _load_json(path)
    if catalog.get("schema_name") != "tfenn_benzene_pair_hyper_catalog":
        raise ValueError("unexpected catalog schema name")
    if catalog.get("schema_version") != 1:
        raise ValueError("unexpected catalog schema version")
    designs_value = catalog.get("designs")
    if not isinstance(designs_value, list) or len(designs_value) != 100:
        raise ValueError("catalog must contain exactly one hundred designs")
    designs = tuple(dict(item) for item in designs_value)
    if catalog.get("design_count") != 100:
        raise ValueError("catalog design_count must equal one hundred")
    if catalog.get("gate_mlp_hidden_width") != 64:
        raise ValueError("catalog must fix the gate MLP hidden width at 64")
    if catalog.get("catalog_sha256") != _canonical_sha256(designs_value):
        raise ValueError("catalog SHA256 does not match its designs")
    trial_ids = tuple(item.get("trial_id") for item in designs)
    candidate_ids = tuple(item.get("candidate_id") for item in designs)
    if len(set(trial_ids)) != 100 or len(set(candidate_ids)) != 100:
        raise ValueError("catalog trial and candidate identifiers must be unique")
    for design in designs:
        expected_hash = _canonical_sha256(
            {key: design[key] for key in CATALOG_HASH_KEYS}
        )
        if design.get("config_hash") != expected_hash:
            raise ValueError(
                f"catalog config hash mismatch for {design.get('trial_id')}"
            )
        pipeline_value = dict(design["pipeline"])
        existing_architecture = pipeline_value.get("architecture_id")
        if existing_architecture not in {None, design["candidate_id"]}:
            raise ValueError("catalog architecture identifier does not match candidate")
        pipeline_value["architecture_id"] = design["candidate_id"]
        pipeline = PairPipelineConfig.from_dict(pipeline_value)
        if any(stage.mlp.hidden_widths != (64,) for stage in pipeline.stages):
            raise ValueError("every catalog gate MLP must have one width 64 layer")
    return catalog, designs


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _validate_dataset_files(config: StudyConfig) -> dict[str, Any]:
    if not config.csv_path.is_file():
        raise FileNotFoundError(config.csv_path)
    metadata_path = config.csv_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not config.validation_path.is_file():
        raise FileNotFoundError(config.validation_path)
    validation = _load_json(config.validation_path)
    if validation.get("schema_name") != "tfenn_benzene_pair_data_validation":
        raise ValueError("unexpected data validation schema")
    if validation.get("passed") is not True:
        raise ValueError("the data validation report did not pass")
    checks = validation.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise ValueError("the data validation checks are incomplete")
    csv_digest = sha256_file(config.csv_path)
    metadata_digest = sha256_file(metadata_path)
    if validation.get("csv_sha256") != csv_digest:
        raise ValueError("data validation CSV SHA256 does not match the current file")
    if validation.get("metadata_sha256") != metadata_digest:
        raise ValueError(
            "data validation metadata SHA256 does not match the current file"
        )
    if validation.get("opls_distribution_version") != config.expected_opls_version:
        raise ValueError("data validation OPLS version does not match the protocol")
    installed_opls = importlib.metadata.version("opls2020-static")
    if installed_opls != config.expected_opls_version:
        raise ValueError("installed OPLS version does not match the protocol")
    return {
        "csv_sha256": csv_digest,
        "metadata_sha256": metadata_digest,
        "validation_sha256": sha256_file(config.validation_path),
        "validation": validation,
    }


def _load_v2_training_data(
    config: StudyConfig,
    *,
    device: str,
    sample_limit: int | None = None,
) -> PairTrainingData:
    arrays = load_benzene_cluster_csv(config.csv_path, dtype=np.float64)
    metadata_path = config.csv_path.with_suffix(".json")
    metadata = load_benzene_cluster_metadata(metadata_path)
    if arrays.molecule_count != 2 or metadata.get("molecule_count") != 2:
        raise ValueError("benzene pair training requires exactly two molecules")
    if metadata.get("schema_version") != 2:
        raise ValueError("training requires data schema version two")
    if metadata.get("dataset_revision") != config.expected_dataset_revision:
        raise ValueError("training requires dataset revision two")
    if metadata.get("sample_count") != config.expected_sample_count:
        raise ValueError("metadata sample count does not match the study protocol")
    if len(arrays) != config.expected_sample_count:
        raise ValueError("CSV sample count does not match the study protocol")
    csv_digest = sha256_file(config.csv_path)
    if metadata.get("csv_sha256") != csv_digest:
        raise ValueError("metadata CSV SHA256 does not match the current file")
    opls = metadata.get("opls")
    if not isinstance(opls, Mapping):
        raise ValueError("metadata OPLS provenance is missing")
    if opls.get("runtime_version") != config.expected_opls_version:
        raise ValueError("metadata OPLS runtime version does not match the protocol")
    if opls.get("distribution_version") != config.expected_opls_version:
        raise ValueError(
            "metadata OPLS distribution version does not match the protocol"
        )
    count = len(arrays) if sample_limit is None else int(sample_limit)
    if count < 3 or count > len(arrays):
        raise ValueError(
            "sample_limit must be at least three and no larger than the dataset"
        )
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    centers = torch.as_tensor(arrays.centers[:count], dtype=dtype, device=device)
    frames = torch.as_tensor(arrays.rotations[:count], dtype=dtype, device=device)
    root_force = torch.as_tensor(arrays.forces[:count, 0], dtype=dtype, device=device)
    if not bool(
        torch.isfinite(centers).all()
        and torch.isfinite(frames).all()
        and torch.isfinite(root_force).all()
    ):
        raise ValueError("training tensors must be finite")
    return PairTrainingData(
        centers=centers,
        frames=frames,
        root_force=root_force,
        metadata=dict(metadata),
        csv_path=config.csv_path,
        metadata_path=metadata_path,
        csv_sha256=csv_digest,
        metadata_sha256=sha256_file(metadata_path),
    )


def _find_design(designs: Sequence[dict[str, Any]], trial_id: str) -> dict[str, Any]:
    matches = tuple(item for item in designs if item.get("trial_id") == trial_id)
    if len(matches) != 1:
        raise ValueError(f"unknown trial identifier {trial_id}")
    return matches[0]


def _pipeline_from_design(design: Mapping[str, Any]) -> PairPipelineConfig:
    value = dict(design["pipeline"])
    existing_architecture = value.get("architecture_id")
    if existing_architecture not in {None, design["candidate_id"]}:
        raise ValueError("catalog architecture identifier does not match candidate")
    value["architecture_id"] = design["candidate_id"]
    return PairPipelineConfig.from_dict(value)


def _trial_training_config(
    study: StudyConfig,
    design: Mapping[str, Any],
    paths: TrialPaths,
    *,
    device: str,
    epochs: int,
) -> TrainingConfig:
    return TrainingConfig(
        csv_path=study.csv_path,
        output_directory=paths.directory,
        epochs=epochs,
        batch_size=int(design["batch_size"]),
        learning_rate=float(design["learning_rate"]),
        weight_decay=float(design["weight_decay"]),
        optimizer=study.optimizer,
        scheduler_step_size=int(design["scheduler_step_size"]),
        scheduler_gamma=float(design["scheduler_gamma"]),
        split_seed=study.split_seed,
        model_seed=study.model_seed,
        split_fractions=study.split_fractions,
        device=device,
        dtype=study.dtype,
        threads=study.threads,
        symmetry_tolerance=study.symmetry_tolerance,
        progress_every=study.progress_every,
        pipeline=_pipeline_from_design(design),
        zero_output_heads=study.zero_output_heads,
        maximum_train_loss_ratio=0.1,
        overwrite=False,
    )


def _trial_hash(
    study: StudyConfig,
    design: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    device: str,
    epochs: int,
    sample_limit: int | None,
) -> str:
    return _canonical_sha256(
        {
            "catalog_config_hash": design["config_hash"],
            "protocol": study.protocol_dict(device=device, epochs=epochs),
            "dataset_csv_sha256": dataset["csv_sha256"],
            "dataset_metadata_sha256": dataset["metadata_sha256"],
            "sample_limit": sample_limit,
            "source_sha256": _source_sha256(),
        }
    )


def _build_trial_definition(
    catalog: Mapping[str, Any],
    design: Mapping[str, Any],
    training: TrainingConfig,
    trial_hash: str,
    dataset: Mapping[str, Any],
    source_sha256: str,
    sample_limit: int | None,
) -> dict[str, Any]:
    return {
        "schema_name": "tfenn_benzene_pair_hyper_trial",
        "schema_version": 1,
        "trial_id": design["trial_id"],
        "candidate_id": design["candidate_id"],
        "trial_hash": trial_hash,
        "catalog_sha256": catalog["catalog_sha256"],
        "catalog_config_hash": design["config_hash"],
        "functional_hash": design["functional_hash"],
        "dataset_csv_sha256": dataset["csv_sha256"],
        "dataset_metadata_sha256": dataset["metadata_sha256"],
        "source_sha256": source_sha256,
        "sample_limit": sample_limit,
        "preflight": design.get("preflight"),
        "training_configuration": training.as_json(),
    }


def _ensure_trial_definition(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        current = _load_json(path)
        if current != dict(value):
            raise RuntimeError("existing trial definition does not match this run")
    else:
        _atomic_json(path, value)


def _history_row(
    epoch: int,
    learning_rate: float,
    train_loss: float,
    validation: Mapping[str, float],
    duration: float,
) -> dict[str, Any]:
    values = (
        float(learning_rate),
        float(train_loss),
        float(validation["normalized_mse"]),
        float(validation["mse"]),
        float(validation["rmse"]),
        float(validation["mae"]),
        float(validation["relative_rmse"]),
        float(validation["r2"]),
        float(duration),
    )
    if not all(math.isfinite(item) for item in values):
        raise RuntimeError("history values must be finite")
    return dict(zip(HISTORY_FIELDS, (int(epoch), *values), strict=True))


def _parameter_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _save_learned_checkpoint(
    path: Path,
    parameter_state: Mapping[str, Tensor],
    *,
    epoch: int,
    target_scale: Tensor,
    config: TrainingConfig,
    metrics: Mapping[str, float],
) -> Path:
    payload = {
        "schema_name": "tfenn_pair_force_checkpoint",
        "schema_version": 2,
        "network_name": config.pipeline.architecture_id,
        "epoch": int(epoch),
        "target_definition": TARGET_DEFINITION,
        "target_scale": float(target_scale.detach().cpu()),
        "configuration": config.as_json(),
        "metrics": dict(metrics),
        "fixed_tensor_artifacts_stored": False,
        "parameter_state_dict": {
            name: value.detach().cpu() for name, value in parameter_state.items()
        },
    }
    return _atomic_torch_save(path, payload)


def _save_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    shuffle_generator: torch.Generator,
    *,
    epoch: int,
    target_scale: Tensor,
    trial_hash: str,
    best_epoch: int,
    best_validation: Mapping[str, float],
    best_parameter_state: Mapping[str, Tensor],
    initial_train_loss: float,
) -> Path:
    payload: dict[str, Any] = {
        "schema_name": "tfenn_benzene_pair_hyper_resume",
        "schema_version": 1,
        "trial_hash": trial_hash,
        "epoch": int(epoch),
        "target_scale": float(target_scale.detach().cpu()),
        "best_epoch": int(best_epoch),
        "best_validation": dict(best_validation),
        "initial_train_normalized_mse": float(initial_train_loss),
        "fixed_tensor_artifacts_stored": False,
        "parameter_state_dict": _parameter_state(model),
        "best_parameter_state_dict": {
            name: value.detach().cpu() for name, value in best_parameter_state.items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "shuffle_generator_state": shuffle_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    return _atomic_torch_save(path, payload)


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if isinstance(value, Tensor):
                state[name] = value.to(device=device)


def _restore_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    shuffle_generator: torch.Generator,
    *,
    trial_hash: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("resume file must contain one object")
    if payload.get("schema_name") != "tfenn_benzene_pair_hyper_resume":
        raise ValueError("unexpected resume schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unexpected resume schema version")
    if payload.get("trial_hash") != trial_hash:
        raise ValueError("resume trial hash does not match the requested trial")
    if payload.get("fixed_tensor_artifacts_stored") is not False:
        raise ValueError("resume file must not contain fixed tensor artifacts")
    learned = payload.get("parameter_state_dict")
    best_learned = payload.get("best_parameter_state_dict")
    parameters = dict(model.named_parameters())
    if not isinstance(learned, Mapping) or set(learned) != set(parameters):
        raise ValueError("resume parameter names do not match the model")
    if not isinstance(best_learned, Mapping) or set(best_learned) != set(parameters):
        raise ValueError("resume best parameter names do not match the model")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = learned[name]
            if not isinstance(value, Tensor) or value.shape != parameter.shape:
                raise ValueError(f"resume parameter shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            best_value = best_learned[name]
            if (
                not isinstance(best_value, Tensor)
                or best_value.shape != parameter.shape
            ):
                raise ValueError(f"resume best parameter shape mismatch for {name}")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _optimizer_to_device(optimizer, next(model.parameters()).device)
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    shuffle_generator.set_state(payload["shuffle_generator_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_states = payload.get("cuda_rng_state_all", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    return payload


def _write_status(
    paths: TrialPaths,
    *,
    trial_id: str,
    candidate_id: str,
    trial_hash: str,
    status: str,
    epoch: int,
    best_epoch: int | None = None,
    train_loss: float | None = None,
    validation_loss: float | None = None,
    message: str | None = None,
) -> None:
    value: dict[str, Any] = {
        "schema_name": "tfenn_benzene_pair_hyper_trial_status",
        "schema_version": 1,
        "updated_at_utc": _utc_now(),
        "trial_id": trial_id,
        "candidate_id": candidate_id,
        "trial_hash": trial_hash,
        "status": status,
        "epoch": int(epoch),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    if best_epoch is not None:
        value["best_epoch"] = int(best_epoch)
    if train_loss is not None:
        value["train_objective_normalized_mse"] = float(train_loss)
    if validation_loss is not None:
        value["validation_normalized_mse"] = float(validation_loss)
    if message is not None:
        value["message"] = message
    _atomic_json(paths.status, value)


def _set_stop_requested(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _configure_runtime(study: StudyConfig, device: str) -> None:
    torch.set_num_threads(study.threads)
    if study.deterministic_algorithms:
        if device.startswith("cuda"):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
    torch.manual_seed(study.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(study.model_seed)


def _train_one_epoch(
    model: nn.Module,
    data: PairTrainingData,
    order: Tensor,
    optimizer: torch.optim.Optimizer,
    target_scale: Tensor,
    batch_size: int,
) -> float:
    model.train()
    squared_error = torch.zeros((), dtype=torch.float64, device=data.root_force.device)
    element_count = 0
    for start in range(0, int(order.numel()), batch_size):
        selection = order[start : start + batch_size].to(data.centers.device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data.centers[selection], data.frames[selection])
        target = data.root_force[selection] / target_scale
        if prediction.shape != target.shape:
            raise ValueError("network prediction and target shapes do not match")
        error = prediction - target
        loss = error.square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("training loss became nonfinite")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (
                parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            ):
                raise RuntimeError(f"gradient is missing or nonfinite for {name}")
        optimizer.step()
        squared_error += error.detach().to(torch.float64).square().sum()
        element_count += error.numel()
    result = float(squared_error / element_count)
    if not math.isfinite(result):
        raise RuntimeError("training objective must be finite")
    return result


def _run_trial(
    study: StudyConfig,
    catalog: Mapping[str, Any],
    design: Mapping[str, Any],
    dataset_record: Mapping[str, Any],
    paths: TrialPaths,
    *,
    device_override: str | None = None,
    epoch_override: int | None = None,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    paths.directory.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_override or study.device)
    epochs = study.epochs if epoch_override is None else int(epoch_override)
    if epochs < 1 or epochs > study.epochs:
        raise ValueError("trial epochs must be between one and the study epoch count")
    source_digest = _source_sha256()
    trial_hash = _trial_hash(
        study,
        design,
        dataset_record,
        device=device,
        epochs=epochs,
        sample_limit=sample_limit,
    )
    training = _trial_training_config(
        study,
        design,
        paths,
        device=device,
        epochs=epochs,
    )
    definition = _build_trial_definition(
        catalog,
        design,
        training,
        trial_hash,
        dataset_record,
        source_digest,
        sample_limit,
    )
    _ensure_trial_definition(paths.definition, definition)
    if paths.summary.is_file():
        existing_summary = _load_json(paths.summary)
        if existing_summary.get("trial_hash") == trial_hash:
            return existing_summary
        raise RuntimeError("existing summary belongs to another trial definition")

    trial_id = str(design["trial_id"])
    candidate_id = str(design["candidate_id"])
    started_at = _utc_now()
    started_clock = time.perf_counter()
    _configure_runtime(study, device)
    previous_sigterm = signal.signal(signal.SIGTERM, _set_stop_requested)
    previous_sigint = signal.signal(signal.SIGINT, _set_stop_requested)
    try:
        data = _load_v2_training_data(study, device=device, sample_limit=sample_limit)
        split = create_split(
            len(data.root_force),
            seed=study.split_seed,
            fractions=study.split_fractions,
        )
        model, zeroed_head_count = _build_model(training)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        expected_parameters = design.get("preflight", {}).get("parameter_count")
        if sample_limit is None and expected_parameters != parameter_count:
            raise RuntimeError(
                "runtime parameter count does not match catalog preflight"
            )
        optimizer, scheduler = _optimizer_and_scheduler(model, training)
        shuffle_generator = torch.Generator().manual_seed(study.shuffle_seed)
        history = _read_history(paths.history)

        if paths.resume.is_file():
            resume = _restore_resume(
                paths.resume,
                model,
                optimizer,
                scheduler,
                shuffle_generator,
                trial_hash=trial_hash,
            )
            start_epoch = int(resume["epoch"])
            target_scale = torch.tensor(
                float(resume["target_scale"]),
                dtype=data.root_force.dtype,
                device=data.root_force.device,
            )
            best_epoch = int(resume["best_epoch"])
            best_validation = dict(resume["best_validation"])
            best_parameters = {
                name: value.detach().cpu().clone()
                for name, value in resume["best_parameter_state_dict"].items()
            }
            initial_train_loss = float(resume["initial_train_normalized_mse"])
            history = [row for row in history if row["epoch"] <= start_epoch]
            if len(history) != start_epoch + 1:
                raise RuntimeError("history does not cover the resume checkpoint")
            _write_history(paths.history, history)
            best_file_epoch = None
            if paths.best.is_file():
                best_file_payload = torch.load(
                    paths.best,
                    map_location="cpu",
                    weights_only=True,
                )
                if isinstance(best_file_payload, Mapping):
                    best_file_epoch = int(best_file_payload.get("epoch", -1))
            if best_file_epoch != best_epoch:
                _save_learned_checkpoint(
                    paths.best,
                    best_parameters,
                    epoch=best_epoch,
                    target_scale=target_scale,
                    config=training,
                    metrics=best_validation,
                )
        else:
            if history:
                raise RuntimeError("history exists without a resume checkpoint")
            start_epoch = 0
            train_selection = split.train.to(data.root_force.device)
            target_scale = torch.sqrt(data.root_force[train_selection].square().mean())
            if not bool(torch.isfinite(target_scale)) or float(target_scale) <= 0.0:
                raise RuntimeError("training target scale must be finite and positive")
            initial_train = _evaluate(
                model,
                data,
                split.train,
                training.batch_size,
                target_scale,
            )
            initial_validation = _evaluate(
                model,
                data,
                split.validation,
                training.batch_size,
                target_scale,
            )
            initial_train_loss = float(initial_train["normalized_mse"])
            best_epoch = 0
            best_validation = dict(initial_validation)
            best_parameters = _parameter_state(model)
            history = [
                _history_row(
                    0,
                    float(optimizer.param_groups[0]["lr"]),
                    initial_train_loss,
                    initial_validation,
                    0.0,
                )
            ]
            _write_history(paths.history, history)
            _save_resume(
                paths.resume,
                model,
                optimizer,
                scheduler,
                shuffle_generator,
                epoch=0,
                target_scale=target_scale,
                trial_hash=trial_hash,
                best_epoch=best_epoch,
                best_validation=best_validation,
                best_parameter_state=best_parameters,
                initial_train_loss=initial_train_loss,
            )
            _save_learned_checkpoint(
                paths.best,
                best_parameters,
                epoch=0,
                target_scale=target_scale,
                config=training,
                metrics=best_validation,
            )

        last_train_loss = float(history[-1]["train_objective_normalized_mse"])
        last_validation_loss = float(history[-1]["validation_normalized_mse"])
        _write_status(
            paths,
            trial_id=trial_id,
            candidate_id=candidate_id,
            trial_hash=trial_hash,
            status="running",
            epoch=start_epoch,
            best_epoch=best_epoch,
            train_loss=last_train_loss,
            validation_loss=last_validation_loss,
        )

        for epoch in range(start_epoch + 1, epochs + 1):
            epoch_clock = time.perf_counter()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            order = split.train[
                torch.randperm(int(split.train.numel()), generator=shuffle_generator)
            ]
            train_loss = _train_one_epoch(
                model,
                data,
                order,
                optimizer,
                target_scale,
                training.batch_size,
            )
            scheduler.step()
            validation = _evaluate(
                model,
                data,
                split.validation,
                training.batch_size,
                target_scale,
            )
            row = _history_row(
                epoch,
                learning_rate,
                train_loss,
                validation,
                time.perf_counter() - epoch_clock,
            )
            history.append(row)
            _write_history(paths.history, history)
            improved = validation["normalized_mse"] < best_validation["normalized_mse"]
            if improved:
                best_epoch = epoch
                best_validation = dict(validation)
                best_parameters = _parameter_state(model)
            if epoch % study.resume_every == 0 or _STOP_REQUESTED:
                _save_resume(
                    paths.resume,
                    model,
                    optimizer,
                    scheduler,
                    shuffle_generator,
                    epoch=epoch,
                    target_scale=target_scale,
                    trial_hash=trial_hash,
                    best_epoch=best_epoch,
                    best_validation=best_validation,
                    best_parameter_state=best_parameters,
                    initial_train_loss=initial_train_loss,
                )
                _save_learned_checkpoint(
                    paths.best,
                    best_parameters,
                    epoch=best_epoch,
                    target_scale=target_scale,
                    config=training,
                    metrics=best_validation,
                )
            _write_status(
                paths,
                trial_id=trial_id,
                candidate_id=candidate_id,
                trial_hash=trial_hash,
                status="interrupted" if _STOP_REQUESTED else "running",
                epoch=epoch,
                best_epoch=best_epoch,
                train_loss=train_loss,
                validation_loss=float(validation["normalized_mse"]),
            )
            if epoch == 1 or epoch % study.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "trial_id": trial_id,
                            "epoch": epoch,
                            "train_objective_normalized_mse": train_loss,
                            "validation_normalized_mse": validation["normalized_mse"],
                            "best_epoch": best_epoch,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if _STOP_REQUESTED:
                raise TrialInterrupted(f"trial interrupted after epoch {epoch}")

        final_validation = _evaluate(
            model,
            data,
            split.validation,
            training.batch_size,
            target_scale,
        )
        _save_learned_checkpoint(
            paths.best,
            best_parameters,
            epoch=best_epoch,
            target_scale=target_scale,
            config=training,
            metrics=best_validation,
        )
        _save_learned_checkpoint(
            paths.final,
            _parameter_state(model),
            epoch=epochs,
            target_scale=target_scale,
            config=training,
            metrics=final_validation,
        )
        selected_model, selected_scale_value, _best_payload = load_trained_model(
            paths.best,
            device=device,
            dtype=study.dtype,
        )
        selected_scale = torch.tensor(
            selected_scale_value,
            dtype=data.root_force.dtype,
            device=data.root_force.device,
        )
        selected_metrics = {
            "train": _evaluate(
                selected_model,
                data,
                split.train,
                training.batch_size,
                selected_scale,
            ),
            "validation": _evaluate(
                selected_model,
                data,
                split.validation,
                training.batch_size,
                selected_scale,
            ),
            "test": _evaluate(
                selected_model,
                data,
                split.test,
                training.batch_size,
                selected_scale,
            ),
        }
        probe_count = min(study.symmetry_probe_count, int(split.validation.numel()))
        probe = split.validation[:probe_count].to(data.centers.device)
        symmetry = symmetry_metrics(
            selected_model,
            data.centers[probe],
            data.frames[probe],
            tolerance=study.symmetry_tolerance,
        )
        parameters_finite = all(
            bool(torch.isfinite(parameter).all())
            for parameter in selected_model.parameters()
        )
        failures = []
        if not parameters_finite:
            failures.append("selected model parameters are nonfinite")
        if not symmetry["passed"]:
            failures.append("selected model symmetry residual exceeds tolerance")
        status = "complete" if not failures else "failed_validation"
        finished_at = _utc_now()
        summary: dict[str, Any] = {
            "schema_name": "tfenn_benzene_pair_hyper_trial_result",
            "schema_version": 1,
            "trial_id": trial_id,
            "candidate_id": candidate_id,
            "trial_hash": trial_hash,
            "status": status,
            "failures": failures,
            "catalog": {
                "path": str(study.catalog_path),
                "sha256": catalog["catalog_sha256"],
                "config_hash": design["config_hash"],
                "functional_hash": design["functional_hash"],
            },
            "data": {
                "csv_path": str(data.csv_path),
                "csv_sha256": data.csv_sha256,
                "metadata_path": str(data.metadata_path),
                "metadata_sha256": data.metadata_sha256,
                "sample_count": len(data.root_force),
                "dataset_revision": data.metadata.get("dataset_revision"),
                "opls_runtime_version": data.metadata.get("opls", {}).get(
                    "runtime_version"
                ),
            },
            "split": {
                "seed": study.split_seed,
                "fractions": list(study.split_fractions),
                **split.as_json(),
            },
            "model": {
                "pipeline": training.pipeline.as_dict(),
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "gate_count": len(model.gates),
                "gate_manifest": list(model.gate_manifest),
                "offline_compilation": model.offline_compilation_summary,
                "zeroed_output_head_count": zeroed_head_count,
                "gate_mlp_hidden_width": 64,
            },
            "training": {
                "epochs_requested": epochs,
                "epochs_completed": epochs,
                "epoch_zero_train_and_validation": True,
                "later_train_metric": "batch_objective_normalized_mse",
                "later_validation_metric": "complete_partition_normalized_mse",
                "test_evaluation_count": 1,
                "optimizer": training.optimizer,
                "learning_rate": training.learning_rate,
                "weight_decay": training.weight_decay,
                "batch_size": training.batch_size,
                "scheduler": "StepLR",
                "scheduler_step_size": training.scheduler_step_size,
                "scheduler_gamma": training.scheduler_gamma,
                "resume_every": study.resume_every,
                "initial_train_normalized_mse": initial_train_loss,
            },
            "selection": {
                "criterion": "minimum validation normalized MSE",
                "best_epoch": best_epoch,
                "best_validation_during_training": best_validation,
                "selected_metrics": selected_metrics,
                "symmetry": symmetry,
                "parameters_finite": parameters_finite,
            },
            "history": {
                "path": str(paths.history),
                "sha256": sha256_file(paths.history),
                "row_count": len(history),
                "first_epoch": 0,
                "last_epoch": epochs,
            },
            "checkpoints": {
                "content": "learned_parameters_only",
                "fixed_tensor_artifacts_stored": False,
                "best": {
                    "path": str(paths.best),
                    "sha256": sha256_file(paths.best),
                    "epoch": best_epoch,
                },
                "final": {
                    "path": str(paths.final),
                    "sha256": sha256_file(paths.final),
                    "epoch": epochs,
                },
                "resume_removed_after_completion": True,
            },
            "runtime": {
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "duration_seconds": time.perf_counter() - started_clock,
                "versions": _runtime_versions(),
                "git": _git_provenance(),
                "source_sha256": source_digest,
                "device": device,
                "dtype": study.dtype,
                "threads": study.threads,
                "deterministic_algorithms": study.deterministic_algorithms,
            },
        }
        _atomic_json(paths.summary, summary)
        paths.resume.unlink(missing_ok=True)
        _write_status(
            paths,
            trial_id=trial_id,
            candidate_id=candidate_id,
            trial_hash=trial_hash,
            status=status,
            epoch=epochs,
            best_epoch=best_epoch,
            train_loss=float(history[-1]["train_objective_normalized_mse"]),
            validation_loss=float(history[-1]["validation_normalized_mse"]),
        )
        return summary
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


def _record_error(
    paths: TrialPaths,
    design: Mapping[str, Any],
    trial_hash: str | None,
    error: BaseException,
    *,
    interrupted: bool,
) -> None:
    status = "interrupted" if interrupted else "error"
    epoch = 0
    if paths.status.is_file():
        try:
            epoch = int(_load_json(paths.status).get("epoch", 0))
        except Exception:
            epoch = 0
    value = {
        "schema_name": "tfenn_benzene_pair_hyper_trial_error",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "trial_id": design.get("trial_id"),
        "candidate_id": design.get("candidate_id"),
        "trial_hash": trial_hash,
        "status": status,
        "epoch": epoch,
        "exception_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "runtime": _runtime_versions(),
    }
    _atomic_json(paths.error, value)
    _write_status(
        paths,
        trial_id=str(design.get("trial_id")),
        candidate_id=str(design.get("candidate_id")),
        trial_hash=trial_hash or "unknown",
        status=status,
        epoch=epoch,
        message=str(error),
    )


class StudyLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream: Any = None

    def __enter__(self) -> StudyLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+", encoding="utf_8")
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                if self.path.stat().st_size == 0:
                    self.stream.write(" ")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.seek(0)
            owner = self.stream.read().strip()
            self.stream.close()
            raise RuntimeError(
                f"study is already locked by {owner or 'another process'}"
            ) from error
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started_at_utc": _utc_now(),
                },
                ensure_ascii=False,
            )
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self.stream is None:
            return
        if os.name == "nt":
            import msvcrt

            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def _ensure_study_files(
    study: StudyConfig,
    catalog: Mapping[str, Any],
    designs: Sequence[dict[str, Any]],
    dataset: Mapping[str, Any],
    *,
    device: str,
) -> None:
    study.study_directory.mkdir(parents=True, exist_ok=True)
    split = create_split(
        study.expected_sample_count,
        seed=study.split_seed,
        fractions=study.split_fractions,
    )
    split_value = {
        "schema_name": "tfenn_benzene_pair_hyper_split",
        "schema_version": 1,
        "dataset_csv_sha256": dataset["csv_sha256"],
        "seed": study.split_seed,
        "fractions": list(study.split_fractions),
        **split.as_json(),
    }
    split_path = study.study_directory / "split.json"
    if split_path.exists() and _load_json(split_path) != split_value:
        raise RuntimeError("existing study split does not match the protocol")
    if not split_path.exists():
        _atomic_json(split_path, split_value)
    manifest = {
        "schema_name": "tfenn_benzene_pair_hyper_study",
        "schema_version": 1,
        "catalog_path": str(study.catalog_path),
        "catalog_sha256": catalog["catalog_sha256"],
        "design_count": len(designs),
        "dataset_csv_path": str(study.csv_path),
        "dataset_csv_sha256": dataset["csv_sha256"],
        "dataset_metadata_sha256": dataset["metadata_sha256"],
        "dataset_validation_sha256": dataset["validation_sha256"],
        "source_sha256": _source_sha256(),
        "protocol": study.protocol_dict(device=device),
        "split_sha256": _canonical_sha256(split_value),
        "search_interpretation": "coupled broad profile search",
        "model_selection": "minimum validation normalized MSE only",
        "test_usage": "one final evaluation of the validation selected model",
        "ranking_rule": "ascending best validation normalized MSE only",
    }
    manifest["study_hash"] = _canonical_sha256(manifest)
    manifest_path = study.study_directory / "manifest.json"
    if manifest_path.exists() and _load_json(manifest_path) != manifest:
        raise RuntimeError(
            "existing study manifest does not match this source or protocol"
        )
    if not manifest_path.exists():
        _atomic_json(manifest_path, manifest)


def _nested(value: Mapping[str, Any], *names: str) -> Any:
    current: Any = value
    for name in names:
        if not isinstance(current, Mapping):
            return None
        current = current.get(name)
    return current


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def _master_row(study: StudyConfig, design: Mapping[str, Any]) -> dict[str, Any]:
    paths = TrialPaths.from_directory(
        study.study_directory / "designs" / str(design["trial_id"])
    )
    status_value = _load_json(paths.status) if paths.status.is_file() else {}
    summary = _load_json(paths.summary) if paths.summary.is_file() else {}
    error = _load_json(paths.error) if paths.error.is_file() else {}
    selected = _nested(summary, "selection", "selected_metrics") or {}
    train = selected.get("train", {}) if isinstance(selected, Mapping) else {}
    validation = selected.get("validation", {}) if isinstance(selected, Mapping) else {}
    test = selected.get("test", {}) if isinstance(selected, Mapping) else {}
    symmetry = _nested(summary, "selection", "symmetry") or {}
    maximum_absolute = (
        max(
            (
                float(value.get("maximum_absolute", 0.0))
                for value in symmetry.values()
                if isinstance(value, Mapping) and "maximum_absolute" in value
            ),
            default="",
        )
        if isinstance(symmetry, Mapping)
        else ""
    )
    return {
        "trial_id": design["trial_id"],
        "candidate_id": design["candidate_id"],
        "status": summary.get("status", status_value.get("status", "planned")),
        "trial_hash": summary.get("trial_hash", status_value.get("trial_hash", "")),
        "epochs_completed": _first_defined(
            _nested(summary, "training", "epochs_completed"),
            status_value.get("epoch", 0),
        ),
        "duration_seconds": _first_defined(
            _nested(summary, "runtime", "duration_seconds")
        ),
        "parameter_count": _nested(summary, "model", "parameter_count")
        or _nested(design, "preflight", "parameter_count")
        or "",
        "gate_count": _nested(summary, "model", "gate_count")
        or _nested(design, "preflight", "gate_count")
        or "",
        "best_epoch": _first_defined(
            _nested(summary, "selection", "best_epoch"),
            status_value.get("best_epoch", ""),
        ),
        "best_validation_normalized_mse": _first_defined(
            _nested(
                summary,
                "selection",
                "best_validation_during_training",
                "normalized_mse",
            )
        ),
        "train_rmse": train.get("rmse", ""),
        "train_mae": train.get("mae", ""),
        "train_r2": train.get("r2", ""),
        "train_relative_rmse": train.get("relative_rmse", ""),
        "validation_rmse": validation.get("rmse", ""),
        "validation_r2": validation.get("r2", ""),
        "test_rmse": test.get("rmse", ""),
        "test_mae": test.get("mae", ""),
        "test_r2": test.get("r2", ""),
        "test_relative_rmse": test.get("relative_rmse", ""),
        "symmetry_maximum_absolute": maximum_absolute,
        "best_checkpoint_path": _nested(summary, "checkpoints", "best", "path") or "",
        "best_checkpoint_sha256": _nested(summary, "checkpoints", "best", "sha256")
        or "",
        "final_checkpoint_path": _nested(summary, "checkpoints", "final", "path") or "",
        "final_checkpoint_sha256": _nested(summary, "checkpoints", "final", "sha256")
        or "",
        "error_type": error.get("exception_type", ""),
        "error_message": error.get("message", ""),
        "error_path": str(paths.error) if error else "",
    }


def _write_master_results(
    study: StudyConfig, designs: Sequence[dict[str, Any]]
) -> Path:
    path = study.study_directory / "master_results.csv"
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    rows = [_master_row(study, design) for design in designs]
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MASTER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return path


def _write_validation_ranking(
    study: StudyConfig,
    designs: Sequence[dict[str, Any]],
) -> Path:
    path = study.study_directory / "validation_ranking.csv"
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    rows = []
    for design in designs:
        row = _master_row(study, design)
        if row["status"] not in {"complete", "failed_validation"}:
            continue
        if row["best_validation_normalized_mse"] == "":
            continue
        rows.append(row)
    rows.sort(key=lambda row: float(row["best_validation_normalized_mse"]))
    ranked = [{"rank": rank, **row} for rank, row in enumerate(rows, start=1)]
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        writer.writerows(ranked)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return path


def _completed_trial(paths: TrialPaths, expected_hash: str) -> bool:
    if not paths.summary.is_file():
        return False
    summary = _load_json(paths.summary)
    if summary.get("trial_hash") != expected_hash:
        raise RuntimeError(f"completed trial hash mismatch in {paths.directory}")
    if summary.get("status") not in {"complete", "failed_validation"}:
        return False
    if not paths.best.is_file() or not paths.final.is_file():
        raise RuntimeError(f"completed trial is missing a model in {paths.directory}")
    paths.resume.unlink(missing_ok=True)
    return True


def _selected_designs(
    designs: Sequence[dict[str, Any]],
    trial_ids: Sequence[str],
    start_index: int,
    limit: int | None,
) -> tuple[dict[str, Any], ...]:
    if trial_ids:
        selected = tuple(_find_design(designs, name) for name in trial_ids)
    else:
        if start_index < 1 or start_index > len(designs):
            raise ValueError("start_index is outside the catalog")
        selected = tuple(designs[start_index - 1 :])
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return selected


def run_study(arguments: argparse.Namespace) -> int:
    study = StudyConfig.from_path(arguments.study_config)
    device = _resolve_device(arguments.device or study.device)
    catalog, designs = _load_catalog(study.catalog_path)
    dataset = _validate_dataset_files(study)
    selected = _selected_designs(
        designs,
        arguments.trial_id,
        arguments.start_index,
        arguments.limit,
    )
    lock_path = study.study_directory / ".runner.lock"
    journal = study.study_directory / "master_journal.jsonl"
    with StudyLock(lock_path):
        _ensure_study_files(study, catalog, designs, dataset, device=device)
        _write_master_results(study, designs)
        _write_validation_ranking(study, designs)
        _append_journal(
            journal,
            {
                "event": "study_started",
                "selected_count": len(selected),
                "device": device,
            },
        )
        for design in selected:
            paths = TrialPaths.from_directory(
                study.study_directory / "designs" / str(design["trial_id"])
            )
            expected_hash = _trial_hash(
                study,
                design,
                dataset,
                device=device,
                epochs=study.epochs,
                sample_limit=None,
            )
            if _completed_trial(paths, expected_hash):
                _append_journal(
                    journal,
                    {"event": "trial_skipped_complete", "trial_id": design["trial_id"]},
                )
                continue
            if paths.status.is_file() and not arguments.retry_errors:
                old_status = _load_json(paths.status).get("status")
                if old_status == "error":
                    _append_journal(
                        journal,
                        {
                            "event": "trial_skipped_error",
                            "trial_id": design["trial_id"],
                        },
                    )
                    continue
            paths.directory.mkdir(parents=True, exist_ok=True)
            _append_journal(
                journal,
                {
                    "event": "trial_started",
                    "trial_id": design["trial_id"],
                    "trial_hash": expected_hash,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "trial_started",
                        "trial_id": design["trial_id"],
                        "candidate_id": design["candidate_id"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            command = [
                sys.executable,
                "-m",
                "experiments.benzene_pair.hyper_search",
                "trial",
                "--study_config",
                str(Path(arguments.study_config).resolve()),
                "--trial_id",
                str(design["trial_id"]),
                "--device",
                device,
            ]
            with (
                paths.stdout.open("a", encoding="utf_8") as stdout,
                paths.stderr.open("a", encoding="utf_8") as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
                try:
                    return_code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    return_code = process.wait()
                    _append_journal(
                        journal,
                        {
                            "event": "study_interrupted",
                            "trial_id": design["trial_id"],
                            "return_code": return_code,
                        },
                    )
                    _write_master_results(study, designs)
                    _write_validation_ranking(study, designs)
                    return 130
            _write_master_results(study, designs)
            _write_validation_ranking(study, designs)
            event = "trial_finished" if return_code == 0 else "trial_error"
            _append_journal(
                journal,
                {
                    "event": event,
                    "trial_id": design["trial_id"],
                    "return_code": return_code,
                },
            )
            print(
                json.dumps(
                    {
                        "event": event,
                        "trial_id": design["trial_id"],
                        "return_code": return_code,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if return_code == 130:
                return 130
        results_path = _write_master_results(study, designs)
        ranking_path = _write_validation_ranking(study, designs)
        _append_journal(
            journal,
            {
                "event": "study_finished",
                "results_path": str(results_path),
                "ranking_path": str(ranking_path),
            },
        )
    return 0


def run_trial_command(arguments: argparse.Namespace) -> int:
    study = StudyConfig.from_path(arguments.study_config)
    catalog, designs = _load_catalog(study.catalog_path)
    design = _find_design(designs, arguments.trial_id)
    dataset = _validate_dataset_files(study)
    directory = (
        Path(arguments.output_directory).resolve()
        if arguments.output_directory is not None
        else study.study_directory / "designs" / str(design["trial_id"])
    )
    paths = TrialPaths.from_directory(directory)
    device = _resolve_device(arguments.device or study.device)
    epochs = study.epochs if arguments.epochs is None else int(arguments.epochs)
    trial_hash: str | None = None
    try:
        trial_hash = _trial_hash(
            study,
            design,
            dataset,
            device=device,
            epochs=epochs,
            sample_limit=arguments.sample_limit,
        )
        summary = _run_trial(
            study,
            catalog,
            design,
            dataset,
            paths,
            device_override=device,
            epoch_override=arguments.epochs,
            sample_limit=arguments.sample_limit,
        )
        print(
            json.dumps(
                {
                    "trial_id": design["trial_id"],
                    "status": summary["status"],
                    "summary_path": str(paths.summary),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except TrialInterrupted as error:
        _record_error(paths, design, trial_hash, error, interrupted=True)
        print(str(error), file=sys.stderr, flush=True)
        return 130
    except BaseException as error:
        _record_error(paths, design, trial_hash, error, interrupted=False)
        traceback.print_exception(error, file=sys.stderr)
        return 1


def run_smoke(arguments: argparse.Namespace) -> int:
    study = StudyConfig.from_path(arguments.study_config)
    _catalog, designs = _load_catalog(study.catalog_path)
    trial_id = arguments.trial_id or str(designs[0]["trial_id"])
    output = (
        Path(arguments.output_directory).resolve()
        if arguments.output_directory is not None
        else Path(tempfile.mkdtemp(prefix="tfenn_hyper_smoke_"))
    )
    command = [
        sys.executable,
        "-m",
        "experiments.benzene_pair.hyper_search",
        "trial",
        "--study_config",
        str(Path(arguments.study_config).resolve()),
        "--trial_id",
        trial_id,
        "--device",
        arguments.device,
        "--epochs",
        str(arguments.epochs),
        "--sample_limit",
        str(arguments.sample_limit),
        "--output_directory",
        str(output),
    ]
    print(
        json.dumps({"event": "smoke_started", "output_directory": str(output)}),
        flush=True,
    )
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    print(
        json.dumps(
            {
                "event": "smoke_finished",
                "return_code": result.returncode,
                "output_directory": str(output),
            }
        ),
        flush=True,
    )
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--study_config", type=Path, default=DEFAULT_STUDY_CONFIG)
    run.add_argument("--device", default=None)
    run.add_argument("--trial_id", action="append", default=[])
    run.add_argument("--start_index", type=int, default=1)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--retry_errors", action="store_true")
    run.set_defaults(handler=run_study)

    trial = commands.add_parser("trial")
    trial.add_argument("--study_config", type=Path, default=DEFAULT_STUDY_CONFIG)
    trial.add_argument("--trial_id", required=True)
    trial.add_argument("--device", default=None)
    trial.add_argument("--epochs", type=int, default=None)
    trial.add_argument("--sample_limit", type=int, default=None)
    trial.add_argument("--output_directory", type=Path, default=None)
    trial.set_defaults(handler=run_trial_command)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--study_config", type=Path, default=DEFAULT_STUDY_CONFIG)
    smoke.add_argument("--trial_id", default=None)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--sample_limit", type=int, default=96)
    smoke.add_argument("--output_directory", type=Path, default=None)
    smoke.set_defaults(handler=run_smoke)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    return int(parsed.handler(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
