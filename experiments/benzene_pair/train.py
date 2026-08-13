"""Train a configured invariant gate pipeline on benzene pair data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from TFENN.data import load_benzene_cluster_csv, load_benzene_cluster_metadata
from TFENN.models import (
    InvariantGatePipeline,
    InvariantGatePipelineV2,
    InvariantGatePipelineV2Config,
    PairPipelineConfig,
    build_invariant_gate_pipeline,
    build_invariant_gate_pipeline_v2,
    default_invariant_gate_pipeline_v2_config,
    default_pair_pipeline_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    REPOSITORY_ROOT / "data" / "benzene_pair" / "benzene_pair_opls_2_0_0_v3.csv"
)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "runs" / "default_v2"
TARGET_DEFINITION = "force on molecule_id 0 in the root coordinate frame"
METRIC_NAMES = (
    "normalized_mse",
    "mse",
    "rmse",
    "mae",
    "relative_rmse",
    "r2",
)
HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    *(f"train_{name}" for name in METRIC_NAMES),
    *(f"validation_{name}" for name in METRIC_NAMES),
)
PipelineConfig = PairPipelineConfig | InvariantGatePipelineV2Config
PipelineModel = InvariantGatePipeline | InvariantGatePipelineV2


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Store deterministic sample indices for three disjoint partitions."""

    train: Tensor
    validation: Tensor
    test: Tensor

    def as_json(self) -> dict[str, Any]:
        return {
            "train_count": int(self.train.numel()),
            "validation_count": int(self.validation.numel()),
            "test_count": int(self.test.numel()),
            "train_indices": self.train.tolist(),
            "validation_indices": self.validation.tolist(),
            "test_indices": self.test.tolist(),
        }


@dataclass(frozen=True, slots=True)
class PairTrainingData:
    """Store tensors and provenance for one pair force training run."""

    centers: Tensor
    frames: Tensor
    root_force: Tensor
    metadata: dict[str, Any]
    csv_path: Path
    metadata_path: Path
    csv_sha256: str
    metadata_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Describe one deterministic training run."""

    csv_path: Path = DEFAULT_DATA_PATH
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 5.0e-3
    weight_decay: float = 1.0e-4
    optimizer: str = "adamw"
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.5
    split_seed: int = 20260802
    model_seed: int = 20260803
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    device: str = "cpu"
    dtype: str = "float32"
    threads: int = 1
    symmetry_tolerance: float = 1.0e-4
    progress_every: int = 10
    pipeline: PipelineConfig = field(
        default_factory=default_invariant_gate_pipeline_v2_config
    )
    dataset_revision: int | None = 3
    zero_output_heads: bool = False
    maximum_train_loss_ratio: float = 0.1
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "csv_path", Path(self.csv_path))
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be adam or adamw")
        if self.scheduler_step_size < 1:
            raise ValueError("scheduler_step_size must be positive")
        if (
            not math.isfinite(self.scheduler_gamma)
            or self.scheduler_gamma <= 0.0
            or self.scheduler_gamma > 1.0
        ):
            raise ValueError("scheduler_gamma must be in the interval zero to one")
        validate_split_fractions(self.split_fractions)
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.threads < 1:
            raise ValueError("threads must be positive")
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive")
        if not math.isfinite(self.symmetry_tolerance) or self.symmetry_tolerance <= 0.0:
            raise ValueError("symmetry_tolerance must be finite and positive")
        if not isinstance(
            self.pipeline,
            (PairPipelineConfig, InvariantGatePipelineV2Config),
        ):
            raise TypeError("pipeline must be a supported pipeline config")
        if self.dataset_revision is not None and (
            not isinstance(self.dataset_revision, Integral)
            or isinstance(self.dataset_revision, bool)
            or self.dataset_revision < 1
        ):
            raise ValueError("dataset_revision must be a positive integer or null")
        if not isinstance(self.zero_output_heads, bool):
            raise TypeError("zero_output_heads must be bool")
        if (
            not math.isfinite(self.maximum_train_loss_ratio)
            or self.maximum_train_loss_ratio <= 0.0
        ):
            raise ValueError("maximum_train_loss_ratio must be finite and positive")

    def as_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["csv_path"] = str(self.csv_path.resolve())
        result["output_directory"] = str(self.output_directory.resolve())
        result["split_fractions"] = list(self.split_fractions)
        result["pipeline"] = self.pipeline.as_dict()
        result["pipeline_version"] = self.pipeline_version
        return result

    @property
    def pipeline_version(self) -> str:
        """Return the stable serialized version for the configured network."""
        if isinstance(self.pipeline, PairPipelineConfig):
            return "v1"
        return "v2"

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        base_directory: Path = REPOSITORY_ROOT,
    ) -> TrainingConfig:
        if not isinstance(value, Mapping):
            raise TypeError("training config must be a mapping")
        defaults = cls()

        def path_value(name: str, default: Path) -> Path:
            result = Path(value.get(name, default))
            return result if result.is_absolute() else base_directory / result

        pipeline_value = value.get("pipeline")
        pipeline_version = value.get("pipeline_version")
        if pipeline_version not in (None, "v1", "v2"):
            raise ValueError("pipeline_version must be v1 or v2")
        if pipeline_value is None:
            pipeline = (
                default_pair_pipeline_config()
                if pipeline_version == "v1"
                else defaults.pipeline
            )
        else:
            if not isinstance(pipeline_value, Mapping):
                raise TypeError("pipeline must be a mapping")
            if pipeline_version is None:
                stages = pipeline_value.get("stages")
                first_stage = stages[0] if isinstance(stages, list) and stages else {}
                pipeline_version = (
                    "v2"
                    if isinstance(first_stage, Mapping)
                    and "source_names" in first_stage
                    else "v1"
                )
            if pipeline_version == "v1":
                pipeline = PairPipelineConfig.from_dict(pipeline_value)
            elif pipeline_version == "v2":
                pipeline = InvariantGatePipelineV2Config.from_dict(pipeline_value)
            else:
                raise ValueError("pipeline_version must be v1 or v2")
        return cls(
            csv_path=path_value("csv_path", defaults.csv_path),
            output_directory=path_value(
                "output_directory",
                defaults.output_directory,
            ),
            epochs=value.get("epochs", defaults.epochs),
            batch_size=value.get("batch_size", defaults.batch_size),
            learning_rate=value.get("learning_rate", defaults.learning_rate),
            weight_decay=value.get("weight_decay", defaults.weight_decay),
            optimizer=value.get("optimizer", defaults.optimizer),
            scheduler_step_size=value.get(
                "scheduler_step_size",
                defaults.scheduler_step_size,
            ),
            scheduler_gamma=value.get("scheduler_gamma", defaults.scheduler_gamma),
            split_seed=value.get("split_seed", defaults.split_seed),
            model_seed=value.get("model_seed", defaults.model_seed),
            split_fractions=tuple(
                value.get("split_fractions", defaults.split_fractions)
            ),
            device=value.get("device", defaults.device),
            dtype=value.get("dtype", defaults.dtype),
            threads=value.get("threads", defaults.threads),
            symmetry_tolerance=value.get(
                "symmetry_tolerance",
                defaults.symmetry_tolerance,
            ),
            progress_every=value.get("progress_every", defaults.progress_every),
            pipeline=pipeline,
            dataset_revision=value.get(
                "dataset_revision",
                (
                    None
                    if isinstance(pipeline, PairPipelineConfig)
                    else defaults.dataset_revision
                ),
            ),
            zero_output_heads=value.get(
                "zero_output_heads",
                defaults.zero_output_heads,
            ),
            maximum_train_loss_ratio=value.get(
                "maximum_train_loss_ratio",
                defaults.maximum_train_loss_ratio,
            ),
            overwrite=value.get("overwrite", defaults.overwrite),
        )


def validate_split_fractions(fractions: Sequence[float]) -> None:
    """Validate three positive fractions whose sum is one."""
    if len(fractions) != 3:
        raise ValueError("split_fractions must contain three values")
    values = tuple(float(value) for value in fractions)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("split fractions must be finite and positive")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("split fractions must sum to one")


def create_split(
    sample_count: int,
    *,
    seed: int,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
) -> DatasetSplit:
    """Create one deterministic partition with exact total coverage."""
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or sample_count < 3
    ):
        raise ValueError("sample_count must be an integer of at least three")
    sample_count = int(sample_count)
    validate_split_fractions(fractions)
    raw_counts = [sample_count * float(value) for value in fractions]
    counts = [math.floor(value) for value in raw_counts]
    remaining = sample_count - sum(counts)
    residual_order = sorted(
        range(3),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in residual_order[:remaining]:
        counts[index] += 1
    if any(count < 1 for count in counts):
        raise ValueError("each split must contain at least one sample")

    permutation = torch.randperm(
        sample_count,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    train_end = counts[0]
    validation_end = train_end + counts[1]
    split = DatasetSplit(
        train=permutation[:train_end],
        validation=permutation[train_end:validation_end],
        test=permutation[validation_end:],
    )
    combined = torch.cat((split.train, split.validation, split.test))
    if not torch.equal(torch.sort(combined).values, torch.arange(sample_count)):
        raise RuntimeError("split indices do not cover the dataset exactly once")
    return split


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 20) -> None:
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


def regression_metrics(
    prediction_normalized: Tensor,
    target_normalized: Tensor,
    target_scale: float | Tensor,
) -> dict[str, float]:
    """Compute normalized and physical force regression metrics."""
    if prediction_normalized.shape != target_normalized.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction_normalized.numel() == 0:
        raise ValueError("prediction and target must not be empty")
    prediction = prediction_normalized.detach().to(dtype=torch.float64)
    target = target_normalized.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise ValueError("prediction and target must be finite")
    scale = float(torch.as_tensor(target_scale).detach().cpu())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("target_scale must be finite and positive")

    error = prediction - target
    normalized_mse = float(error.square().mean())
    physical_error = error * scale
    physical_target = target * scale
    mse = float(physical_error.square().mean())
    rmse = math.sqrt(mse)
    mae = float(physical_error.abs().mean())
    target_rms = float(torch.sqrt(physical_target.square().mean()))
    if target_rms <= 0.0:
        raise ValueError("relative RMSE requires a nonzero target")
    relative_rmse = rmse / target_rms

    if physical_target.ndim == 1:
        centered_target = physical_target - physical_target.mean()
    else:
        dimensions = tuple(range(physical_target.ndim - 1))
        centered_target = physical_target - physical_target.mean(
            dim=dimensions,
            keepdim=True,
        )
    residual_sum = float(physical_error.square().sum())
    total_sum = float(centered_target.square().sum())
    if total_sum <= torch.finfo(torch.float64).eps:
        r2 = 1.0 if residual_sum <= torch.finfo(torch.float64).eps else 0.0
    else:
        r2 = 1.0 - residual_sum / total_sum

    metrics = {
        "normalized_mse": normalized_mse,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "relative_rmse": relative_rmse,
        "r2": r2,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("regression metrics must be finite")
    return metrics


def make_history_row(
    epoch: int,
    learning_rate: float,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
) -> dict[str, float | int]:
    """Create one complete history record."""
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    row: dict[str, float | int] = {
        "epoch": int(epoch),
        "learning_rate": float(learning_rate),
    }
    for prefix, values in (
        ("train", train_metrics),
        ("validation", validation_metrics),
    ):
        missing = set(METRIC_NAMES) - set(values)
        if missing:
            raise ValueError(f"missing {prefix} metrics: {sorted(missing)}")
        for name in METRIC_NAMES:
            value = float(values[name])
            if not math.isfinite(value):
                raise ValueError(f"{prefix}_{name} must be finite")
            row[f"{prefix}_{name}"] = value
    return row


def write_history_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, float | int]],
) -> Path:
    """Write all epoch records with an atomic replacement."""
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = history_path.with_name(f"{history_path.name}.partial")
    with partial_path.open("w", newline="", encoding="utf_8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            if set(row) != set(HISTORY_FIELDS):
                raise ValueError("history row fields do not match the schema")
            writer.writerow(row)
    _replace_with_retry(partial_path, history_path)
    return history_path


def append_history_row(
    path: str | Path,
    row: Mapping[str, float | int],
) -> Path:
    history_path = Path(path)
    if tuple(row.keys()) != HISTORY_FIELDS:
        raise ValueError("history row fields do not match the schema")
    for attempt in range(20):
        try:
            with history_path.open("a", newline="", encoding="utf_8") as stream:
                csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writerow(row)
            return history_path
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))
    raise RuntimeError("history append retry loop did not return")


def write_summary_json(path: str | Path, summary: Mapping[str, Any]) -> Path:
    """Write one strict JSON summary with an atomic replacement."""
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = summary_path.with_name(f"{summary_path.name}.partial")
    partial_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf_8",
    )
    _replace_with_retry(partial_path, summary_path)
    return summary_path


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    value: Any = metadata
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def load_training_data(config: TrainingConfig) -> PairTrainingData:
    """Load one schema two pair dataset and select the root force target."""
    csv_path = config.csv_path.resolve()
    metadata_path = csv_path.with_suffix(".json")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    arrays = load_benzene_cluster_csv(csv_path, dtype=np.float64)
    metadata = load_benzene_cluster_metadata(metadata_path)
    if arrays.molecule_count != 2:
        raise ValueError("benzene pair training requires two molecules")
    if metadata.get("schema_version") != 2:
        raise ValueError("training requires schema_version two")
    if (
        config.dataset_revision is not None
        and metadata.get("dataset_revision") != config.dataset_revision
    ):
        raise ValueError(
            "training data revision does not match the configured dataset_revision"
        )
    if metadata.get("sample_count") != len(arrays):
        raise ValueError("metadata sample_count does not match the CSV")
    if metadata.get("molecule_count") != 2:
        raise ValueError("metadata molecule_count must equal two")

    csv_sha256 = sha256_file(csv_path)
    recorded_sha256 = metadata.get("csv_sha256")
    if recorded_sha256 is not None and recorded_sha256 != csv_sha256:
        raise ValueError("metadata csv_sha256 does not match the CSV")

    dtype = _resolve_dtype(config.dtype)
    device = torch.device(config.device)
    centers = torch.as_tensor(arrays.centers, dtype=dtype, device=device)
    frames = torch.as_tensor(arrays.rotations, dtype=dtype, device=device)
    root_force = torch.as_tensor(arrays.forces[:, 0], dtype=dtype, device=device)
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
        metadata=metadata,
        csv_path=csv_path,
        metadata_path=metadata_path,
        csv_sha256=csv_sha256,
        metadata_sha256=sha256_file(metadata_path),
    )


def _proper_d6_generators() -> Tensor:
    cosine = math.cos(math.pi / 3.0)
    sine = math.sin(math.pi / 3.0)
    sixfold = torch.tensor(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )
    twofold = torch.diag(torch.tensor((1.0, -1.0, -1.0), dtype=torch.float64))
    return torch.stack((sixfold, twofold))


def _build_model(config: TrainingConfig) -> tuple[PipelineModel, int]:
    generators = _proper_d6_generators()
    generator_names = ("sixfold", "twofold")
    if isinstance(config.pipeline, PairPipelineConfig):
        model = build_invariant_gate_pipeline(
            generators,
            config.pipeline,
            generator_names=generator_names,
        )
    else:
        model = build_invariant_gate_pipeline_v2(
            generators,
            config.pipeline,
            generator_names=generator_names,
        )
    model = model.to(
        device=torch.device(config.device),
        dtype=_resolve_dtype(config.dtype),
    )
    zeroed_head_count = 0
    if config.zero_output_heads:
        zero_method = getattr(model, "zero_output_heads", None)
        if zero_method is None:
            raise ValueError(
                "zero_output_heads is not supported by the configured pipeline"
            )
        zeroed_head_count = int(zero_method())
    return model, zeroed_head_count


def load_trained_model(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[PipelineModel, float, dict[str, Any]]:
    """Recompile fixed tensors and restore only learned checkpoint parameters."""
    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("checkpoint must use schema version two")
    if payload.get("fixed_tensor_artifacts_stored") is not False:
        raise ValueError("checkpoint must not store fixed tensor artifacts")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("checkpoint configuration is missing")
    config = TrainingConfig.from_dict(configuration)
    config = replace(
        config,
        device=device,
        dtype=dtype,
        zero_output_heads=False,
    )
    model, _count = _build_model(config)
    learned = payload.get("parameter_state_dict")
    if not isinstance(learned, Mapping):
        raise ValueError("checkpoint parameter state is missing")
    parameters = dict(model.named_parameters())
    if set(learned) != set(parameters):
        raise ValueError("checkpoint parameter names do not match the pipeline")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = learned[name]
            if not isinstance(value, Tensor) or value.shape != parameter.shape:
                raise ValueError(f"checkpoint parameter {name} has an invalid shape")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    target_scale = float(payload["target_scale"])
    return model, target_scale, payload


def _predict(
    model: nn.Module,
    data: PairTrainingData,
    indices: Tensor,
    batch_size: int,
) -> Tensor:
    was_training = model.training
    model.eval()
    predictions: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, int(indices.numel()), batch_size):
            selection = indices[start : start + batch_size].to(data.centers.device)
            prediction = model(data.centers[selection], data.frames[selection])
            expected_shape = (int(selection.numel()), 3)
            if prediction.shape != expected_shape:
                raise ValueError(
                    f"network output shape must be {expected_shape}, got "
                    f"{tuple(prediction.shape)}"
                )
            predictions.append(prediction.detach())
    model.train(was_training)
    return torch.cat(predictions, dim=0)


def _evaluate(
    model: nn.Module,
    data: PairTrainingData,
    indices: Tensor,
    batch_size: int,
    target_scale: Tensor,
) -> dict[str, float]:
    prediction = _predict(model, data, indices, batch_size)
    selection = indices.to(data.root_force.device)
    target = data.root_force[selection] / target_scale
    return regression_metrics(prediction, target, target_scale)


def _optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("network must have trainable parameters")
    optimizer_class = (
        torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
    )
    optimizer = optimizer_class(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )
    return optimizer, scheduler


def _save_checkpoint(
    path: Path,
    model: nn.Module,
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
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
        },
    }
    partial_path = path.with_name(f"{path.name}.partial")
    torch.save(payload, partial_path)
    _replace_with_retry(partial_path, path)
    return path


def _model_manifest(model: PipelineModel) -> list[dict[str, Any]]:
    """Return the version specific compiled path audit as plain JSON data."""
    if isinstance(model, InvariantGatePipeline):
        return list(model.gate_manifest)
    return list(model.candidate_manifest)


def _model_builder_name(model: PipelineModel) -> str:
    if isinstance(model, InvariantGatePipeline):
        return "build_invariant_gate_pipeline"
    return "build_invariant_gate_pipeline_v2"


def _matrix_residual(actual: Tensor, expected: Tensor) -> dict[str, float]:
    difference = actual - expected
    denominator = max(float(torch.linalg.vector_norm(expected)), 1.0e-12)
    return {
        "maximum_absolute": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference)) / denominator,
    }


def _fixed_rotation(dtype: torch.dtype, device: torch.device) -> Tensor:
    vector = torch.tensor((0.23, -0.31, 0.17), dtype=dtype, device=device)
    x, y, z = vector.unbind()
    zero = torch.zeros((), dtype=dtype, device=device)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    return torch.matrix_exp(skew)


def _benzene_group(dtype: torch.dtype, device: torch.device) -> Tensor:
    sixfold, twofold = _proper_d6_generators().to(dtype=dtype, device=device)
    identity = torch.eye(3, dtype=dtype, device=device)
    powers = [identity]
    for _ in range(5):
        powers.append(powers[-1] @ sixfold)
    return torch.stack(tuple(powers) + tuple(value @ twofold for value in powers))


def symmetry_metrics(
    model: nn.Module,
    centers: Tensor,
    frames: Tensor,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Measure translation, world rotation, and independent D6 residuals."""
    if centers.ndim != 3 or centers.shape[1:] != (2, 3):
        raise ValueError("centers must have shape batch by two by three")
    if frames.shape != centers.shape[:2] + (3, 3):
        raise ValueError("frames must match centers with rotation matrices")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        reference = model(centers, frames)
        translation = torch.tensor(
            (0.4, -0.7, 0.2),
            dtype=centers.dtype,
            device=centers.device,
        )
        translated = model(centers + translation, frames)

        rotation = _fixed_rotation(centers.dtype, centers.device)
        rotated_centers = centers @ rotation.T
        rotated_frames = torch.einsum("ij,bmjk->bmik", rotation, frames)
        rotated = model(rotated_centers, rotated_frames)
        expected_rotated = reference @ rotation.T

        group = _benzene_group(centers.dtype, centers.device)
        base_centers = centers[:1]
        base_frames = frames[:1]
        reference_local = model.forward_local(base_centers, base_frames)
        moved_frames = []
        expected_local = []
        for root_gauge in group:
            for sender_gauge in group:
                moved_frames.append(
                    torch.stack(
                        (
                            base_frames[0, 0] @ root_gauge,
                            base_frames[0, 1] @ sender_gauge,
                        )
                    )
                )
                expected_local.append(reference_local[0] @ root_gauge)
        gauge_frames = torch.stack(moved_frames)
        gauge_centers = base_centers.expand(len(moved_frames), 2, 3).clone()
        gauged_local = model.forward_local(gauge_centers, gauge_frames)
        gauged_world = model(gauge_centers, gauge_frames)
    model.train(was_training)

    translation_result = _matrix_residual(translated, reference)
    rotation_result = _matrix_residual(rotated, expected_rotated)
    local_gauge_result = _matrix_residual(
        gauged_local,
        torch.stack(expected_local),
    )
    world_gauge_result = _matrix_residual(
        gauged_world,
        reference[:1].expand_as(gauged_world),
    )
    passed = all(
        result["relative_l2"] <= tolerance
        for result in (
            translation_result,
            rotation_result,
            local_gauge_result,
            world_gauge_result,
        )
    )
    return {
        "tolerance": float(tolerance),
        "translation_invariance": translation_result,
        "world_rotation_covariance": rotation_result,
        "d6_gauge_combinations": len(moved_frames),
        "local_d6_covariance": local_gauge_result,
        "world_d6_gauge_invariance": world_gauge_result,
        "passed": passed,
    }


def _finite_state(model: nn.Module) -> dict[str, Any]:
    parameters = tuple(model.parameters())
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    gradients = tuple(parameter.grad for parameter in trainable)
    parameters_finite = all(
        bool(torch.isfinite(parameter).all()) for parameter in parameters
    )
    gradients_present = all(gradient is not None for gradient in gradients)
    gradients_finite = gradients_present and all(
        bool(torch.isfinite(gradient).all())
        for gradient in gradients
        if gradient is not None
    )
    gradient_norm = math.sqrt(
        sum(
            float(gradient.detach().to(torch.float64).square().sum())
            for gradient in gradients
            if gradient is not None
        )
    )
    return {
        "parameters_finite": parameters_finite,
        "gradients_present": gradients_present,
        "gradients_finite": gradients_finite,
        "gradient_l2": gradient_norm,
        "passed": parameters_finite and gradients_present and gradients_finite,
    }


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _git_provenance() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ("git", "-C", str(REPOSITORY_ROOT), *arguments),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "source_tree_dirty": bool(status) if status is not None else None,
    }


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tfenn": _distribution_version("TFENN"),
        "opls": _distribution_version("opls2020-static"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV")
        or Path(sys.prefix).name,
        "python_prefix": sys.prefix,
        "cuda_runtime": torch.version.cuda,
    }


def _prepare_output_directory(config: TrainingConfig) -> Path:
    output_directory = config.output_directory.resolve()
    expected = (
        output_directory / "history.csv",
        output_directory / "summary.json",
        output_directory / "best.pt",
        output_directory / "final.pt",
    )
    existing = [path for path in expected if path.exists()]
    if existing and not config.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"output artifacts already exist: {names}")
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def _accuracy_report(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    return {
        partition: {
            "r2": float(values["r2"]),
            "relative_rmse": float(values["relative_rmse"]),
        }
        for partition, values in metrics.items()
    }


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Run all requested epochs and write checkpoints plus structured logs."""
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    git_provenance = _git_provenance()
    runtime_versions = _runtime_versions()
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.model_seed)

    output_directory = _prepare_output_directory(config)
    history_path = output_directory / "history.csv"
    summary_path = output_directory / "summary.json"
    best_path = output_directory / "best.pt"
    final_path = output_directory / "final.pt"
    data = load_training_data(config)
    split = create_split(
        len(data.root_force),
        seed=config.split_seed,
        fractions=config.split_fractions,
    )
    model, zeroed_output_head_count = _build_model(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if not all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    ):
        raise RuntimeError("initial model parameters must be finite")

    train_selection = split.train.to(data.root_force.device)
    target_scale = torch.sqrt(data.root_force[train_selection].square().mean())
    if not bool(torch.isfinite(target_scale)) or float(target_scale) <= 0.0:
        raise RuntimeError("training target scale must be finite and positive")
    optimizer, scheduler = _optimizer_and_scheduler(model, config)
    shuffle_generator = torch.Generator().manual_seed(config.model_seed + 1)

    train_metrics = _evaluate(
        model,
        data,
        split.train,
        config.batch_size,
        target_scale,
    )
    validation_metrics = _evaluate(
        model,
        data,
        split.validation,
        config.batch_size,
        target_scale,
    )
    initial_train_normalized_mse = train_metrics["normalized_mse"]
    history: list[dict[str, float | int]] = [
        make_history_row(
            0,
            optimizer.param_groups[0]["lr"],
            train_metrics,
            validation_metrics,
        )
    ]
    write_history_csv(history_path, history)
    best_epoch = 0
    best_validation_metrics = validation_metrics
    _save_checkpoint(
        best_path,
        model,
        epoch=0,
        target_scale=target_scale,
        config=config,
        metrics=validation_metrics,
    )

    for epoch in range(1, config.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        order = split.train[
            torch.randperm(int(split.train.numel()), generator=shuffle_generator)
        ]
        model.train()
        for start in range(0, int(order.numel()), config.batch_size):
            selection = order[start : start + config.batch_size].to(data.centers.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(data.centers[selection], data.frames[selection])
            expected_shape = (int(selection.numel()), 3)
            if prediction.shape != expected_shape:
                raise ValueError(
                    f"network output shape must be {expected_shape}, got "
                    f"{tuple(prediction.shape)}"
                )
            target = data.root_force[selection] / target_scale
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"loss became nonfinite at epoch {epoch}")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and (
                    parameter.grad is None
                    or not bool(torch.isfinite(parameter.grad).all())
                ):
                    raise RuntimeError(f"gradient is missing or nonfinite for {name}")
            optimizer.step()
        scheduler.step()

        train_metrics = _evaluate(
            model,
            data,
            split.train,
            config.batch_size,
            target_scale,
        )
        validation_metrics = _evaluate(
            model,
            data,
            split.validation,
            config.batch_size,
            target_scale,
        )
        history_row = make_history_row(
            epoch,
            learning_rate,
            train_metrics,
            validation_metrics,
        )
        history.append(history_row)
        append_history_row(history_path, history_row)
        if epoch == 1 or epoch % config.progress_every == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_normalized_mse": train_metrics["normalized_mse"],
                        "validation_normalized_mse": validation_metrics[
                            "normalized_mse"
                        ],
                        "learning_rate": learning_rate,
                    }
                ),
                flush=True,
            )
        if (
            validation_metrics["normalized_mse"]
            < best_validation_metrics["normalized_mse"]
        ):
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            _save_checkpoint(
                best_path,
                model,
                epoch=epoch,
                target_scale=target_scale,
                config=config,
                metrics=validation_metrics,
            )

    final_metrics = {
        "train": _evaluate(
            model,
            data,
            split.train,
            config.batch_size,
            target_scale,
        ),
        "validation": _evaluate(
            model,
            data,
            split.validation,
            config.batch_size,
            target_scale,
        ),
        "test": _evaluate(
            model,
            data,
            split.test,
            config.batch_size,
            target_scale,
        ),
    }
    probe_count = min(8, int(split.test.numel()))
    probe_indices = split.test[:probe_count].to(data.centers.device)
    final_symmetry = symmetry_metrics(
        model,
        data.centers[probe_indices],
        data.frames[probe_indices],
        tolerance=config.symmetry_tolerance,
    )
    finite_state = _finite_state(model)
    _save_checkpoint(
        final_path,
        model,
        epoch=config.epochs,
        target_scale=target_scale,
        config=config,
        metrics=final_metrics["validation"],
    )

    final_train_normalized_mse = final_metrics["train"]["normalized_mse"]
    required_train_normalized_mse = (
        config.maximum_train_loss_ratio * initial_train_normalized_mse
    )
    accuracy_passed = final_train_normalized_mse <= required_train_normalized_mse
    failures: list[str] = []
    if not accuracy_passed:
        failures.append("final train normalized MSE exceeds the configured ratio")
    if not finite_state["passed"]:
        failures.append("final parameters or gradients are not finite")
    if not final_symmetry["passed"]:
        failures.append("final symmetry residual exceeds tolerance")

    finished_at = datetime.now(timezone.utc)
    history_sha256 = sha256_file(history_path)
    best_sha256 = sha256_file(best_path)
    final_sha256 = sha256_file(final_path)
    opls_model = _metadata_value(data.metadata, "opls", "model")
    summary: dict[str, Any] = {
        "schema_name": "tfenn_benzene_pair_training_run",
        "schema_version": 1,
        "experiment_id": (
            f"{config.pipeline.architecture_id}_opls_2_0_0_"
            f"revision{data.metadata.get('dataset_revision')}"
        ),
        "status": "complete" if not failures else "failed_validation",
        "target": {
            "definition": TARGET_DEFINITION,
            "array_selection": "forces[:, 0, :]",
            "molecule_id": 0,
            "uses_moment": False,
            "normalization": "scalar RMS fitted from training targets only",
            "target_scale": float(target_scale.detach().cpu()),
        },
        "data": {
            "csv_path": str(data.csv_path),
            "metadata_path": str(data.metadata_path),
            "csv_sha256": data.csv_sha256,
            "metadata_sha256": data.metadata_sha256,
            "sample_count": len(data.root_force),
            "molecule_count": 2,
            "schema_version": data.metadata.get("schema_version"),
            "dataset_revision": data.metadata.get("dataset_revision"),
            "opls_runtime_version": _metadata_value(
                data.metadata,
                "opls",
                "runtime_version",
            ),
            "opls_distribution_version": _metadata_value(
                data.metadata,
                "opls",
                "distribution_version",
            ),
            "opls_model_semantics_id": (
                opls_model.get("model_semantics_id")
                if isinstance(opls_model, Mapping)
                else None
            ),
        },
        "split": {
            "seed": config.split_seed,
            "fractions": list(config.split_fractions),
            **split.as_json(),
        },
        "runtime": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": time.perf_counter() - started_clock,
            "versions": runtime_versions,
            "git": git_provenance,
            "device": config.device,
            "dtype": config.dtype,
            "threads": config.threads,
        },
        "model": {
            "name": config.pipeline.architecture_id,
            "pipeline_version": config.pipeline_version,
            "builder": _model_builder_name(model),
            "class": type(model).__name__,
            "pipeline": config.pipeline.as_dict(),
            "compiled_path_manifest": _model_manifest(model),
            "gate_manifest": (
                list(model.gate_manifest)
                if isinstance(model, InvariantGatePipeline)
                else []
            ),
            "candidate_manifest": (
                list(model.candidate_manifest)
                if isinstance(model, InvariantGatePipelineV2)
                else []
            ),
            "offline_compilation": model.offline_compilation_summary,
            "checkpoint_content": "learned_parameters_only",
            "proper_d6_generator_names": ["sixfold", "twofold"],
            "proper_d6_generator_dtype": "float64",
            "zeroed_output_head_count": zeroed_output_head_count,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
        },
        "training": {
            "epochs_requested": config.epochs,
            "epochs_completed": config.epochs,
            "history_includes_epoch_zero_baseline": True,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "scheduler": "StepLR",
            "scheduler_step_size": config.scheduler_step_size,
            "scheduler_gamma": config.scheduler_gamma,
            "no_early_stopping": True,
        },
        "configuration": config.as_json(),
        "history": {
            "path": str(history_path),
            "sha256": history_sha256,
            "row_count": len(history),
            "first_epoch": 0,
            "last_epoch": config.epochs,
        },
        "best": {
            "epoch": best_epoch,
            "validation": best_validation_metrics,
            "checkpoint_path": str(best_path),
            "checkpoint_sha256": best_sha256,
        },
        "final": {
            "metrics": final_metrics,
            "accuracy": _accuracy_report(final_metrics),
            "symmetry": final_symmetry,
            "finite_state": finite_state,
            "checkpoint_path": str(final_path),
            "checkpoint_sha256": final_sha256,
        },
        "accuracy_requirement": {
            "initial_train_normalized_mse": initial_train_normalized_mse,
            "maximum_train_loss_ratio": config.maximum_train_loss_ratio,
            "maximum_final_train_normalized_mse": required_train_normalized_mse,
            "actual_final_train_normalized_mse": final_train_normalized_mse,
            "passed": accuracy_passed,
            "reported_by": ("r2", "relative_rmse"),
        },
        "failures": failures,
    }
    write_summary_json(summary_path, summary)
    if failures:
        raise RuntimeError("; ".join(failures))
    return summary


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load one complete experiment definition from JSON."""
    config_path = Path(path).resolve()
    value = json.loads(config_path.read_text(encoding="utf_8"))
    if not isinstance(value, dict):
        raise ValueError("training config JSON must contain one object")
    return TrainingConfig.from_dict(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_v2.json",
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run training from command line arguments."""
    parsed = build_parser().parse_args(arguments)
    config = load_training_config(parsed.config)
    changes: dict[str, Any] = {}
    if parsed.output_directory is not None:
        changes["output_directory"] = parsed.output_directory
    if parsed.epochs is not None:
        changes["epochs"] = parsed.epochs
    if parsed.overwrite:
        changes["overwrite"] = True
    if changes:
        config = replace(config, **changes)
    summary = run_training(config)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary_path": str(config.output_directory.resolve() / "summary.json"),
                "history_path": str(config.output_directory.resolve() / "history.csv"),
                "accuracy": summary["final"]["accuracy"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
