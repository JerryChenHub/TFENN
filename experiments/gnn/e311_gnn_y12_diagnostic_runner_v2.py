"""Deterministic step based runner for the Y diagnostic study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from experiments.gnn.e311_gnn_12_experiment_core_v1 import SupervisionV1
from experiments.gnn.e311_gnn_12_experiment_runner_v1 import (
    FIVE_BENZENE_CSV,
    TWO_BENZENE_CSV,
    GraphBucketV1,
    SplitIndicesV1,
    _base_bucket,
    _load_five_pair_forces,
    _validate_cluster_files,
    deterministic_group_split_v1,
    scatter_pair_forces_v1,
)
from experiments.gnn.e311_gnn_y12_diagnostic_core_v2 import (
    CURRENT_X01_STEPS,
    LEGACY_5K_V3_STEPS,
    E311YDiagnosticCoreV2,
    assert_one_layer_stack_preflight_v2,
    build_y_diagnostic_core_v2,
    get_y_experiment_spec_v2,
)


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parents[1]
DEFAULT_OUTPUT_ROOT = MODULE_DIRECTORY / "runs" / "e311_gnn_y12_fast_v2"
LEGACY_PAIR_CSV = (
    MODULE_DIRECTORY.parent
    / "benzene_pair"
    / "data"
    / "benzene_pair_opls_2_0_0_v3.csv"
)
DEFAULT_SPLIT_SEED = 20260824
DEFAULT_MODEL_SEED = 20260824
DEFAULT_SHUFFLE_SEED = 20260824
DEFAULT_COMET_PROJECT = "tfenn_e311_gnn_y12_diagnostic_v2"
RUNNER_VARIANT = "fast_v2"
PSTAR_SCHEMA_NAME = "tfenn_e311_gnn_y12_pstar_selection_fast_v2"
CHECKPOINT_SCHEMA_NAME = "tfenn_e311_gnn_y12_checkpoint_fast_v2"
RESULT_SCHEMA_NAME = "tfenn_e311_gnn_y12_result_fast_v2"
SAMPLER_SCHEMA_NAME = "tfenn_stateful_batch_sampler_fast_v2"
RECORD_EVERY_EVALUATIONS = 10
GRADIENT_CHECK_EVERY_STEPS = 100


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf_8")
    return hashlib.sha256(payload).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(
            _json_ready(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf_8",
    )
    partial.replace(path)


HISTORY_FIELDS_V2 = (
    "global_step",
    "evaluation_index",
    "train_loss",
    "train_loss_source",
    "validation_loss",
    "epoch_duration_seconds",
    "learning_rate",
    "sampler_cycle",
    "recorded_at_utc",
)


def _write_history_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("history must contain at least one row")
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf_8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS_V2)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def _read_history(path: Path, maximum_step: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf_8") as stream:
        for row in csv.DictReader(stream):
            step = int(row["global_step"])
            if step > maximum_step:
                continue
            result.append(
                {
                    "global_step": step,
                    "evaluation_index": int(row["evaluation_index"]),
                    "train_loss": float(row["train_loss"]),
                    "train_loss_source": row["train_loss_source"],
                    "validation_loss": float(row["validation_loss"]),
                    "epoch_duration_seconds": (
                        None
                        if row["epoch_duration_seconds"] == ""
                        else float(row["epoch_duration_seconds"])
                    ),
                    "learning_rate": float(row["learning_rate"]),
                    "sampler_cycle": int(row["sampler_cycle"]),
                    "recorded_at_utc": row["recorded_at_utc"],
                }
            )
    return result


@dataclass(frozen=True, slots=True)
class OptimizerProtocolV2:
    name: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    scheduler: str
    evaluation_cadence_steps: int
    scheduler_step_updates: int | None = None
    scheduler_gamma: float | None = None
    scheduler_factor: float | None = None
    scheduler_patience: int | None = None
    minimum_learning_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("protocol name must not be empty")
        if self.optimizer != "AdamW":
            raise ValueError("the Y study requires AdamW")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer values are invalid")
        if self.batch_size < 1 or self.evaluation_cadence_steps < 1:
            raise ValueError("batch size and cadence must be positive")


P0_CURRENT_X01_PROTOCOL = OptimizerProtocolV2(
    name="current_x01_protocol",
    optimizer="AdamW",
    learning_rate=0.002,
    weight_decay=0.000001,
    batch_size=100,
    scheduler="ReduceLROnPlateau",
    evaluation_cadence_steps=16,
    scheduler_factor=0.5,
    scheduler_patience=20,
    minimum_learning_rate=0.00001,
)


P1_LEGACY_5K_V3_PROTOCOL = OptimizerProtocolV2(
    name="legacy_5k_v3_protocol",
    optimizer="AdamW",
    learning_rate=0.005,
    weight_decay=0.0001,
    batch_size=64,
    scheduler="StepLR",
    evaluation_cadence_steps=63,
    scheduler_step_updates=6300,
    scheduler_gamma=0.5,
)


PROTOCOLS_BY_NAME_V2: Mapping[str, OptimizerProtocolV2] = {
    P0_CURRENT_X01_PROTOCOL.name: P0_CURRENT_X01_PROTOCOL,
    P1_LEGACY_5K_V3_PROTOCOL.name: P1_LEGACY_5K_V3_PROTOCOL,
}


@dataclass(frozen=True, slots=True)
class PreparedYDataV2:
    experiment_id: str
    dataset_id: str
    supervision: SupervisionV1
    train: GraphBucketV1
    validation: GraphBucketV1
    test: GraphBucketV1
    force_scale: float
    split: SplitIndicesV1
    records: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PhysicalMetricsV2:
    final_test_mae: float
    final_test_sae: float
    residual_frobenius_norm: float
    target_frobenius_norm: float
    component_count: int


@dataclass(frozen=True, slots=True)
class YRunnerConfigV2:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    split_seed: int = DEFAULT_SPLIT_SEED
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED
    device: str = "cuda"
    comet_project: str = DEFAULT_COMET_PROJECT
    comet_workspace: str | None = None
    comet_enabled: bool = True
    pstar_selection_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.pstar_selection_path is not None:
            object.__setattr__(
                self,
                "pstar_selection_path",
                Path(self.pstar_selection_path),
            )
        if not self.comet_project.strip() and self.comet_enabled:
            raise ValueError("enabled Comet logging requires a project")


@dataclass(frozen=True, slots=True)
class TrainStepResultV2:
    loss: Tensor
    component_count: int
    loss_finite: Tensor


class ProtocolEpochLossAccumulatorV2:
    def __init__(self, device: torch.device) -> None:
        self.weighted_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.loss_finite = torch.ones((), dtype=torch.bool, device=device)
        self.component_count = 0

    def add(self, result: TrainStepResultV2) -> None:
        self.weighted_loss_sum.add_(
            result.loss.to(dtype=torch.float64) * result.component_count
        )
        self.loss_finite.logical_and_(result.loss_finite)
        self.component_count += result.component_count

    def finish(self) -> float:
        if self.component_count < 1:
            raise ValueError("protocol epoch contains no loss components")
        packed = torch.stack(
            (
                self.weighted_loss_sum,
                self.loss_finite.to(dtype=torch.float64),
            )
        ).detach().cpu()
        if float(packed[1]) != 1.0:
            raise RuntimeError("training loss became nonfinite")
        result = float(packed[0]) / self.component_count
        if not math.isfinite(result):
            raise RuntimeError("training loss became nonfinite")
        return result


def should_check_gradients_v2(global_step: int) -> bool:
    if global_step < 1:
        return False
    return global_step <= 4 or global_step % GRADIENT_CHECK_EVERY_STEPS == 0


def should_persist_evaluation_v2(
    evaluation_index: int,
    global_step: int,
    target_steps: int,
) -> bool:
    if evaluation_index < 1:
        raise ValueError("evaluation index must be positive")
    return (
        evaluation_index == 1
        or evaluation_index % RECORD_EVERY_EVALUATIONS == 0
        or global_step == target_steps
    )


class StatefulBatchSamplerV2:
    """Own the exact shuffled batch continuation state."""

    def __init__(self, sample_count: int, batch_size: int, seed: int) -> None:
        if sample_count < 1 or batch_size < 1:
            raise ValueError("sampler sizes must be positive")
        self.sample_count = int(sample_count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.permutation = torch.empty((0,), dtype=torch.int64)
        self.position = 0
        self.cycle = 0
        self.last_base_seed: int | None = None

    def _begin_cycle(self) -> None:
        self.last_base_seed = int(
            torch.empty((), dtype=torch.int64)
            .random_(generator=self.generator)
            .item()
        )
        self.permutation = torch.randperm(
            self.sample_count,
            generator=self.generator,
        )
        self.position = 0
        self.cycle += 1

    def next_indices(self) -> Tensor:
        if self.position >= int(self.permutation.numel()):
            self._begin_cycle()
        end = min(self.position + self.batch_size, self.sample_count)
        result = self.permutation[self.position:end].clone()
        self.position = end
        if self.position == self.sample_count:
            torch.randperm(self.sample_count, generator=self.generator)
        return result

    def consumed_batch_count(self) -> int:
        if self.cycle == 0:
            return 0
        batches_per_cycle = math.ceil(self.sample_count / self.batch_size)
        current_cycle_batches = math.ceil(self.position / self.batch_size)
        return (self.cycle - 1) * batches_per_cycle + current_cycle_batches

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_name": SAMPLER_SCHEMA_NAME,
            "sample_count": self.sample_count,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "generator_state": self.generator.get_state().clone(),
            "permutation": self.permutation.clone(),
            "position": self.position,
            "cycle": self.cycle,
            "last_base_seed": self.last_base_seed,
            "consumed_batch_count": self.consumed_batch_count(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_name") != SAMPLER_SCHEMA_NAME:
            raise ValueError("sampler schema is invalid")
        expected = (self.sample_count, self.batch_size, self.seed)
        actual = (
            int(state.get("sample_count", -1)),
            int(state.get("batch_size", -1)),
            int(state.get("seed", -1)),
        )
        if actual != expected:
            raise ValueError("sampler identity does not match the checkpoint")
        permutation = state.get("permutation")
        generator_state = state.get("generator_state")
        if not isinstance(permutation, Tensor) or permutation.dtype != torch.int64:
            raise TypeError("sampler permutation is invalid")
        if not isinstance(generator_state, Tensor):
            raise TypeError("sampler generator state is invalid")
        cycle = int(state.get("cycle", -1))
        if cycle < 0:
            raise ValueError("sampler cycle is invalid")
        position = int(state.get("position", -1))
        if cycle == 0:
            if permutation.numel() != 0 or position != 0:
                raise ValueError("initial sampler state is invalid")
            if state.get("last_base_seed") is not None:
                raise ValueError("initial sampler base seed is invalid")
        else:
            if permutation.numel() != self.sample_count:
                raise ValueError("sampler permutation length is invalid")
            if position < 1 or position > self.sample_count:
                raise ValueError("sampler position is invalid")
            if position != self.sample_count and position % self.batch_size != 0:
                raise ValueError("sampler position is not a batch boundary")
            if bool((permutation < 0).any()) or bool((permutation >= self.sample_count).any()):
                raise ValueError("sampler permutation range is invalid")
            if int(torch.unique(permutation).numel()) != self.sample_count:
                raise ValueError("sampler permutation is not unique")
            base_seed = state.get("last_base_seed")
            if not isinstance(base_seed, int) or base_seed < 0:
                raise ValueError("sampler base seed is invalid")
        if position < 0 or position > int(permutation.numel()):
            raise ValueError("sampler position is invalid")
        self.permutation = permutation.cpu().clone()
        self.position = position
        self.cycle = cycle
        base_seed = state.get("last_base_seed")
        self.last_base_seed = None if base_seed is None else int(base_seed)
        self.generator.set_state(generator_state.cpu())
        if int(state.get("consumed_batch_count", -1)) != self.consumed_batch_count():
            raise ValueError("sampler consumed batch count is invalid")


@dataclass(frozen=True, slots=True)
class TensorBucketV2:
    centers: Tensor
    frames: Tensor
    normalized_target: Tensor
    physical_node_target: Tensor
    pair_index: Tensor


def y_supervision_v2(experiment_id: str) -> SupervisionV1:
    normalized = get_y_experiment_spec_v2(experiment_id).experiment_id
    if normalized in {f"Y{index:02d}" for index in range(1, 6)}:
        return SupervisionV1.PAIR_FORCE
    return SupervisionV1.NODE_FORCE


def _dataset_id_v2(experiment_id: str) -> str:
    normalized = get_y_experiment_spec_v2(experiment_id).experiment_id
    if normalized == "Y04":
        return "pair_5k_legacy_v3"
    if normalized in {f"Y{index:02d}" for index in range(1, 6)}:
        return "pair_2k_current"
    return "five_benzene_1k"


def _force_scale(bucket: GraphBucketV1, supervision: SupervisionV1) -> float:
    if supervision is SupervisionV1.PAIR_FORCE:
        if bucket.pair_force_world is None:
            raise ValueError("pair labels are required")
        target = bucket.pair_force_world
    else:
        target = bucket.node_force_world
    result = float(np.sqrt(np.square(target).mean()))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("force scale must be finite and positive")
    return result


def prepare_y_experiment_data_v2(
    experiment_id: str,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> PreparedYDataV2:
    spec = get_y_experiment_spec_v2(experiment_id)
    supervision = y_supervision_v2(spec.experiment_id)
    dataset_id = _dataset_id_v2(spec.experiment_id)
    records: dict[str, Any] = {
        "dataset_id": dataset_id,
        "split_seed": split_seed,
        "supervision": supervision.value,
    }

    if dataset_id in {"pair_2k_current", "pair_5k_legacy_v3"}:
        csv_path = TWO_BENZENE_CSV if dataset_id == "pair_2k_current" else LEGACY_PAIR_CSV
        arrays, file_record = _validate_cluster_files(csv_path)
        pair_index = np.asarray(((0,), (1,)), dtype=np.int64)
        pair_force = np.ascontiguousarray(arrays.forces[:, 0, None, :])
        bucket = _base_bucket("N2", arrays, pair_index, pair_force)
        records[dataset_id] = file_record
    else:
        arrays, file_record = _validate_cluster_files(FIVE_BENZENE_CSV)
        pair_index, pair_force, pair_record = _load_five_pair_forces()
        bucket = _base_bucket("N5", arrays, pair_index, pair_force)
        reconstructed = scatter_pair_forces_v1(pair_force, pair_index, 5)
        maximum_error = float(np.max(np.abs(reconstructed - bucket.node_force_world)))
        if maximum_error > 0.000000001:
            raise ValueError("five benzene pair labels do not reconstruct node labels")
        records[dataset_id] = file_record
        records["five_benzene_pair"] = pair_record
        records["pair_scatter_maximum_absolute_error"] = maximum_error

    split = deterministic_group_split_v1(bucket.group_id, split_seed)
    train = bucket.subset(split.train)
    validation = bucket.subset(split.validation)
    test = bucket.subset(split.test)
    scale = _force_scale(train, supervision)
    records["split"] = {
        "train_count": train.sample_count,
        "validation_count": validation.sample_count,
        "test_count": test.sample_count,
        "train_indices_sha256": _array_sha256(split.train),
        "validation_indices_sha256": _array_sha256(split.validation),
        "test_indices_sha256": _array_sha256(split.test),
    }
    records["force_scale"] = scale
    return PreparedYDataV2(
        spec.experiment_id,
        dataset_id,
        supervision,
        train,
        validation,
        test,
        scale,
        split,
        records,
    )


def _tensor_bucket(bucket: GraphBucketV1, data: PreparedYDataV2) -> TensorBucketV2:
    if data.supervision is SupervisionV1.PAIR_FORCE:
        if bucket.pair_force_world is None:
            raise ValueError("pair labels are required")
        target = bucket.pair_force_world
    else:
        target = bucket.node_force_world
    return TensorBucketV2(
        centers=torch.from_numpy(np.ascontiguousarray(bucket.centers_world, dtype=np.float32)),
        frames=torch.from_numpy(
            np.ascontiguousarray(bucket.frames_body_to_world, dtype=np.float32)
        ),
        normalized_target=torch.from_numpy(
            np.ascontiguousarray(target / data.force_scale, dtype=np.float32)
        ),
        physical_node_target=torch.from_numpy(
            np.ascontiguousarray(bucket.node_force_world, dtype=np.float32)
        ),
        pair_index=torch.from_numpy(bucket.pair_index.copy()),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_model(
    experiment_id: str,
    force_scale: float,
    model_seed: int,
    device: torch.device,
) -> E311YDiagnosticCoreV2:
    _set_seed(model_seed)
    model = build_y_diagnostic_core_v2(experiment_id, force_scale, torch.float32)
    with torch.no_grad():
        for block in model.message_blocks:
            block.pair_kernel._runtime_reference.zero_()
    return model.to(device)


def _build_optimizer_and_scheduler(
    model: nn.Module,
    protocol: OptimizerProtocolV2,
) -> tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    if protocol.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(protocol.scheduler_factor),
            patience=int(protocol.scheduler_patience),
            min_lr=float(protocol.minimum_learning_rate),
        )
    elif protocol.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(protocol.scheduler_step_updates),
            gamma=float(protocol.scheduler_gamma),
        )
    else:
        raise ValueError("unknown scheduler")
    return optimizer, scheduler


def _normalized_prediction(
    model: E311YDiagnosticCoreV2,
    centers: Tensor,
    frames: Tensor,
    pair_index: Tensor,
    supervision: SupervisionV1,
) -> Tensor:
    output = model.core_output(centers, frames, pair_index)
    if supervision is SupervisionV1.PAIR_FORCE:
        return output.normalized_pair_force_world
    return output.normalized_node_force_world


def _train_step(
    model: E311YDiagnosticCoreV2,
    tensor_data: TensorBucketV2,
    selection: Tensor,
    supervision: SupervisionV1,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_step: int,
) -> TrainStepResultV2:
    model.train()
    centers = tensor_data.centers[selection].to(device)
    frames = tensor_data.frames[selection].to(device)
    target = tensor_data.normalized_target[selection].to(device)
    pair_index = tensor_data.pair_index.to(device)
    optimizer.zero_grad(set_to_none=True)
    difference = _normalized_prediction(
        model,
        centers,
        frames,
        pair_index,
        supervision,
    ) - target
    loss = difference.square().mean()
    loss_finite = torch.isfinite(loss.detach())
    loss.backward()
    if should_check_gradients_v2(global_step):
        gradient_finite = loss_finite.clone()
        gradient_seen = False
        for parameter in model.parameters():
            if parameter.grad is not None:
                gradient_finite.logical_and_(torch.isfinite(parameter.grad).all())
                gradient_seen = True
        if not gradient_seen or not bool(gradient_finite.detach().cpu()):
            raise RuntimeError("a gradient became nonfinite")
    optimizer.step()
    return TrainStepResultV2(
        loss=loss.detach(),
        component_count=difference.numel(),
        loss_finite=loss_finite,
    )


def _evaluate_loss(
    model: E311YDiagnosticCoreV2,
    tensor_data: TensorBucketV2,
    supervision: SupervisionV1,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    squared_sum = torch.zeros((), dtype=torch.float64, device=device)
    component_count = 0
    pair_index = tensor_data.pair_index.to(device)
    with torch.inference_mode():
        for start in range(0, len(tensor_data.centers), batch_size):
            end = min(start + batch_size, len(tensor_data.centers))
            centers = tensor_data.centers[start:end].to(device)
            frames = tensor_data.frames[start:end].to(device)
            target = tensor_data.normalized_target[start:end].to(device)
            difference = _normalized_prediction(
                model,
                centers,
                frames,
                pair_index,
                supervision,
            ) - target
            squared_sum.add_(difference.to(dtype=torch.float64).square().sum())
            component_count += difference.numel()
    if component_count == 0:
        raise ValueError("evaluation data is empty")
    result = (squared_sum / component_count).detach().cpu()
    value = float(result)
    if not math.isfinite(value):
        raise RuntimeError("evaluation loss became nonfinite")
    return value


def _physical_metrics(
    model: E311YDiagnosticCoreV2,
    tensor_data: TensorBucketV2,
    force_scale: float,
    batch_size: int,
    device: torch.device,
) -> PhysicalMetricsV2:
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    pair_index = tensor_data.pair_index.to(device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensor_data.centers), batch_size):
            end = min(start + batch_size, len(tensor_data.centers))
            centers = tensor_data.centers[start:end].to(device)
            frames = tensor_data.frames[start:end].to(device)
            output = model.core_output(centers, frames, pair_index)
            predictions.append(
                (output.normalized_node_force_world * force_scale).detach().cpu()
            )
            targets.append(tensor_data.physical_node_target[start:end])
    prediction = torch.cat(predictions).double()
    target = torch.cat(targets).double()
    difference = prediction - target
    return PhysicalMetricsV2(
        final_test_mae=float(difference.abs().mean()),
        final_test_sae=float(difference.abs().sum()),
        residual_frobenius_norm=float(difference.square().sum().sqrt()),
        target_frobenius_norm=float(target.square().sum().sqrt()),
        component_count=difference.numel(),
    )


def _finite_optimizer_smoke_v2(
    experiment_id: str,
    data: PreparedYDataV2,
    device: torch.device,
) -> dict[str, Any]:
    model = _build_model(experiment_id, data.force_scale, DEFAULT_MODEL_SEED, device)
    tensor_data = _tensor_bucket(data.train, data)
    optimizer, _ = _build_optimizer_and_scheduler(model, P1_LEGACY_5K_V3_PROTOCOL)
    sample_count = min(2, data.train.sample_count)
    selection = torch.arange(sample_count, dtype=torch.int64)
    centers = tensor_data.centers[selection].to(device)
    frames = tensor_data.frames[selection].to(device)
    target = tensor_data.normalized_target[selection].to(device)
    pair_index = tensor_data.pair_index.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = _normalized_prediction(
        model,
        centers,
        frames,
        pair_index,
        data.supervision,
    )
    loss = (prediction - target).square().mean()
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(loss)):
        raise RuntimeError(f"{experiment_id} preflight forward became nonfinite")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
        raise RuntimeError(f"{experiment_id} preflight backward became nonfinite")
    optimizer.step()
    _synchronize(device)
    floating_state = [
        value
        for value in model.state_dict().values()
        if isinstance(value, Tensor) and (value.is_floating_point() or value.is_complex())
    ]
    if any(not bool(torch.isfinite(value).all()) for value in floating_state):
        raise RuntimeError(f"{experiment_id} preflight optimizer step became nonfinite")
    optimizer_steps = [
        float(state["step"].detach().cpu())
        if isinstance(state.get("step"), Tensor)
        else float(state["step"])
        for state in optimizer.state.values()
        if "step" in state
    ]
    if not optimizer_steps or any(value != 1.0 for value in optimizer_steps):
        raise RuntimeError(f"{experiment_id} preflight optimizer did not advance once")
    architecture = dict(model.architecture_record())
    return {
        "experiment_id": experiment_id,
        "device": str(device),
        "cuda_executed": device.type == "cuda",
        "sample_count": sample_count,
        "layer_count": int(architecture["layer_count"]),
        "ema_layers": architecture["ema_layers"],
        "running_rms_policy_by_layer": architecture["running_rms_policy_by_layer"],
        "loss": float(loss.detach().cpu()),
        "prediction_maximum_absolute_value": float(prediction.detach().abs().max().cpu()),
        "gradient_parameter_count": len(gradients),
        "optimizer_state_count": len(optimizer_steps),
        "optimizer_step_minimum": min(optimizer_steps),
        "optimizer_step_maximum": max(optimizer_steps),
        "forward_finite": True,
        "backward_finite": True,
        "optimizer_state_finite": True,
    }


def _run_directory(config: YRunnerConfigV2, experiment_id: str, seed: int) -> Path:
    return config.output_root.resolve() / experiment_id / f"seed_{seed}"


def _default_pstar_path(config: YRunnerConfigV2, seed: int) -> Path:
    if config.pstar_selection_path is not None:
        return config.pstar_selection_path.resolve()
    return config.output_root.resolve() / f"pstar_seed_{seed}.json"


def _summary_path(config: YRunnerConfigV2, experiment_id: str, seed: int) -> Path:
    return _run_directory(config, experiment_id, seed) / "summary.json"


def _required_sha256(value: Any, field_name: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} is not a SHA256 digest")
    return normalized


def _pstar_candidate_evidence(
    summary_path: Path,
    experiment_id: str,
    model_seed: int,
    protocol: OptimizerProtocolV2,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf_8"))
    if summary.get("schema_name") != RESULT_SCHEMA_NAME:
        raise ValueError(f"{experiment_id} result schema is not fast v2")
    if summary.get("runner_variant") != RUNNER_VARIANT:
        raise ValueError(f"{experiment_id} runner variant is not fast v2")
    if summary.get("status") != "complete":
        raise ValueError(f"{experiment_id} is not complete")
    experiment = summary.get("experiment", {})
    if experiment.get("experiment_id") != experiment_id:
        raise ValueError(f"{experiment_id} summary identity changed")
    if experiment.get("dataset") != "pair_2k_current":
        raise ValueError(f"{experiment_id} experiment dataset is not current pair 2k")
    if int(experiment.get("optimizer_steps", -1)) != LEGACY_5K_V3_STEPS:
        raise ValueError(f"{experiment_id} experiment step contract changed")
    if int(summary.get("model_seed", -1)) != model_seed:
        raise ValueError(f"{experiment_id} model seed changed")

    training = summary.get("training", {})
    if int(training.get("target_steps", -1)) != LEGACY_5K_V3_STEPS:
        raise ValueError(f"{experiment_id} target steps are not 31500")
    if int(training.get("global_steps_completed", -1)) != LEGACY_5K_V3_STEPS:
        raise ValueError(f"{experiment_id} did not complete exactly 31500 updates")
    if training.get("resolved_protocol") != asdict(protocol):
        raise ValueError(f"{experiment_id} optimizer protocol changed")
    validation_loss = float(training["final_validation_loss"])
    if not math.isfinite(validation_loss) or validation_loss < 0.0:
        raise ValueError("Pstar validation loss is invalid")

    data = summary.get("data", {})
    if data.get("dataset_id") != "pair_2k_current":
        raise ValueError(f"{experiment_id} data is not current pair 2k")
    if data.get("supervision") != SupervisionV1.PAIR_FORCE.value:
        raise ValueError(f"{experiment_id} does not use pair supervision")
    if int(data.get("split_seed", -1)) != int(summary.get("split_seed", -2)):
        raise ValueError(f"{experiment_id} split seed record is inconsistent")
    split = data.get("split", {})
    split_hashes = {
        name: _required_sha256(split.get(name), f"{experiment_id}.{name}")
        for name in (
            "train_indices_sha256",
            "validation_indices_sha256",
            "test_indices_sha256",
        )
    }
    dataset = data.get("pair_2k_current", {})
    dataset_hashes = {
        name: _required_sha256(dataset.get(name), f"{experiment_id}.{name}")
        for name in ("csv_sha256", "metadata_sha256", "validation_sha256")
    }
    force_scale = float(data["force_scale"])
    if not math.isfinite(force_scale) or force_scale <= 0.0:
        raise ValueError(f"{experiment_id} force scale is invalid")

    architecture = summary.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError(f"{experiment_id} architecture record is missing")
    architecture_fingerprint = _json_sha256(architecture)
    if summary.get("architecture_fingerprint_sha256") != architecture_fingerprint:
        raise ValueError(f"{experiment_id} architecture fingerprint changed")

    return {
        "experiment_id": experiment_id,
        "model_seed": model_seed,
        "protocol_name": protocol.name,
        "protocol_parameters": asdict(protocol),
        "target_steps": LEGACY_5K_V3_STEPS,
        "global_steps_completed": LEGACY_5K_V3_STEPS,
        "final_validation_loss": validation_loss,
        "dataset_id": "pair_2k_current",
        "supervision": SupervisionV1.PAIR_FORCE.value,
        "split_seed": int(summary["split_seed"]),
        "shuffle_seed": int(summary["shuffle_seed"]),
        "split_hashes": split_hashes,
        "dataset_hashes": dataset_hashes,
        "force_scale": force_scale,
        "architecture_fingerprint_sha256": architecture_fingerprint,
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
    }


def _pstar_common_evidence(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(candidates) != 2:
        raise ValueError("Pstar requires exactly two candidates")
    common_names = (
        "model_seed",
        "target_steps",
        "global_steps_completed",
        "dataset_id",
        "supervision",
        "split_seed",
        "shuffle_seed",
        "split_hashes",
        "dataset_hashes",
        "force_scale",
        "architecture_fingerprint_sha256",
    )
    first = candidates[0]
    second = candidates[1]
    result: dict[str, Any] = {}
    for name in common_names:
        if first.get(name) != second.get(name):
            raise ValueError(f"Pstar candidate common fact changed: {name}")
        result[name] = first[name]
    return result


def select_pstar_protocol_v2(
    config: YRunnerConfigV2,
    model_seed: int = DEFAULT_MODEL_SEED,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    path = _default_pstar_path(config, model_seed) if output_path is None else Path(output_path)
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    candidates: list[dict[str, Any]] = []
    expected = {
        "Y02": P0_CURRENT_X01_PROTOCOL,
        "Y03": P1_LEGACY_5K_V3_PROTOCOL,
    }
    for experiment_id in ("Y02", "Y03"):
        summary_path = _summary_path(config, experiment_id, model_seed)
        candidates.append(
            _pstar_candidate_evidence(
                summary_path,
                experiment_id,
                model_seed,
                expected[experiment_id],
            )
        )
    common_evidence = _pstar_common_evidence(candidates)
    selected = min(candidates, key=lambda item: (item["final_validation_loss"], item["experiment_id"]))
    record = {
        "schema_name": PSTAR_SCHEMA_NAME,
        "created_at_utc": _utc_now(),
        "model_seed": model_seed,
        "criterion": "minimum final validation loss on current pair data",
        "selected_protocol": selected["protocol_name"],
        "selected_protocol_parameters": selected["protocol_parameters"],
        "selected_from_experiment": selected["experiment_id"],
        "common_evidence": common_evidence,
        "candidates": candidates,
    }
    _write_json_atomic(path, record)
    return record


def load_pstar_protocol_v2(
    path: Path,
    *,
    expected_model_seed: int | None = None,
) -> tuple[OptimizerProtocolV2, dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text(encoding="utf_8"))
    if record.get("schema_name") != PSTAR_SCHEMA_NAME:
        raise ValueError("Pstar selection schema is invalid")
    model_seed = int(record.get("model_seed", -1))
    if expected_model_seed is not None and model_seed != expected_model_seed:
        raise ValueError("Pstar model seed does not match the requested run")
    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 2:
        raise ValueError("Pstar requires exactly two candidate records")
    by_experiment = {
        candidate.get("experiment_id"): candidate
        for candidate in raw_candidates
        if isinstance(candidate, Mapping)
    }
    if set(by_experiment) != {"Y02", "Y03"}:
        raise ValueError("Pstar candidates must be exactly Y02 and Y03")
    rebuilt_candidates = []
    expected = {
        "Y02": P0_CURRENT_X01_PROTOCOL,
        "Y03": P1_LEGACY_5K_V3_PROTOCOL,
    }
    for experiment_id in ("Y02", "Y03"):
        candidate = by_experiment[experiment_id]
        summary_path = Path(candidate["summary_path"])
        if not summary_path.is_file() or _sha256(summary_path) != candidate.get("summary_sha256"):
            raise ValueError("Pstar source summary changed after selection")
        rebuilt = _pstar_candidate_evidence(
            summary_path,
            experiment_id,
            model_seed,
            expected[experiment_id],
        )
        if candidate != rebuilt:
            raise ValueError("Pstar candidate evidence changed")
        rebuilt_candidates.append(rebuilt)
    common_evidence = _pstar_common_evidence(rebuilt_candidates)
    if record.get("common_evidence") != common_evidence:
        raise ValueError("Pstar common evidence changed")
    selected = min(
        rebuilt_candidates,
        key=lambda item: (item["final_validation_loss"], item["experiment_id"]),
    )
    name = record.get("selected_protocol")
    if name != selected["protocol_name"]:
        raise ValueError("Pstar selected protocol is inconsistent")
    if record.get("selected_from_experiment") != selected["experiment_id"]:
        raise ValueError("Pstar selected experiment is inconsistent")
    protocol = expected[selected["experiment_id"]]
    if record.get("selected_protocol_parameters") != asdict(protocol):
        raise ValueError("Pstar selected protocol parameters changed")
    return protocol, record


def resolve_protocol_v2(
    experiment_id: str,
    config: YRunnerConfigV2,
    model_seed: int,
) -> tuple[OptimizerProtocolV2, dict[str, Any] | None]:
    normalized = get_y_experiment_spec_v2(experiment_id).experiment_id
    if normalized in {"Y01", "Y02"}:
        return P0_CURRENT_X01_PROTOCOL, None
    if normalized in {"Y03", "Y04", "Y05"}:
        return P1_LEGACY_5K_V3_PROTOCOL, None
    protocol, record = load_pstar_protocol_v2(
        _default_pstar_path(config, model_seed),
        expected_model_seed=model_seed,
    )
    return protocol, record


def target_steps_v2(experiment_id: str) -> int:
    normalized = get_y_experiment_spec_v2(experiment_id).experiment_id
    return CURRENT_X01_STEPS if normalized == "Y01" else LEGACY_5K_V3_STEPS


def _write_split_artifact(path: Path, split: SplitIndicesV1) -> dict[str, Any]:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as stream:
        np.savez(
            stream,
            train_sample_id=split.train,
            validation_sample_id=split.validation,
            test_sample_id=split.test,
        )
    partial.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }


def _source_records() -> dict[str, Any]:
    core_path = Path(__import__(
        "experiments.gnn.e311_gnn_y12_diagnostic_core_v2",
        fromlist=["__file__"],
    ).__file__).resolve()
    x_runner_path = Path(__import__(
        "experiments.gnn.e311_gnn_12_experiment_runner_v1",
        fromlist=["__file__"],
    ).__file__).resolve()
    runner_path = Path(__file__).resolve()
    return {
        "runner": {"path": str(runner_path), "sha256": _sha256(runner_path)},
        "core": {"path": str(core_path), "sha256": _sha256(core_path)},
        "x_data_support": {
            "path": str(x_runner_path),
            "sha256": _sha256(x_runner_path),
        },
    }


def _checkpoint_payload(
    *,
    experiment_id: str,
    model_seed: int,
    global_step: int,
    target_steps: int,
    model: E311YDiagnosticCoreV2,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: StatefulBatchSamplerV2,
    protocol: OptimizerProtocolV2,
    data_records: Mapping[str, Any],
    best_step: int,
    best_validation_loss: float,
    train_loss: float,
    train_loss_source: str,
    validation_loss: float,
    evaluation_index: int,
) -> dict[str, Any]:
    return {
        "schema_name": CHECKPOINT_SCHEMA_NAME,
        "experiment_id": experiment_id,
        "model_seed": model_seed,
        "global_step": global_step,
        "target_steps": target_steps,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "sampler_state": sampler.state_dict(),
        "protocol": asdict(protocol),
        "data_records": dict(data_records),
        "architecture": dict(model.architecture_record()),
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "train_loss": train_loss,
        "train_loss_source": train_loss_source,
        "validation_loss": validation_loss,
        "evaluation_index": evaluation_index,
        "runner_variant": RUNNER_VARIANT,
        "saved_at_utc": _utc_now(),
    }


def _save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(payload), partial)
    partial.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    model: E311YDiagnosticCoreV2,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: StatefulBatchSamplerV2,
    expected_protocol: OptimizerProtocolV2,
    expected_model_seed: int,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_name") != CHECKPOINT_SCHEMA_NAME:
        raise ValueError("checkpoint schema is invalid")
    if payload.get("runner_variant") != RUNNER_VARIANT:
        raise ValueError("checkpoint runner variant is invalid")
    if int(payload.get("model_seed", -1)) != expected_model_seed:
        raise ValueError("checkpoint model seed changed")
    if payload.get("protocol") != asdict(expected_protocol):
        raise ValueError("checkpoint protocol changed")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    sampler.load_state_dict(payload["sampler_state"])
    if int(payload.get("global_step", -1)) != sampler.consumed_batch_count():
        raise ValueError("checkpoint global step does not match sampler progress")
    return payload


_SENSITIVE_FRAGMENTS = ("api_key", "apikey", "password", "secret", "token")


def _sensitive_name(name: object) -> bool:
    normalized = str(name).strip().lower().replace(" ", "_")
    return any(fragment in normalized for fragment in _SENSITIVE_FRAGMENTS)


def _flatten_parameters(
    values: Mapping[str, Any],
    *,
    api_key: str,
    prefix: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_name, raw_value in values.items():
        if _sensitive_name(raw_name):
            raise ValueError("sensitive parameter names are forbidden")
        name = f"{prefix}.{raw_name}" if prefix else str(raw_name)
        if isinstance(raw_value, Mapping):
            result.update(_flatten_parameters(raw_value, api_key=api_key, prefix=name))
            continue
        value = _json_ready(raw_value)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if api_key and api_key in str(value):
            raise ValueError("Comet parameters contain the credential")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Comet parameters must be finite")
        result[name] = value
    return result


@dataclass(frozen=True, slots=True)
class YCometConfigV2:
    project: str
    workspace: str | None
    tags: tuple[str, ...]
    enabled: bool


class NullYCometLoggerV2:
    enabled = False

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "project": None,
            "experiment_name": None,
            "experiment_key": None,
            "url": None,
        }

    def log_parameters(self, parameters: Mapping[str, Any]) -> None:
        del parameters

    def log_evaluation(
        self,
        *,
        global_step: int,
        train_loss: float,
        validation_loss: float,
        epoch_duration_seconds: float,
    ) -> None:
        del global_step, train_loss, validation_loss, epoch_duration_seconds

    def log_final(
        self,
        *,
        global_step: int,
        final_test_mae: float,
        final_test_sae: float,
    ) -> None:
        del global_step, final_test_mae, final_test_sae

    def finish(self) -> None:
        return None


class YCometLoggerV2:
    enabled = True

    def __init__(
        self,
        backend: Any,
        config: YCometConfigV2,
        experiment_name: str,
        api_key: str,
    ) -> None:
        self.backend = backend
        self.config = config
        self.experiment_name = experiment_name
        self.api_key = api_key
        self.finished = False

    @property
    def identity(self) -> dict[str, Any]:
        key_method = getattr(self.backend, "get_key", None)
        experiment_key = key_method() if callable(key_method) else None
        return {
            "enabled": True,
            "project": self.config.project,
            "workspace": self.config.workspace,
            "experiment_name": self.experiment_name,
            "experiment_key": experiment_key,
            "url": getattr(self.backend, "url", None),
        }

    def log_parameters(self, parameters: Mapping[str, Any]) -> None:
        flattened = _flatten_parameters(parameters, api_key=self.api_key)
        if not flattened:
            raise ValueError("Comet parameters must not be empty")
        self.backend.log_parameters(flattened)

    def log_evaluation(
        self,
        *,
        global_step: int,
        train_loss: float,
        validation_loss: float,
        epoch_duration_seconds: float,
    ) -> None:
        metrics = {
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "epoch_duration_seconds": float(epoch_duration_seconds),
        }
        if set(metrics) != {
            "train_loss",
            "validation_loss",
            "epoch_duration_seconds",
        }:
            raise RuntimeError("Comet evaluation metric registry changed")
        if any(not math.isfinite(value) or value < 0.0 for value in metrics.values()):
            raise ValueError("Comet metrics must be finite and nonnegative")
        self.backend.log_metrics(metrics, step=global_step)

    def log_final(
        self,
        *,
        global_step: int,
        final_test_mae: float,
        final_test_sae: float,
    ) -> None:
        metrics = {
            "final_test_mae": float(final_test_mae),
            "final_test_sae": float(final_test_sae),
        }
        if set(metrics) != {"final_test_mae", "final_test_sae"}:
            raise RuntimeError("Comet final metric registry changed")
        if any(not math.isfinite(value) or value < 0.0 for value in metrics.values()):
            raise ValueError("Comet metrics must be finite and nonnegative")
        self.backend.log_metrics(metrics, step=global_step)

    def finish(self) -> None:
        if not self.finished:
            self.backend.end()
            self.finished = True


def _default_comet_backend_factory(
    *,
    config: YCometConfigV2,
    experiment_name: str,
    api_key: str,
) -> Any:
    try:
        import comet_ml
    except ImportError as error:
        raise RuntimeError("Comet logging is enabled but comet_ml is unavailable") from error
    experiment_config = comet_ml.ExperimentConfig(
        disabled=False,
        name=experiment_name,
        tags=list(config.tags),
        log_code=False,
        log_graph=False,
        parse_args=False,
        display_summary_level=0,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_details=False,
        log_env_gpu=False,
        log_env_host=False,
        log_env_cpu=False,
        log_env_network=False,
        log_env_disk=False,
        auto_output_logging=False,
        auto_param_logging=False,
        auto_metric_logging=False,
        auto_log_co2=False,
        auto_metric_step_rate=0,
        auto_histogram_epoch_rate=0,
        auto_histogram_gradient_logging=False,
        auto_histogram_activation_logging=False,
        auto_histogram_tensorboard_logging=False,
        auto_histogram_weight_logging=False,
    )
    return comet_ml.start(
        api_key=api_key,
        project_name=config.project,
        workspace=config.workspace,
        online=True,
        mode="create",
        experiment_config=experiment_config,
    )


def create_y_comet_logger_v2(
    config: YCometConfigV2,
    *,
    experiment_name: str,
    backend_factory: Callable[..., Any] | None = None,
) -> YCometLoggerV2 | NullYCometLoggerV2:
    if not config.enabled:
        return NullYCometLoggerV2()
    api_key = os.environ.get("COMET_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("COMET_API_KEY must be set")
    factory = _default_comet_backend_factory if backend_factory is None else backend_factory
    backend = factory(config=config, experiment_name=experiment_name, api_key=api_key)
    if backend is None or bool(getattr(backend, "disabled", False)):
        raise RuntimeError("Comet did not create an enabled experiment")
    return YCometLoggerV2(backend, config, experiment_name, api_key)


def _comet_config(config: YRunnerConfigV2, experiment_id: str, seed: int) -> YCometConfigV2:
    return YCometConfigV2(
        project=config.comet_project.strip(),
        workspace=config.comet_workspace,
        tags=("e311_gnn_y12_v2", RUNNER_VARIANT, experiment_id, f"seed_{seed}"),
        enabled=config.comet_enabled,
    )


def _create_and_record_comet(
    output: Path,
    config: YRunnerConfigV2,
    experiment_id: str,
    model_seed: int,
    backend_factory: Callable[..., Any] | None,
) -> YCometLoggerV2 | NullYCometLoggerV2:
    logger = create_y_comet_logger_v2(
        _comet_config(config, experiment_id, model_seed),
        experiment_name=f"{experiment_id}_seed_{model_seed}_{RUNNER_VARIANT}",
        backend_factory=backend_factory,
    )
    _write_json_atomic(output / "comet.json", logger.identity)
    return logger


def _status_record(
    *,
    status: str,
    experiment_id: str,
    model_seed: int,
    global_step: int,
    target_steps: int,
    protocol: OptimizerProtocolV2,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "runner_variant": RUNNER_VARIANT,
        "experiment_id": experiment_id,
        "model_seed": model_seed,
        "global_step": global_step,
        "target_steps": target_steps,
        "protocol": protocol.name,
        "updated_at_utc": _utc_now(),
        **extra,
    }


def _checkpoint_matches_data(payload: Mapping[str, Any], data_records: Mapping[str, Any]) -> None:
    source = dict(payload.get("data_records", {}))
    current = dict(data_records)
    source.pop("split_artifact", None)
    current.pop("split_artifact", None)
    if source != current:
        raise ValueError("checkpoint data provenance changed")


def run_y_experiment_v2(
    experiment_id: str,
    model_seed: int,
    config: YRunnerConfigV2,
    *,
    resume: bool = False,
    comet_backend_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    spec = get_y_experiment_spec_v2(experiment_id)
    experiment_id = spec.experiment_id
    target_steps = target_steps_v2(experiment_id)
    if int(spec.optimizer_steps) != target_steps:
        raise ValueError("core optimizer step declaration changed")
    protocol, pstar_record = resolve_protocol_v2(experiment_id, config, model_seed)
    device = _resolve_device(config.device)
    output = _run_directory(config, experiment_id, model_seed)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"completed run already exists at {output}")
    status_path = output / "status.json"
    history_path = output / "history.csv"
    resume_path = output / "resume_checkpoint.pt"
    best_path = output / "best_checkpoint.pt"
    final_path = output / "final_checkpoint.pt"

    data = prepare_y_experiment_data_v2(experiment_id, split_seed=config.split_seed)
    split_artifact = _write_split_artifact(output / "split_indices.npz", data.split)
    data_records = dict(data.records)
    data_records["split_artifact"] = split_artifact
    source_records = _source_records()
    model = _build_model(experiment_id, data.force_scale, model_seed, device)
    frozen_architecture_record = dict(model.architecture_record())
    architecture_fingerprint = _json_sha256(frozen_architecture_record)
    optimizer, scheduler = _build_optimizer_and_scheduler(model, protocol)
    sampler = StatefulBatchSamplerV2(
        data.train.sample_count,
        protocol.batch_size,
        config.shuffle_seed,
    )
    train_tensor = _tensor_bucket(data.train, data)
    validation_tensor = _tensor_bucket(data.validation, data)
    test_tensor = _tensor_bucket(data.test, data)
    dataset_steps_per_pass = math.ceil(data.train.sample_count / protocol.batch_size)
    initial_train_loss_source = (
        "inherited_Y01_final_full_train_evaluation"
        if experiment_id == "Y02"
        else "initial_full_train_evaluation"
    )

    global_step = 0
    best_step = 0
    best_validation_loss = math.inf
    train_loss = math.inf
    validation_loss = math.inf
    continuation_source: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    checkpoint_evaluation_index = 0

    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        payload = _load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_protocol=protocol,
            expected_model_seed=model_seed,
        )
        if payload.get("experiment_id") != experiment_id:
            raise ValueError("resume checkpoint belongs to another experiment")
        if int(payload.get("target_steps", -1)) != target_steps:
            raise ValueError("resume checkpoint target steps changed")
        if payload.get("architecture") != frozen_architecture_record:
            raise ValueError("resume checkpoint architecture changed")
        _checkpoint_matches_data(payload, data_records)
        global_step = int(payload["global_step"])
        best_step = int(payload["best_step"])
        best_validation_loss = float(payload["best_validation_loss"])
        train_loss = float(payload["train_loss"])
        validation_loss = float(payload["validation_loss"])
        checkpoint_evaluation_index = int(payload.get("evaluation_index", -1))
        history = _read_history(history_path, global_step)
        if (
            not history
            or int(history[-1]["global_step"]) != global_step
            or int(history[-1]["evaluation_index"]) != checkpoint_evaluation_index
        ):
            raise ValueError("resume history does not match the checkpoint")
        continuation_source = {
            "kind": "same_run_resume",
            "path": str(resume_path.resolve()),
        }
    elif experiment_id == "Y02":
        source_path = _run_directory(config, "Y01", model_seed) / "final_checkpoint.pt"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        payload = _load_checkpoint(
            source_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_protocol=P0_CURRENT_X01_PROTOCOL,
            expected_model_seed=model_seed,
        )
        if payload.get("experiment_id") != "Y01":
            raise ValueError("Y02 source is not a Y01 checkpoint")
        if int(payload.get("global_step", -1)) != CURRENT_X01_STEPS:
            raise ValueError("Y01 did not finish at exactly 8000 updates")
        if int(payload.get("target_steps", -1)) != CURRENT_X01_STEPS:
            raise ValueError("Y01 checkpoint target steps changed")
        if payload.get("architecture") != frozen_architecture_record:
            raise ValueError("Y01 checkpoint architecture changed")
        if payload.get("train_loss_source") != "final_full_train_evaluation":
            raise ValueError("Y01 final train loss source changed")
        _checkpoint_matches_data(payload, data_records)
        global_step = int(payload["global_step"])
        train_loss = float(payload["train_loss"])
        validation_loss = float(payload["validation_loss"])
        best_step = global_step
        best_validation_loss = validation_loss
        continuation_source = {
            "kind": "Y01_final_exact_state",
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_path),
            "source_global_step": global_step,
            "restored_states": ["model", "optimizer", "scheduler", "sampler"],
        }

    if global_step > target_steps:
        raise ValueError("checkpoint is beyond the requested step budget")

    _write_json_atomic(
        status_path,
        _status_record(
            status="running",
            experiment_id=experiment_id,
            model_seed=model_seed,
            global_step=global_step,
            target_steps=target_steps,
            protocol=protocol,
            started_at_utc=_utc_now(),
        ),
    )
    logger: YCometLoggerV2 | NullYCometLoggerV2 | None = None
    try:
        logger = _create_and_record_comet(
            output,
            config,
            experiment_id,
            model_seed,
            comet_backend_factory,
        )
        parameter_record = {
            "runner_variant": RUNNER_VARIANT,
            "train_curve_source": "component_weighted_optimization_loss_per_protocol_epoch",
            "full_train_evaluation": "fresh_run_initial_and_final",
            "record_cadence": "evaluation_index_1_every_10_and_final",
            "gradient_finite_check_cadence": "global_steps_1_to_4_and_every_100",
            "validation_cadence": "every_protocol_epoch",
            "protocol_epoch_steps": protocol.evaluation_cadence_steps,
            "dataset_steps_per_pass": dataset_steps_per_pass,
            "initial_train_loss_source": initial_train_loss_source,
            "experiment": asdict(spec),
            "model_seed": model_seed,
            "split_seed": config.split_seed,
            "shuffle_seed": config.shuffle_seed,
            "target_steps": target_steps,
            "resolved_protocol": asdict(protocol),
            "supervision": data.supervision.value,
            "dataset": data_records,
            "architecture": frozen_architecture_record,
            "architecture_fingerprint_sha256": architecture_fingerprint,
            "sources": source_records,
            "continuation": continuation_source,
            "pstar_selection": pstar_record,
            "device": str(device),
        }
        logger.log_parameters(parameter_record)

        evaluation_index = checkpoint_evaluation_index
        if not history:
            if not math.isfinite(train_loss):
                train_loss = _evaluate_loss(
                    model,
                    train_tensor,
                    data.supervision,
                    protocol.batch_size,
                    device,
                )
                validation_loss = _evaluate_loss(
                    model,
                    validation_tensor,
                    data.supervision,
                    protocol.batch_size,
                    device,
                )
                best_step = global_step
                best_validation_loss = validation_loss
            history.append(
                {
                    "global_step": global_step,
                    "evaluation_index": 0,
                    "train_loss": train_loss,
                    "train_loss_source": initial_train_loss_source,
                    "validation_loss": validation_loss,
                    "epoch_duration_seconds": None,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "sampler_cycle": sampler.cycle,
                    "recorded_at_utc": _utc_now(),
                }
            )
            evaluation_index = 0
            _write_history_atomic(history_path, history)
            initial_payload = _checkpoint_payload(
                experiment_id=experiment_id,
                model_seed=model_seed,
                global_step=global_step,
                target_steps=target_steps,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                protocol=protocol,
                data_records=data_records,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                train_loss=train_loss,
                train_loss_source=initial_train_loss_source,
                validation_loss=validation_loss,
                evaluation_index=evaluation_index,
            )
            _save_checkpoint_atomic(best_path, initial_payload)
            _save_checkpoint_atomic(resume_path, initial_payload)

        interval_started = time.perf_counter()
        epoch_loss = ProtocolEpochLossAccumulatorV2(device)
        while global_step < target_steps:
            selection = sampler.next_indices()
            next_global_step = global_step + 1
            step_result = _train_step(
                model,
                train_tensor,
                selection,
                data.supervision,
                optimizer,
                device,
                next_global_step,
            )
            epoch_loss.add(step_result)
            global_step = next_global_step
            if protocol.scheduler == "StepLR":
                scheduler.step()
            should_evaluate = (
                global_step % protocol.evaluation_cadence_steps == 0
                or global_step == target_steps
            )
            if not should_evaluate:
                continue
            train_loss = epoch_loss.finish()
            validation_loss = _evaluate_loss(
                model,
                validation_tensor,
                data.supervision,
                protocol.batch_size,
                device,
            )
            if protocol.scheduler == "ReduceLROnPlateau":
                scheduler.step(validation_loss)
            duration = time.perf_counter() - interval_started
            evaluation_index += 1
            row = {
                "global_step": global_step,
                "evaluation_index": evaluation_index,
                "train_loss": train_loss,
                "train_loss_source": "protocol_epoch_optimization_loss",
                "validation_loss": validation_loss,
                "epoch_duration_seconds": duration,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "sampler_cycle": sampler.cycle,
                "recorded_at_utc": _utc_now(),
            }
            history.append(row)
            improved = validation_loss < best_validation_loss
            if improved:
                best_step = global_step
                best_validation_loss = validation_loss
            should_persist = should_persist_evaluation_v2(
                evaluation_index,
                global_step,
                target_steps,
            )
            payload = None
            if improved or should_persist:
                payload = _checkpoint_payload(
                    experiment_id=experiment_id,
                    model_seed=model_seed,
                    global_step=global_step,
                    target_steps=target_steps,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    protocol=protocol,
                    data_records=data_records,
                    best_step=best_step,
                    best_validation_loss=best_validation_loss,
                    train_loss=train_loss,
                    train_loss_source="protocol_epoch_optimization_loss",
                    validation_loss=validation_loss,
                    evaluation_index=evaluation_index,
                )
            if improved:
                if payload is None:
                    raise RuntimeError("best checkpoint payload is missing")
                _save_checkpoint_atomic(best_path, payload)
            if should_persist:
                if payload is None:
                    raise RuntimeError("resume checkpoint payload is missing")
                _write_history_atomic(history_path, history)
                _save_checkpoint_atomic(resume_path, payload)
                _write_json_atomic(
                    status_path,
                    _status_record(
                        status="running",
                        experiment_id=experiment_id,
                        model_seed=model_seed,
                        global_step=global_step,
                        target_steps=target_steps,
                        protocol=protocol,
                        best_step=best_step,
                        best_validation_loss=best_validation_loss,
                        evaluation_index=evaluation_index,
                    ),
                )
                logger.log_evaluation(
                    global_step=global_step,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    epoch_duration_seconds=duration,
                )
            interval_started = time.perf_counter()
            epoch_loss = ProtocolEpochLossAccumulatorV2(device)

        final_full_train_loss = _evaluate_loss(
            model,
            train_tensor,
            data.supervision,
            protocol.batch_size,
            device,
        )
        final_payload = _checkpoint_payload(
            experiment_id=experiment_id,
            model_seed=model_seed,
            global_step=global_step,
            target_steps=target_steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            protocol=protocol,
            data_records=data_records,
            best_step=best_step,
            best_validation_loss=best_validation_loss,
            train_loss=final_full_train_loss,
            train_loss_source="final_full_train_evaluation",
            validation_loss=validation_loss,
            evaluation_index=evaluation_index,
        )
        _write_history_atomic(history_path, history)
        _save_checkpoint_atomic(final_path, final_payload)
        _save_checkpoint_atomic(resume_path, final_payload)
        metrics = _physical_metrics(
            model,
            test_tensor,
            data.force_scale,
            protocol.batch_size,
            device,
        )
        logger.log_final(
            global_step=global_step,
            final_test_mae=metrics.final_test_mae,
            final_test_sae=metrics.final_test_sae,
        )
        summary = {
            "schema_name": RESULT_SCHEMA_NAME,
            "runner_variant": RUNNER_VARIANT,
            "status": "complete",
            "experiment": asdict(spec),
            "model_seed": model_seed,
            "split_seed": config.split_seed,
            "shuffle_seed": config.shuffle_seed,
            "training": {
                "train_curve_source": "component_weighted_optimization_loss_per_protocol_epoch",
                "full_train_evaluation": "fresh_run_initial_and_final",
                "record_cadence": "evaluation_index_1_every_10_and_final",
                "gradient_finite_check_cadence": "global_steps_1_to_4_and_every_100",
                "validation_cadence": "every_protocol_epoch",
                "protocol_epoch_steps": protocol.evaluation_cadence_steps,
                "dataset_steps_per_pass": dataset_steps_per_pass,
                "initial_train_loss_source": initial_train_loss_source,
                "global_steps_completed": global_step,
                "target_steps": target_steps,
                "resolved_protocol": asdict(protocol),
                "initial_train_loss": history[0]["train_loss"],
                "initial_validation_loss": history[0]["validation_loss"],
                "final_train_loss": final_full_train_loss,
                "final_train_loss_source": "final_full_train_evaluation",
                "final_validation_loss": validation_loss,
                "best_step": best_step,
                "best_validation_loss": best_validation_loss,
            },
            "final_test": asdict(metrics),
            "data": data_records,
            "architecture": frozen_architecture_record,
            "architecture_fingerprint_sha256": architecture_fingerprint,
            "sources": source_records,
            "continuation": continuation_source,
            "pstar_selection": pstar_record,
            "comet": logger.identity,
            "artifacts": {
                "history": str(history_path.resolve()),
                "status": str(status_path.resolve()),
                "comet": str((output / "comet.json").resolve()),
                "best_checkpoint": str(best_path.resolve()),
                "final_checkpoint": str(final_path.resolve()),
                "resume_checkpoint": str(resume_path.resolve()),
                "split_indices": split_artifact,
            },
            "completed_at_utc": _utc_now(),
        }
        _write_json_atomic(summary_path, summary)
        _write_json_atomic(
            status_path,
            _status_record(
                status="complete",
                experiment_id=experiment_id,
                model_seed=model_seed,
                global_step=global_step,
                target_steps=target_steps,
                protocol=protocol,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                completed_at_utc=_utc_now(),
            ),
        )
        logger.finish()
        return summary
    except Exception as error:
        _write_json_atomic(
            status_path,
            _status_record(
                status="failed",
                experiment_id=experiment_id,
                model_seed=model_seed,
                global_step=global_step,
                target_steps=target_steps,
                protocol=protocol,
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        if logger is not None:
            logger.finish()
        raise


def run_y_group_v2(
    experiment_ids: Sequence[str],
    model_seed: int,
    config: YRunnerConfigV2,
) -> list[dict[str, Any]]:
    normalized = [get_y_experiment_spec_v2(value).experiment_id for value in experiment_ids]
    order = {f"Y{index:02d}": index for index in range(1, 13)}
    normalized = sorted(dict.fromkeys(normalized), key=order.__getitem__)
    results = []
    for experiment_id in normalized:
        if int(experiment_id[1:]) >= 6 and not _default_pstar_path(config, model_seed).is_file():
            select_pstar_protocol_v2(config, model_seed)
        results.append(run_y_experiment_v2(experiment_id, model_seed, config))
    return results


def preflight_v2(config: YRunnerConfigV2) -> dict[str, Any]:
    device = _resolve_device(config.device)
    pair_data = prepare_y_experiment_data_v2("Y01", split_seed=config.split_seed)
    legacy_data = prepare_y_experiment_data_v2("Y04", split_seed=config.split_seed)
    five_data = prepare_y_experiment_data_v2("Y06", split_seed=config.split_seed)
    model = _build_model("Y01", pair_data.force_scale, DEFAULT_MODEL_SEED, device)
    tensor_data = _tensor_bucket(pair_data.train, pair_data)
    centers = tensor_data.centers[:2].to(device)
    frames = tensor_data.frames[:2].to(device)
    pair_index = tensor_data.pair_index.to(device)
    assert_one_layer_stack_preflight_v2(model, centers, frames, pair_index)
    sampler = StatefulBatchSamplerV2(17, 5, 31)
    for _ in range(5):
        sampler.next_indices()
    state = sampler.state_dict()
    expected_next = sampler.next_indices()
    restored = StatefulBatchSamplerV2(17, 5, 31)
    restored.load_state_dict(state)
    if not torch.equal(expected_next, restored.next_indices()):
        raise RuntimeError("sampler continuation preflight failed")
    finite_optimizer_smoke = [
        _finite_optimizer_smoke_v2(experiment_id, five_data, device)
        for experiment_id in ("Y07", "Y12")
    ]
    return {
        "status": "passed",
        "device": str(device),
        "protocols": {
            "P0": asdict(P0_CURRENT_X01_PROTOCOL),
            "P1": asdict(P1_LEGACY_5K_V3_PROTOCOL),
        },
        "datasets": {
            "pair_2k_current": dict(pair_data.records),
            "pair_5k_legacy_v3": dict(legacy_data.records),
            "five_benzene_1k": dict(five_data.records),
        },
        "one_layer_architecture": dict(model.architecture_record()),
        "sampler_state_round_trip": True,
        "finite_optimizer_smoke": finite_optimizer_smoke,
        "sources": _source_records(),
        "completed_at_utc": _utc_now(),
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--comet-project", default=DEFAULT_COMET_PROJECT)
    parser.add_argument("--comet-workspace")
    parser.add_argument("--disable-comet", action="store_true")
    parser.add_argument("--pstar-selection", type=Path)


def build_argument_parser_v2() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_common_arguments(preflight)
    run = subparsers.add_parser("run")
    run.add_argument("experiment_id")
    run.add_argument("--seed", type=int, default=DEFAULT_MODEL_SEED)
    run.add_argument("--resume", action="store_true")
    _add_common_arguments(run)
    group = subparsers.add_parser("run-group")
    group.add_argument("--experiments", nargs="+", required=True)
    group.add_argument("--seed", type=int, default=DEFAULT_MODEL_SEED)
    _add_common_arguments(group)
    select = subparsers.add_parser("select-pstar")
    select.add_argument("--seed", type=int, default=DEFAULT_MODEL_SEED)
    select.add_argument("--selection-output", type=Path)
    select.add_argument("--overwrite", action="store_true")
    _add_common_arguments(select)
    return parser


def _config_from_args(args: argparse.Namespace) -> YRunnerConfigV2:
    return YRunnerConfigV2(
        output_root=args.output_root,
        split_seed=args.split_seed,
        shuffle_seed=args.shuffle_seed,
        device=args.device,
        comet_project=args.comet_project,
        comet_workspace=args.comet_workspace,
        comet_enabled=not args.disable_comet,
        pstar_selection_path=args.pstar_selection,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser_v2().parse_args(argv)
    config = _config_from_args(args)
    if args.command == "preflight":
        result: Any = preflight_v2(config)
    elif args.command == "run":
        result = run_y_experiment_v2(
            args.experiment_id,
            args.seed,
            config,
            resume=args.resume,
        )
    elif args.command == "run-group":
        result = run_y_group_v2(args.experiments, args.seed, config)
    else:
        result = select_pstar_protocol_v2(
            config,
            args.seed,
            output_path=args.selection_output,
            overwrite=args.overwrite,
        )
    print(json.dumps(_json_ready(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_COMET_PROJECT",
    "DEFAULT_MODEL_SEED",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SHUFFLE_SEED",
    "DEFAULT_SPLIT_SEED",
    "OptimizerProtocolV2",
    "P0_CURRENT_X01_PROTOCOL",
    "P1_LEGACY_5K_V3_PROTOCOL",
    "PhysicalMetricsV2",
    "PreparedYDataV2",
    "ProtocolEpochLossAccumulatorV2",
    "RUNNER_VARIANT",
    "StatefulBatchSamplerV2",
    "TrainStepResultV2",
    "YCometConfigV2",
    "YRunnerConfigV2",
    "build_argument_parser_v2",
    "create_y_comet_logger_v2",
    "load_pstar_protocol_v2",
    "prepare_y_experiment_data_v2",
    "preflight_v2",
    "resolve_protocol_v2",
    "run_y_experiment_v2",
    "run_y_group_v2",
    "select_pstar_protocol_v2",
    "should_check_gradients_v2",
    "should_persist_evaluation_v2",
    "target_steps_v2",
    "y_supervision_v2",
]
