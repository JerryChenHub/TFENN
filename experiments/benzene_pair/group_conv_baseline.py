"""Train the network level finite group convolution comparison model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import socket
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from experiments.benzene_pair.hyper_search import (
    _load_v2_training_data,
    _validate_dataset_files,
)
from experiments.benzene_pair.train import (
    PairTrainingData,
    TARGET_DEFINITION,
    _git_provenance,
    _runtime_versions,
    create_split,
    regression_metrics,
    sha256_file,
    symmetry_metrics,
    validate_split_fractions,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "group_conv_config.json"
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
COMPONENT_LABELS = ("Fx", "Fy", "Fz")
_STOP_REQUESTED = False


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Describe the reproducible group convolution comparison run."""

    csv_path: Path
    validation_path: Path
    output_directory: Path
    architecture_id: str
    hidden_widths: tuple[int, ...]
    expected_parameter_count: int
    epochs: int
    resume_every: int
    split_seed: int
    model_seed: int
    shuffle_seed: int
    split_fractions: tuple[float, float, float]
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    scheduler_step_size: int
    scheduler_gamma: float
    device: str
    dtype: str
    threads: int
    symmetry_tolerance: float
    symmetry_probe_count: int
    progress_every: int
    expected_sample_count: int
    expected_dataset_revision: int
    expected_opls_version: str
    deterministic_algorithms: bool
    zero_output_head: bool
    mape_zero_threshold: float

    @classmethod
    def from_path(cls, path: str | Path) -> BaselineConfig:
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf_8"))
        if not isinstance(value, Mapping):
            raise TypeError("baseline config must contain one object")
        if value.get("schema_name") != "tfenn_benzene_pair_group_conv_baseline":
            raise ValueError("unexpected baseline config schema name")
        if value.get("schema_version") != 1:
            raise ValueError("unexpected baseline config schema version")

        def resolved(name: str) -> Path:
            result = Path(str(value[name]))
            return (
                result.resolve() if result.is_absolute() else REPOSITORY_ROOT / result
            )

        result = cls(
            csv_path=resolved("csv_path"),
            validation_path=resolved("validation_path"),
            output_directory=resolved("output_directory"),
            architecture_id=str(value["architecture_id"]),
            hidden_widths=tuple(int(item) for item in value["hidden_widths"]),
            expected_parameter_count=int(value["expected_parameter_count"]),
            epochs=int(value["epochs"]),
            resume_every=int(value["resume_every"]),
            split_seed=int(value["split_seed"]),
            model_seed=int(value["model_seed"]),
            shuffle_seed=int(value["shuffle_seed"]),
            split_fractions=tuple(float(item) for item in value["split_fractions"]),
            optimizer=str(value["optimizer"]),
            learning_rate=float(value["learning_rate"]),
            weight_decay=float(value["weight_decay"]),
            batch_size=int(value["batch_size"]),
            scheduler_step_size=int(value["scheduler_step_size"]),
            scheduler_gamma=float(value["scheduler_gamma"]),
            device=str(value["device"]),
            dtype=str(value["dtype"]),
            threads=int(value["threads"]),
            symmetry_tolerance=float(value["symmetry_tolerance"]),
            symmetry_probe_count=int(value["symmetry_probe_count"]),
            progress_every=int(value["progress_every"]),
            expected_sample_count=int(value["expected_sample_count"]),
            expected_dataset_revision=int(value["expected_dataset_revision"]),
            expected_opls_version=str(value["expected_opls_version"]),
            deterministic_algorithms=bool(value["deterministic_algorithms"]),
            zero_output_head=bool(value["zero_output_head"]),
            mape_zero_threshold=float(value["mape_zero_threshold"]),
        )
        result.validate()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BaselineConfig:
        fields = dict(value)
        for name in ("csv_path", "validation_path", "output_directory"):
            fields[name] = Path(str(fields[name]))
        fields["hidden_widths"] = tuple(int(item) for item in fields["hidden_widths"])
        fields["split_fractions"] = tuple(
            float(item) for item in fields["split_fractions"]
        )
        result = cls(**fields)
        result.validate()
        return result

    def validate(self) -> None:
        if not self.architecture_id:
            raise ValueError("architecture_id must be nonempty")
        if self.hidden_widths != (96, 96, 96):
            raise ValueError("the comparison model requires three width 96 layers")
        if self.expected_parameter_count != 20160:
            raise ValueError("the comparison model requires exactly 20160 parameters")
        if self.epochs != 1500:
            raise ValueError("the formal comparison requires exactly 1500 epochs")
        if self.resume_every != 25:
            raise ValueError(
                "the formal comparison must save resume state every 25 epochs"
            )
        validate_split_fractions(self.split_fractions)
        if (self.split_seed, self.model_seed, self.shuffle_seed) != (
            20260813,
            20260814,
            20260815,
        ):
            raise ValueError("baseline seeds must match the hyperparameter study")
        if self.optimizer != "adamw":
            raise ValueError("the comparison optimizer must be adamw")
        if self.learning_rate != 0.002 or self.weight_decay != 0.00001:
            raise ValueError("optimizer values must match trial 029")
        if self.batch_size != 128:
            raise ValueError("batch size must match trial 029")
        if self.scheduler_step_size != 400 or self.scheduler_gamma != 0.5:
            raise ValueError("scheduler values must match trial 029")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.threads < 1 or self.progress_every < 1:
            raise ValueError("threads and progress_every must be positive")
        if self.symmetry_probe_count < 1 or self.symmetry_tolerance <= 0.0:
            raise ValueError("symmetry settings must be positive")
        if self.expected_sample_count != 5000:
            raise ValueError("the formal comparison requires 5000 samples")
        if self.expected_dataset_revision != 2:
            raise ValueError("the formal comparison requires dataset revision two")
        if self.expected_opls_version != "2.0.0":
            raise ValueError("the formal comparison requires OPLS 2.0.0")
        if self.mape_zero_threshold < 0.0 or not math.isfinite(
            self.mape_zero_threshold
        ):
            raise ValueError("mape_zero_threshold must be finite and nonnegative")

    def as_json(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("csv_path", "validation_path", "output_directory"):
            value[name] = str(Path(value[name]).resolve())
        value["hidden_widths"] = list(self.hidden_widths)
        value["split_fractions"] = list(self.split_fractions)
        return value

    def protocol_dict(
        self,
        *,
        epochs: int,
        sample_limit: int | None,
        device: str,
    ) -> dict[str, Any]:
        value = self.as_json()
        value["epochs"] = int(epochs)
        value["sample_limit"] = sample_limit
        value["device"] = device
        value.pop("output_directory")
        return value


@dataclass(frozen=True, slots=True)
class BaselinePaths:
    directory: Path
    definition: Path
    status: Path
    history: Path
    best: Path
    final: Path
    resume: Path
    summary: Path
    error: Path

    @classmethod
    def from_directory(cls, directory: str | Path) -> BaselinePaths:
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
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf_8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf_8",
    )
    os.replace(partial, path)
    return path


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    torch.save(dict(value), partial)
    os.replace(partial, path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf_8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one object")
    return value


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", newline="", encoding="utf_8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            if tuple(row) != HISTORY_FIELDS:
                raise ValueError("history fields do not match the required schema")
            writer.writerow(row)
    os.replace(partial, path)
    return path


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf_8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
            raise ValueError("history header does not match the required schema")
        rows = []
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    **{
                        name: float(row[name])
                        for name in HISTORY_FIELDS
                        if name != "epoch"
                    },
                }
            )
    if [row["epoch"] for row in rows] != list(range(len(rows))):
        raise ValueError("history epochs must be contiguous from zero")
    return rows


def _history_row(
    epoch: int,
    learning_rate: float,
    train_loss: float,
    validation: Mapping[str, float],
    duration: float,
) -> dict[str, Any]:
    values = (
        int(epoch),
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
    if not all(math.isfinite(item) for item in values[1:]):
        raise RuntimeError("history values must be finite")
    return dict(zip(HISTORY_FIELDS, values, strict=True))


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _configure_runtime(config: BaselineConfig, device: str) -> None:
    torch.set_num_threads(config.threads)
    if config.deterministic_algorithms:
        if device.startswith("cuda"):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.model_seed)


def _build_model(config: BaselineConfig, *, device: str, dtype: str) -> nn.Module:
    from TFENN.models import (
        NetworkGroupConvConfig,
        build_benzene_pair_network_group_conv_mlp,
    )

    model_config = NetworkGroupConvConfig(
        hidden_widths=config.hidden_widths,
        activation="silu",
        distance_scale=10.0,
        use_hidden_bias=True,
        architecture_id=config.architecture_id,
    )
    torch.manual_seed(config.model_seed)
    model = build_benzene_pair_network_group_conv_mlp(model_config)
    if config.zero_output_head:
        zeroed = model.zero_output_head()
        if zeroed != 1:
            raise RuntimeError("the comparison model must zero exactly one output head")
    return model.to(device=torch.device(device), dtype=_resolve_dtype(dtype))


def _parameter_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _restore_parameter_state(model: nn.Module, learned: Mapping[str, Any]) -> None:
    parameters = dict(model.named_parameters())
    if set(learned) != set(parameters):
        raise ValueError("checkpoint parameter names do not match the model")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = learned[name]
            if not isinstance(value, Tensor) or value.shape != parameter.shape:
                raise ValueError(f"checkpoint parameter {name} has an invalid shape")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _save_learned_checkpoint(
    path: Path,
    parameter_state: Mapping[str, Tensor],
    *,
    epoch: int,
    target_scale: Tensor,
    config: BaselineConfig,
    metrics: Mapping[str, float],
) -> Path:
    return _atomic_torch_save(
        path,
        {
            "schema_name": "tfenn_pair_group_conv_force_checkpoint",
            "schema_version": 1,
            "network_name": config.architecture_id,
            "epoch": int(epoch),
            "target_definition": TARGET_DEFINITION,
            "target_scale": float(target_scale.detach().cpu()),
            "configuration": config.as_json(),
            "metrics": dict(metrics),
            "fixed_tensor_artifacts_stored": False,
            "parameter_state_dict": {
                name: value.detach().cpu() for name, value in parameter_state.items()
            },
        },
    )


def load_trained_group_conv(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[nn.Module, float, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain one object")
    if payload.get("schema_name") != "tfenn_pair_group_conv_force_checkpoint":
        raise ValueError("unexpected group convolution checkpoint schema")
    if payload.get("schema_version") != 1:
        raise ValueError("unexpected group convolution checkpoint version")
    if payload.get("fixed_tensor_artifacts_stored") is not False:
        raise ValueError("checkpoint must not store fixed tensor artifacts")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("checkpoint configuration is missing")
    config = BaselineConfig.from_dict(configuration)
    config = replace(config, device=device, dtype=dtype, zero_output_head=False)
    model = _build_model(config, device=device, dtype=dtype)
    learned = payload.get("parameter_state_dict")
    if not isinstance(learned, Mapping):
        raise ValueError("checkpoint parameter state is missing")
    _restore_parameter_state(model, learned)
    target_scale = float(payload["target_scale"])
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise ValueError("checkpoint target scale must be finite and positive")
    return model, target_scale, payload


def _optimizer_and_scheduler(
    model: nn.Module,
    config: BaselineConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("network must have trainable parameters")
    optimizer = torch.optim.AdamW(
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


def _predict(
    model: nn.Module,
    data: PairTrainingData,
    indices: Tensor,
    batch_size: int,
) -> Tensor:
    was_training = model.training
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, int(indices.numel()), batch_size):
            selection = indices[start : start + batch_size].to(data.centers.device)
            prediction = model(data.centers[selection], data.frames[selection])
            if prediction.shape != (int(selection.numel()), 3):
                raise ValueError("network output must have shape batch by three")
            values.append(prediction.detach())
    model.train(was_training)
    return torch.cat(values, dim=0)


def _evaluate(
    model: nn.Module,
    data: PairTrainingData,
    indices: Tensor,
    batch_size: int,
    target_scale: Tensor,
) -> dict[str, float]:
    prediction = _predict(model, data, indices, batch_size)
    target = data.root_force[indices.to(data.root_force.device)] / target_scale
    return regression_metrics(prediction, target, target_scale)


def _component_errors(
    model: nn.Module,
    data: PairTrainingData,
    indices: Tensor,
    batch_size: int,
    target_scale: Tensor,
    *,
    zero_threshold: float,
) -> dict[str, Any]:
    prediction = _predict(model, data, indices, batch_size).to(torch.float64)
    selection = indices.to(data.root_force.device)
    target = (data.root_force[selection] / target_scale).to(torch.float64)
    error = prediction - target
    relative = torch.sqrt(error.square().mean(dim=0)) / torch.sqrt(
        target.square().mean(dim=0)
    )
    mape = []
    median = []
    valid_counts = []
    zero_counts = []
    for component in range(3):
        denominator = target[:, component].abs()
        valid = denominator > zero_threshold
        valid_count = int(valid.sum())
        if valid_count == 0:
            raise ValueError("component percentage error has no valid targets")
        percentages = error[valid, component].abs() / denominator[valid] * 100.0
        mape.append(float(percentages.mean()))
        median.append(float(percentages.median()))
        valid_counts.append(valid_count)
        zero_counts.append(int((~valid).sum()))
    result = {
        "component_labels": list(COMPONENT_LABELS),
        "component_relative_rmse_percent": [
            float(item) for item in relative.mul(100.0)
        ],
        "component_mape_percent": mape,
        "component_median_ape_percent": median,
        "mape_valid_count": valid_counts,
        "mape_excluded_count": zero_counts,
        "mape_denominator_rule": "absolute target component greater than threshold",
        "mape_zero_threshold_normalized": float(zero_threshold),
    }
    numerical = (
        result["component_relative_rmse_percent"]
        + result["component_mape_percent"]
        + result["component_median_ape_percent"]
    )
    if not all(math.isfinite(value) for value in numerical):
        raise RuntimeError("component error metrics must be finite")
    return result


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


def _save_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    shuffle_generator: torch.Generator,
    *,
    epoch: int,
    target_scale: Tensor,
    run_hash: str,
    best_epoch: int,
    best_validation: Mapping[str, float],
    best_parameters: Mapping[str, Tensor],
    initial_train_loss: float,
) -> Path:
    return _atomic_torch_save(
        path,
        {
            "schema_name": "tfenn_pair_group_conv_resume",
            "schema_version": 1,
            "run_hash": run_hash,
            "epoch": int(epoch),
            "target_scale": float(target_scale.detach().cpu()),
            "best_epoch": int(best_epoch),
            "best_validation": dict(best_validation),
            "initial_train_normalized_mse": float(initial_train_loss),
            "fixed_tensor_artifacts_stored": False,
            "parameter_state_dict": _parameter_state(model),
            "best_parameter_state_dict": {
                name: value.detach().cpu() for name, value in best_parameters.items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "shuffle_generator_state": shuffle_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        },
    )


def _restore_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    shuffle_generator: torch.Generator,
    *,
    run_hash: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("resume file must contain one object")
    if payload.get("schema_name") != "tfenn_pair_group_conv_resume":
        raise ValueError("unexpected resume schema")
    if payload.get("schema_version") != 1 or payload.get("run_hash") != run_hash:
        raise ValueError("resume state does not match this run")
    if payload.get("fixed_tensor_artifacts_stored") is not False:
        raise ValueError("resume file must not store fixed tensor artifacts")
    learned = payload.get("parameter_state_dict")
    best = payload.get("best_parameter_state_dict")
    if not isinstance(learned, Mapping) or not isinstance(best, Mapping):
        raise ValueError("resume parameter state is missing")
    _restore_parameter_state(model, learned)
    if set(best) != set(dict(model.named_parameters())):
        raise ValueError("resume best parameter names do not match the model")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    device = next(model.parameters()).device
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if isinstance(value, Tensor):
                state[name] = value.to(device=device)
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    shuffle_generator.set_state(payload["shuffle_generator_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_states = payload.get("cuda_rng_state_all", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    return payload


def _write_status(
    paths: BaselinePaths,
    *,
    run_hash: str,
    status: str,
    epoch: int,
    best_epoch: int | None = None,
    train_loss: float | None = None,
    validation_loss: float | None = None,
    message: str | None = None,
) -> None:
    value: dict[str, Any] = {
        "schema_name": "tfenn_pair_group_conv_status",
        "schema_version": 1,
        "updated_at_utc": _utc_now(),
        "architecture_id": "pair_network_group_conv_mlp_v1",
        "run_hash": run_hash,
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


def _source_sha256() -> str:
    from TFENN.models import network_group_conv

    files = (
        Path(__file__).resolve(),
        Path(network_group_conv.__file__).resolve(),
        DEFAULT_CONFIG_PATH.resolve(),
    )
    return _canonical_sha256(
        {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in files}
    )


def _ensure_definition(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if _load_json(path) != dict(value):
            raise RuntimeError("existing run definition does not match this run")
    else:
        _atomic_json(path, value)


def _run_hash(
    config: BaselineConfig,
    dataset_record: Mapping[str, Any],
    *,
    epochs: int,
    sample_limit: int | None,
    device: str,
    source_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "protocol": config.protocol_dict(
                epochs=epochs,
                sample_limit=sample_limit,
                device=device,
            ),
            "dataset_csv_sha256": dataset_record["csv_sha256"],
            "dataset_metadata_sha256": dataset_record["metadata_sha256"],
            "source_sha256": source_sha256,
        }
    )


def run_baseline(
    config: BaselineConfig,
    *,
    epochs_override: int | None = None,
    sample_limit: int | None = None,
    output_directory: str | Path | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Run or resume the fixed group convolution comparison."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    epochs = config.epochs if epochs_override is None else int(epochs_override)
    if epochs < 1 or epochs > config.epochs:
        raise ValueError("epochs_override must be between one and 1500")
    if sample_limit is not None and (sample_limit < 3 or sample_limit > 5000):
        raise ValueError("sample_limit must be between three and 5000")
    device = _resolve_device(device_override or config.device)
    paths = BaselinePaths.from_directory(
        output_directory if output_directory is not None else config.output_directory
    )
    paths.directory.mkdir(parents=True, exist_ok=True)
    dataset_record = _validate_dataset_files(config)
    source_digest = _source_sha256()
    run_hash = _run_hash(
        config,
        dataset_record,
        epochs=epochs,
        sample_limit=sample_limit,
        device=device,
        source_sha256=source_digest,
    )
    definition = {
        "schema_name": "tfenn_pair_group_conv_run",
        "schema_version": 1,
        "architecture_id": config.architecture_id,
        "run_hash": run_hash,
        "source_sha256": source_digest,
        "dataset_csv_sha256": dataset_record["csv_sha256"],
        "dataset_metadata_sha256": dataset_record["metadata_sha256"],
        "sample_limit": sample_limit,
        "training_configuration": {
            **config.as_json(),
            "epochs": epochs,
            "output_directory": str(paths.directory),
            "device": device,
        },
    }
    _ensure_definition(paths.definition, definition)
    if paths.summary.is_file():
        summary = _load_json(paths.summary)
        if summary.get("run_hash") != run_hash:
            raise RuntimeError("existing summary belongs to another run")
        return summary

    _configure_runtime(config, device)
    started_at = _utc_now()
    started_clock = time.perf_counter()
    previous_sigterm = signal.signal(signal.SIGTERM, _set_stop_requested)
    previous_sigint = signal.signal(signal.SIGINT, _set_stop_requested)
    try:
        data = _load_v2_training_data(config, device=device, sample_limit=sample_limit)
        split = create_split(
            len(data.root_force),
            seed=config.split_seed,
            fractions=config.split_fractions,
        )
        model = _build_model(config, device=device, dtype=config.dtype)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if parameter_count != config.expected_parameter_count:
            raise RuntimeError(
                f"model has {parameter_count} parameters, expected "
                f"{config.expected_parameter_count}"
            )
        if trainable_parameter_count != parameter_count:
            raise RuntimeError("all comparison model parameters must be trainable")
        optimizer, scheduler = _optimizer_and_scheduler(model, config)
        shuffle_generator = torch.Generator().manual_seed(config.shuffle_seed)
        history = _read_history(paths.history)

        if paths.resume.is_file():
            resume = _restore_resume(
                paths.resume,
                model,
                optimizer,
                scheduler,
                shuffle_generator,
                run_hash=run_hash,
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
                config.batch_size,
                target_scale,
            )
            initial_validation = _evaluate(
                model,
                data,
                split.validation,
                config.batch_size,
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
                run_hash=run_hash,
                best_epoch=best_epoch,
                best_validation=best_validation,
                best_parameters=best_parameters,
                initial_train_loss=initial_train_loss,
            )
            _save_learned_checkpoint(
                paths.best,
                best_parameters,
                epoch=0,
                target_scale=target_scale,
                config=config,
                metrics=best_validation,
            )

        _write_status(
            paths,
            run_hash=run_hash,
            status="running",
            epoch=start_epoch,
            best_epoch=best_epoch,
            train_loss=float(history[-1]["train_objective_normalized_mse"]),
            validation_loss=float(history[-1]["validation_normalized_mse"]),
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
                config.batch_size,
            )
            scheduler.step()
            validation = _evaluate(
                model,
                data,
                split.validation,
                config.batch_size,
                target_scale,
            )
            history.append(
                _history_row(
                    epoch,
                    learning_rate,
                    train_loss,
                    validation,
                    time.perf_counter() - epoch_clock,
                )
            )
            _write_history(paths.history, history)
            if validation["normalized_mse"] < best_validation["normalized_mse"]:
                best_epoch = epoch
                best_validation = dict(validation)
                best_parameters = _parameter_state(model)
            if epoch % config.resume_every == 0 or _STOP_REQUESTED:
                _save_resume(
                    paths.resume,
                    model,
                    optimizer,
                    scheduler,
                    shuffle_generator,
                    epoch=epoch,
                    target_scale=target_scale,
                    run_hash=run_hash,
                    best_epoch=best_epoch,
                    best_validation=best_validation,
                    best_parameters=best_parameters,
                    initial_train_loss=initial_train_loss,
                )
                _save_learned_checkpoint(
                    paths.best,
                    best_parameters,
                    epoch=best_epoch,
                    target_scale=target_scale,
                    config=config,
                    metrics=best_validation,
                )
            _write_status(
                paths,
                run_hash=run_hash,
                status="interrupted" if _STOP_REQUESTED else "running",
                epoch=epoch,
                best_epoch=best_epoch,
                train_loss=train_loss,
                validation_loss=float(validation["normalized_mse"]),
            )
            if epoch == 1 or epoch % config.progress_every == 0:
                print(
                    json.dumps(
                        {
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
                raise KeyboardInterrupt(f"run interrupted after epoch {epoch}")

        final_validation = _evaluate(
            model,
            data,
            split.validation,
            config.batch_size,
            target_scale,
        )
        _save_learned_checkpoint(
            paths.best,
            best_parameters,
            epoch=best_epoch,
            target_scale=target_scale,
            config=config,
            metrics=best_validation,
        )
        _save_learned_checkpoint(
            paths.final,
            _parameter_state(model),
            epoch=epochs,
            target_scale=target_scale,
            config=config,
            metrics=final_validation,
        )
        selected_model, selected_scale_value, _ = load_trained_group_conv(
            paths.best,
            device=device,
            dtype=config.dtype,
        )
        selected_scale = torch.tensor(
            selected_scale_value,
            dtype=data.root_force.dtype,
            device=data.root_force.device,
        )
        selected_metrics = {}
        percentage_errors = {}
        for name, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            overall = _evaluate(
                selected_model,
                data,
                indices,
                config.batch_size,
                selected_scale,
            )
            selected_metrics[name] = {
                **overall,
                "relative_rmse_percent": overall["relative_rmse"] * 100.0,
            }
            percentage_errors[name] = _component_errors(
                selected_model,
                data,
                indices,
                config.batch_size,
                selected_scale,
                zero_threshold=config.mape_zero_threshold,
            )
        probe_count = min(config.symmetry_probe_count, int(split.validation.numel()))
        probe = split.validation[:probe_count].to(data.centers.device)
        symmetry = symmetry_metrics(
            selected_model,
            data.centers[probe],
            data.frames[probe],
            tolerance=config.symmetry_tolerance,
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
            "schema_name": "tfenn_benzene_pair_group_conv_result",
            "schema_version": 1,
            "architecture_id": config.architecture_id,
            "run_hash": run_hash,
            "status": status,
            "failures": failures,
            "comparison_reference": {
                "trial_id": "trial_029",
                "candidate_id": "pair_hp_t06_p03",
                "matched_training_protocol": True,
            },
            "data": {
                "csv_path": str(data.csv_path),
                "csv_sha256": data.csv_sha256,
                "metadata_path": str(data.metadata_path),
                "metadata_sha256": data.metadata_sha256,
                "validation_path": str(config.validation_path),
                "validation_sha256": dataset_record["validation_sha256"],
                "sample_count": len(data.root_force),
                "dataset_revision": data.metadata.get("dataset_revision"),
                "opls_runtime_version": data.metadata.get("opls", {}).get(
                    "runtime_version"
                ),
            },
            "split": {
                "seed": config.split_seed,
                "fractions": list(config.split_fractions),
                **split.as_json(),
            },
            "model": {
                "model_family": "network_level_group_convolution_mlp",
                "hidden_widths": list(config.hidden_widths),
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "group_order": 12,
                "reynolds_action_count": 144,
                "fixed_tensor_artifacts_stored": False,
                "zero_output_head": config.zero_output_head,
            },
            "training": {
                "epochs_requested": epochs,
                "epochs_completed": epochs,
                "epoch_zero_train_and_validation": True,
                "later_train_metric": "batch_objective_normalized_mse",
                "later_validation_metric": "complete_partition_normalized_mse",
                "target_scale": float(target_scale.detach().cpu()),
                "target_scale_definition": "train partition force component RMS",
                "optimizer": config.optimizer,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "scheduler": "StepLR",
                "scheduler_step_size": config.scheduler_step_size,
                "scheduler_gamma": config.scheduler_gamma,
                "resume_every": config.resume_every,
                "split_seed": config.split_seed,
                "model_seed": config.model_seed,
                "shuffle_seed": config.shuffle_seed,
                "initial_train_normalized_mse": initial_train_loss,
            },
            "selection": {
                "criterion": "minimum validation normalized MSE",
                "best_epoch": best_epoch,
                "best_validation_during_training": best_validation,
                "selected_metrics": selected_metrics,
                "percentage_errors": percentage_errors,
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
                "dtype": config.dtype,
                "threads": config.threads,
                "deterministic_algorithms": config.deterministic_algorithms,
            },
        }
        _atomic_json(paths.summary, summary)
        paths.resume.unlink(missing_ok=True)
        paths.error.unlink(missing_ok=True)
        _write_status(
            paths,
            run_hash=run_hash,
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
    paths: BaselinePaths,
    error: BaseException,
    *,
    run_hash: str = "",
) -> None:
    value = {
        "schema_name": "tfenn_pair_group_conv_error",
        "schema_version": 1,
        "recorded_at_utc": _utc_now(),
        "run_hash": run_hash,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    _atomic_json(paths.error, value)
    epoch = 0
    if paths.status.is_file():
        epoch = int(_load_json(paths.status).get("epoch", 0))
    _write_status(
        paths,
        run_hash=run_hash,
        status="error",
        epoch=epoch,
        message=str(error),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--output_directory", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    config = BaselineConfig.from_path(parsed.config)
    epochs = parsed.epochs
    sample_limit = parsed.sample_limit
    output = parsed.output_directory
    if parsed.smoke:
        epochs = 1 if epochs is None else epochs
        sample_limit = 96 if sample_limit is None else sample_limit
        output = (
            REPOSITORY_ROOT / "tmp" / "group_conv_baseline_smoke"
            if output is None
            else output
        )
    paths = BaselinePaths.from_directory(output or config.output_directory)
    try:
        summary = run_baseline(
            config,
            epochs_override=epochs,
            sample_limit=sample_limit,
            output_directory=output,
            device_override=parsed.device,
        )
    except BaseException as error:
        if not isinstance(error, (KeyboardInterrupt, SystemExit)):
            _record_error(paths, error)
        raise
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary_path": str(paths.summary),
                "parameter_count": summary["model"]["parameter_count"],
                "best_epoch": summary["selection"]["best_epoch"],
                "selected_metrics": summary["selection"]["selected_metrics"],
                "percentage_errors": summary["selection"]["percentage_errors"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
