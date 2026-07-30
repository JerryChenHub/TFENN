from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from TFENN.data import BenzenePairDataset
from TFENN.models import (
    D6GroupAverageNetV1,
    D6SymmetrizedMLPBaselineV1,
    D6TensorBasisNetV1,
    D6TensorBasisNetV2,
    MLPBaselineV1,
)
from TFENN.symmetry import d6_rotations


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "benzene_pair"
    / "Benzene_10000_6.0_10.0_4.0_gamma1.csv"
)
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"
MODEL_NAMES = (
    "D6TensorBasisNetV1",
    "D6TensorBasisNetV2",
    "D6GroupAverageNetV1",
    "D6SymmetrizedMLPBaselineV1",
    "MLPBaselineV1",
)


def _construct_model(
    model_class: type[nn.Module],
    requested_parameters: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    signature = inspect.signature(model_class.__init__)
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_keywords:
        used_parameters = dict(requested_parameters)
    else:
        used_parameters = {
            name: value
            for name, value in requested_parameters.items()
            if name in signature.parameters
        }
    return model_class(**used_parameters), used_parameters


def _equivariant_parameters(config: argparse.Namespace) -> dict[str, Any]:
    return {
        "x_in_channels": 1,
        "r_in_channels": 1,
        "x_hidden_channels": config.x_hidden_channels,
        "r_hidden_channels": config.r_hidden_channels,
        "num_x_layers": config.num_x_layers,
        "num_r_layers": config.num_r_layers,
        "r_to_x_channels": config.r_to_x_channels,
        "out_channels": 2,
        "vector_activation": config.vector_activation,
        "matrix_gate": config.matrix_gate,
        "head_activation": config.head_activation,
        "num_head_layers": config.num_head_layers,
        "head_hidden_channels": config.head_hidden_channels,
        "init_policy": config.init_policy,
    }


def build_d6_tensor_basis_net_v1(
    config: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    requested = _equivariant_parameters(config)
    model, used = _construct_model(D6TensorBasisNetV1, requested)
    return model, requested, used


def build_d6_tensor_basis_net_v2(
    config: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    requested = _equivariant_parameters(config)
    model, used = _construct_model(D6TensorBasisNetV2, requested)
    return model, requested, used


def build_d6_group_average_net_v1(
    config: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    requested = _equivariant_parameters(config)
    model, used = _construct_model(D6GroupAverageNetV1, requested)
    return model, requested, used


def build_d6_symmetrized_mlp_baseline_v1(
    config: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    requested = {
        "hidden_dim": config.hidden_dim,
        "num_hidden_layers": config.num_hidden_layers,
        "activation": config.activation,
        "out_channels": 2,
        "init_policy": config.init_policy,
    }
    model, used = _construct_model(D6SymmetrizedMLPBaselineV1, requested)
    return model, requested, used


def build_mlp_baseline_v1(
    config: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    requested = {
        "hidden_dim": config.hidden_dim,
        "num_hidden_layers": config.num_hidden_layers,
        "activation": config.activation,
        "out_channels": 2,
        "init_policy": config.init_policy,
    }
    model, used = _construct_model(MLPBaselineV1, requested)
    return model, requested, used


MODEL_BUILDERS: dict[
    str,
    Callable[
        [argparse.Namespace],
        tuple[nn.Module, dict[str, Any], dict[str, Any]],
    ],
] = {
    "D6TensorBasisNetV1": build_d6_tensor_basis_net_v1,
    "D6TensorBasisNetV2": build_d6_tensor_basis_net_v2,
    "D6GroupAverageNetV1": build_d6_group_average_net_v1,
    "D6SymmetrizedMLPBaselineV1": build_d6_symmetrized_mlp_baseline_v1,
    "MLPBaselineV1": build_mlp_baseline_v1,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_datasets(
    dataset: BenzenePairDataset,
    subset_size: int | None,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset, Subset, list[int]]:
    available = len(dataset)
    selected_count = available if subset_size is None else min(subset_size, available)
    if selected_count < 2:
        raise ValueError("At least two samples are required")

    generator = torch.Generator().manual_seed(seed)
    selected_indices = torch.randperm(available, generator=generator)[
        :selected_count
    ].tolist()
    validation_count = max(1, round(selected_count * validation_fraction))
    training_count = selected_count - validation_count
    if training_count < 1:
        raise ValueError("validation_fraction leaves no training samples")

    training_indices = selected_indices[:training_count]
    validation_indices = selected_indices[training_count:]
    return (
        Subset(dataset, training_indices),
        Subset(dataset, validation_indices),
        selected_indices,
    )


def _normalize_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 2 and target.shape[-1] == 6:
        return target.reshape(target.shape[0], 2, 3)
    if target.ndim == 3 and target.shape[-2:] == (2, 3):
        return target
    raise ValueError(f"Expected target shape (B, 6) or (B, 2, 3), got {target.shape}")


def _predict(
    model: nn.Module,
    displacement: torch.Tensor,
    relative_rotation: torch.Tensor,
) -> torch.Tensor:
    prediction = model(displacement, relative_rotation)
    return _normalize_target(prediction)


def _evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for (displacement, relative_rotation), target in data_loader:
            displacement = displacement.to(device)
            relative_rotation = relative_rotation.to(device)
            target = _normalize_target(target.to(device))
            prediction = _predict(model, displacement, relative_rotation)
            batch_loss = criterion(prediction, target)
            total_loss += batch_loss.item() * displacement.shape[0]
            total_samples += displacement.shape[0]
    return total_loss / total_samples


def _train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for (displacement, relative_rotation), target in data_loader:
        displacement = displacement.to(device)
        relative_rotation = relative_rotation.to(device)
        target = _normalize_target(target.to(device))

        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, displacement, relative_rotation)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * displacement.shape[0]
        total_samples += displacement.shape[0]
    return total_loss / total_samples


def measure_d6_equivariance(
    model: nn.Module,
    displacement: torch.Tensor,
    relative_rotation: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    group = d6_rotations(
        dtype=displacement.dtype,
        device=displacement.device,
    )
    identity = torch.eye(
        3,
        dtype=displacement.dtype,
        device=displacement.device,
    )
    identity_index = int(
        (group - identity).square().sum(dim=(-2, -1)).argmin()
    )

    with torch.inference_mode():
        reference = _predict(model, displacement, relative_rotation)
        group_order = group.shape[0]
        sample_count = displacement.shape[0]

        transformed_displacement = torch.einsum(
            "bi,lij->lbj",
            displacement,
            group,
        )
        transformed_displacement = (
            transformed_displacement[:, None]
            .expand(group_order, group_order, sample_count, 3)
            .reshape(group_order * group_order * sample_count, 3)
        )
        transformed_rotation = torch.einsum(
            "lji,bjk,rkm->lrbim",
            group,
            relative_rotation,
            group,
        ).reshape(group_order * group_order * sample_count, 3, 3)
        transformed = _predict(
            model,
            transformed_displacement,
            transformed_rotation,
        ).reshape(group_order, group_order, sample_count, 2, 3)

        expected = torch.einsum(
            "bci,lij->lbcj",
            reference,
            group,
        )
        expected = expected[:, None].expand(
            group_order,
            group_order,
            sample_count,
            2,
            3,
        )
        absolute_error = (transformed - expected).abs()
        joint_max = float(absolute_error.max())
        left_max = float(absolute_error[:, identity_index].max())
        right_max = float(absolute_error[identity_index].max())
        reference_scale = max(
            float(reference.abs().max()),
            torch.finfo(reference.dtype).eps,
        )

    return {
        "convention": "row_vectors",
        "group_order": int(group_order),
        "checked_group_pairs": int(group_order * group_order),
        "checked_samples_per_pair": int(sample_count),
        "joint_max_abs": joint_max,
        "joint_max_relative": joint_max / reference_scale,
        "left_equivariance_max_abs": left_max,
        "right_invariance_max_abs": right_max,
    }


def _read_data_metadata(dataset: BenzenePairDataset) -> Any:
    metadata = getattr(dataset, "metadata", None)
    if metadata is None:
        return None
    return _json_ready(metadata)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y_%m_%dT%H_%M_%SZ")


def _safe_experiment_name(name: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in name)
    return cleaned.strip("_") or "experiment"


def _resolve_log_path(config: argparse.Namespace, experiment_name: str) -> Path:
    if config.log_path is not None:
        log_path = config.log_path.resolve()
    else:
        log_path = DEFAULT_LOG_DIR / f"{_safe_experiment_name(experiment_name)}.json"
    if log_path.exists():
        raise FileExistsError(f"Log already exists: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def _write_log(log_path: Path, content: dict[str, Any]) -> None:
    log_path.write_text(
        json.dumps(_json_ready(content), indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )


def run_experiment(config: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    started_at = _utc_timestamp()
    experiment_name = config.experiment_name or (
        f"{config.model_name}_{started_at}"
    )
    log_path = _resolve_log_path(config, experiment_name)

    _set_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    device = torch.device(config.device)
    dtype = torch.float64

    data_path = config.data_path.resolve()
    try:
        data_log_path = data_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        data_log_path = str(data_path)
    dataset = BenzenePairDataset(data_path, dtype=dtype)
    training_dataset, validation_dataset, selected_indices = _prepare_datasets(
        dataset,
        config.subset_size,
        config.validation_fraction,
        config.seed,
    )

    loader_generator = torch.Generator().manual_seed(config.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    training_evaluation_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model, requested_parameters, used_parameters = MODEL_BUILDERS[
        config.model_name
    ](config)
    model = model.to(device=device, dtype=dtype)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    criterion = nn.SmoothL1Loss(beta=config.smooth_l1_beta)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    initial_training_loss = _evaluate_loss(
        model,
        training_evaluation_loader,
        criterion,
        device,
    )
    initial_validation_loss = _evaluate_loss(
        model,
        validation_loader,
        criterion,
        device,
    )
    history = [
        {
            "epoch": 0,
            "training_loss": initial_training_loss,
            "validation_loss": initial_validation_loss,
        }
    ]

    training_started = time.perf_counter()
    final_training_loss = initial_training_loss
    final_validation_loss = initial_validation_loss
    for epoch in range(1, config.epochs + 1):
        optimization_loss = _train_one_epoch(
            model,
            training_loader,
            criterion,
            optimizer,
            device,
        )
        final_training_loss = _evaluate_loss(
            model,
            training_evaluation_loader,
            criterion,
            device,
        )
        final_validation_loss = _evaluate_loss(
            model,
            validation_loader,
            criterion,
            device,
        )
        if epoch % config.history_stride == 0 or epoch == config.epochs:
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": final_training_loss,
                    "validation_loss": final_validation_loss,
                    "optimization_loss": optimization_loss,
                }
            )
        print(
            f"epoch={epoch:04d} "
            f"training_loss={final_training_loss:.8g} "
            f"validation_loss={final_validation_loss:.8g}"
        )
    training_seconds = time.perf_counter() - training_started

    symmetry_loader = DataLoader(
        validation_dataset,
        batch_size=min(config.symmetry_samples, len(validation_dataset)),
        shuffle=False,
    )
    (symmetry_x, symmetry_r), _ = next(iter(symmetry_loader))
    symmetry_errors = measure_d6_equivariance(
        model,
        symmetry_x.to(device),
        symmetry_r.to(device),
    )
    expected_by_design = config.model_name != "MLPBaselineV1"
    symmetry_errors["expected_by_design"] = expected_by_design
    symmetry_errors["tolerance"] = config.symmetry_tolerance
    symmetry_errors["passed"] = (
        symmetry_errors["joint_max_abs"] <= config.symmetry_tolerance
    )

    training_losses = [entry["training_loss"] for entry in history]
    validation_losses = [entry["validation_loss"] for entry in history]
    training_loss_decreased = final_training_loss < initial_training_loss
    validation_loss_decreased = final_validation_loss < initial_validation_loss

    log = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_timestamp(),
        "model": {
            "name": config.model_name,
            "structure": (
                model.architecture_summary()
                if hasattr(model, "architecture_summary")
                else str(model)
            ),
            "parameter_count": parameter_count,
            "parameters_requested": requested_parameters,
            "parameters_used": used_parameters,
        },
        "data": {
            "path": data_log_path,
            "available_samples": len(dataset),
            "selected_samples": len(selected_indices),
            "training_samples": len(training_dataset),
            "validation_samples": len(validation_dataset),
            "selected_index_min": min(selected_indices),
            "selected_index_max": max(selected_indices),
            "generation_metadata": _read_data_metadata(dataset),
        },
        "training": {
            "seed": config.seed,
            "optimizer": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "loss": "SmoothL1Loss",
            "smooth_l1_beta": config.smooth_l1_beta,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "validation_fraction": config.validation_fraction,
            "history_stride": config.history_stride,
            "duration_seconds": training_seconds,
            "loss_history": history,
            "initial_training_loss": initial_training_loss,
            "final_training_loss": final_training_loss,
            "minimum_recorded_training_loss": min(training_losses),
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "minimum_recorded_validation_loss": min(validation_losses),
            "loss_decreased": training_loss_decreased,
            "validation_loss_decreased": validation_loss_decreased,
        },
        "symmetry": symmetry_errors,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "dtype": str(dtype),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "num_threads": torch.get_num_threads(),
        },
    }
    _write_log(log_path, log)

    if config.require_loss_decrease and not training_loss_decreased:
        raise RuntimeError(f"Training loss did not decrease. Log: {log_path}")
    if (
        config.require_symmetry
        and expected_by_design
        and not symmetry_errors["passed"]
    ):
        raise RuntimeError(f"D6 equivariance check failed. Log: {log_path}")
    return log, log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare benzene pair force and moment models.",
    )
    parser.add_argument("--model_name", choices=MODEL_NAMES, default=MODEL_NAMES[0])
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--experiment_name")
    parser.add_argument("--log_path", type=Path)
    parser.add_argument("--subset_size", type=int, default=512)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--smooth_l1_beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--history_stride", type=int, default=1)
    parser.add_argument("--symmetry_samples", type=int, default=4)
    parser.add_argument("--symmetry_tolerance", type=float, default=1e-9)
    parser.add_argument("--require_loss_decrease", action="store_true")
    parser.add_argument("--require_symmetry", action="store_true")

    parser.add_argument("--x_hidden_channels", type=int, default=16)
    parser.add_argument("--r_hidden_channels", type=int, default=16)
    parser.add_argument("--num_x_layers", type=int, default=2)
    parser.add_argument("--num_r_layers", type=int, default=1)
    parser.add_argument("--r_to_x_channels", type=int, default=16)
    parser.add_argument("--num_head_layers", type=int, default=2)
    parser.add_argument("--head_hidden_channels", type=int, default=16)
    parser.add_argument("--vector_activation", default="sigmoid_centered")
    parser.add_argument("--matrix_gate", default="block_norm")
    parser.add_argument("--head_activation", default="leaky_relu")
    parser.add_argument("--init_policy", default="xavier_uniform")

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--activation", default="leaky_relu")
    return parser


def _validate_config(config: argparse.Namespace) -> None:
    if config.subset_size is not None and config.subset_size < 2:
        raise ValueError("subset_size must be at least two")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.history_stride < 1:
        raise ValueError("history_stride must be positive")
    if config.symmetry_samples < 1:
        raise ValueError("symmetry_samples must be positive")
    if config.num_threads < 1:
        raise ValueError("num_threads must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")


def main() -> int:
    config = build_parser().parse_args()
    _validate_config(config)
    log, log_path = run_experiment(config)
    summary = {
        "log_path": str(log_path),
        "model": log["model"]["name"],
        "loss_decreased": log["training"]["loss_decreased"],
        "symmetry_passed": log["symmetry"]["passed"],
        "joint_max_abs": log["symmetry"]["joint_max_abs"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
