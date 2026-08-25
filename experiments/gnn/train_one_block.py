from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.benzene_pair.data.benzene_cluster import (
    load_benzene_cluster_csv,
)

from .one_block_model import OneBlockForceGNN


DEFAULT_DATA = (
    Path(__file__).resolve().parent
    / "data"
    / "five_benzene_opls_2_0_0_1k_v1.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "runs"
    / "one_block_seed_20260824_v2"
)


@dataclass(frozen=True)
class Evaluation:
    normalized_mse: float
    physical_mse: float
    physical_component_mae: float
    physical_vector_mae: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<i8").tobytes()
    ).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _split_indices(
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_count < 10:
        raise ValueError("at least ten complete samples are required")
    order = np.random.default_rng(seed).permutation(sample_count)
    train_count = int(math.floor(0.8 * sample_count))
    validation_count = int(math.floor(0.1 * sample_count))
    train = order[:train_count]
    validation = order[train_count : train_count + validation_count]
    test = order[train_count + validation_count :]
    return train, validation, test


def _tensor_subset(value: np.ndarray, indices: np.ndarray) -> Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(value[indices], dtype=np.float64)
    )


def _loader(
    centers: Tensor,
    rotations: Tensor,
    forces: Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(centers, rotations, forces),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _evaluate(
    model: OneBlockForceGNN,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> Evaluation:
    model.eval()
    squared_sum = 0.0
    absolute_sum = 0.0
    vector_sum = 0.0
    component_count = 0
    vector_count = 0
    scale = float(model.force_scale.detach().cpu())
    with torch.no_grad():
        for centers, rotations, target in loader:
            centers = centers.to(device)
            rotations = rotations.to(device)
            target = target.to(device)
            prediction = model.normalized_forces(centers, rotations)
            difference = prediction - target
            squared_sum += float(difference.square().sum().cpu())
            physical_difference = difference * model.force_scale
            absolute_sum += float(physical_difference.abs().sum().cpu())
            vector_sum += float(
                torch.linalg.vector_norm(
                    physical_difference,
                    dim=-1,
                ).sum().cpu()
            )
            component_count += difference.numel()
            vector_count += difference.shape[0] * difference.shape[1]
    normalized_mse = squared_sum / component_count
    return Evaluation(
        normalized_mse,
        normalized_mse * scale * scale,
        absolute_sum / component_count,
        vector_sum / vector_count,
    )


def _evaluation_dict(value: Evaluation) -> dict[str, float]:
    return {
        "normalized_mse": value.normalized_mse,
        "physical_mse_kcal2_mol2_A2": value.physical_mse,
        "physical_component_mae_kcal_mol_A": value.physical_component_mae,
        "physical_vector_mae_kcal_mol_A": value.physical_vector_mae,
    }


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf_8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    data_path = args.data.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"output path already exists: {output_path}")
    output_path.mkdir(parents=True)
    arrays = load_benzene_cluster_csv(data_path)
    if arrays.molecule_count != 5:
        raise ValueError("the one block experiment requires five molecules")
    train_indices, validation_indices, test_indices = _split_indices(
        len(arrays),
        args.seed,
    )
    train_centers = _tensor_subset(arrays.centers, train_indices)
    train_rotations = _tensor_subset(arrays.rotations, train_indices)
    train_forces_physical = _tensor_subset(arrays.forces, train_indices)
    force_scale = float(torch.sqrt(train_forces_physical.square().mean()))
    train_forces = train_forces_physical / force_scale
    validation_centers = _tensor_subset(arrays.centers, validation_indices)
    validation_rotations = _tensor_subset(arrays.rotations, validation_indices)
    validation_forces = (
        _tensor_subset(arrays.forces, validation_indices) / force_scale
    )
    test_centers = _tensor_subset(arrays.centers, test_indices)
    test_rotations = _tensor_subset(arrays.rotations, test_indices)
    test_forces = _tensor_subset(arrays.forces, test_indices) / force_scale
    train_loader = _loader(
        train_centers,
        train_rotations,
        train_forces,
        args.batch_size,
        True,
        args.seed,
    )
    train_evaluation_loader = _loader(
        train_centers,
        train_rotations,
        train_forces,
        args.batch_size,
        False,
        args.seed,
    )
    validation_loader = _loader(
        validation_centers,
        validation_rotations,
        validation_forces,
        args.batch_size,
        False,
        args.seed,
    )
    test_loader = _loader(
        test_centers,
        test_rotations,
        test_forces,
        args.batch_size,
        False,
        args.seed,
    )
    device = torch.device(args.device)
    model = OneBlockForceGNN(
        force_scale,
        molecule_count=5,
        distance_scale=args.distance_scale,
        gate_width=args.gate_width,
        dtype=torch.float64,
    ).to(device)
    if model.message_block_count != 1:
        raise RuntimeError("the experiment must contain exactly one Message Block")
    initial_train = _evaluate(model, train_evaluation_loader, device)
    initial_validation = _evaluate(model, validation_loader, device)
    if initial_validation.normalized_mse <= 0.0:
        raise RuntimeError("initial validation loss must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    history: list[dict[str, float | int]] = [
        {
            "epoch": 0,
            "train_normalized_mse": initial_train.normalized_mse,
            "validation_normalized_mse": initial_validation.normalized_mse,
            "validation_physical_mse": initial_validation.physical_mse,
            "validation_reduction_fraction": 0.0,
            "epoch_seconds": 0.0,
        }
    ]
    best_loss = initial_validation.normalized_mse
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        squared_sum = 0.0
        component_count = 0
        for centers, rotations, target in train_loader:
            centers = centers.to(device)
            rotations = rotations.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.normalized_forces(centers, rotations)
            loss = loss_function(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training loss became nonfinite")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise RuntimeError("a parameter gradient became nonfinite")
            optimizer.step()
            squared_sum += float(loss.detach().cpu()) * target.numel()
            component_count += target.numel()
        train_loss = squared_sum / component_count
        validation = _evaluate(model, validation_loader, device)
        elapsed = time.perf_counter() - started
        reduction = 1.0 - (
            validation.normalized_mse / initial_validation.normalized_mse
        )
        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": train_loss,
                "validation_normalized_mse": validation.normalized_mse,
                "validation_physical_mse": validation.physical_mse,
                "validation_reduction_fraction": reduction,
                "epoch_seconds": elapsed,
            }
        )
        if validation.normalized_mse < best_loss:
            best_loss = validation.normalized_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_normalized_mse": train_loss,
                    "validation_normalized_mse": validation.normalized_mse,
                    "validation_reduction_percent": 100.0 * reduction,
                    "epoch_seconds": elapsed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    model.load_state_dict(best_state)
    final_train = _evaluate(model, train_evaluation_loader, device)
    best_validation = _evaluate(model, validation_loader, device)
    test = _evaluate(model, test_loader, device)
    reduction_fraction = 1.0 - (
        best_validation.normalized_mse / initial_validation.normalized_mse
    )
    passed = reduction_fraction >= args.required_reduction
    history_path = output_path / "history.csv"
    summary_path = output_path / "summary.json"
    checkpoint_path = output_path / "best_checkpoint.pt"
    _write_history(history_path, history)
    checkpoint = {
        "model_state": best_state,
        "optimizer_state": best_optimizer_state,
        "best_epoch": best_epoch,
        "force_scale": force_scale,
        "model": {
            "message_block_count": model.message_block_count,
            "molecule_count": model.molecule_count,
            "distance_scale": model.distance_scale,
            "gate_width": args.gate_width,
            "dtype": "float64",
        },
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "schema_name": "tfenn_minimal_one_block_gnn_training",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "acceptance": {
            "required_validation_loss_reduction_fraction": args.required_reduction,
            "achieved_validation_loss_reduction_fraction": reduction_fraction,
            "best_loss_at_most_initial_times": 1.0 - args.required_reduction,
        },
        "data": {
            "path": str(data_path),
            "sha256": _sha256(data_path),
            "sample_count": len(arrays),
            "molecule_count": arrays.molecule_count,
            "force_scale_kcal_mol_A": force_scale,
            "target_scaling": "one shared scalar across Cartesian components",
        },
        "split": {
            "unit": "complete sample_id graph",
            "seed": args.seed,
            "train_count": len(train_indices),
            "validation_count": len(validation_indices),
            "test_count": len(test_indices),
            "train_indices_sha256": _index_sha256(train_indices),
            "validation_indices_sha256": _index_sha256(validation_indices),
            "test_indices_sha256": _index_sha256(test_indices),
        },
        "model": {
            "message_block_count": model.message_block_count,
            "trainable_parameter_count": model.trainable_parameter_count,
            "graph": "complete directed graph without self edges",
            "edge_count": model.edge_count,
            "aggregation": "typed incoming sum",
            "node_update": "none after the terminal block",
            "gate_trunk": "Linear then SiLU",
            "gate_width": args.gate_width,
            "invariants": [
                "constant",
                "scaled edge distance",
                "same TypeKey B_0 pose inner product",
                "same TypeKey B_1 pose inner product",
            ],
            "distance_scale_A": args.distance_scale,
            "a_mid_channels": 1,
            "b_wide_channels_per_TypeKey": 2,
            "b_out_channels_per_TypeKey": 1,
            "a_out_channels": 1,
            "covariant_path_arity": "unary only",
            "coefficient_output": "signed linear identity",
            "channel_mixing": "dense",
            "flow_policy": "EXPLICIT_REUSE_FLOW",
            "raw_covariant_access": "EXPLICIT_RAW_REREAD",
            "invariant_context": "RAW_ONLY_INVARIANTS",
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "device": str(device),
            "torch_version": torch.__version__,
            "best_epoch": best_epoch,
        },
        "metrics": {
            "initial_train": _evaluation_dict(initial_train),
            "initial_validation": _evaluation_dict(initial_validation),
            "best_train": _evaluation_dict(final_train),
            "best_validation": _evaluation_dict(best_validation),
            "test_at_best_validation": _evaluation_dict(test),
        },
        "artifacts": {
            "history": str(history_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--distance-scale", type=float, default=8.0)
    parser.add_argument("--gate-width", type=int, default=8)
    parser.add_argument("--required-reduction", type=float, default=0.20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if arguments.learning_rate <= 0.0:
        parser.error("learning rate must be positive")
    if not 0.0 < arguments.required_reduction < 1.0:
        parser.error("required reduction must lie between zero and one")
    return arguments


if __name__ == "__main__":
    result = train(parse_args())
    if not result["passed"]:
        raise SystemExit(2)
