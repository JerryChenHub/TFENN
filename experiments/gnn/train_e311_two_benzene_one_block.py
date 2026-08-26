from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .e311_one_block_gnn import (
    E311OneBlockGNN,
    E311OneBlockGNNConfig,
)
from .two_benzene_training_support import (
    DEFAULT_DATA,
    DEFAULT_METADATA,
    DEFAULT_SEED,
    DEFAULT_VALIDATION,
    TEST_COUNT,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    Evaluation,
    _evaluate,
    _evaluation_dict,
    _index_sha256,
    _load_prepared_data,
    _loader,
    _normalized_single_edge_prediction,
    _set_seed,
    _sha256,
    _single_edge_targets,
    _split_sample_ids,
    _tensor_subset,
    _write_history,
    _write_json,
)


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent.parent
DEFAULT_OUTPUT = (
    MODULE_DIRECTORY
    / "runs"
    / "two_benzene_e311_multibody_one_block_seed_20260824_300e"
)
SPECIFIED_DESIGN_SOURCE = (
    REPOSITORY_ROOT
    / "src"
    / "TFENN"
    / "models"
    / "e311_multibody_message_block_v1.py"
)
WRAPPER_SOURCE = MODULE_DIRECTORY / "e311_one_block_gnn.py"
SHARED_TRAINING_SOURCE = MODULE_DIRECTORY / "two_benzene_training_support.py"
SPECIFIED_DESIGN_SHA256 = (
    "14d08905d591faf0595119472685391c5c982e74546c18f190d30c76dff287ff"
)
MODEL_FAMILY = "e311_multibody_one_block_v1"
FORMAL_EPOCHS = 300
BATCH_SIZE = 100
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1.0e-6
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 20
MINIMUM_LEARNING_RATE = 1.0e-5
TRAIN_LOSS_FRACTION = 0.05
THREADS = 8
DTYPE_NAME = "float32"
DEVICE_NAME = "cpu"


def _source_provenance() -> dict[str, Any]:
    actual_design_hash = _sha256(SPECIFIED_DESIGN_SOURCE)
    if actual_design_hash != SPECIFIED_DESIGN_SHA256:
        raise RuntimeError(
            "the E311 multibody design source does not match the specified SHA256"
        )
    trainer_source = Path(__file__).resolve()
    return {
        "model_family": MODEL_FAMILY,
        "specified_design_source": {
            "path": str(SPECIFIED_DESIGN_SOURCE.resolve()),
            "expected_sha256": SPECIFIED_DESIGN_SHA256,
            "actual_sha256": actual_design_hash,
            "verification_status": "matched",
            "verified": True,
        },
        "wrapper_source": {
            "path": str(WRAPPER_SOURCE.resolve()),
            "sha256": _sha256(WRAPPER_SOURCE),
        },
        "trainer_source": {
            "path": str(trainer_source),
            "sha256": _sha256(trainer_source),
        },
        "shared_training_utility_source": {
            "path": str(SHARED_TRAINING_SOURCE.resolve()),
            "sha256": _sha256(SHARED_TRAINING_SOURCE),
        },
    }


def _checkpoint_payload(
    *,
    purpose: str,
    epoch: int,
    model: E311OneBlockGNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    force_scale: float,
    config: E311OneBlockGNNConfig,
    data_sha256: str,
    split_hashes: dict[str, str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "purpose": purpose,
        "epoch": epoch,
        "model_family": MODEL_FAMILY,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "force_scale": force_scale,
        "model_config": asdict(config),
        "dtype": DTYPE_NAME,
        "device": DEVICE_NAME,
        "seed": DEFAULT_SEED,
        "data_sha256": data_sha256,
        "split_indices_sha256": split_hashes,
        "source_provenance": provenance,
    }


def _evaluate_model(
    model: E311OneBlockGNN,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> Evaluation:
    return _evaluate(cast(Any, model), loader, device)


def train(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(DEFAULT_SEED)
    torch.set_num_threads(THREADS)
    data_path = DEFAULT_DATA.resolve()
    metadata_path = DEFAULT_METADATA.resolve()
    validation_path = DEFAULT_VALIDATION.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"output path already exists: {output_path}")
    provenance = _source_provenance()
    arrays, metadata, validation, data_hash = _load_prepared_data(
        data_path,
        metadata_path,
        validation_path,
    )
    targets_physical = _single_edge_targets(arrays)
    train_ids, validation_ids, test_ids = _split_sample_ids(
        len(arrays),
        DEFAULT_SEED,
    )
    if (
        len(train_ids) != TRAIN_COUNT
        or len(validation_ids) != VALIDATION_COUNT
        or len(test_ids) != TEST_COUNT
    ):
        raise RuntimeError("the fixed complete graph split counts are invalid")
    split_hashes = {
        "train": _index_sha256(train_ids),
        "validation": _index_sha256(validation_ids),
        "test": _index_sha256(test_ids),
    }
    force_scale = float(
        np.sqrt(np.mean(np.square(targets_physical[train_ids])))
    )
    if not math.isfinite(force_scale) or force_scale <= 0.0:
        raise ValueError("training force scale must be finite and positive")
    output_path.mkdir(parents=True)
    split_path = output_path / "split_indices.npz"
    run_manifest_path = output_path / "run_manifest.json"
    history_path = output_path / "history.csv"
    best_checkpoint_path = output_path / "best_checkpoint.pt"
    final_checkpoint_path = output_path / "final_checkpoint.pt"
    summary_path = output_path / "summary.json"
    np.savez_compressed(
        split_path,
        train_sample_id=train_ids,
        validation_sample_id=validation_ids,
        test_sample_id=test_ids,
    )
    run_manifest = {
        "schema_name": "tfenn_e311_two_benzene_one_block_run_manifest",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": MODEL_FAMILY,
        "source_provenance": provenance,
        "data": {
            "path": str(data_path),
            "sha256": data_hash,
            "metadata_path": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "validation_path": str(validation_path),
            "validation_sha256": _sha256(validation_path),
            "validation_passed": validation["passed"],
            "sample_count": len(arrays),
            "molecule_count": arrays.molecule_count,
            "dataset": metadata.get("dataset"),
            "dataset_revision": metadata.get("dataset_revision"),
            "force_target": "forces[:, 0, None, :]",
            "force_scale_kcal_mol_A": force_scale,
        },
        "split": {
            "unit": "complete sample_id graph",
            "seed": DEFAULT_SEED,
            "train_count": len(train_ids),
            "validation_count": len(validation_ids),
            "test_count": len(test_ids),
            "indices_sha256": split_hashes,
            "artifact": str(split_path),
        },
        "training_protocol": {
            "requested_epochs": args.epochs,
            "formal_epochs": FORMAL_EPOCHS,
            "formal_protocol": args.epochs == FORMAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "minimum_learning_rate": MINIMUM_LEARNING_RATE,
            "dtype": DTYPE_NAME,
            "device": DEVICE_NAME,
            "threads": THREADS,
            "early_stopping": False,
        },
    }
    _write_json(run_manifest_path, run_manifest)
    dtype = torch.float32

    def subset_loader(
        indices: np.ndarray,
        *,
        shuffle: bool,
    ) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        return _loader(
            _tensor_subset(arrays.centers, indices, dtype),
            _tensor_subset(arrays.rotations, indices, dtype),
            _tensor_subset(targets_physical, indices, dtype) / force_scale,
            BATCH_SIZE,
            shuffle,
            DEFAULT_SEED,
        )

    train_loader = subset_loader(train_ids, shuffle=True)
    train_evaluation_loader = subset_loader(train_ids, shuffle=False)
    validation_loader = subset_loader(validation_ids, shuffle=False)
    test_loader = subset_loader(test_ids, shuffle=False)
    device = torch.device(DEVICE_NAME)
    config = E311OneBlockGNNConfig()
    model = E311OneBlockGNN(force_scale, config, dtype).to(device)
    if model.message_block_count != 1:
        raise RuntimeError("the experiment must contain one E311 Message Block")
    if model.pair_count != 1 or model.config.molecule_count != 2:
        raise RuntimeError("the experiment must contain two nodes and one pair")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MINIMUM_LEARNING_RATE,
    )
    loss_function = nn.MSELoss()
    initial_train = _evaluate_model(model, train_evaluation_loader, device)
    initial_validation = _evaluate_model(model, validation_loader, device)
    history: list[dict[str, float | int | None]] = [
        {
            "epoch": 0,
            "optimization_normalized_mse": None,
            "train_normalized_mse": initial_train.normalized_mse,
            "validation_normalized_mse": initial_validation.normalized_mse,
            "train_ratio_to_initial": 1.0,
            "learning_rate": LEARNING_RATE,
            "epoch_seconds": 0.0,
        }
    ]
    _write_history(history_path, history)
    best_epoch = 0
    best_validation_loss = initial_validation.normalized_mse
    torch.save(
        _checkpoint_payload(
            purpose="lowest validation loss checkpoint",
            epoch=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            force_scale=force_scale,
            config=config,
            data_sha256=data_hash,
            split_hashes=split_hashes,
            provenance=provenance,
        ),
        best_checkpoint_path,
    )
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        squared_error_sum = 0.0
        component_count = 0
        for centers, rotations, target in train_loader:
            centers = centers.to(device)
            rotations = rotations.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _normalized_single_edge_prediction(
                cast(Any, model),
                centers,
                rotations,
            )
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
            squared_error_sum += float(loss.detach().cpu()) * target.numel()
            component_count += target.numel()
        optimization_loss = squared_error_sum / component_count
        train_evaluation = _evaluate_model(
            model,
            train_evaluation_loader,
            device,
        )
        validation_evaluation = _evaluate_model(
            model,
            validation_loader,
            device,
        )
        scheduler.step(validation_evaluation.normalized_mse)
        elapsed = time.perf_counter() - started
        train_ratio = (
            train_evaluation.normalized_mse / initial_train.normalized_mse
        )
        history.append(
            {
                "epoch": epoch,
                "optimization_normalized_mse": optimization_loss,
                "train_normalized_mse": train_evaluation.normalized_mse,
                "validation_normalized_mse": (
                    validation_evaluation.normalized_mse
                ),
                "train_ratio_to_initial": train_ratio,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": elapsed,
            }
        )
        _write_history(history_path, history)
        if validation_evaluation.normalized_mse < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_evaluation.normalized_mse
            torch.save(
                _checkpoint_payload(
                    purpose="lowest validation loss checkpoint",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    force_scale=force_scale,
                    config=config,
                    data_sha256=data_hash,
                    split_hashes=split_hashes,
                    provenance=provenance,
                ),
                best_checkpoint_path,
            )
        print(
            json.dumps(
                {
                    "model_family": MODEL_FAMILY,
                    "epoch": epoch,
                    "requested_final_epoch": args.epochs,
                    "optimization_normalized_mse": optimization_loss,
                    "train_normalized_mse": train_evaluation.normalized_mse,
                    "validation_normalized_mse": (
                        validation_evaluation.normalized_mse
                    ),
                    "train_ratio_to_initial": train_ratio,
                    "train_loss_at_or_below_five_percent": (
                        train_ratio <= TRAIN_LOSS_FRACTION
                    ),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": elapsed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if len(history) != args.epochs + 1:
        raise RuntimeError("history does not cover epoch zero through final epoch")
    final_train = _evaluate_model(model, train_evaluation_loader, device)
    final_validation = _evaluate_model(model, validation_loader, device)
    final_test = _evaluate_model(model, test_loader, device)
    final_ratio = final_train.normalized_mse / initial_train.normalized_mse
    criterion_passed = final_ratio <= TRAIN_LOSS_FRACTION
    torch.save(
        _checkpoint_payload(
            purpose="model at the exact requested final epoch",
            epoch=args.epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            force_scale=force_scale,
            config=config,
            data_sha256=data_hash,
            split_hashes=split_hashes,
            provenance=provenance,
        ),
        final_checkpoint_path,
    )
    summary = {
        "schema_name": "tfenn_e311_two_benzene_one_block_training",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "model_family": MODEL_FAMILY,
        "source_provenance": provenance,
        "train_loss_criterion_passed": criterion_passed,
        "criterion": {
            "metric": "full training split normalized single edge force MSE",
            "initial_epoch": 0,
            "final_epoch": args.epochs,
            "required_fraction_of_initial": TRAIN_LOSS_FRACTION,
            "initial_train_normalized_mse": initial_train.normalized_mse,
            "target_train_normalized_mse": (
                TRAIN_LOSS_FRACTION * initial_train.normalized_mse
            ),
            "final_train_normalized_mse": final_train.normalized_mse,
            "final_to_initial_ratio": final_ratio,
            "passed": criterion_passed,
        },
        "data": run_manifest["data"],
        "split": run_manifest["split"],
        "model": {
            **model.block_configuration(),
            "model_family": MODEL_FAMILY,
            "pair_count": model.pair_count,
            "trainable_parameter_count": model.trainable_parameter_count,
            "initial_hidden_b": "all zero receiver local state",
            "initial_edge_a": "all zero world frame state",
            "invariant_normalization": (
                "cumulative RunningRMS updated only during training mode"
            ),
        },
        "training": cast(dict[str, Any], run_manifest["training_protocol"])
        | {
            "completed_epochs": args.epochs,
            "termination": "fixed epoch budget completed",
            "torch_version": torch.__version__,
            "best_epoch": best_epoch,
            "best_validation_normalized_mse": best_validation_loss,
        },
        "metrics": {
            "definitions": {
                "relative_frobenius_norm_error_percent": (
                    "100 * norm(prediction minus target) / norm(target) over "
                    "the full split"
                ),
                "per_graph_relative_norm_error_percent": (
                    "100 * norm(prediction minus target) / norm(target) for "
                    "each graph"
                ),
                "per_graph_force_magnitude_percent_error": (
                    "100 * abs(norm(prediction) minus norm(target)) / "
                    "norm(target) for each graph"
                ),
            },
            "initial_train": _evaluation_dict(initial_train),
            "initial_validation": _evaluation_dict(initial_validation),
            "final_train": _evaluation_dict(final_train),
            "final_validation": _evaluation_dict(final_validation),
            "test_at_requested_final_epoch": {
                "epoch": args.epochs,
                **_evaluation_dict(final_test),
            },
        },
        "artifacts": {
            "run_manifest": str(run_manifest_path),
            "history": str(history_path),
            "split_indices": str(split_path),
            "best_checkpoint": str(best_checkpoint_path),
            "final_checkpoint": str(final_checkpoint_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    arguments = parser.parse_args(argv)
    if arguments.epochs < 1:
        parser.error("epochs must be positive")
    return arguments


if __name__ == "__main__":
    train(parse_args())
