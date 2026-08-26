"""Shared data and metric support for the confirmed two benzene experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from experiments.benzene_pair.data.benzene_cluster import (
    BenzeneClusterArrays,
    load_benzene_cluster_csv,
)


MODULE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATA = (
    MODULE_DIRECTORY / "data" / "two_benzene_opls_2_0_0_2k_v1.csv"
)
DEFAULT_METADATA = DEFAULT_DATA.with_suffix(".json")
DEFAULT_VALIDATION = DEFAULT_DATA.with_suffix(".validation.json")
DEFAULT_SEED = 20260824
TRAIN_COUNT = 1600
VALIDATION_COUNT = 200
TEST_COUNT = 200


class PairForceModel(Protocol):
    pair_count: int
    force_scale: Tensor

    def eval(self) -> PairForceModel: ...

    def normalized_forces_and_pairs_world(
        self,
        centers_world: Tensor,
        rotations_world_from_body: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class Evaluation:
    normalized_mse: float
    physical_mse_kcal2_mol2_A2: float
    physical_component_mae_kcal_mol_A: float
    physical_vector_mae_kcal_mol_A: float
    residual_frobenius_norm_kcal_mol_A: float
    target_frobenius_norm_kcal_mol_A: float
    relative_frobenius_norm_error_percent: float
    per_graph_relative_norm_error_percent_mean: float
    per_graph_relative_norm_error_percent_median: float
    per_graph_relative_norm_error_percent_p95: float
    per_graph_relative_norm_error_percent_maximum: float
    per_graph_force_magnitude_percent_error_mean: float
    per_graph_force_magnitude_percent_error_median: float
    per_graph_force_magnitude_percent_error_p95: float
    per_graph_force_magnitude_percent_error_maximum: float


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


def _split_sample_ids(
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_count = TRAIN_COUNT + VALIDATION_COUNT + TEST_COUNT
    if sample_count != expected_count:
        raise ValueError(
            f"the experiment requires {expected_count} samples, got "
            f"{sample_count}"
        )
    order = np.random.default_rng(seed).permutation(sample_count)
    train = order[:TRAIN_COUNT]
    validation = order[TRAIN_COUNT : TRAIN_COUNT + VALIDATION_COUNT]
    test = order[TRAIN_COUNT + VALIDATION_COUNT :]
    combined = np.concatenate((train, validation, test))
    if not np.array_equal(np.sort(combined), np.arange(sample_count)):
        raise RuntimeError("sample splits are not a complete disjoint partition")
    return train, validation, test


def _single_edge_targets(arrays: BenzeneClusterArrays) -> np.ndarray:
    if arrays.molecule_count != 2:
        raise ValueError("the experiment requires exactly two molecules")
    conservation_error = float(
        np.max(np.abs(arrays.forces[:, 0] + arrays.forces[:, 1]))
    )
    if conservation_error > 1.0e-10:
        raise ValueError(
            "the two molecule forces do not form equal and opposite targets"
        )
    targets = arrays.forces[:, 0, None, :]
    if not np.isfinite(targets).all():
        raise ValueError("single edge targets contain nonfinite values")
    if np.any(np.linalg.vector_norm(targets[:, 0], axis=-1) == 0.0):
        raise ValueError("single edge targets contain a zero force graph")
    return np.ascontiguousarray(targets)


def _tensor_subset(
    value: np.ndarray,
    indices: np.ndarray,
    dtype: torch.dtype,
) -> Tensor:
    resolved_dtype = np.float32 if dtype == torch.float32 else np.float64
    return torch.from_numpy(
        np.ascontiguousarray(value[indices], dtype=resolved_dtype)
    )


def _loader(
    centers: Tensor,
    rotations: Tensor,
    targets: Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    dataset = cast(
        Dataset[tuple[Tensor, Tensor, Tensor]],
        TensorDataset(centers, rotations, targets),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _normalized_single_edge_prediction(
    model: PairForceModel,
    centers: Tensor,
    rotations: Tensor,
) -> Tensor:
    if model.pair_count != 1:
        raise RuntimeError("the two benzene model must contain one pair")
    _, pair_prediction = model.normalized_forces_and_pairs_world(
        centers,
        rotations,
    )
    if pair_prediction.shape[-2:] != (1, 3):
        raise RuntimeError("the model returned an invalid single edge shape")
    return pair_prediction


def _evaluate(
    model: PairForceModel,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> Evaluation:
    model.eval()
    normalized_squared_error_sum = 0.0
    physical_absolute_error_sum = 0.0
    physical_vector_error_sum = 0.0
    physical_squared_error_sum = 0.0
    physical_target_squared_sum = 0.0
    component_count = 0
    vector_count = 0
    per_graph_percent: list[np.ndarray] = []
    per_graph_magnitude_percent: list[np.ndarray] = []
    scale = float(model.force_scale.detach().cpu())
    with torch.inference_mode():
        for centers, rotations, target in loader:
            centers = centers.to(device)
            rotations = rotations.to(device)
            target = target.to(device)
            prediction = _normalized_single_edge_prediction(
                model,
                centers,
                rotations,
            )
            difference = prediction - target
            physical_difference = difference * model.force_scale
            physical_target = target * model.force_scale
            normalized_squared_error_sum += float(
                difference.square().sum().cpu()
            )
            physical_absolute_error_sum += float(
                physical_difference.abs().sum().cpu()
            )
            physical_vector_error_sum += float(
                torch.linalg.vector_norm(
                    physical_difference,
                    dim=-1,
                ).sum().cpu()
            )
            physical_squared_error_sum += float(
                physical_difference.square().sum().cpu()
            )
            physical_target_squared_sum += float(
                physical_target.square().sum().cpu()
            )
            target_norm = torch.linalg.vector_norm(
                physical_target,
                dim=(-2, -1),
            )
            if bool((target_norm <= 0.0).any()):
                raise ValueError("relative error is undefined for a zero target")
            residual_norm = torch.linalg.vector_norm(
                physical_difference,
                dim=(-2, -1),
            )
            prediction_norm = torch.linalg.vector_norm(
                prediction * model.force_scale,
                dim=(-2, -1),
            )
            per_graph_percent.append(
                (100.0 * residual_norm / target_norm).cpu().numpy()
            )
            per_graph_magnitude_percent.append(
                (
                    100.0
                    * torch.abs(prediction_norm - target_norm)
                    / target_norm
                ).cpu().numpy()
            )
            component_count += target.numel()
            vector_count += target.shape[0] * target.shape[1]
    if component_count == 0 or physical_target_squared_sum <= 0.0:
        raise ValueError("evaluation loader is empty or has zero target norm")
    graph_values = np.concatenate(per_graph_percent).astype(
        np.float64,
        copy=False,
    )
    magnitude_values = np.concatenate(per_graph_magnitude_percent).astype(
        np.float64,
        copy=False,
    )
    normalized_mse = normalized_squared_error_sum / component_count
    physical_mse = normalized_mse * scale * scale
    residual_norm = math.sqrt(physical_squared_error_sum)
    target_norm = math.sqrt(physical_target_squared_sum)
    return Evaluation(
        normalized_mse=normalized_mse,
        physical_mse_kcal2_mol2_A2=physical_mse,
        physical_component_mae_kcal_mol_A=(
            physical_absolute_error_sum / component_count
        ),
        physical_vector_mae_kcal_mol_A=(
            physical_vector_error_sum / vector_count
        ),
        residual_frobenius_norm_kcal_mol_A=residual_norm,
        target_frobenius_norm_kcal_mol_A=target_norm,
        relative_frobenius_norm_error_percent=(
            100.0 * residual_norm / target_norm
        ),
        per_graph_relative_norm_error_percent_mean=float(
            np.mean(graph_values)
        ),
        per_graph_relative_norm_error_percent_median=float(
            np.median(graph_values)
        ),
        per_graph_relative_norm_error_percent_p95=float(
            np.percentile(graph_values, 95.0)
        ),
        per_graph_relative_norm_error_percent_maximum=float(
            np.max(graph_values)
        ),
        per_graph_force_magnitude_percent_error_mean=float(
            np.mean(magnitude_values)
        ),
        per_graph_force_magnitude_percent_error_median=float(
            np.median(magnitude_values)
        ),
        per_graph_force_magnitude_percent_error_p95=float(
            np.percentile(magnitude_values, 95.0)
        ),
        per_graph_force_magnitude_percent_error_maximum=float(
            np.max(magnitude_values)
        ),
    )


def _evaluation_dict(value: Evaluation) -> dict[str, float]:
    return asdict(value)


def _write_history(
    path: Path,
    rows: list[dict[str, float | int | None]],
) -> None:
    with path.open("w", newline="", encoding="utf_8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )


def _load_prepared_data(
    data_path: Path,
    metadata_path: Path,
    validation_path: Path,
) -> tuple[BenzeneClusterArrays, dict[str, Any], dict[str, Any], str]:
    data_hash = _sha256(data_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf_8"))
    validation = json.loads(validation_path.read_text(encoding="utf_8"))
    if metadata.get("csv_sha256") != data_hash:
        raise ValueError("metadata CSV hash does not match the selected data")
    if validation.get("csv_sha256") != data_hash:
        raise ValueError("validation CSV hash does not match the selected data")
    if validation.get("passed") is not True:
        raise ValueError("the selected data validation did not pass")
    arrays = load_benzene_cluster_csv(data_path)
    if len(arrays) != TRAIN_COUNT + VALIDATION_COUNT + TEST_COUNT:
        raise ValueError("the selected data must contain exactly 2000 graphs")
    if arrays.molecule_count != 2:
        raise ValueError("the selected data must contain two molecules per graph")
    if metadata.get("sample_count") != len(arrays):
        raise ValueError("metadata sample count does not match the CSV")
    if metadata.get("molecule_count") != arrays.molecule_count:
        raise ValueError("metadata molecule count does not match the CSV")
    return arrays, metadata, validation, data_hash


__all__ = [
    "DEFAULT_DATA",
    "DEFAULT_METADATA",
    "DEFAULT_SEED",
    "DEFAULT_VALIDATION",
    "Evaluation",
    "PairForceModel",
    "TEST_COUNT",
    "TRAIN_COUNT",
    "VALIDATION_COUNT",
]
