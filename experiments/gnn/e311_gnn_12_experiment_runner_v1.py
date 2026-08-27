"""Deterministic training and audit runner for the twelve E311 GNN experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from experiments.benzene_pair.data.benzene_cluster import (
    BenzeneClusterArrays,
    load_benzene_cluster_csv,
)
from experiments.gnn.e311_gnn_12_experiment_core_v1 import (
    E311GNNExperimentSpecV1,
    E311MessageStackCoreV1,
    SupervisionV1,
    build_experiment_core_v1,
    complete_pair_index_v1,
    get_experiment_spec_v1,
)
from experiments.gnn.e311_gnn_comet_v1 import (
    E311GNNCometConfigV1,
    create_e311_gnn_comet_logger_v1,
)


MODULE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = MODULE_DIRECTORY / "data"
DEFAULT_OUTPUT_ROOT = MODULE_DIRECTORY / "runs" / "e311_gnn_12_experiment_v1"
TWO_BENZENE_CSV = DATA_DIRECTORY / "two_benzene_opls_2_0_0_2k_v1.csv"
FIVE_BENZENE_CSV = DATA_DIRECTORY / "five_benzene_opls_2_0_0_1k_v1.csv"
FIVE_BENZENE_PAIR_NPZ = (
    DATA_DIRECTORY / "five_benzene_opls_2_0_0_1k_v1_pair_forces.npz"
)

DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 100
DEFAULT_FIVE_BENZENE_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.002
DEFAULT_WEIGHT_DECAY = 1.0e-6
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 20
MINIMUM_LEARNING_RATE = 1.0e-5
DEFAULT_MODEL_SEEDS = (20260824,)
DEFAULT_SPLIT_SEED = 20260824
DEFAULT_SHUFFLE_SEED_OFFSET = 0
DEFAULT_THREE_BODY_BASE_COUNT = 1_000
DEFAULT_THREE_BODY_INTERVENTIONS = 4
DEFAULT_THREE_BODY_SEED = 20260824
DEFAULT_COMET_PROJECT = "tfenn_e311_gnn_12_v1"
FIVE_BENZENE_EXPERIMENT_IDS = frozenset(
    f"X{index:02d}" for index in range(2, 11)
)


@dataclass(frozen=True, slots=True)
class SplitIndicesV1:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True, slots=True)
class GraphBucketV1:
    name: str
    centers_world: np.ndarray
    frames_body_to_world: np.ndarray
    node_force_world: np.ndarray
    pair_index: np.ndarray
    pair_force_world: np.ndarray | None
    group_id: np.ndarray

    def __post_init__(self) -> None:
        sample_count = len(self.centers_world)
        if self.centers_world.ndim != 3 or self.centers_world.shape[-1] != 3:
            raise ValueError("centers_world must have shape (S, N, 3)")
        if self.frames_body_to_world.shape != self.centers_world.shape + (3,):
            raise ValueError("frames_body_to_world must have shape (S, N, 3, 3)")
        if self.node_force_world.shape != self.centers_world.shape:
            raise ValueError("node_force_world must match centers_world")
        if self.pair_index.ndim != 2 or self.pair_index.shape[0] != 2:
            raise ValueError("pair_index must have shape (2, E)")
        if self.pair_index.dtype != np.int64:
            raise TypeError("pair_index must use int64")
        if self.pair_force_world is not None and self.pair_force_world.shape != (
            sample_count,
            self.pair_index.shape[1],
            3,
        ):
            raise ValueError("pair_force_world has the wrong shape")
        if self.group_id.shape != (sample_count,):
            raise ValueError("group_id must have shape (S,)")
        arrays = (
            self.centers_world,
            self.frames_body_to_world,
            self.node_force_world,
        )
        if self.pair_force_world is not None:
            arrays += (self.pair_force_world,)
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("graph bucket contains nonfinite values")

    @property
    def sample_count(self) -> int:
        return len(self.centers_world)

    @property
    def node_count(self) -> int:
        return self.centers_world.shape[1]

    def subset(self, indices: np.ndarray, name: str | None = None) -> GraphBucketV1:
        selected = np.asarray(indices, dtype=np.int64)
        return GraphBucketV1(
            self.name if name is None else name,
            np.ascontiguousarray(self.centers_world[selected]),
            np.ascontiguousarray(self.frames_body_to_world[selected]),
            np.ascontiguousarray(self.node_force_world[selected]),
            self.pair_index.copy(),
            None
            if self.pair_force_world is None
            else np.ascontiguousarray(self.pair_force_world[selected]),
            np.ascontiguousarray(self.group_id[selected]),
        )


@dataclass(frozen=True, slots=True)
class PreparedExperimentDataV1:
    spec: E311GNNExperimentSpecV1
    train: tuple[GraphBucketV1, ...]
    validation: tuple[GraphBucketV1, ...]
    test: tuple[GraphBucketV1, ...]
    force_scale: float | None
    records: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NodeMetricsV1:
    final_test_mae: float
    final_normal_f_difference: float
    residual_frobenius_norm: float
    target_frobenius_norm: float
    component_count: int


@dataclass(frozen=True, slots=True)
class BucketLoaderV1:
    name: str
    pair_index: Tensor
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]]


@dataclass(frozen=True, slots=True)
class RunnerConfigV1:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    five_benzene_batch_size: int = DEFAULT_FIVE_BENZENE_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    split_seed: int = DEFAULT_SPLIT_SEED
    device: str = "cuda"
    three_body_base_count: int = DEFAULT_THREE_BODY_BASE_COUNT
    three_body_interventions: int = DEFAULT_THREE_BODY_INTERVENTIONS
    comet_project: str = DEFAULT_COMET_PROJECT
    comet_workspace: str | None = None
    comet_enabled: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.five_benzene_batch_size < 1:
            raise ValueError("five_benzene_batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.three_body_base_count < 10:
            raise ValueError("three_body_base_count must be at least ten")
        if self.three_body_interventions < 2:
            raise ValueError("three_body_interventions must be at least two")

    def batch_size_for(self, experiment_id: str) -> int:
        normalized = get_experiment_spec_v1(experiment_id).experiment_id
        if normalized in FIVE_BENZENE_EXPERIMENT_IDS:
            return self.five_benzene_batch_size
        return self.batch_size


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    partial.replace(path)


def _write_history_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("history rows cannot be empty")
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf_8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def deterministic_group_split_v1(group_id: np.ndarray, seed: int) -> SplitIndicesV1:
    groups = np.asarray(group_id, dtype=np.int64)
    if groups.ndim != 1:
        raise ValueError("group_id must be one dimensional")
    unique = np.unique(groups)
    if len(unique) < 10:
        raise ValueError("at least ten independent groups are required")
    order = np.random.default_rng(seed).permutation(unique)
    train_count = int(math.floor(0.8 * len(order)))
    validation_count = int(math.floor(0.1 * len(order)))
    train_groups = order[:train_count]
    validation_groups = order[train_count : train_count + validation_count]
    test_groups = order[train_count + validation_count :]
    result = SplitIndicesV1(
        np.flatnonzero(np.isin(groups, train_groups)).astype(np.int64),
        np.flatnonzero(np.isin(groups, validation_groups)).astype(np.int64),
        np.flatnonzero(np.isin(groups, test_groups)).astype(np.int64),
    )
    selected_groups = tuple(set(groups[index].tolist()) for index in asdict(result).values())
    if selected_groups[0] & selected_groups[1]:
        raise RuntimeError("train and validation groups overlap")
    if selected_groups[0] & selected_groups[2]:
        raise RuntimeError("train and test groups overlap")
    if selected_groups[1] & selected_groups[2]:
        raise RuntimeError("validation and test groups overlap")
    combined = np.concatenate((result.train, result.validation, result.test))
    if not np.array_equal(np.sort(combined), np.arange(len(groups))):
        raise RuntimeError("split is not a complete partition")
    return result


def _validate_cluster_files(csv_path: Path) -> tuple[BenzeneClusterArrays, dict[str, Any]]:
    metadata_path = csv_path.with_suffix(".json")
    validation_path = csv_path.with_suffix(".validation.json")
    for path in (csv_path, metadata_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    csv_hash = _sha256(csv_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf_8"))
    validation = json.loads(validation_path.read_text(encoding="utf_8"))
    if metadata.get("csv_sha256") != csv_hash:
        raise ValueError(f"metadata hash mismatch for {csv_path.name}")
    if validation.get("csv_sha256") != csv_hash or validation.get("passed") is not True:
        raise ValueError(f"validation did not pass for {csv_path.name}")
    arrays = load_benzene_cluster_csv(csv_path)
    if metadata.get("sample_count") != len(arrays):
        raise ValueError("metadata sample count mismatch")
    if metadata.get("molecule_count") != arrays.molecule_count:
        raise ValueError("metadata molecule count mismatch")
    return arrays, {
        "csv_path": str(csv_path.resolve()),
        "csv_sha256": csv_hash,
        "metadata_sha256": _sha256(metadata_path),
        "validation_sha256": _sha256(validation_path),
        "sample_count": len(arrays),
        "molecule_count": arrays.molecule_count,
    }


def _load_five_pair_forces() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata_path = FIVE_BENZENE_PAIR_NPZ.with_suffix(".json")
    if not FIVE_BENZENE_PAIR_NPZ.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(FIVE_BENZENE_PAIR_NPZ)
    metadata = json.loads(metadata_path.read_text(encoding="utf_8"))
    npz_hash = _sha256(FIVE_BENZENE_PAIR_NPZ)
    if metadata.get("artifacts", {}).get("npz_sha256") != npz_hash:
        raise ValueError("five benzene pair force hash mismatch")
    with np.load(FIVE_BENZENE_PAIR_NPZ, allow_pickle=False) as archive:
        sample_id = np.asarray(archive["sample_id"], dtype=np.int64)
        pair_index = np.asarray(archive["pair_index"], dtype=np.int64)
        pair_force = np.asarray(archive["pair_force_kcal_mol_A"], dtype=np.float64)
    if pair_index.ndim != 2 or pair_index.shape[1] != 2:
        raise ValueError("stored pair_index must have shape (E, 2)")
    if not np.array_equal(sample_id, np.arange(len(sample_id))):
        raise ValueError("pair force sample IDs are not canonical")
    return pair_index.T.copy(), np.ascontiguousarray(pair_force), {
        "pair_npz_path": str(FIVE_BENZENE_PAIR_NPZ.resolve()),
        "pair_npz_sha256": npz_hash,
        "pair_metadata_sha256": _sha256(metadata_path),
    }


def scatter_pair_forces_v1(
    pair_force_world: np.ndarray,
    pair_index: np.ndarray,
    node_count: int,
) -> np.ndarray:
    pair_force = np.asarray(pair_force_world)
    edges = np.asarray(pair_index, dtype=np.int64)
    if pair_force.ndim != 3 or pair_force.shape[-1] != 3:
        raise ValueError("pair_force_world must have shape (S, E, 3)")
    if edges.shape != (2, pair_force.shape[1]):
        raise ValueError("pair_index does not match pair force edge count")
    result = np.zeros((len(pair_force), node_count, 3), dtype=pair_force.dtype)
    for edge_id, (receiver, sender) in enumerate(edges.T):
        result[:, receiver] += pair_force[:, edge_id]
        result[:, sender] -= pair_force[:, edge_id]
    return np.ascontiguousarray(result)


def _base_bucket(
    name: str,
    arrays: BenzeneClusterArrays,
    pair_index: np.ndarray,
    pair_force: np.ndarray | None,
) -> GraphBucketV1:
    if arrays.molecule_count != int(pair_index.max()) + 1:
        raise ValueError("pair index does not cover the expected molecule count")
    return GraphBucketV1(
        name,
        np.ascontiguousarray(arrays.centers, dtype=np.float64),
        np.ascontiguousarray(arrays.rotations, dtype=np.float64),
        np.ascontiguousarray(arrays.forces, dtype=np.float64),
        np.ascontiguousarray(pair_index, dtype=np.int64),
        None if pair_force is None else np.ascontiguousarray(pair_force, dtype=np.float64),
        np.arange(len(arrays), dtype=np.int64),
    )


def _truncate_five_bucket(bucket: GraphBucketV1, node_count: int) -> GraphBucketV1:
    if bucket.node_count != 5 or node_count not in (3, 4, 5):
        raise ValueError("five benzene truncation supports N equal to three, four, or five")
    if bucket.pair_force_world is None:
        raise ValueError("pair decomposition is required for truncation")
    keep = np.flatnonzero(
        (bucket.pair_index[0] < node_count) & (bucket.pair_index[1] < node_count)
    )
    pair_index = np.ascontiguousarray(bucket.pair_index[:, keep])
    pair_force = np.ascontiguousarray(bucket.pair_force_world[:, keep])
    node_force = scatter_pair_forces_v1(pair_force, pair_index, node_count)
    return GraphBucketV1(
        f"N{node_count}",
        np.ascontiguousarray(bucket.centers_world[:, :node_count]),
        np.ascontiguousarray(bucket.frames_body_to_world[:, :node_count]),
        node_force,
        pair_index,
        pair_force,
        bucket.group_id.copy(),
    )


def _random_rotation_v1(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    quaternion = np.asarray(
        (
            math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
            math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
            math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
            math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
        ),
        dtype=np.float64,
    )
    x, y, z, w = quaternion
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def generate_three_body_chain_v1(
    base_count: int = DEFAULT_THREE_BODY_BASE_COUNT,
    interventions: int = DEFAULT_THREE_BODY_INTERVENTIONS,
    seed: int = DEFAULT_THREE_BODY_SEED,
    *,
    coupling: float = 5.0,
    target_cosine: float = -1.0 / 3.0,
    cutoff: float = 7.0,
) -> GraphBucketV1:
    if base_count < 10 or interventions < 2:
        raise ValueError("three body data requires ten bases and two interventions")
    sample_count = base_count * interventions
    centers = np.empty((sample_count, 3, 3), dtype=np.float64)
    frames = np.empty((sample_count, 3, 3, 3), dtype=np.float64)
    group_id = np.repeat(np.arange(base_count, dtype=np.int64), interventions)
    for base_id in range(base_count):
        rng = np.random.default_rng(np.random.SeedSequence((seed, base_id)))
        translation = rng.uniform(-1.0, 1.0, size=3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        r01 = rng.uniform(4.5, 5.8)
        x0 = translation
        x1 = translation + r01 * axis
        frame0 = _random_rotation_v1(rng)
        frame1 = _random_rotation_v1(rng)
        incoming = -axis
        for intervention in range(interventions):
            index = base_id * interventions + intervention
            candidate = rng.normal(size=3)
            perpendicular = candidate - np.dot(candidate, incoming) * incoming
            if np.linalg.norm(perpendicular) < 1.0e-8:
                candidate = np.roll(incoming, 1)
                perpendicular = candidate - np.dot(candidate, incoming) * incoming
            perpendicular /= np.linalg.norm(perpendicular)
            cosine = rng.uniform(-0.75, 0.75)
            direction = cosine * incoming + math.sqrt(1.0 - cosine * cosine) * perpendicular
            r12 = rng.uniform(4.5, 5.8)
            centers[index] = np.stack((x0, x1, x1 + r12 * direction))
            frames[index] = np.stack((frame0, frame1, _random_rotation_v1(rng)))

    positions = torch.tensor(centers, dtype=torch.float64, requires_grad=True)
    r10 = positions[:, 0] - positions[:, 1]
    r12 = positions[:, 2] - positions[:, 1]
    distance10 = torch.linalg.vector_norm(r10, dim=-1)
    distance12 = torch.linalg.vector_norm(r12, dim=-1)
    cosine = (r10 * r12).sum(dim=-1) / (distance10 * distance12)
    cutoff10 = 0.5 * (torch.cos(math.pi * distance10 / cutoff) + 1.0)
    cutoff12 = 0.5 * (torch.cos(math.pi * distance12 / cutoff) + 1.0)
    energy = coupling * cutoff10 * cutoff12 * (cosine - target_cosine).square()
    force = -torch.autograd.grad(energy.sum(), positions)[0]
    forces = force.detach().cpu().numpy()
    residual = float(np.max(np.abs(forces.sum(axis=1))))
    if residual > 1.0e-10:
        raise RuntimeError("three body force conservation failed")
    return GraphBucketV1(
        "three_body_chain",
        centers,
        frames,
        np.ascontiguousarray(forces),
        np.asarray(((0, 1), (1, 2)), dtype=np.int64),
        None,
        group_id,
    )


def _force_scale(buckets: Iterable[GraphBucketV1], supervision: SupervisionV1) -> float:
    squared_sum = 0.0
    component_count = 0
    for bucket in buckets:
        if supervision is SupervisionV1.PAIR_FORCE:
            if bucket.pair_force_world is None:
                raise ValueError("pair force supervision requires pair labels")
            target = bucket.pair_force_world
        else:
            target = bucket.node_force_world
        squared_sum += float(np.square(target).sum())
        component_count += target.size
    if component_count == 0:
        raise ValueError("force scale requires nonempty training data")
    value = math.sqrt(squared_sum / component_count)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("force scale must be finite and positive")
    return value


def prepare_experiment_data_v1(
    experiment_id: str,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    three_body_base_count: int = DEFAULT_THREE_BODY_BASE_COUNT,
    three_body_interventions: int = DEFAULT_THREE_BODY_INTERVENTIONS,
) -> PreparedExperimentDataV1:
    spec = get_experiment_spec_v1(experiment_id)
    records: dict[str, Any] = {"split_seed": split_seed}
    if spec.experiment_id == "X01":
        arrays, record = _validate_cluster_files(TWO_BENZENE_CSV)
        pair_index = np.asarray(((0,), (1,)), dtype=np.int64)
        pair_force = np.ascontiguousarray(arrays.forces[:, 0, None, :])
        bucket = _base_bucket("N2", arrays, pair_index, pair_force)
        split = deterministic_group_split_v1(bucket.group_id, split_seed)
        train = (bucket.subset(split.train),)
        validation = (bucket.subset(split.validation),)
        test = (bucket.subset(split.test),)
        records["two_benzene"] = record
    elif spec.experiment_id in {"X02", "X03", "X04", "X05", "X06", "X07", "X08", "X09", "X10"}:
        arrays, record = _validate_cluster_files(FIVE_BENZENE_CSV)
        pair_index, pair_force, pair_record = _load_five_pair_forces()
        bucket = _base_bucket("N5", arrays, pair_index, pair_force)
        reconstruction = scatter_pair_forces_v1(pair_force, pair_index, 5)
        if float(np.max(np.abs(reconstruction - bucket.node_force_world))) > 1.0e-9:
            raise ValueError("five benzene pair labels do not reconstruct node labels")
        split = deterministic_group_split_v1(bucket.group_id, split_seed)
        records["five_benzene"] = record
        records["five_benzene_pair"] = pair_record
        if spec.experiment_id == "X02":
            train = ()
            validation = ()
            test = tuple(
                _truncate_five_bucket(bucket, node_count).subset(split.test)
                for node_count in (3, 4, 5)
            )
        elif spec.experiment_id in {"X09", "X10"}:
            train = tuple(
                _truncate_five_bucket(bucket, node_count).subset(split.train)
                for node_count in (3, 4)
            )
            validation = tuple(
                _truncate_five_bucket(bucket, node_count).subset(split.validation)
                for node_count in (3, 4)
            )
            test = (_truncate_five_bucket(bucket, 5).subset(split.test),)
        else:
            train = (bucket.subset(split.train),)
            validation = (bucket.subset(split.validation),)
            test = (bucket.subset(split.test),)
    else:
        bucket = generate_three_body_chain_v1(
            three_body_base_count,
            three_body_interventions,
            DEFAULT_THREE_BODY_SEED,
        )
        split = deterministic_group_split_v1(bucket.group_id, split_seed)
        train = (bucket.subset(split.train),)
        validation = (bucket.subset(split.validation),)
        test = (bucket.subset(split.test),)
        records["synthetic_three_body"] = {
            "base_count": three_body_base_count,
            "interventions_per_base": three_body_interventions,
            "generation_seed": DEFAULT_THREE_BODY_SEED,
            "coupling": 5.0,
            "target_cosine": -1.0 / 3.0,
            "cutoff": 7.0,
            "distance_range": [4.5, 5.8],
        }
    if spec.supervision is SupervisionV1.AUDIT_ONLY:
        scale = None
    elif spec.experiment_id == "X04":
        scale = _force_scale(train, SupervisionV1.NODE_FORCE)
    else:
        scale = _force_scale(train, spec.supervision)
    records["split_counts"] = {
        "train": sum(bucket.sample_count for bucket in train),
        "validation": sum(bucket.sample_count for bucket in validation),
        "test": sum(bucket.sample_count for bucket in test),
    }
    return PreparedExperimentDataV1(spec, train, validation, test, scale, records)


def _target_for_supervision(bucket: GraphBucketV1, supervision: SupervisionV1) -> np.ndarray:
    if supervision is SupervisionV1.PAIR_FORCE:
        if bucket.pair_force_world is None:
            raise ValueError("pair force labels are unavailable")
        return bucket.pair_force_world
    if supervision is SupervisionV1.NODE_FORCE:
        return bucket.node_force_world
    raise ValueError("audit only data has no training target")


def _bucket_loaders(
    buckets: Sequence[GraphBucketV1],
    supervision: SupervisionV1,
    force_scale: float,
    batch_size: int,
    shuffle: bool,
    shuffle_seed: int,
) -> tuple[BucketLoaderV1, ...]:
    result = []
    for bucket_index, bucket in enumerate(buckets):
        loss_target = _target_for_supervision(bucket, supervision)
        dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(bucket.centers_world, dtype=np.float32)),
            torch.from_numpy(
                np.ascontiguousarray(bucket.frames_body_to_world, dtype=np.float32)
            ),
            torch.from_numpy(np.ascontiguousarray(loss_target / force_scale, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(bucket.node_force_world, dtype=np.float32)),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=torch.Generator().manual_seed(
                shuffle_seed + bucket_index * 100_003
            ),
        )
        result.append(
            BucketLoaderV1(
                bucket.name,
                torch.from_numpy(bucket.pair_index.copy()),
                loader,
            )
        )
    return tuple(result)


def _set_model_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_seeded_experiment_model_v1(
    experiment_id: str,
    force_scale: float,
    model_seed: int,
    device: torch.device | str = "cpu",
) -> E311MessageStackCoreV1:
    _set_model_seed(model_seed)
    model = build_experiment_core_v1(experiment_id, force_scale, torch.float32)
    with torch.no_grad():
        for block in model.message_blocks:
            block.pair_kernel._runtime_reference.zero_()
    return model.to(torch.device(device))


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


def _normalized_prediction(
    model: E311MessageStackCoreV1,
    centers: Tensor,
    frames: Tensor,
    pair_index: Tensor,
    supervision: SupervisionV1,
) -> Tensor:
    output = model.core_output(centers, frames, pair_index)
    if supervision is SupervisionV1.PAIR_FORCE:
        return output.normalized_pair_force_world
    if supervision is SupervisionV1.NODE_FORCE:
        return output.normalized_node_force_world
    raise ValueError("audit only is not a training supervision")


def _evaluate_loss(
    model: E311MessageStackCoreV1,
    loaders: Sequence[BucketLoaderV1],
    supervision: SupervisionV1,
    device: torch.device,
) -> float:
    model.eval()
    squared_sum = 0.0
    graph_count = 0
    with torch.inference_mode():
        for bucket in loaders:
            pair_index = bucket.pair_index.to(device)
            for centers, frames, target, _ in bucket.loader:
                centers = centers.to(device)
                frames = frames.to(device)
                target = target.to(device)
                difference = _normalized_prediction(
                    model, centers, frames, pair_index, supervision
                ) - target
                squared_sum += float(difference.flatten(1).square().mean(1).sum().cpu())
                graph_count += len(centers)
    if graph_count == 0:
        raise ValueError("evaluation data is empty")
    return squared_sum / graph_count


def _train_epoch(
    model: E311MessageStackCoreV1,
    loaders: Sequence[BucketLoaderV1],
    supervision: SupervisionV1,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    squared_sum = 0.0
    graph_count = 0
    for bucket in loaders:
        pair_index = bucket.pair_index.to(device)
        for centers, frames, target, _ in bucket.loader:
            centers = centers.to(device)
            frames = frames.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            difference = _normalized_prediction(
                model, centers, frames, pair_index, supervision
            ) - target
            per_graph = difference.flatten(1).square().mean(1)
            loss = per_graph.mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training loss became nonfinite")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise RuntimeError("a parameter gradient became nonfinite")
            optimizer.step()
            squared_sum += float(per_graph.detach().sum().cpu())
            graph_count += len(centers)
    if graph_count == 0:
        raise ValueError("training data is empty")
    return squared_sum / graph_count


def compute_node_metrics_v1(prediction: Tensor, target: Tensor) -> NodeMetricsV1:
    if prediction.shape != target.shape or prediction.ndim < 2 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have matching vector shapes")
    difference = prediction - target
    target_squared = float(target.double().square().sum().cpu())
    residual_squared = float(difference.double().square().sum().cpu())
    if target_squared <= 0.0:
        raise ValueError("normal F difference is undefined for a zero target")
    return NodeMetricsV1(
        final_test_mae=float(difference.abs().double().mean().cpu()),
        final_normal_f_difference=100.0 * math.sqrt(residual_squared / target_squared),
        residual_frobenius_norm=math.sqrt(residual_squared),
        target_frobenius_norm=math.sqrt(target_squared),
        component_count=target.numel(),
    )


def _evaluate_node_metrics(
    model: E311MessageStackCoreV1,
    buckets: Sequence[GraphBucketV1],
    batch_size: int,
    device: torch.device,
) -> NodeMetricsV1:
    predictions = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for bucket in buckets:
            dataset = TensorDataset(
                torch.from_numpy(
                    np.ascontiguousarray(bucket.centers_world, dtype=np.float32)
                ),
                torch.from_numpy(
                    np.ascontiguousarray(bucket.frames_body_to_world, dtype=np.float32)
                ),
                torch.from_numpy(
                    np.ascontiguousarray(bucket.node_force_world, dtype=np.float32)
                ),
            )
            pair_index = torch.from_numpy(bucket.pair_index.copy()).to(device)
            for centers, frames, target in DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0
            ):
                centers = centers.to(device)
                frames = frames.to(device)
                prediction = model.normalized_forces_world(
                    centers, frames, pair_index
                ) * model.force_scale
                predictions.append(prediction.cpu().reshape(-1, 3))
                targets.append(target.reshape(-1, 3))
    if not predictions:
        raise ValueError("test data is empty")
    return compute_node_metrics_v1(torch.cat(predictions), torch.cat(targets))


def _checkpoint_payload(
    experiment_id: str,
    model_seed: int,
    epoch: int,
    model: E311MessageStackCoreV1,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    config: RunnerConfigV1,
    data: PreparedExperimentDataV1,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "schema_name": "tfenn_e311_gnn_12_checkpoint_v1",
        "experiment_id": experiment_id,
        "model_seed": model_seed,
        "epoch": epoch,
        "force_scale": float(model.force_scale.detach().cpu()),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "runner_config": {
            **_json_ready(asdict(config)),
            "actual_batch_size": batch_size,
        },
        "architecture": dict(model.architecture_record()),
        "data_records": dict(data.records),
    }


def _save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(payload), partial)
    partial.replace(path)


def _legacy_x01_state_v1(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if any(key.startswith("message_blocks.0.") for key in state):
        return dict(state)
    remapped = {
        "message_blocks.0." + key[len("message_block.") :]: value
        for key, value in state.items()
        if key.startswith("message_block.")
    }
    if "force_scale" in state:
        remapped["force_scale"] = state["force_scale"]
    if not remapped:
        raise ValueError("checkpoint does not contain an X01 model state")
    return remapped


def _load_x01_for_audit(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[E311MessageStackCoreV1, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("X01 checkpoint payload is invalid")
    force_scale = float(payload["force_scale"])
    model = build_experiment_core_v1("X02", force_scale, torch.float32)
    model.load_state_dict(_legacy_x01_state_v1(payload["model_state"]), strict=True)
    with torch.no_grad():
        for block in model.message_blocks:
            block.pair_kernel._runtime_reference.zero_()
    model.to(device).eval()
    return model, payload


def _audit_explicit_pairs(
    model: E311MessageStackCoreV1,
    buckets: Sequence[GraphBucketV1],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    maximum_absolute = 0.0
    residual_squared = 0.0
    reference_squared = 0.0
    model.eval()
    with torch.inference_mode():
        for bucket in buckets:
            dataset = TensorDataset(
                torch.from_numpy(
                    np.ascontiguousarray(bucket.centers_world, dtype=np.float32)
                ),
                torch.from_numpy(
                    np.ascontiguousarray(bucket.frames_body_to_world, dtype=np.float32)
                ),
            )
            graph_pair_index = torch.from_numpy(bucket.pair_index.copy()).to(device)
            single_pair_index = complete_pair_index_v1(2, device)
            for centers, frames in DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0
            ):
                centers = centers.to(device)
                frames = frames.to(device)
                graph_pairs = model.core_output(
                    centers, frames, graph_pair_index
                ).normalized_pair_force_world
                explicit = []
                for receiver, sender in bucket.pair_index.T:
                    pair_centers = centers[:, (int(receiver), int(sender))]
                    pair_frames = frames[:, (int(receiver), int(sender))]
                    pair_prediction = model.core_output(
                        pair_centers, pair_frames, single_pair_index
                    ).normalized_pair_force_world
                    explicit.append(pair_prediction[:, 0])
                explicit_pairs = torch.stack(explicit, dim=1)
                difference = graph_pairs - explicit_pairs
                maximum_absolute = max(maximum_absolute, float(difference.abs().max().cpu()))
                residual_squared += float(difference.double().square().sum().cpu())
                reference_squared += float(explicit_pairs.double().square().sum().cpu())
    relative = 0.0 if reference_squared == 0.0 else 100.0 * math.sqrt(
        residual_squared / reference_squared
    )
    return {
        "maximum_absolute_normalized_pair_difference": maximum_absolute,
        "relative_frobenius_pair_difference_percent": relative,
    }


def _run_directory(config: RunnerConfigV1, experiment_id: str, model_seed: int) -> Path:
    return config.output_root.resolve() / experiment_id / f"seed_{model_seed}"


def _comet_config(config: RunnerConfigV1, experiment_id: str, model_seed: int) -> E311GNNCometConfigV1:
    return E311GNNCometConfigV1(
        project_name=config.comet_project,
        workspace=config.comet_workspace,
        tags=("e311_gnn_12_v1", experiment_id, f"seed_{model_seed}"),
        enabled=config.comet_enabled,
    )


def _parameter_record(
    data: PreparedExperimentDataV1,
    config: RunnerConfigV1,
    model_seed: int,
    shuffle_seed: int,
    architecture: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    return {
        "experiment_id": data.spec.experiment_id,
        "model_seed": model_seed,
        "split_seed": config.split_seed,
        "shuffle_seed": shuffle_seed,
        "epochs": 0 if data.spec.supervision is SupervisionV1.AUDIT_ONLY else config.epochs,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "device": config.device,
        "supervision": data.spec.supervision.value,
        "train_node_counts": data.spec.train_node_counts,
        "test_node_counts": data.spec.test_node_counts,
        **dict(architecture),
    }


def run_experiment_v1(
    experiment_id: str,
    model_seed: int,
    config: RunnerConfigV1,
    *,
    comet_backend_factory: Any | None = None,
) -> dict[str, Any]:
    spec = get_experiment_spec_v1(experiment_id)
    batch_size = config.batch_size_for(spec.experiment_id)
    device = _resolve_device(config.device)
    output = _run_directory(config, spec.experiment_id, model_seed)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if (output / "summary.json").exists():
        raise FileExistsError(f"completed run already exists at {output}")
    _write_json_atomic(
        status_path,
        {
            "status": "running",
            "experiment_id": spec.experiment_id,
            "model_seed": model_seed,
            "batch_size": batch_size,
            "started_at_utc": _utc_now(),
            "last_completed_epoch": 0,
        },
    )
    logger = None
    try:
        data = prepare_experiment_data_v1(
            spec.experiment_id,
            split_seed=config.split_seed,
            three_body_base_count=config.three_body_base_count,
            three_body_interventions=config.three_body_interventions,
        )
        shuffle_seed = model_seed + DEFAULT_SHUFFLE_SEED_OFFSET
        if spec.supervision is SupervisionV1.AUDIT_ONLY:
            checkpoint_path = _run_directory(config, "X01", model_seed) / "best_checkpoint.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"X02 requires the matching X01 best checkpoint at {checkpoint_path}"
                )
            model, checkpoint = _load_x01_for_audit(checkpoint_path, device)
            logger = create_e311_gnn_comet_logger_v1(
                _comet_config(config, spec.experiment_id, model_seed),
                experiment_name=f"{spec.experiment_id}_seed_{model_seed}",
                backend_factory=comet_backend_factory,
            )
            logger.log_parameters(
                parameters=_parameter_record(
                    data,
                    config,
                    model_seed,
                    shuffle_seed,
                    model.architecture_record(),
                    batch_size,
                )
            )
            audit = _audit_explicit_pairs(model, data.test, batch_size, device)
            metrics = _evaluate_node_metrics(model, data.test, batch_size, device)
            logger.log_final(
                final_test_mae=metrics.final_test_mae,
                final_normal_f_difference=metrics.final_normal_f_difference,
            )
            summary = {
                "schema_name": "tfenn_e311_gnn_12_result_v1",
                "status": "complete",
                "experiment": asdict(spec),
                "model_seed": model_seed,
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": _sha256(checkpoint_path),
                    "source_epoch": checkpoint.get("epoch"),
                },
                "audit": audit,
                "final_test": asdict(metrics),
                "architecture": dict(model.architecture_record()),
                "evaluation": {"batch_size": batch_size},
                "data": dict(data.records),
                "comet_identity": logger.identity,
                "completed_at_utc": _utc_now(),
            }
            _write_json_atomic(output / "summary.json", summary)
            _write_json_atomic(
                status_path,
                {
                    "status": "complete",
                    "experiment_id": spec.experiment_id,
                    "model_seed": model_seed,
                    "batch_size": batch_size,
                    "last_completed_epoch": 0,
                    "completed_at_utc": _utc_now(),
                },
            )
            logger.finish(status="complete")
            return summary

        if data.force_scale is None:
            raise RuntimeError("training data did not define a force scale")
        model = build_seeded_experiment_model_v1(
            spec.experiment_id, data.force_scale, model_seed, device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=MINIMUM_LEARNING_RATE,
        )
        train_loaders = _bucket_loaders(
            data.train,
            spec.supervision,
            data.force_scale,
            batch_size,
            True,
            shuffle_seed,
        )
        train_evaluation_loaders = _bucket_loaders(
            data.train,
            spec.supervision,
            data.force_scale,
            batch_size,
            False,
            shuffle_seed,
        )
        validation_loaders = _bucket_loaders(
            data.validation,
            spec.supervision,
            data.force_scale,
            batch_size,
            False,
            shuffle_seed,
        )
        logger = create_e311_gnn_comet_logger_v1(
            _comet_config(config, spec.experiment_id, model_seed),
            experiment_name=f"{spec.experiment_id}_seed_{model_seed}",
            backend_factory=comet_backend_factory,
        )
        logger.log_parameters(
            parameters=_parameter_record(
                data,
                config,
                model_seed,
                shuffle_seed,
                model.architecture_record(),
                batch_size,
            )
        )
        initial_train_loss = _evaluate_loss(
            model, train_evaluation_loaders, spec.supervision, device
        )
        initial_validation_loss = _evaluate_loss(
            model, validation_loaders, spec.supervision, device
        )
        history: list[dict[str, float | int | None]] = [
            {
                "epoch": 0,
                "optimization_loss": None,
                "train_loss": initial_train_loss,
                "validation_loss": initial_validation_loss,
                "epoch_duration_seconds": None,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        ]
        _write_history_atomic(output / "history.csv", history)
        best_validation = initial_validation_loss
        best_epoch = 0
        _save_checkpoint_atomic(
            output / "best_checkpoint.pt",
            _checkpoint_payload(
                spec.experiment_id,
                model_seed,
                0,
                model,
                optimizer,
                scheduler,
                config,
                data,
                batch_size,
            ),
        )
        for epoch in range(1, config.epochs + 1):
            _synchronize(device)
            started = time.perf_counter()
            optimization_loss = _train_epoch(
                model, train_loaders, spec.supervision, optimizer, device
            )
            train_loss = _evaluate_loss(
                model, train_evaluation_loaders, spec.supervision, device
            )
            validation_loss = _evaluate_loss(
                model, validation_loaders, spec.supervision, device
            )
            scheduler.step(validation_loss)
            _synchronize(device)
            duration = time.perf_counter() - started
            row = {
                "epoch": epoch,
                "optimization_loss": optimization_loss,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "epoch_duration_seconds": duration,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            _write_history_atomic(output / "history.csv", history)
            logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                epoch_duration_seconds=duration,
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_epoch = epoch
                _save_checkpoint_atomic(
                    output / "best_checkpoint.pt",
                    _checkpoint_payload(
                        spec.experiment_id,
                        model_seed,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        data,
                        batch_size,
                    ),
                )
            _write_json_atomic(
                status_path,
                {
                    "status": "running",
                    "experiment_id": spec.experiment_id,
                    "model_seed": model_seed,
                    "batch_size": batch_size,
                    "last_completed_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation,
                    "updated_at_utc": _utc_now(),
                },
            )
        _save_checkpoint_atomic(
            output / "final_checkpoint.pt",
            _checkpoint_payload(
                spec.experiment_id,
                model_seed,
                config.epochs,
                model,
                optimizer,
                scheduler,
                config,
                data,
                batch_size,
            ),
        )
        metrics = _evaluate_node_metrics(model, data.test, batch_size, device)
        logger.log_final(
            final_test_mae=metrics.final_test_mae,
            final_normal_f_difference=metrics.final_normal_f_difference,
        )
        summary = {
            "schema_name": "tfenn_e311_gnn_12_result_v1",
            "status": "complete",
            "experiment": asdict(spec),
            "model_seed": model_seed,
            "split_seed": config.split_seed,
            "shuffle_seed": shuffle_seed,
            "training": {
                "epochs": config.epochs,
                "batch_size": batch_size,
                "optimizer": "AdamW",
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "scheduler": "ReduceLROnPlateau",
                "scheduler_factor": SCHEDULER_FACTOR,
                "scheduler_patience": SCHEDULER_PATIENCE,
                "minimum_learning_rate": MINIMUM_LEARNING_RATE,
                "initial_train_loss": initial_train_loss,
                "initial_validation_loss": initial_validation_loss,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
                "final_train_loss": history[-1]["train_loss"],
                "final_validation_loss": history[-1]["validation_loss"],
            },
            "final_test": asdict(metrics),
            "architecture": dict(model.architecture_record()),
            "data": dict(data.records),
            "artifacts": {
                "history": str(output / "history.csv"),
                "best_checkpoint": str(output / "best_checkpoint.pt"),
                "final_checkpoint": str(output / "final_checkpoint.pt"),
            },
            "comet_identity": logger.identity,
            "completed_at_utc": _utc_now(),
        }
        _write_json_atomic(output / "summary.json", summary)
        _write_json_atomic(
            status_path,
            {
                "status": "complete",
                "experiment_id": spec.experiment_id,
                "model_seed": model_seed,
                "batch_size": batch_size,
                "last_completed_epoch": config.epochs,
                "best_epoch": best_epoch,
                "completed_at_utc": _utc_now(),
            },
        )
        logger.finish(status="complete")
        return summary
    except Exception as error:
        _write_json_atomic(
            status_path,
            {
                "status": "failed",
                "experiment_id": spec.experiment_id,
                "model_seed": model_seed,
                "batch_size": batch_size,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_utc": _utc_now(),
            },
        )
        if logger is not None:
            logger.finish(status="error")
        raise


def run_group_v1(
    experiment_ids: Sequence[str],
    model_seeds: Sequence[int],
    config: RunnerConfigV1,
) -> list[dict[str, Any]]:
    normalized = [get_experiment_spec_v1(value).experiment_id for value in experiment_ids]
    if "X02" in normalized and "X01" in normalized:
        normalized.remove("X01")
        normalized.remove("X02")
        normalized = ["X01", "X02", *normalized]
    results = []
    for experiment_id in normalized:
        for model_seed in model_seeds:
            results.append(run_experiment_v1(experiment_id, model_seed, config))
    return results


def preflight_v1(config: RunnerConfigV1) -> dict[str, Any]:
    device = _resolve_device(config.device)
    x01_data = prepare_experiment_data_v1("X01", split_seed=config.split_seed)
    if x01_data.force_scale is None:
        raise RuntimeError("X01 force scale is unavailable")
    x01_model = build_seeded_experiment_model_v1(
        "X01", x01_data.force_scale, DEFAULT_MODEL_SEEDS[0], device
    )
    bucket = x01_data.train[0]
    centers = torch.from_numpy(bucket.centers_world[:1].astype(np.float32)).to(device)
    frames = torch.from_numpy(bucket.frames_body_to_world[:1].astype(np.float32)).to(device)
    pair_index = torch.from_numpy(bucket.pair_index.copy()).to(device)
    with torch.inference_mode():
        x01_output = x01_model.core_output(centers, frames, pair_index)
    if not bool(torch.isfinite(x01_output.normalized_node_force_world).all()):
        raise RuntimeError("preflight output is nonfinite")

    x06_data = prepare_experiment_data_v1("X06", split_seed=config.split_seed)
    if x06_data.force_scale is None:
        raise RuntimeError("X06 force scale is unavailable")
    x06_model = build_seeded_experiment_model_v1(
        "X06", x06_data.force_scale, DEFAULT_MODEL_SEEDS[0], device
    )
    x06_bucket = x06_data.train[0]
    centers = torch.from_numpy(
        x06_bucket.centers_world[:2].astype(np.float32)
    ).to(device)
    frames = torch.from_numpy(
        x06_bucket.frames_body_to_world[:2].astype(np.float32)
    ).to(device)
    target = torch.from_numpy(
        (x06_bucket.node_force_world[:2] / x06_data.force_scale).astype(np.float32)
    ).to(device)
    pair_index = torch.from_numpy(x06_bucket.pair_index.copy()).to(device)
    prediction = x06_model.normalized_forces_world(centers, frames, pair_index)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    finite_gradient_count = 0
    for parameter in x06_model.parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("preflight gradient is nonfinite")
            finite_gradient_count += 1
    if finite_gradient_count == 0:
        raise RuntimeError("preflight did not produce parameter gradients")
    return {
        "status": "passed",
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "core_source_sha256": _sha256(
            MODULE_DIRECTORY / "e311_gnn_12_experiment_core_v1.py"
        ),
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "x01_architecture": dict(x01_model.architecture_record()),
        "x01_output_shape": tuple(x01_output.normalized_node_force_world.shape),
        "x01_force_scale": x01_data.force_scale,
        "x06_architecture": dict(x06_model.architecture_record()),
        "x06_output_shape": tuple(prediction.shape),
        "x06_force_scale": x06_data.force_scale,
        "finite_gradient_parameter_count": finite_gradient_count,
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        "--study-root",
        dest="output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--five-benzene-batch-size",
        type=int,
        default=DEFAULT_FIVE_BENZENE_BATCH_SIZE,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--three-body-base-count", type=int, default=DEFAULT_THREE_BODY_BASE_COUNT
    )
    parser.add_argument(
        "--three-body-interventions",
        type=int,
        default=DEFAULT_THREE_BODY_INTERVENTIONS,
    )
    parser.add_argument("--comet-project", default=DEFAULT_COMET_PROJECT)
    parser.add_argument("--comet-workspace")
    parser.add_argument("--disable-comet", action="store_true")


def build_argument_parser_v1() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_common_arguments(preflight)
    run = subparsers.add_parser("run")
    run.add_argument("experiment_id")
    run.add_argument("--seed", type=int, default=DEFAULT_MODEL_SEEDS[0])
    _add_common_arguments(run)
    run_group = subparsers.add_parser("run-group")
    run_group.add_argument(
        "--experiments",
        nargs="+",
        default=[f"X{index:02d}" for index in range(1, 13)],
    )
    run_group.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_MODEL_SEEDS)
    )
    _add_common_arguments(run_group)
    return parser


def _config_from_args(args: argparse.Namespace) -> RunnerConfigV1:
    return RunnerConfigV1(
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        five_benzene_batch_size=args.five_benzene_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        split_seed=args.split_seed,
        device=args.device,
        three_body_base_count=args.three_body_base_count,
        three_body_interventions=args.three_body_interventions,
        comet_project=args.comet_project,
        comet_workspace=args.comet_workspace,
        comet_enabled=not args.disable_comet,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser_v1().parse_args(argv)
    config = _config_from_args(args)
    if args.command == "preflight":
        print(json.dumps(_json_ready(preflight_v1(config)), indent=2))
    elif args.command == "run":
        result = run_experiment_v1(args.experiment_id, args.seed, config)
        print(json.dumps(_json_ready(result), indent=2))
    else:
        results = run_group_v1(args.experiments, args.seeds, config)
        print(json.dumps(_json_ready(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_FIVE_BENZENE_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MODEL_SEEDS",
    "DEFAULT_WEIGHT_DECAY",
    "GraphBucketV1",
    "NodeMetricsV1",
    "PreparedExperimentDataV1",
    "RunnerConfigV1",
    "SplitIndicesV1",
    "build_argument_parser_v1",
    "build_seeded_experiment_model_v1",
    "compute_node_metrics_v1",
    "deterministic_group_split_v1",
    "generate_three_body_chain_v1",
    "prepare_experiment_data_v1",
    "preflight_v1",
    "run_experiment_v1",
    "run_group_v1",
    "scatter_pair_forces_v1",
]
