"""Core runners for the Y13--Y15 exact-E311 pair controls.

Y13 delegates to the unchanged E-series runner. Y14 reuses the E-series data,
split, optimizer, scheduler, normalization, checkpoint selection, and metrics,
changing only the model to the parameter-free OddGraph wrapper. Y15 uses the
same optimizer budget per unordered edge on configurable five-benzene data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from experiments.benzene_pair import sweep30 as e_common
from experiments.benzene_pair.comet_logging import NullCometTrialLogger
from experiments.benzene_pair.data.benzene_cluster import (
    load_benzene_cluster_csv,
)
from experiments.benzene_pair.e_series import runner as e_runner
from experiments.gnn.e311_y13_y15_pair_control_core_v1 import (
    E311OddGraphCoreV1,
    E311TwoNodeOddControlV1,
    HISTORICAL_E311_MODEL_ID,
    HISTORICAL_E311_PARAMETER_COUNT,
    HISTORICAL_MODEL_SEED,
    HISTORICAL_SHUFFLE_SEED,
    HISTORICAL_SPLIT_SEED,
    assert_historical_e311_definition_v1,
    build_e311_odd_graph_core_v1,
    build_y14_two_node_control_v1,
    complete_pair_index_v1,
    get_y_pair_control_spec_v1,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "gnn" / "runs" / "e311_y13_y15_pair_controls_v1"
)
Y15_LEARNING_RATE = 0.003
Y15_WEIGHT_DECAY = 1.0e-4
Y15_SCHEDULER_GAMMA = 0.5
Y15_TF32 = True
Y15_RECONSTRUCTION_TOLERANCE = 1.0e-9
Y15_OPLS_VERSION = "2.0.0"
Y15_OPLS_COMMIT = "a5f874ed00152b156cd2525c961bd81030237e31"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("e311_y13_y15_pair_control_core_v1.py"),
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "e_series" / "catalog.py",
        REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "e_series"
        / "model_factory.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "e_series" / "runner.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "sweep30.py",
        REPOSITORY_ROOT / "src" / "TFENN" / "models" / "invariant_gate_pipeline_v2.py",
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    partial.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf_8",
    )
    os.replace(partial, path)


def _write_history_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("history cannot be empty")
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ("git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _assert_y13_preflight(study_root: Path) -> Mapping[str, Any]:
    preflight = e_runner._require_preflight(study_root)
    record = e_runner._preflight_record(preflight, HISTORICAL_E311_MODEL_ID)
    if (
        record.get("status") != "passed"
        or record.get("planned_parameter_count")
        != HISTORICAL_E311_PARAMETER_COUNT
        or record.get("actual_parameter_count")
        != HISTORICAL_E311_PARAMETER_COUNT
    ):
        raise RuntimeError(f"Y13 E311 preflight mismatch: {record}")
    return preflight


def run_y13_exact_reproduction_v1(study_root: Path, device: str) -> Path:
    """Run the exact historical non-GNN E311 path without reimplementation."""

    assert_historical_e311_definition_v1()
    study_root = study_root.resolve()
    completed = (
        study_root
        / "e3_path_gate_width"
        / "models"
        / HISTORICAL_E311_MODEL_ID
        / "summary.json"
    )
    if completed.exists():
        raise FileExistsError(
            "Y13 requires a fresh study root; an E311 summary already exists"
        )
    if not os.environ.get("COMET_API_KEY", "").strip():
        raise RuntimeError("formal exact Y13 requires COMET_API_KEY")
    e_runner.main(("prepare", "--study_root", str(study_root)))
    preflight = _assert_y13_preflight(study_root)
    alias = {
        "schema_name": "tfenn_y13_exact_e311_alias",
        "schema_version": 1,
        "experiment": get_y_pair_control_spec_v1("Y13").as_dict(),
        "delegated_module": "experiments.benzene_pair.e_series.runner",
        "delegated_command": {
            "experiment": 3,
            "model": HISTORICAL_E311_MODEL_ID,
            "device": device,
        },
        "preflight_hash": preflight["preflight_hash"],
        "source_sha256": _source_sha256(),
        "git_commit": _git_commit(),
        "created_at_utc": _utc_now(),
    }
    _write_json_atomic(study_root / "y13_exact_e311_alias.json", alias)
    e_runner.main(
        (
            "run",
            "--experiment",
            "3",
            "--study_root",
            str(study_root),
            "--model",
            HISTORICAL_E311_MODEL_ID,
            "--device",
            device,
        )
    )
    if not completed.is_file():
        raise RuntimeError("the delegated E311 run did not produce a summary")
    return completed


def _build_y14_model(_spec: object, device: str) -> E311TwoNodeOddControlV1:
    result = build_y14_two_node_control_v1(
        dtype=torch.float32,
        device=device,
        seed=HISTORICAL_MODEL_SEED,
    )
    if _parameter_count(result) != HISTORICAL_E311_PARAMETER_COUNT:
        raise RuntimeError("Y14 parameter count changed")
    return result


def _sum_abs(difference: Tensor) -> tuple[float, int]:
    return float(difference.abs().sum().detach().cpu()), difference.numel()


def _y14_selected_audit(
    *,
    model: nn.Module,
    data: e_common.TrainingData,
    split: e_common.SplitIndices,
    config: e_common.SweepConfig,
    device: str,
    target_scale: Tensor,
    **_unused: Any,
) -> dict[str, Any]:
    if not isinstance(model, E311TwoNodeOddControlV1):
        raise TypeError("Y14 audit received the wrong model")
    scale = float(target_scale.detach().cpu())
    totals = {
        "raw_forward": 0.0,
        "raw_reverse": 0.0,
        "odd_pair": 0.0,
        "even_leakage": 0.0,
        "direct_parity": 0.0,
    }
    count = 0
    max_net_force = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, int(split.test.numel()), config.micro_batch_size):
            selection = split.test[start : start + config.micro_batch_size]
            centers = data.centers[selection].to(device)
            frames = data.frames[selection].to(device)
            target = data.root_force[selection].to(device) / target_scale
            output = model.core_output(centers, frames)
            forward = output.raw_forward_world[..., 0, :]
            reverse = output.raw_reverse_world[..., 0, :]
            odd = output.normalized_pair_force_world[..., 0, :]
            direct = model.pair_kernel(centers, frames)
            pieces = {
                "raw_forward": forward - target,
                "raw_reverse": reverse + target,
                "odd_pair": odd - target,
                "even_leakage": 0.5 * (forward + reverse),
                "direct_parity": forward - direct,
            }
            for name, difference in pieces.items():
                value, piece_count = _sum_abs(difference)
                totals[name] += value
                if name == "odd_pair":
                    count += piece_count
            net = output.normalized_node_force_world.sum(dim=-2)
            max_net_force = max(
                max_net_force,
                float(net.abs().max().detach().cpu()),
            )
    model.train(was_training)
    return {
        "metric_unit": "kcal_per_mol_per_angstrom",
        "raw_forward_mae": totals["raw_forward"] * scale / count,
        "raw_reverse_mae": totals["raw_reverse"] * scale / count,
        "odd_pair_mae": totals["odd_pair"] * scale / count,
        "even_leakage_mae": totals["even_leakage"] * scale / count,
        "raw_forward_direct_call_mae": totals["direct_parity"] * scale / count,
        "maximum_normalized_net_force_component": max_net_force,
        "test_component_count": count,
        "two_directions_share_one_kernel_call": True,
        "running_rms_population": "both_endpoint_orientations",
    }


def run_y14_odd_graph_400k_v1(
    e_study_root: Path,
    output_directory: Path,
    device: str,
) -> Path:
    """Run Y14 on the exact Y13 data and split using the common E runner."""

    spec = get_y_pair_control_spec_v1("Y14")
    e_study_root = e_study_root.resolve()
    split_directory = e_runner._shared_split_directory(e_study_root)
    split, split_manifest = e_common._load_split(split_directory)
    config = e_runner.make_config(3, study_root=e_study_root)
    if (
        config.expected_sample_count != spec.sample_count
        or config.epochs != spec.epochs
        or config.effective_batch_size != spec.graph_batch_size
        or config.micro_batch_size != spec.graph_batch_size
        or config.learning_rate != 0.003
        or config.weight_decay != 1.0e-4
        or config.scheduler_step_size != spec.scheduler_step_size
        or config.scheduler_gamma != 0.5
        or config.validation_every != 1
        or config.split_seed != HISTORICAL_SPLIT_SEED
        or config.model_seed != HISTORICAL_MODEL_SEED
        or config.shuffle_seed != HISTORICAL_SHUFFLE_SEED
        or config.split_fractions != (0.8, 0.1, 0.1)
        or config.dtype != "float32"
        or not config.enable_tf32
        or config.expected_dataset_revision != 3
        or config.expected_opls_version != "2.0.0"
    ):
        raise RuntimeError("Y14 no longer matches the historical E311 protocol")
    output_directory = output_directory.resolve()
    paths = e_common.TrialPaths.create(output_directory)
    if paths.summary.exists():
        raise FileExistsError(paths.summary)
    config = replace(config, study_directory=output_directory.parent)
    e_common.run_trial(
        config,
        spec,
        paths,
        split,
        split_manifest,
        NullCometTrialLogger(),
        device=str(_resolve_device(device)),
        epochs=spec.epochs,
        model_builder=_build_y14_model,
        selected_model_audit_hook=_y14_selected_audit,
        source_sha256=_source_sha256(),
        study_metadata={
            "series": "Y13_Y15_pair_controls_v1",
            "experiment_id": "Y14",
            "shared_split_source": str(split_directory),
            "shared_split_manifest_hash": split_manifest["manifest_hash"],
            "only_trainable_module": "historical_E311",
            "running_rms_population": "both_endpoint_orientations",
            "train_labeled_pair_exposures": 160_000_000,
            "train_ordered_kernel_evaluations": 320_000_000,
        },
    )
    if not paths.summary.is_file():
        raise RuntimeError("Y14 did not produce a summary")
    return paths.summary


@dataclass(frozen=True, slots=True)
class Y15ArraysV1:
    centers_world: np.ndarray
    frames_body_to_world: np.ndarray
    pair_force_world: np.ndarray
    node_force_world: np.ndarray
    pair_index: np.ndarray
    group_id: np.ndarray
    records: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.centers_world.shape[0])


@dataclass(frozen=True, slots=True)
class SplitIndicesV1:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "validation": int(self.validation.size),
            "test": int(self.test.size),
        }


def _deterministic_group_split(
    group_id: np.ndarray,
    seed: int,
) -> SplitIndicesV1:
    groups = np.asarray(group_id, dtype=np.int64)
    if groups.ndim != 1:
        raise ValueError("group_id must be one-dimensional")
    unique = np.unique(groups)
    if unique.size < 10:
        raise ValueError("at least ten independent groups are required")
    order = np.random.default_rng(seed).permutation(unique)
    train_count = math.floor(0.8 * len(order))
    validation_count = math.floor(0.1 * len(order))
    train_groups = order[:train_count]
    validation_groups = order[train_count : train_count + validation_count]
    test_groups = order[train_count + validation_count :]
    result = SplitIndicesV1(
        np.flatnonzero(np.isin(groups, train_groups)).astype(np.int64),
        np.flatnonzero(np.isin(groups, validation_groups)).astype(np.int64),
        np.flatnonzero(np.isin(groups, test_groups)).astype(np.int64),
    )
    selected = (
        set(groups[result.train].tolist()),
        set(groups[result.validation].tolist()),
        set(groups[result.test].tolist()),
    )
    if (
        selected[0] & selected[1]
        or selected[0] & selected[2]
        or selected[1] & selected[2]
    ):
        raise RuntimeError("group split leaked configurations")
    combined = np.concatenate((result.train, result.validation, result.test))
    if not np.array_equal(np.sort(combined), np.arange(len(groups))):
        raise RuntimeError("split is not a complete partition")
    return result


def _numpy_signed_scatter(
    pair_force: np.ndarray,
    pair_index: np.ndarray,
    node_count: int,
) -> np.ndarray:
    result = np.zeros(
        (pair_force.shape[0], node_count, 3),
        dtype=pair_force.dtype,
    )
    for edge, (first, second) in enumerate(pair_index.T):
        result[:, first] += pair_force[:, edge]
        result[:, second] -= pair_force[:, edge]
    return result


def load_y15_arrays_v1(
    csv_path: Path,
    pair_npz_path: Path,
    *,
    expected_sample_count: int = 100_000,
) -> Y15ArraysV1:
    """Load and strictly validate five-benzene pair supervision."""

    csv_path = csv_path.resolve()
    pair_npz_path = pair_npz_path.resolve()
    csv_metadata_path = csv_path.with_suffix(".json")
    csv_validation_path = csv_path.with_suffix(".validation.json")
    pair_metadata_path = pair_npz_path.with_suffix(".json")
    for path in (
        csv_path,
        csv_metadata_path,
        csv_validation_path,
        pair_npz_path,
        pair_metadata_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    csv_sha = _sha256(csv_path)
    csv_metadata = json.loads(csv_metadata_path.read_text(encoding="utf_8"))
    csv_validation = json.loads(csv_validation_path.read_text(encoding="utf_8"))
    csv_generator = (
        REPOSITORY_ROOT / "experiments" / "gnn" / "data" / "five_benzene.py"
    )
    if (
        csv_metadata.get("schema_name") != "tfenn_five_benzene_gnn_dataset"
        or csv_metadata.get("schema_version") != 1
        or csv_metadata.get("dataset_revision") != 1
        or csv_metadata.get("sample_count") != expected_sample_count
        or csv_metadata.get("molecule_count") != 5
        or csv_metadata.get("row_count") != expected_sample_count * 5
        or csv_metadata.get("rows_per_sample") != 5
        or csv_metadata.get("csv_sha256") != csv_sha
        or csv_metadata.get("generator_sha256") != _sha256(csv_generator)
        or csv_metadata.get("opls", {}).get("runtime_version")
        != Y15_OPLS_VERSION
        or csv_metadata.get("opls", {}).get("source_commit")
        != Y15_OPLS_COMMIT
        or csv_metadata.get("coordinate_convention", {}).get("frame")
        != "single_root_molecule_0_body_frame"
        or csv_metadata.get("coordinate_convention", {}).get(
            "rotation_semantics"
        )
        != "active_body_to_root_frame"
        or csv_metadata.get("units", {}).get("force")
        != "kcal_per_mol_per_angstrom"
        or csv_metadata.get("statistics", {}).get("unique_configuration_count")
        != expected_sample_count
    ):
        raise ValueError("five-benzene CSV metadata hash mismatch")
    if (
        csv_validation.get("passed") is not True
        or csv_validation.get("csv_sha256") != csv_sha
    ):
        raise ValueError("five-benzene CSV validation is missing or stale")
    arrays = load_benzene_cluster_csv(csv_path)
    if arrays.molecule_count != 5 or len(arrays) != expected_sample_count:
        raise ValueError("Y15 requires exactly 100k five-benzene configurations")

    pair_sha = _sha256(pair_npz_path)
    pair_metadata = json.loads(pair_metadata_path.read_text(encoding="utf_8"))
    expected_pair_index = complete_pair_index_v1(5).T.cpu().numpy()
    expected_pair_list = expected_pair_index.tolist()
    pair_generator = (
        REPOSITORY_ROOT
        / "experiments"
        / "gnn"
        / "data"
        / "derive_pair_forces.py"
    )
    pair_contract = pair_metadata.get("pair_contract", {})
    pair_validation = pair_metadata.get("validation", {})
    pair_arrays = pair_metadata.get("arrays", {})
    try:
        reverse_residual = float(
            pair_validation.get("maximum_reverse_component_residual", math.nan)
        )
        reaggregation_residual = float(
            pair_validation.get(
                "maximum_reaggregation_component_residual",
                math.nan,
            )
        )
    except (TypeError, ValueError):
        reverse_residual = math.nan
        reaggregation_residual = math.nan
    if (
        pair_metadata.get("schema_name")
        != "tfenn_five_benzene_pair_force_supervision"
        or pair_metadata.get("schema_version") != 1
        or pair_metadata.get("source", {}).get("csv_sha256") != csv_sha
        or pair_metadata.get("source", {}).get("sample_count")
        != expected_sample_count
        or pair_metadata.get("source", {}).get("molecule_count") != 5
        or pair_metadata.get("opls_runtime", {}).get("runtime_version")
        != Y15_OPLS_VERSION
        or pair_metadata.get("opls_runtime", {}).get("source_commit")
        != Y15_OPLS_COMMIT
        or pair_contract.get("pair_count") != 10
        or pair_contract.get("pair_index") != expected_pair_list
        or pair_contract.get("orientation")
        != "first index receives force from second index"
        or pair_contract.get("reverse_force") != "negative of stored force"
        or pair_contract.get("aggregation")
        != (
            "add stored force to first index and subtract it from second index"
        )
        or pair_arrays.get("sample_id")
        != {"shape": [expected_sample_count], "dtype": "int64"}
        or pair_arrays.get("pair_index")
        != {"shape": [10, 2], "dtype": "int64"}
        or pair_arrays.get("pair_force_kcal_mol_A")
        != {
            "shape": [expected_sample_count, 10, 3],
            "dtype": "float64",
            "unit": "kcal_per_mol_per_angstrom",
        }
        or pair_validation.get("passed") is not True
        or not math.isfinite(reverse_residual)
        or reverse_residual < 0.0
        or reverse_residual > Y15_RECONSTRUCTION_TOLERANCE
        or not math.isfinite(reaggregation_residual)
        or reaggregation_residual < 0.0
        or reaggregation_residual > Y15_RECONSTRUCTION_TOLERANCE
        or pair_metadata.get("artifacts", {}).get("npz_sha256") != pair_sha
        or pair_metadata.get("artifacts", {}).get("generator_sha256")
        != _sha256(pair_generator)
    ):
        raise ValueError("pair-force provenance or contract mismatch")
    with np.load(pair_npz_path, allow_pickle=False) as archive:
        sample_id = np.asarray(archive["sample_id"])
        stored_pair_index = np.asarray(archive["pair_index"])
        pair_force = np.asarray(archive["pair_force_kcal_mol_A"])
        group_id = np.asarray(
            archive["group_id"] if "group_id" in archive.files else sample_id
        )
    if (
        sample_id.dtype != np.dtype(np.int64)
        or stored_pair_index.dtype != np.dtype(np.int64)
        or group_id.dtype != np.dtype(np.int64)
        or pair_force.dtype != np.dtype(np.float64)
    ):
        raise ValueError("Y15 NPZ array dtypes do not match the locked contract")
    if not np.array_equal(stored_pair_index, expected_pair_index):
        raise ValueError("Y15 pair_index must be the canonical ten-edge complete graph")
    if not np.array_equal(sample_id, np.arange(expected_sample_count)):
        raise ValueError("pair-force sample IDs are not canonical")
    if pair_force.shape != (expected_sample_count, 10, 3):
        raise ValueError("pair-force array must have shape (100000, 10, 3)")
    if group_id.shape != (expected_sample_count,):
        raise ValueError("group_id must have one entry per configuration")
    if not np.array_equal(group_id, sample_id):
        raise ValueError(
            "formal Y15 requires independent configurations "
            "with group_id equal to sample_id"
        )
    if not np.isfinite(pair_force).all():
        raise ValueError("pair-force labels contain nonfinite values")

    pair_index = stored_pair_index.T.copy()
    reconstructed = _numpy_signed_scatter(pair_force, pair_index, 5)
    node_force = np.asarray(arrays.forces, dtype=np.float64)
    reconstruction_residual = float(np.max(np.abs(reconstructed - node_force)))
    if reconstruction_residual > Y15_RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            "pair-force labels do not reconstruct five-benzene node forces: "
            f"{reconstruction_residual}"
        )
    return Y15ArraysV1(
        centers_world=np.ascontiguousarray(arrays.centers, dtype=np.float32),
        frames_body_to_world=np.ascontiguousarray(
            arrays.rotations,
            dtype=np.float32,
        ),
        pair_force_world=np.ascontiguousarray(pair_force, dtype=np.float32),
        node_force_world=np.ascontiguousarray(node_force, dtype=np.float32),
        pair_index=np.ascontiguousarray(pair_index, dtype=np.int64),
        group_id=np.ascontiguousarray(group_id, dtype=np.int64),
        records={
            "csv_path": str(csv_path),
            "csv_sha256": csv_sha,
            "csv_metadata_sha256": _sha256(csv_metadata_path),
            "csv_validation_sha256": _sha256(csv_validation_path),
            "pair_npz_path": str(pair_npz_path),
            "pair_npz_sha256": pair_sha,
            "pair_metadata_sha256": _sha256(pair_metadata_path),
            "sample_count": expected_sample_count,
            "molecule_count": 5,
            "pair_count": 10,
            "pair_to_node_maximum_component_residual": reconstruction_residual,
            "group_source": "one_independent_group_per_configuration",
        },
    )


def _make_loader(
    dataset: TensorDataset,
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    subset = Subset(dataset, indices.tolist())
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _move_batch(
    batch: Sequence[Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return tuple(
        item.to(device=device, non_blocking=device.type == "cuda")
        for item in batch
    )  # type: ignore[return-value]


def _evaluate_y15_loss(
    model: E311OddGraphCoreV1,
    loader: DataLoader[Any],
    pair_index: Tensor,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    squared_sum = 0.0
    component_count = 0
    with torch.no_grad():
        for batch in loader:
            centers, frames, pair_target, _node_target = _move_batch(batch, device)
            prediction = model(centers, frames, pair_index)
            difference = prediction - pair_target
            squared_sum += float(difference.square().sum().cpu())
            component_count += difference.numel()
    model.train(was_training)
    return squared_sum / component_count


def _save_y15_checkpoint(
    path: Path,
    model: E311OddGraphCoreV1,
    *,
    epoch: int,
    validation_loss: float,
    target_scale: float,
    source_sha256: str,
) -> None:
    e_common._atomic_torch_save(
        path,
        {
            "schema_name": "tfenn_y15_selected_checkpoint",
            "schema_version": 1,
            "epoch": epoch,
            "validation_normalized_mse": validation_loss,
            "target_scale": target_scale,
            "source_sha256": source_sha256,
            "parameter_state_dict": e_common._parameter_state(model),
            "normalization_state_dict": e_common._normalization_state(model),
            "calibration_state_dict": e_common._calibration_state(model),
        },
    )


def _restore_y15_checkpoint(
    path: Path,
    model: E311OddGraphCoreV1,
    *,
    source_sha256: str,
) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("source_sha256") != source_sha256:
        raise RuntimeError("Y15 checkpoint source hash changed")
    e_common._restore_model_state(
        model,
        payload["parameter_state_dict"],
        payload["normalization_state_dict"],
        payload.get("calibration_state_dict"),
    )
    return payload


def _selected_y15_metrics(
    model: E311OddGraphCoreV1,
    loader: DataLoader[Any],
    pair_index: Tensor,
    device: torch.device,
    target_scale: float,
) -> dict[str, Any]:
    totals = {
        "pair_abs": 0.0,
        "pair_square": 0.0,
        "node_abs": 0.0,
        "node_square": 0.0,
        "raw_forward_abs": 0.0,
        "raw_reverse_abs": 0.0,
        "even_abs": 0.0,
        "even_square": 0.0,
    }
    pair_components = 0
    node_components = 0
    max_net_force = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            centers, frames, pair_target_normalized, node_target = _move_batch(
                batch,
                device,
            )
            output = model.core_output(centers, frames, pair_index)
            pair_prediction = output.normalized_pair_force_world * target_scale
            pair_target = pair_target_normalized * target_scale
            node_prediction = output.normalized_node_force_world * target_scale
            pair_difference = pair_prediction - pair_target
            node_difference = node_prediction - node_target
            raw_forward = output.raw_forward_world * target_scale
            raw_reverse = output.raw_reverse_world * target_scale
            totals["pair_abs"] += float(pair_difference.abs().sum().cpu())
            totals["pair_square"] += float(pair_difference.square().sum().cpu())
            totals["node_abs"] += float(node_difference.abs().sum().cpu())
            totals["node_square"] += float(node_difference.square().sum().cpu())
            totals["raw_forward_abs"] += float(
                (raw_forward - pair_target).abs().sum().cpu()
            )
            totals["raw_reverse_abs"] += float(
                (raw_reverse + pair_target).abs().sum().cpu()
            )
            even_leakage = 0.5 * (raw_forward + raw_reverse)
            totals["even_abs"] += float(even_leakage.abs().sum().cpu())
            totals["even_square"] += float(even_leakage.square().sum().cpu())
            pair_components += pair_difference.numel()
            node_components += node_difference.numel()
            net = node_prediction.sum(dim=-2)
            max_net_force = max(
                max_net_force,
                float(net.abs().max().cpu()),
            )
    model.train(was_training)
    return {
        "pair_force": {
            "mae": totals["pair_abs"] / pair_components,
            "rmse": math.sqrt(totals["pair_square"] / pair_components),
            "normalized_mse": (
                totals["pair_square"]
                / pair_components
                / (target_scale * target_scale)
            ),
            "component_count": pair_components,
        },
        "node_force_after_sum": {
            "mae": totals["node_abs"] / node_components,
            "rmse": math.sqrt(totals["node_square"] / node_components),
            "normalized_mse_using_pair_component_rms": (
                totals["node_square"]
                / node_components
                / (target_scale * target_scale)
            ),
            "component_count": node_components,
        },
        "direction_audit": {
            "raw_forward_mae": totals["raw_forward_abs"] / pair_components,
            "raw_reverse_mae": totals["raw_reverse_abs"] / pair_components,
            "even_leakage_mae": totals["even_abs"] / pair_components,
            "even_leakage_rmse": math.sqrt(
                totals["even_square"] / pair_components
            ),
        },
        "maximum_net_force_component": max_net_force,
        "unit": "kcal_per_mol_per_angstrom",
    }


def _tensor_residual_v1(actual: Tensor, expected: Tensor) -> dict[str, float]:
    difference = actual - expected
    difference_l2 = torch.linalg.vector_norm(difference)
    expected_l2 = torch.linalg.vector_norm(expected)
    return {
        "maximum_absolute": float(difference.abs().max().cpu()),
        "relative_l2": float(
            (difference_l2 / expected_l2.clamp_min(1.0e-12)).cpu()
        ),
    }


def _selected_y15_symmetry_audit_v1(
    model: E311OddGraphCoreV1,
    loader: DataLoader[Any],
    pair_index: Tensor,
    device: torch.device,
    *,
    tolerance: float = 1.0e-4,
) -> dict[str, Any]:
    """Audit the selected graph model once, without changing its statistics."""

    try:
        batch = next(iter(loader))
    except StopIteration as error:
        raise RuntimeError("Y15 test loader is empty") from error
    centers, frames, _pair_target, _node_target = _move_batch(batch, device)
    audit_count = min(4, int(centers.shape[0]))
    centers = centers[:audit_count]
    frames = frames[:audit_count]
    was_training = model.training
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    model.eval()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            reference = model.core_output(centers, frames, pair_index)

            angle = torch.tensor(0.37, device=device, dtype=centers.dtype)
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            zero = torch.zeros((), device=device, dtype=centers.dtype)
            identity = torch.eye(3, device=device, dtype=centers.dtype)
            axis = torch.tensor(
                (0.31, -0.47, 0.82),
                device=device,
                dtype=centers.dtype,
            )
            axis = axis / torch.linalg.vector_norm(axis)
            axis_x, axis_y, axis_z = axis.unbind()
            cross_matrix = torch.stack(
                (
                    torch.stack((zero, -axis_z, axis_y)),
                    torch.stack((axis_z, zero, -axis_x)),
                    torch.stack((-axis_y, axis_x, zero)),
                )
            )
            rotation = (
                cosine * identity
                + (1.0 - cosine) * torch.outer(axis, axis)
                + sine * cross_matrix
            )
            translation = torch.tensor(
                (0.41, -0.73, 1.17),
                device=device,
                dtype=centers.dtype,
            )
            transformed_centers = torch.einsum(
                "ij,bnj->bni",
                rotation,
                centers,
            ) + translation
            transformed_frames = torch.einsum(
                "ij,bnjk->bnik",
                rotation,
                frames,
            )
            transformed = model.core_output(
                transformed_centers,
                transformed_frames,
                pair_index,
            )
            expected_transformed_node = torch.einsum(
                "ij,bnj->bni",
                rotation,
                reference.normalized_node_force_world,
            )

            generators = e_common._proper_d6_generators().to(
                device=device,
                dtype=centers.dtype,
            )
            gauges = torch.stack(
                (
                    identity,
                    generators[0],
                    generators[1],
                    generators[0] @ generators[0],
                    generators[0] @ generators[1],
                )
            )
            gauged_frames = torch.einsum("bnij,njk->bnik", frames, gauges)
            gauged = model.core_output(centers, gauged_frames, pair_index)

            permutation = torch.tensor(
                (2, 4, 0, 3, 1),
                device=device,
                dtype=torch.int64,
            )
            permuted = model.core_output(
                centers.index_select(-2, permutation),
                frames.index_select(-3, permutation),
                pair_index,
            )
            expected_permuted_node = (
                reference.normalized_node_force_world.index_select(
                    -2,
                    permutation,
                )
            )

            odd_definition = 0.5 * (
                reference.raw_forward_world - reference.raw_reverse_world
            )
            zero_net_force = torch.zeros_like(
                reference.normalized_node_force_world.sum(dim=-2)
            )

        residuals = {
            "global_se3_node_force": _tensor_residual_v1(
                transformed.normalized_node_force_world,
                expected_transformed_node,
            ),
            "independent_d6_gauge_node_force": _tensor_residual_v1(
                gauged.normalized_node_force_world,
                reference.normalized_node_force_world,
            ),
            "node_permutation": _tensor_residual_v1(
                permuted.normalized_node_force_world,
                expected_permuted_node,
            ),
            "oddpair_definition": _tensor_residual_v1(
                reference.normalized_pair_force_world,
                odd_definition,
            ),
            "zero_total_force": _tensor_residual_v1(
                reference.normalized_node_force_world.sum(dim=-2),
                zero_net_force,
            ),
        }
        return {
            "passed": all(
                record["maximum_absolute"] <= tolerance
                for record in residuals.values()
            ),
            "absolute_tolerance": tolerance,
            "sample_count": audit_count,
            "tf32_disabled_during_audit": device.type == "cuda",
            "residuals": residuals,
        }
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        model.train(was_training)


def run_y15_odd_graph_5b100k_v1(
    csv_path: Path,
    pair_npz_path: Path,
    output_directory: Path,
    device_value: str,
) -> Path:
    """Train the shared E311 OddGraph on 100k five-benzene configurations."""

    spec = get_y_pair_control_spec_v1("Y15")
    device = _resolve_device(device_value)
    output_directory = output_directory.resolve()
    summary_path = output_directory / "summary.json"
    history_path = output_directory / "history.csv"
    status_path = output_directory / "status.json"
    best_path = output_directory / "best.pt"
    final_path = output_directory / "final.pt"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    source_sha = _source_sha256()
    arrays = load_y15_arrays_v1(
        csv_path,
        pair_npz_path,
        expected_sample_count=spec.sample_count,
    )
    split = _deterministic_group_split(arrays.group_id, HISTORICAL_SPLIT_SEED)
    if split.counts() != {"train": 80_000, "validation": 10_000, "test": 10_000}:
        raise RuntimeError("Y15 formal split counts changed")
    target_scale = float(
        np.sqrt(np.mean(np.square(arrays.pair_force_world[split.train])))
    )
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise RuntimeError("Y15 target RMS is invalid")
    dataset = TensorDataset(
        torch.from_numpy(arrays.centers_world),
        torch.from_numpy(arrays.frames_body_to_world),
        torch.from_numpy(arrays.pair_force_world / target_scale),
        torch.from_numpy(arrays.node_force_world),
    )
    pin_memory = device.type == "cuda"
    train_loader = _make_loader(
        dataset,
        split.train,
        spec.graph_batch_size,
        shuffle=True,
        seed=HISTORICAL_SHUFFLE_SEED,
        pin_memory=pin_memory,
    )
    warm_loader = _make_loader(
        dataset,
        split.train,
        spec.graph_batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
    )
    validation_loader = _make_loader(
        dataset,
        split.validation,
        spec.graph_batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
    )
    test_loader = _make_loader(
        dataset,
        split.test,
        spec.graph_batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
    )

    _set_seed(HISTORICAL_MODEL_SEED)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Y15_TF32
        torch.backends.cudnn.allow_tf32 = Y15_TF32
    model = build_e311_odd_graph_core_v1(
        dtype=torch.float32,
        device=device,
        seed=HISTORICAL_MODEL_SEED,
    )
    if _parameter_count(model) != HISTORICAL_E311_PARAMETER_COUNT:
        raise RuntimeError("Y15 parameter count changed")
    pair_index = torch.from_numpy(arrays.pair_index).to(device)
    model.reset_normalization_stats()
    model.train()
    with torch.no_grad():
        for batch in warm_loader:
            centers, frames, _pair_target, _node_target = _move_batch(batch, device)
            model(centers, frames, pair_index)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Y15_LEARNING_RATE,
        weight_decay=Y15_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=spec.scheduler_step_size,
        gamma=Y15_SCHEDULER_GAMMA,
    )
    best_epoch = 0
    best_validation = _evaluate_y15_loss(
        model,
        validation_loader,
        pair_index,
        device,
    )
    _save_y15_checkpoint(
        best_path,
        model,
        epoch=best_epoch,
        validation_loss=best_validation,
        target_scale=target_scale,
        source_sha256=source_sha,
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "learning_rate": Y15_LEARNING_RATE,
            "train_normalized_mse": "",
            "validation_normalized_mse": best_validation,
            "epoch_duration_seconds": 0.0,
        }
    ]
    _write_history_atomic(history_path, history)
    _write_json_atomic(
        status_path,
        {
            "status": "running",
            "experiment_id": "Y15",
            "epoch": 0,
            "epochs": spec.epochs,
            "best_epoch": 0,
            "best_validation_normalized_mse": best_validation,
        },
    )

    global_step = 0
    for epoch in range(1, spec.epochs + 1):
        started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        squared_sum = 0.0
        component_count = 0
        for batch in train_loader:
            centers, frames, pair_target, _node_target = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(centers, frames, pair_index)
            loss = torch.nn.functional.mse_loss(prediction, pair_target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Y15 training loss became nonfinite")
            loss.backward()
            global_step += 1
            if global_step <= 4 or global_step % 100 == 0:
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                if any(
                    gradient is None or not bool(torch.isfinite(gradient).all())
                    for gradient in gradients
                ):
                    raise RuntimeError(
                        "Y15 encountered a missing or nonfinite gradient"
                    )
            optimizer.step()
            difference = prediction.detach() - pair_target
            squared_sum += float(difference.square().sum().cpu())
            component_count += difference.numel()
        scheduler.step()
        train_loss = squared_sum / component_count
        validation_loss = _evaluate_y15_loss(
            model,
            validation_loader,
            pair_index,
            device,
        )
        if validation_loss < best_validation:
            best_epoch = epoch
            best_validation = validation_loss
            _save_y15_checkpoint(
                best_path,
                model,
                epoch=epoch,
                validation_loss=validation_loss,
                target_scale=target_scale,
                source_sha256=source_sha,
            )
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_normalized_mse": train_loss,
            "validation_normalized_mse": validation_loss,
            "epoch_duration_seconds": time.perf_counter() - started,
        }
        history.append(row)
        _write_history_atomic(history_path, history)
        _write_json_atomic(
            status_path,
            {
                "status": "running",
                "experiment_id": "Y15",
                "epoch": epoch,
                "epochs": spec.epochs,
                "best_epoch": best_epoch,
                "best_validation_normalized_mse": best_validation,
            },
        )
        print(json.dumps({"experiment_id": "Y15", **row}), flush=True)

    selected = _restore_y15_checkpoint(
        best_path,
        model,
        source_sha256=source_sha,
    )
    selected_metrics = _selected_y15_metrics(
        model,
        test_loader,
        pair_index,
        device,
        target_scale,
    )
    selected_symmetry_audit = _selected_y15_symmetry_audit_v1(
        model,
        test_loader,
        pair_index,
        device,
    )
    e_common._atomic_torch_save(final_path, dict(selected))
    train_graphs = split.counts()["train"]
    optimizer_updates = spec.epochs * math.ceil(
        train_graphs / spec.graph_batch_size
    )
    if global_step != optimizer_updates:
        raise RuntimeError("Y15 optimizer update count changed")
    summary = {
        "schema_name": "tfenn_y15_e311_odd_graph_result",
        "schema_version": 1,
        "status": "complete",
        "experiment": spec.as_dict(),
        "model": {
            "parameter_count": _parameter_count(model),
            "architecture": dict(model.architecture_metadata),
            "pair_kernel": HISTORICAL_E311_MODEL_ID,
        },
        "protocol": {
            "epochs": spec.epochs,
            "graph_batch_size": spec.graph_batch_size,
            "unordered_edges_per_graph": spec.unordered_edge_count,
            "ordered_kernel_evaluations_per_graph": 2
            * spec.unordered_edge_count,
            "learning_rate": Y15_LEARNING_RATE,
            "weight_decay": Y15_WEIGHT_DECAY,
            "scheduler": "StepLR",
            "scheduler_step_size": spec.scheduler_step_size,
            "scheduler_gamma": Y15_SCHEDULER_GAMMA,
            "target_scale_component_rms": target_scale,
            "dtype": "float32",
            "enable_tf32_during_training": Y15_TF32,
            "torch_threads": 4,
            "split_seed": HISTORICAL_SPLIT_SEED,
            "model_seed": HISTORICAL_MODEL_SEED,
            "shuffle_seed": HISTORICAL_SHUFFLE_SEED,
            "optimizer_updates": optimizer_updates,
            "train_unordered_edge_exposures": (
                spec.epochs * train_graphs * spec.unordered_edge_count
            ),
            "train_ordered_kernel_evaluations": (
                2 * spec.epochs * train_graphs * spec.unordered_edge_count
            ),
            "running_rms_population": "both_endpoint_orientations",
        },
        "split": split.counts(),
        "selected_checkpoint": {
            "rule": "minimum validation normalized pair MSE",
            "best_epoch": int(selected["epoch"]),
            "best_validation_normalized_mse": float(
                selected["validation_normalized_mse"]
            ),
            "test_metrics_evaluated_once": True,
            "symmetry_audit_reuses_four_unlabeled_test_geometries": True,
        },
        "selected_test": selected_metrics,
        "selected_symmetry_audit": selected_symmetry_audit,
        "data": dict(arrays.records),
        "source_sha256": source_sha,
        "git_commit": _git_commit(),
        "artifacts": {
            "history": str(history_path),
            "history_sha256": _sha256(history_path),
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": _sha256(best_path),
            "final_checkpoint": str(final_path),
            "final_checkpoint_sha256": _sha256(final_path),
        },
        "completed_at_utc": _utc_now(),
    }
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(
        status_path,
        {
            "status": "complete",
            "experiment_id": "Y15",
            "epoch": spec.epochs,
            "best_epoch": best_epoch,
            "best_validation_normalized_mse": best_validation,
            "completed_at_utc": _utc_now(),
        },
    )
    return summary_path


def build_argument_parser_v1() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    y13 = commands.add_parser("y13", help="exact historical E311 reproduction")
    y13.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "Y13_exact_e311_400k",
    )
    y13.add_argument("--device", default="cuda")

    y14 = commands.add_parser("y14", help="two-node E311 OddGraph control")
    y14.add_argument("--e-study-root", type=Path, required=True)
    y14.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "Y14_e311_odd_graph_400k",
    )
    y14.add_argument("--device", default="cuda")

    y15 = commands.add_parser("y15", help="five-node E311 OddGraph control")
    y15.add_argument("--csv", type=Path, required=True)
    y15.add_argument("--pair-npz", type=Path, required=True)
    y15.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "Y15_e311_odd_graph_5b100k",
    )
    y15.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser_v1().parse_args(argv)
    if arguments.command == "y13":
        result = run_y13_exact_reproduction_v1(
            arguments.study_root,
            arguments.device,
        )
    elif arguments.command == "y14":
        result = run_y14_odd_graph_400k_v1(
            arguments.e_study_root,
            arguments.output_directory,
            arguments.device,
        )
    else:
        result = run_y15_odd_graph_5b100k_v1(
            arguments.csv,
            arguments.pair_npz,
            arguments.output_directory,
            arguments.device,
        )
    print(json.dumps({"status": "complete", "result": str(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SplitIndicesV1",
    "Y15ArraysV1",
    "build_argument_parser_v1",
    "load_y15_arrays_v1",
    "main",
    "run_y13_exact_reproduction_v1",
    "run_y14_odd_graph_400k_v1",
    "run_y15_odd_graph_5b100k_v1",
]
