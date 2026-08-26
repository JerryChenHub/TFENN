from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA = (
    Path(__file__).resolve().parent
    / "data"
    / "two_benzene_opls_2_0_0_2k_v1.csv"
)
DEFAULT_SPLIT_SEED = 20260824
DEFAULT_OUTPUT_NAME = "rotation_evaluation.json"
CHECKPOINT_NAMES = ("final_checkpoint.pt", "best_checkpoint.pt")
E311_MODEL_FAMILY = "e311_multibody_one_block_v1"


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


def _split_indices(sample_count: int, seed: int) -> tuple[np.ndarray, ...]:
    order = np.random.default_rng(seed).permutation(sample_count)
    train_count = int(math.floor(0.8 * sample_count))
    validation_count = int(math.floor(0.1 * sample_count))
    return (
        order[:train_count],
        order[train_count : train_count + validation_count],
        order[train_count + validation_count :],
    )


def _rotation_z(angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _frobenius_residual(
    actual: np.ndarray,
    expected: np.ndarray,
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        raise ValueError("actual and expected arrays must have the same shape")
    difference = np.subtract(actual_array, expected_array)
    absolute = math.sqrt(
        math.fsum(float(value) * float(value) for value in difference.flat)
    )
    reference = math.sqrt(
        math.fsum(float(value) * float(value) for value in expected_array.flat)
    )
    relative = absolute / max(reference, np.finfo(np.float64).tiny)
    return {
        "absolute_frobenius_residual": absolute,
        "relative_frobenius_residual": relative,
        "relative_frobenius_residual_percent": 100.0 * relative,
        "maximum_component_absolute_residual": float(
            np.max(np.abs(difference), initial=0.0)
        ),
        "reference_frobenius_norm": reference,
    }


def _matrix_product_3x3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != (3, 3) or right_array.shape != (3, 3):
        raise ValueError("matrix product requires two arrays with shape (3, 3)")
    return np.asarray(
        tuple(
            tuple(
                math.fsum(
                    float(left_array[row, inner])
                    * float(right_array[inner, column])
                    for inner in range(3)
                )
                for column in range(3)
            )
            for row in range(3)
        ),
        dtype=np.float64,
    )


def _row_vector_product(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vector_array = np.asarray(vector, dtype=np.float64)
    matrix_array = np.asarray(matrix, dtype=np.float64)
    if vector_array.shape != (3,) or matrix_array.shape != (3, 3):
        raise ValueError("row vector product requires shapes (3,) and (3, 3)")
    return np.asarray(
        tuple(
            math.fsum(
                float(vector_array[inner]) * float(matrix_array[inner, column])
                for inner in range(3)
            )
            for column in range(3)
        ),
        dtype=np.float64,
    )


def _right_rotated_orientations(
    rotations: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    result = np.asarray(rotations, dtype=np.float64).copy()
    if result.shape != (2, 3, 3):
        raise ValueError("two benzene rotations must have shape (2, 3, 3)")
    result[1] = _matrix_product_3x3(result[1], _rotation_z(angle_degrees))
    return result


def _naive_partial_covariance_expected(
    baseline_forces: np.ndarray,
    molecule_orientation: np.ndarray,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    body_rotation = _rotation_z(angle_degrees)
    orientation = np.asarray(molecule_orientation, dtype=np.float64)
    world_rotation = _matrix_product_3x3(
        _matrix_product_3x3(orientation, body_rotation),
        orientation.T,
    )
    expected = np.asarray(baseline_forces, dtype=np.float64).copy()
    expected[1] = _row_vector_product(expected[1], world_rotation.T)
    return expected, world_rotation


def _resolve_checkpoint(run_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        candidate = requested
        if not candidate.is_absolute() and not candidate.exists():
            candidate = run_path / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {candidate}")
        return candidate
    for name in CHECKPOINT_NAMES:
        candidate = run_path / name
        if candidate.is_file():
            return candidate.resolve()
    names = ", ".join(CHECKPOINT_NAMES)
    raise FileNotFoundError(f"run contains none of the expected checkpoints: {names}")


def _read_summary(run_path: Path) -> dict[str, Any] | None:
    summary_path = run_path / "summary.json"
    if not summary_path.is_file():
        return None
    value = json.loads(summary_path.read_text(encoding="utf_8"))
    if not isinstance(value, dict):
        raise ValueError("summary.json must contain a JSON object")
    return value


def _validate_split(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
) -> np.ndarray:
    values = np.asarray(actual)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a one dimensional integer array")
    resolved = values.astype(np.int64, copy=False)
    if not np.array_equal(resolved, expected):
        raise ValueError(
            f"{name} does not match the fixed seed {DEFAULT_SPLIT_SEED} split"
        )
    return resolved


def _fixed_split_record(
    run_path: Path,
    sample_count: int,
    checkpoint: dict[str, Any],
    summary: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_train, expected_validation, expected_test = _split_indices(
        sample_count,
        DEFAULT_SPLIT_SEED,
    )
    expected_hashes = {
        "train": _index_sha256(expected_train),
        "validation": _index_sha256(expected_validation),
        "test": _index_sha256(expected_test),
    }
    split_path = run_path / "split_indices.npz"
    source = "reproduced_numpy_default_rng"
    split_sha256 = None
    if split_path.is_file():
        with np.load(split_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            expected_keys = {
                "train_sample_id",
                "validation_sample_id",
                "test_sample_id",
            }
            if not expected_keys.issubset(keys):
                raise ValueError(
                    "split_indices.npz must contain train_sample_id, "
                    "validation_sample_id, and test_sample_id"
                )
            _validate_split(
                archive["train_sample_id"],
                expected_train,
                "train_sample_id",
            )
            _validate_split(
                archive["validation_sample_id"],
                expected_validation,
                "validation_sample_id",
            )
            test_indices = _validate_split(
                archive["test_sample_id"],
                expected_test,
                "test_sample_id",
            )
        source = "split_indices.npz_verified_against_fixed_seed"
        split_sha256 = _sha256(split_path)
    else:
        test_indices = expected_test

    checkpoint_seed = checkpoint.get("seed")
    summary_seed = None
    if summary is not None:
        split_summary = summary.get("split")
        if isinstance(split_summary, dict):
            summary_seed = split_summary.get("seed")
    for label, value in (
        ("checkpoint seed", checkpoint_seed),
        ("summary split seed", summary_seed),
    ):
        if value is not None and int(value) != DEFAULT_SPLIT_SEED:
            raise ValueError(
                f"{label} {value} does not match fixed seed {DEFAULT_SPLIT_SEED}"
            )

    checkpoint_hashes = checkpoint.get("split_indices_sha256")
    if checkpoint_hashes is not None and checkpoint_hashes != expected_hashes:
        raise ValueError("checkpoint split hashes do not match the fixed split")
    summary_hashes = None
    if summary is not None:
        split_summary = summary.get("split")
        if isinstance(split_summary, dict):
            summary_hashes = split_summary.get("indices_sha256")
    if summary_hashes is not None and summary_hashes != expected_hashes:
        raise ValueError("summary split hashes do not match the fixed split")

    record = {
        "seed": DEFAULT_SPLIT_SEED,
        "algorithm": "numpy_default_rng_permutation_80_10_10",
        "source": source,
        "split_indices_path": str(split_path.resolve()) if split_path.is_file() else None,
        "split_indices_sha256": split_sha256,
        "train_count": len(expected_train),
        "validation_count": len(expected_validation),
        "test_count": len(expected_test),
        "indices_sha256": expected_hashes,
        "first_test_sample_id": int(test_indices[0]),
    }
    return test_indices, record


def _opls_evaluate_forces(
    centers: np.ndarray,
    rotations: np.ndarray,
    configuration_id: str,
) -> np.ndarray:
    import opls2020
    from opls2020 import (
        DEFAULT_MODEL,
        DEFAULT_PARAMETER_CATALOG,
        MoleculeInstance,
        Pose,
        StaticEngine,
        SystemSpec,
        benzene,
    )

    distribution_version = version("opls2020-static")
    if opls2020.__version__ != "2.0.0" or distribution_version != "2.0.0":
        raise RuntimeError(
            "OPLS runtime and distribution must both be version 2.0.0"
        )
    species = benzene()
    molecules = tuple(
        MoleculeInstance(
            f"benzene_{index}",
            species.species_id,
            Pose.from_matrix(center, rotation),
        )
        for index, (center, rotation) in enumerate(
            zip(centers, rotations, strict=True)
        )
    )
    system = SystemSpec(
        configuration_id=configuration_id,
        species={species.species_id: species},
        molecules=molecules,
        random_seed=None,
    )
    engine = StaticEngine(
        model=DEFAULT_MODEL,
        parameters=DEFAULT_PARAMETER_CATALOG,
        use_neighbor_list=False,
    )
    result = engine.evaluate(system)
    return np.asarray(result.molecular_forces_kcal_mol_A, dtype=np.float64)


def _opls_worker(payload_text: str) -> int:
    payload = json.loads(payload_text)
    centers = np.asarray(payload["centers"], dtype=np.float64)
    rotations = np.asarray(payload["rotations"], dtype=np.float64)
    stored_forces = np.asarray(payload["stored_forces"], dtype=np.float64)
    sample_id = int(payload["sample_id"])
    if centers.shape != (2, 3) or rotations.shape != (2, 3, 3):
        raise ValueError("OPLS worker received invalid two benzene geometry")

    baseline = _opls_evaluate_forces(
        centers,
        rotations,
        f"rotation_audit_{sample_id}_0",
    )
    angle_60 = _opls_evaluate_forces(
        centers,
        _right_rotated_orientations(rotations, 60.0),
        f"rotation_audit_{sample_id}_60",
    )
    angle_45 = _opls_evaluate_forces(
        centers,
        _right_rotated_orientations(rotations, 45.0),
        f"rotation_audit_{sample_id}_45",
    )
    naive_expected, world_rotation = _naive_partial_covariance_expected(
        baseline,
        rotations[1],
        45.0,
    )
    result = {
        "status": "available",
        "runtime_version": "2.0.0",
        "distribution_version": version("opls2020-static"),
        "evaluation_process": "separate_process_without_torch",
        "baseline_forces_kcal_mol_A": baseline.tolist(),
        "angle_60_forces_kcal_mol_A": angle_60.tolist(),
        "angle_45_forces_kcal_mol_A": angle_45.tolist(),
        "stored_baseline": _frobenius_residual(baseline, stored_forces),
        "angle_60_ground_truth_change": _frobenius_residual(angle_60, baseline),
        "angle_45_ground_truth_change": _frobenius_residual(angle_45, baseline),
        "angle_45_naive_partial_covariance": _frobenius_residual(
            angle_45,
            naive_expected,
        ),
        "angle_45_naive_world_rotation": world_rotation.tolist(),
    }
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    return 0


def _run_opls_recompute(
    centers: np.ndarray,
    rotations: np.ndarray,
    stored_forces: np.ndarray,
    sample_id: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "centers": centers.tolist(),
            "rotations": rotations.tolist(),
            "stored_forces": stored_forces.tolist(),
            "sample_id": sample_id,
        },
        separators=(",", ":"),
    )
    repository_root = Path(__file__).resolve().parents[2]
    command = (
        sys.executable,
        "-m",
        "experiments.gnn.evaluate_two_benzene_rotations",
        "--opls-worker",
        payload,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"OPLS worker exited with code {completed.returncode}: {error[-2000:]}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("OPLS worker returned no JSON result")
        value = json.loads(lines[-1])
        if not isinstance(value, dict) or value.get("status") != "available":
            raise RuntimeError("OPLS worker returned an invalid result")
        value["model_test_is_physical_correctness_test"] = False
        return value
    except Exception as error:
        return {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "error": str(error),
            "model_test_is_physical_correctness_test": False,
            "interpretation": (
                "Model symmetry metrics alone do not establish physical correctness."
            ),
        }


def _dtype_from_checkpoint(torch_module: Any, value: Any) -> Any:
    name = str(value)
    if name in {"float32", "torch.float32"}:
        return torch_module.float32
    if name in {"float64", "torch.float64"}:
        return torch_module.float64
    raise ValueError(f"unsupported checkpoint dtype: {value}")


def _resolve_model_family(checkpoint: dict[str, Any]) -> tuple[str, str]:
    value = checkpoint.get("model_family")
    if not isinstance(value, str):
        raise ValueError("checkpoint model_family must identify the E311 model")
    if value != E311_MODEL_FAMILY:
        raise ValueError(f"unsupported checkpoint model_family: {value}")
    return E311_MODEL_FAMILY, "explicit_checkpoint_field"


def _running_rms_buffer_bytes(model: Any) -> dict[str, tuple[Any, ...]]:
    snapshot: dict[str, tuple[Any, ...]] = {}
    for name, buffer in model.named_buffers():
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name not in {"mean_square", "sample_count"}:
            continue
        value = buffer.detach().cpu().contiguous()
        snapshot[name] = (
            str(value.dtype),
            tuple(value.shape),
            value.numpy().tobytes(),
        )
    return snapshot


def _opls_model_comparison(
    opls_ground_truth: dict[str, Any],
    baseline: np.ndarray,
    angle_60: np.ndarray,
    angle_45: np.ndarray,
) -> None:
    if opls_ground_truth.get("status") != "available":
        return
    truth_baseline = np.asarray(
        opls_ground_truth["baseline_forces_kcal_mol_A"],
        dtype=np.float64,
    )
    truth_angle_60 = np.asarray(
        opls_ground_truth["angle_60_forces_kcal_mol_A"],
        dtype=np.float64,
    )
    truth_angle_45 = np.asarray(
        opls_ground_truth["angle_45_forces_kcal_mol_A"],
        dtype=np.float64,
    )
    expected_shape = np.asarray(baseline).shape
    for name, value in (
        ("baseline", truth_baseline),
        ("angle_60", truth_angle_60),
        ("angle_45", truth_angle_45),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"OPLS {name} forces have shape {value.shape}, expected {expected_shape}"
            )

    angle_0_error = _frobenius_residual(baseline, truth_baseline)
    angle_60_error = _frobenius_residual(angle_60, truth_angle_60)
    angle_45_error = _frobenius_residual(angle_45, truth_angle_45)
    delta_model = np.subtract(angle_45, baseline)
    delta_opls = np.subtract(truth_angle_45, truth_baseline)
    delta_error = _frobenius_residual(delta_model, delta_opls)
    opls_ground_truth["model_comparison"] = {
        "angle_0_model_vs_opls": angle_0_error,
        "angle_60_model_vs_opls": angle_60_error,
        "angle_45_model_vs_opls": angle_45_error,
        "angle_45_delta_model_vs_delta_opls": delta_error,
        "relative_frobenius_errors": {
            "angle_0": angle_0_error["relative_frobenius_residual"],
            "angle_60": angle_60_error["relative_frobenius_residual"],
            "angle_45": angle_45_error["relative_frobenius_residual"],
            "angle_45_delta": delta_error["relative_frobenius_residual"],
        },
        "relative_frobenius_errors_percent": {
            "angle_0": angle_0_error["relative_frobenius_residual_percent"],
            "angle_60": angle_60_error["relative_frobenius_residual_percent"],
            "angle_45": angle_45_error["relative_frobenius_residual_percent"],
            "angle_45_delta": delta_error[
                "relative_frobenius_residual_percent"
            ],
        },
        "angle_45_delta_model_kcal_mol_A": delta_model.tolist(),
        "angle_45_delta_opls_kcal_mol_A": delta_opls.tolist(),
    }
    opls_ground_truth["direct_model_physical_comparison_available"] = True


def evaluate(
    run_path: Path,
    data_path: Path = DEFAULT_DATA,
    *,
    checkpoint_path: Path | None = None,
    sample_id: int | None = None,
    output_path: Path | None = None,
    absolute_tolerance: float = 2.0e-5,
    relative_tolerance: float = 2.0e-5,
    recompute_opls: bool = True,
    opls_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    import torch

    from experiments.benzene_pair.data.benzene_cluster import (
        load_benzene_cluster_csv,
    )

    run = run_path.resolve()
    if not run.is_dir():
        raise NotADirectoryError(f"run path does not exist: {run}")
    data = data_path.resolve()
    checkpoint_file = _resolve_checkpoint(run, checkpoint_path)
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")
    required = {"model_state", "model_config", "force_scale", "dtype"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"checkpoint is missing fields: {sorted(missing)}")

    arrays = load_benzene_cluster_csv(data)
    if arrays.molecule_count != 2:
        raise ValueError("rotation evaluation requires exactly two benzene molecules")
    data_sha256 = _sha256(data)
    expected_data_sha256 = checkpoint.get("data_sha256")
    if (
        expected_data_sha256 is not None
        and str(expected_data_sha256).lower() != data_sha256.lower()
    ):
        raise ValueError("checkpoint data_sha256 does not match the selected dataset")

    summary = _read_summary(run)
    test_indices, split_record = _fixed_split_record(
        run,
        len(arrays),
        checkpoint,
        summary,
    )
    selected_sample_id = int(test_indices[0]) if sample_id is None else int(sample_id)
    if not 0 <= selected_sample_id < len(arrays):
        raise IndexError(f"sample_id is outside the dataset: {selected_sample_id}")

    model_config = checkpoint["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model_config must contain a mapping")
    model_family, model_family_resolution = _resolve_model_family(checkpoint)
    from .e311_one_block_gnn import E311OneBlockGNN, E311OneBlockGNNConfig

    config = E311OneBlockGNNConfig(**model_config)
    if config.molecule_count != 2:
        raise ValueError("checkpoint model_config must specify two molecules")
    dtype = _dtype_from_checkpoint(torch, checkpoint["dtype"])
    model = E311OneBlockGNN(
        float(checkpoint["force_scale"]),
        config,
        dtype,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    centers_numpy = np.asarray(
        arrays.centers[selected_sample_id],
        dtype=np.float64,
    )
    rotations_numpy = np.asarray(
        arrays.rotations[selected_sample_id],
        dtype=np.float64,
    )
    stored_forces = np.asarray(
        arrays.forces[selected_sample_id],
        dtype=np.float64,
    )
    centers = torch.as_tensor(centers_numpy, dtype=dtype).unsqueeze(0)
    rotations = torch.as_tensor(rotations_numpy, dtype=dtype).unsqueeze(0)
    rotation_60 = torch.as_tensor(_rotation_z(60.0), dtype=dtype)
    rotation_45 = torch.as_tensor(_rotation_z(45.0), dtype=dtype)
    rotations_60 = rotations.clone()
    rotations_45 = rotations.clone()
    rotations_60[:, 1] = rotations[:, 1] @ rotation_60
    rotations_45[:, 1] = rotations[:, 1] @ rotation_45
    running_rms_before = _running_rms_buffer_bytes(model)
    if not running_rms_before:
        raise AssertionError("E311 model exposes no RunningRMS state buffers")
    with torch.inference_mode():
        baseline_tensor = model.forward_world(centers, rotations)
        angle_60_tensor = model.forward_world(centers, rotations_60)
        angle_45_tensor = model.forward_world(centers, rotations_45)
    running_rms_after = _running_rms_buffer_bytes(model)
    if running_rms_after != running_rms_before:
        changed_names = sorted(
            name
            for name in set(running_rms_before).union(running_rms_after)
            if running_rms_before.get(name) != running_rms_after.get(name)
        )
        raise AssertionError(
            "RunningRMS buffers changed during eval forwards: "
            + ", ".join(changed_names)
        )
    baseline = baseline_tensor[0].detach().cpu().to(torch.float64).numpy()
    angle_60 = angle_60_tensor[0].detach().cpu().to(torch.float64).numpy()
    angle_45 = angle_45_tensor[0].detach().cpu().to(torch.float64).numpy()

    angle_60_residual = _frobenius_residual(angle_60, baseline)
    angle_45_change = _frobenius_residual(angle_45, baseline)
    naive_expected, naive_world_rotation = _naive_partial_covariance_expected(
        baseline,
        rotations_numpy[1],
        45.0,
    )
    angle_45_naive_residual = _frobenius_residual(angle_45, naive_expected)
    comparison_threshold = (
        absolute_tolerance
        + relative_tolerance * angle_60_residual["reference_frobenius_norm"]
    )
    checks = {
        "angle_60_d6_world_force_invariance": (
            angle_60_residual["absolute_frobenius_residual"]
            <= comparison_threshold
        ),
        "angle_45_output_change_resolved": (
            angle_45_change["absolute_frobenius_residual"]
            > comparison_threshold
        ),
        "angle_45_naive_partial_covariance_rejected": (
            angle_45_naive_residual["absolute_frobenius_residual"]
            > comparison_threshold
        ),
    }

    opls_ground_truth: dict[str, Any]
    if recompute_opls:
        opls_ground_truth = _run_opls_recompute(
            centers_numpy,
            rotations_numpy,
            stored_forces,
            selected_sample_id,
            opls_timeout_seconds,
        )
    else:
        opls_ground_truth = {
            "status": "skipped",
            "model_test_is_physical_correctness_test": False,
            "interpretation": (
                "Model symmetry metrics alone do not establish physical correctness."
            ),
        }
    _opls_model_comparison(
        opls_ground_truth,
        baseline,
        angle_60,
        angle_45,
    )

    running_rms_record = {
        "applicable": True,
        "checked": True,
        "model_eval_mode": not model.training,
        "comparison": "byte_exact_before_and_after_three_forwards",
        "buffer_count": len(running_rms_before),
        "buffer_names": sorted(running_rms_before),
        "unchanged": running_rms_after == running_rms_before,
    }

    result: dict[str, Any] = {
        "schema_name": "tfenn_two_benzene_rotation_evaluation",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "run_path": str(run),
        "checkpoint": {
            "path": str(checkpoint_file),
            "name": checkpoint_file.name,
            "sha256": _sha256(checkpoint_file),
            "epoch": checkpoint.get("epoch", checkpoint.get("best_epoch")),
            "purpose": checkpoint.get("purpose"),
            "model_family": model_family,
            "model_family_checkpoint_value": checkpoint.get("model_family"),
            "model_family_resolution": model_family_resolution,
            "dtype": str(checkpoint["dtype"]),
            "force_scale_kcal_mol_A": float(checkpoint["force_scale"]),
            "message_block_count": model.message_block_count,
            "design_source_verification": checkpoint.get(
                "design_source_verification"
            ),
            "source_provenance": checkpoint.get("source_provenance"),
        },
        "data": {
            "path": str(data),
            "sha256": data_sha256,
            "checkpoint_sha256_match": expected_data_sha256 is None
            or str(expected_data_sha256).lower() == data_sha256.lower(),
            "sample_count": len(arrays),
            "molecule_count": arrays.molecule_count,
        },
        "split": split_record,
        "sample": {
            "sample_id": selected_sample_id,
            "selection": (
                "fixed_test_split_first_sample"
                if sample_id is None
                else "explicit_sample_id"
            ),
            "is_in_fixed_test_split": bool(
                np.any(test_indices == selected_sample_id)
            ),
            "centers_A": centers_numpy.tolist(),
            "orientations_active_body_to_root": rotations_numpy.tolist(),
            "stored_forces_kcal_mol_A": stored_forces.tolist(),
        },
        "transformation": {
            "molecule_id": 1,
            "center_changed": False,
            "orientation_law": "O_1_prime_equals_O_1_right_multiply_Rz_body",
            "angle_60_degrees": 60.0,
            "angle_45_degrees": 45.0,
        },
        "tolerance": {
            "absolute_frobenius_kcal_mol_A": absolute_tolerance,
            "relative_frobenius": relative_tolerance,
            "combined_absolute_threshold_kcal_mol_A": comparison_threshold,
        },
        "model": {
            "running_rms_eval_state": running_rms_record,
            "baseline_forces_kcal_mol_A": baseline.tolist(),
            "angle_60_forces_kcal_mol_A": angle_60.tolist(),
            "angle_45_forces_kcal_mol_A": angle_45.tolist(),
            "angle_60": {
                "expected_world_relation": "F_60_equals_F_0",
                "meaning": "internal_D6_gauge_invariance",
                "absolute_frobenius_residual_kcal_mol_A": angle_60_residual[
                    "absolute_frobenius_residual"
                ],
                **angle_60_residual,
            },
            "angle_45": {
                "symmetry_obligation": "none_for_single_molecule_rotation",
                "relative_output_change": angle_45_change[
                    "relative_frobenius_residual"
                ],
                "relative_output_change_percent": angle_45_change[
                    "relative_frobenius_residual_percent"
                ],
                "output_change": angle_45_change,
                "naive_partial_covariance_definition": (
                    "rotate_only_molecule_1_force_by_O1_Rz45_O1_transpose_and_"
                    "leave_molecule_0_force_unchanged"
                ),
                "naive_partial_covariance_expected_forces_kcal_mol_A": (
                    naive_expected.tolist()
                ),
                "naive_partial_covariance_world_rotation": (
                    naive_world_rotation.tolist()
                ),
                "naive_partial_covariance_residual": (
                    angle_45_naive_residual["relative_frobenius_residual"]
                ),
                "naive_partial_covariance_residual_percent": (
                    angle_45_naive_residual[
                        "relative_frobenius_residual_percent"
                    ]
                ),
                "naive_partial_covariance": angle_45_naive_residual,
            },
        },
        "opls_ground_truth": opls_ground_truth,
        "interpretation": {
            "angle_60": (
                "Right multiplication by a 60 degree benzene body symmetry "
                "represents the same physical molecule, so world forces are invariant."
            ),
            "angle_45": (
                "A 45 degree right rotation is not in benzene D6. Its output change "
                "and naive covariance residual are empirical diagnostics for this "
                "sample, not a general group law."
            ),
            "physical_correctness": (
                "The model metrics do not establish physical correctness; use the "
                "separately recomputed OPLS ground truth when available."
            ),
        },
    }
    destination = (
        output_path.resolve()
        if output_path is not None
        else (run / DEFAULT_OUTPUT_NAME).resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result["output_path"] = str(destination)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf_8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample-id", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--skip-opls", action="store_true")
    parser.add_argument("--opls-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--opls-worker", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.opls_worker is None and arguments.run is None:
        parser.error("run is required")
    for name in (
        "absolute_tolerance",
        "relative_tolerance",
        "opls_timeout_seconds",
    ):
        value = getattr(arguments, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    return arguments


def main() -> int:
    arguments = parse_args()
    if arguments.opls_worker is not None:
        return _opls_worker(arguments.opls_worker)
    result = evaluate(
        arguments.run,
        arguments.data,
        checkpoint_path=arguments.checkpoint,
        sample_id=arguments.sample_id,
        output_path=arguments.output,
        absolute_tolerance=arguments.absolute_tolerance,
        relative_tolerance=arguments.relative_tolerance,
        recompute_opls=not arguments.skip_opls,
        opls_timeout_seconds=arguments.opls_timeout_seconds,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
