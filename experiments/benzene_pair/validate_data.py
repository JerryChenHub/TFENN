"""Validate one OPLS benzene pair dataset without importing torch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.benzene_pair.data import (
    load_benzene_cluster_csv,
    load_benzene_cluster_metadata,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(value: np.ndarray) -> dict[str, float]:
    levels = (0, 1, 50, 95, 99, 100)
    numbers = np.percentile(value, levels)
    return {
        f"p{level}": float(number)
        for level, number in zip(levels, numbers, strict=True)
    }


def _recompute_samples(
    centers: np.ndarray,
    rotations: np.ndarray,
    forces: np.ndarray,
    moments: np.ndarray,
    count: int,
) -> dict[str, Any]:
    from opls2020 import (
        DEFAULT_MODEL,
        DEFAULT_PARAMETER_CATALOG,
        MoleculeInstance,
        Pose,
        StaticEngine,
        SystemSpec,
        benzene,
    )

    species = benzene()
    catalog = {species.species_id: species}
    engine = StaticEngine(
        model=DEFAULT_MODEL,
        parameters=DEFAULT_PARAMETER_CATALOG,
        use_neighbor_list=False,
    )
    indices = np.unique(
        np.linspace(0, len(centers) - 1, min(count, len(centers)), dtype=np.int64)
    )
    maximum_force_error = 0.0
    maximum_moment_error = 0.0
    records = []
    for sample_index in indices:
        molecules = tuple(
            MoleculeInstance(
                f"benzene_{molecule_index}",
                species.species_id,
                Pose.from_matrix(
                    centers[sample_index, molecule_index],
                    rotations[sample_index, molecule_index],
                ),
            )
            for molecule_index in range(2)
        )
        system = SystemSpec(
            configuration_id=f"tfenn_validate_{sample_index:08d}",
            species=catalog,
            molecules=molecules,
        )
        result = engine.evaluate(system)
        force_error = float(
            np.max(
                np.abs(
                    np.asarray(result.molecular_forces_kcal_mol_A)
                    - forces[sample_index]
                )
            )
        )
        moment_error = float(
            np.max(
                np.abs(
                    np.asarray(result.molecular_torques_kcal_mol)
                    - moments[sample_index]
                )
            )
        )
        maximum_force_error = max(maximum_force_error, force_error)
        maximum_moment_error = max(maximum_moment_error, moment_error)
        records.append(
            {
                "sample_index": int(sample_index),
                "maximum_force_error": force_error,
                "maximum_moment_error": moment_error,
            }
        )
    return {
        "sample_count": len(records),
        "maximum_force_error": maximum_force_error,
        "maximum_moment_error": maximum_moment_error,
        "records": records,
    }


def validate_pair_dataset(
    csv_path: str | Path,
    *,
    expected_sample_count: int = 5_000,
    recompute_count: int = 16,
    numerical_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Return a complete numerical and provenance validation report."""
    path = Path(csv_path).resolve()
    metadata_path = path.with_suffix(".json")
    arrays = load_benzene_cluster_csv(path, dtype=np.float64)
    metadata = load_benzene_cluster_metadata(metadata_path)
    centers = arrays.centers
    rotations = arrays.rotations
    forces = arrays.forces
    moments = arrays.moments
    csv_sha256 = sha256_file(path)
    package_version = importlib.metadata.version("opls2020-static")

    identity = np.eye(3, dtype=np.float64)
    orthogonality = rotations @ np.swapaxes(rotations, -1, -2)
    determinant = np.linalg.det(rotations)
    center_distance = np.linalg.norm(centers[:, 1] - centers[:, 0], axis=1)
    total_force = forces.sum(axis=1)
    total_angular = moments.sum(axis=1) + np.cross(centers, forces).sum(axis=1)

    from opls2020 import benzene

    reference = np.asarray(benzene().reference_coordinates_A, dtype=np.float64)
    positions = (
        np.einsum("aj,nmij->nmai", reference, rotations) + centers[:, :, None, :]
    )
    atom_delta = positions[:, 0, :, None, :] - positions[:, 1, None, :, :]
    minimum_atomic_distance = np.linalg.norm(atom_delta, axis=-1).min(axis=(1, 2))
    pose_rows = np.concatenate(
        (centers[:, 1], rotations[:, 1].reshape(len(arrays), 9)),
        axis=1,
    )
    unique_pose_count = int(np.unique(pose_rows, axis=0).shape[0])
    force_norm = np.linalg.norm(forces, axis=-1)
    moment_norm = np.linalg.norm(moments, axis=-1)
    recomputed = _recompute_samples(
        centers,
        rotations,
        forces,
        moments,
        recompute_count,
    )

    sampling = metadata.get("sampling", {})
    distance_range = tuple(sampling.get("distance_range_A", ()))
    minimum_required = sampling.get("min_interatomic_distance_A")
    target_health = sampling.get("target_health", {})
    maximum_force_norm = target_health.get("max_force_norm_kcal_mol_A")
    maximum_moment_norm = target_health.get("max_moment_norm_kcal_mol")
    recorded_opls = metadata.get("opls", {})
    checks = {
        "sample_count": len(arrays) == expected_sample_count,
        "molecule_count": arrays.molecule_count == 2,
        "metadata_sample_count": metadata.get("sample_count") == len(arrays),
        "metadata_molecule_count": metadata.get("molecule_count") == 2,
        "csv_sha256": metadata.get("csv_sha256") == csv_sha256,
        "opls_runtime_version": recorded_opls.get("runtime_version") == package_version,
        "finite": bool(
            np.isfinite(centers).all()
            and np.isfinite(rotations).all()
            and np.isfinite(forces).all()
            and np.isfinite(moments).all()
        ),
        "root_center": float(np.max(np.abs(centers[:, 0]))) <= numerical_tolerance,
        "root_rotation": float(np.max(np.abs(rotations[:, 0] - identity)))
        <= numerical_tolerance,
        "rotation_orthogonality": float(np.max(np.abs(orthogonality - identity)))
        <= numerical_tolerance,
        "rotation_determinant": float(np.max(np.abs(determinant - 1.0)))
        <= numerical_tolerance,
        "distance_range": len(distance_range) == 2
        and float(center_distance.min())
        >= float(distance_range[0]) - numerical_tolerance
        and float(center_distance.max())
        <= float(distance_range[1]) + numerical_tolerance,
        "minimum_atomic_distance": minimum_required is not None
        and float(minimum_atomic_distance.min())
        >= float(minimum_required) - numerical_tolerance,
        "force_conservation": float(np.max(np.linalg.norm(total_force, axis=1)))
        <= numerical_tolerance,
        "angular_conservation": float(np.max(np.linalg.norm(total_angular, axis=1)))
        <= numerical_tolerance,
        "unique_poses": unique_pose_count == len(arrays),
        "nonzero_force_targets": int(np.count_nonzero(force_norm[:, 0])) == len(arrays),
        "force_norm_health": maximum_force_norm is None
        or float(force_norm.max()) <= float(maximum_force_norm) + numerical_tolerance,
        "moment_norm_health": maximum_moment_norm is None
        or float(moment_norm.max()) <= float(maximum_moment_norm) + numerical_tolerance,
        "opls_force_recompute": recomputed["maximum_force_error"]
        <= numerical_tolerance,
        "opls_moment_recompute": recomputed["maximum_moment_error"]
        <= numerical_tolerance,
    }
    return {
        "schema_name": "tfenn_benzene_pair_data_validation",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(path),
        "metadata_path": str(metadata_path),
        "csv_sha256": csv_sha256,
        "metadata_sha256": sha256_file(metadata_path),
        "opls_distribution_version": package_version,
        "passed": all(checks.values()),
        "checks": checks,
        "statistics": {
            "sample_count": len(arrays),
            "unique_pose_count": unique_pose_count,
            "center_distance_A": _percentiles(center_distance),
            "minimum_atomic_distance_A": _percentiles(minimum_atomic_distance),
            "force_norm_kcal_mol_A": _percentiles(force_norm),
            "moment_norm_kcal_mol": _percentiles(moment_norm),
            "maximum_root_center_error": float(np.max(np.abs(centers[:, 0]))),
            "maximum_root_rotation_error": float(
                np.max(np.abs(rotations[:, 0] - identity))
            ),
            "maximum_rotation_orthogonality_error": float(
                np.max(np.abs(orthogonality - identity))
            ),
            "maximum_rotation_determinant_error": float(
                np.max(np.abs(determinant - 1.0))
            ),
            "maximum_total_force_norm": float(
                np.max(np.linalg.norm(total_force, axis=1))
            ),
            "maximum_total_angular_norm": float(
                np.max(np.linalg.norm(total_angular, axis=1))
            ),
        },
        "recomputed": recomputed,
        "thresholds": {
            "expected_sample_count": expected_sample_count,
            "numerical_tolerance": numerical_tolerance,
            "recompute_count": recompute_count,
            "max_force_norm_kcal_mol_A": maximum_force_norm,
            "max_moment_norm_kcal_mol": maximum_moment_norm,
        },
    }


def write_validation_report(
    csv_path: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    report = validate_pair_dataset(csv_path, **kwargs)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    partial.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    partial.replace(target)
    if not report["passed"]:
        failed = tuple(name for name, passed in report["checks"].items() if not passed)
        raise RuntimeError(f"dataset validation failed: {failed}")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--expected_sample_count", type=int, default=5_000)
    parser.add_argument("--recompute_count", type=int, default=16)
    parser.add_argument("--numerical_tolerance", type=float, default=1.0e-10)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    path = write_validation_report(
        arguments.csv_path,
        arguments.output_path,
        expected_sample_count=arguments.expected_sample_count,
        recompute_count=arguments.recompute_count,
        numerical_tolerance=arguments.numerical_tolerance,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
