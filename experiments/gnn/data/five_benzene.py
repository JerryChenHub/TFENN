from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MOLECULE_COUNT = 5
OPLS_VERSION = "2.0.0"
OPLS_COMMIT = "a5f874ed00152b156cd2525c961bd81030237e31"
DEFAULT_DENSITY_G_CM3 = 0.70
DEFAULT_BASE_SEED = 20260824
DEFAULT_MAX_MOLECULAR_FORCE = 4.0
DEFAULT_MAX_MOLECULAR_TORQUE = 5.0
DEFAULT_MAX_PAIR_FORCE = 5.0
DEFAULT_MAX_ATOMIC_FORCE = 5.0
PERCENTILE_LEVELS = (0, 1, 5, 25, 50, 75, 95, 99, 100)
ROTATION_COLUMNS = tuple(
    f"R{row}{column}" for row in range(1, 4) for column in range(1, 4)
)
COLUMNS = (
    ("sample_id", "molecule_id", "cx", "cy", "cz")
    + ROTATION_COLUMNS
    + ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
)


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    sample_count: int
    base_seed: int
    density_g_cm3: float
    max_molecular_force: float
    max_molecular_torque: float
    max_pair_force: float
    max_atomic_force: float
    max_candidate_attempts: int


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    centers: np.ndarray
    rotations: np.ndarray
    forces: np.ndarray
    moments: np.ndarray
    source_molecule_order: tuple[int, ...]
    seed: int
    candidate_attempt: int
    candidate_rejections: dict[str, int]
    quality: dict[str, object]
    generation: dict[str, object]
    total_force_residual: float
    total_angular_residual: float


_GENERATION_SETTINGS: GenerationSettings | None = None
_GENERATION_POLICY: Any = None
_GENERATION_ENGINE: Any = None
_GENERATION_SPECIES: Any = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    numbers = np.percentile(np.asarray(values, dtype=np.float64), PERCENTILE_LEVELS)
    return {
        f"p{level}": float(number)
        for level, number in zip(PERCENTILE_LEVELS, numbers, strict=True)
    }


def _candidate_seed(base_seed: int, sample_index: int, attempt: int) -> int:
    state = np.random.SeedSequence((base_seed, sample_index, attempt)).generate_state(
        1,
        dtype=np.uint64,
    )
    return int(state[0])


def _source_molecule_order(seed: int) -> tuple[int, ...]:
    label_rng = np.random.default_rng(
        np.random.SeedSequence((seed, MOLECULE_COUNT, DEFAULT_BASE_SEED))
    )
    return tuple(int(item) for item in label_rng.permutation(MOLECULE_COUNT))


def _initialize_generation(settings: GenerationSettings) -> None:
    global _GENERATION_SETTINGS
    global _GENERATION_POLICY
    global _GENERATION_ENGINE
    global _GENERATION_SPECIES
    from opls2020 import GenerationPolicy, QualityLimits, StaticEngine, benzene

    quality = QualityLimits(
        max_pair_force_kcal_mol_A=settings.max_pair_force,
        max_atomic_force_kcal_mol_A=settings.max_atomic_force,
    )
    _GENERATION_SETTINGS = settings
    _GENERATION_POLICY = GenerationPolicy(
        target_mass_density_g_cm3=settings.density_g_cm3,
        quality=quality,
        equilibration_sweeps=0,
        production_sweeps=0,
    )
    _GENERATION_ENGINE = StaticEngine(use_neighbor_list=False)
    _GENERATION_SPECIES = benzene()


def _root_normalize(
    system: Any,
    result: Any,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    world_centers = np.stack([item.pose.center for item in system.molecules])
    world_rotations = np.stack([item.pose.orientation for item in system.molecules])
    world_forces = np.asarray(result.molecular_forces_kcal_mol_A, dtype=np.float64)
    world_moments = np.asarray(result.molecular_torques_kcal_mol, dtype=np.float64)
    source_order = _source_molecule_order(seed)
    source_order_array = np.asarray(source_order, dtype=np.int64)
    world_centers = world_centers[source_order_array]
    world_rotations = world_rotations[source_order_array]
    world_forces = world_forces[source_order_array]
    world_moments = world_moments[source_order_array]
    root_center = world_centers[0]
    root_rotation = world_rotations[0]
    centers = np.ascontiguousarray((world_centers - root_center) @ root_rotation)
    rotations = np.ascontiguousarray(root_rotation.T @ world_rotations)
    forces = np.ascontiguousarray(world_forces @ root_rotation)
    moments = np.ascontiguousarray(world_moments @ root_rotation)
    centers[0] = 0.0
    rotations[0] = np.eye(3, dtype=np.float64)
    return centers, rotations, forces, moments, source_order


def _generate_sample(sample_index: int) -> GeneratedSample:
    from opls2020 import SamplingError, assess_configuration, generate_molecular_cloud

    settings = _GENERATION_SETTINGS
    if (
        settings is None
        or _GENERATION_POLICY is None
        or _GENERATION_ENGINE is None
        or _GENERATION_SPECIES is None
    ):
        raise RuntimeError("generation worker is not initialized")
    rejection_counts: Counter[str] = Counter()
    for candidate_attempt in range(settings.max_candidate_attempts):
        seed = _candidate_seed(settings.base_seed, sample_index, candidate_attempt)
        try:
            system = generate_molecular_cloud(
                f"five_benzene_{sample_index:06d}",
                ((_GENERATION_SPECIES, MOLECULE_COUNT),),
                _GENERATION_POLICY,
                seed,
            )
        except SamplingError:
            rejection_counts.update(("sampling_error",))
            continue
        result = _GENERATION_ENGINE.evaluate(system)
        report = assess_configuration(result, _GENERATION_POLICY.quality)
        if not report.passed:
            rejection_counts.update(("quality",))
            continue
        force_norm = np.linalg.norm(result.molecular_forces_kcal_mol_A, axis=1)
        moment_norm = np.linalg.norm(result.molecular_torques_kcal_mol, axis=1)
        if float(force_norm.max()) > settings.max_molecular_force:
            rejection_counts.update(("molecular_force",))
            continue
        if float(moment_norm.max()) > settings.max_molecular_torque:
            rejection_counts.update(("molecular_torque",))
            continue
        centers, rotations, forces, moments, source_order = _root_normalize(
            system,
            result,
            seed,
        )
        world_centers = np.stack([item.pose.center for item in system.molecules])
        total_angular = np.sum(result.molecular_torques_kcal_mol, axis=0) + np.sum(
            np.cross(world_centers, result.molecular_forces_kcal_mol_A),
            axis=0,
        )
        generation = system.generation
        if generation is None:
            raise RuntimeError("OPLS cloud generation metadata is missing")
        return GeneratedSample(
            centers=centers,
            rotations=rotations,
            forces=forces,
            moments=moments,
            source_molecule_order=source_order,
            seed=seed,
            candidate_attempt=candidate_attempt,
            candidate_rejections=dict(rejection_counts),
            quality=report.as_dict(),
            generation=generation.as_dict(),
            total_force_residual=float(np.linalg.norm(result.total_force)),
            total_angular_residual=float(np.linalg.norm(total_angular)),
        )
    raise RuntimeError(
        f"sample {sample_index} exhausted {settings.max_candidate_attempts} candidates"
    )


def _git_provenance(root: Path) -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "source_tree_dirty": bool(status) if status is not None else None,
    }


def _opls_provenance() -> dict[str, object]:
    import opls2020
    from opls2020 import DEFAULT_MODEL, benzene
    from opls2020.io import SCHEMA_VERSION, code_provenance

    package = importlib.metadata.distribution("opls2020-static")
    direct_text = package.read_text("direct_url.json")
    direct_url = json.loads(direct_text) if direct_text else None
    commit = None
    if isinstance(direct_url, dict):
        vcs_info = direct_url.get("vcs_info")
        if isinstance(vcs_info, dict):
            commit = vcs_info.get("commit_id")
    if opls2020.__version__ != OPLS_VERSION or package.version != OPLS_VERSION:
        raise RuntimeError("OPLS runtime does not match the verified version")
    if commit != OPLS_COMMIT:
        raise RuntimeError("OPLS runtime does not match the verified source commit")
    return {
        "runtime_version": opls2020.__version__,
        "distribution_version": package.version,
        "result_schema_version": SCHEMA_VERSION,
        "source_commit": commit,
        "direct_url": direct_url,
        "code_provenance": code_provenance(),
        "model": DEFAULT_MODEL.as_dict(),
        "species": benzene().as_dict(),
    }


def _iter_center_distances(centers: np.ndarray) -> Iterable[float]:
    for first in range(MOLECULE_COUNT):
        for second in range(first + 1, MOLECULE_COUNT):
            yield float(np.linalg.norm(centers[first] - centers[second]))


def generate_dataset(
    csv_path: Path,
    settings: GenerationSettings,
    workers: int,
) -> tuple[Path, Path]:
    if settings.sample_count < 1:
        raise ValueError("sample_count must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("output path must have csv suffix")
    metadata_path = csv_path.with_suffix(".json")
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError("output dataset already exists")
    generator_path = Path(__file__).resolve()
    generator_sha256 = _sha256(generator_path)
    opls_provenance = _opls_provenance()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = csv_path.with_name(f"{csv_path.name}.partial")
    force_norms: list[float] = []
    moment_norms: list[float] = []
    center_distances: list[float] = []
    accepted_seeds: list[int] = []
    source_molecule_orders: list[tuple[int, ...]] = []
    candidate_attempts: list[int] = []
    rejection_counts: Counter[str] = Counter()
    quality_observations: list[dict[str, object]] = []
    generation_records: list[dict[str, object]] = []
    force_residuals: list[float] = []
    angular_residuals: list[float] = []
    configurations: list[np.ndarray] = []
    try:
        with partial_path.open("w", newline="", encoding="utf_8") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_generation,
                initargs=(settings,),
            ) as executor:
                for sample_index, sample in enumerate(
                    executor.map(
                        _generate_sample,
                        range(settings.sample_count),
                        chunksize=max(1, settings.sample_count // (workers * 8)),
                    )
                ):
                    for molecule_index in range(MOLECULE_COUNT):
                        writer.writerow(
                            (
                                sample_index,
                                molecule_index,
                                *sample.centers[molecule_index],
                                *sample.rotations[molecule_index].reshape(9),
                                *sample.forces[molecule_index],
                                *sample.moments[molecule_index],
                            )
                        )
                    force_norms.extend(np.linalg.norm(sample.forces, axis=1))
                    moment_norms.extend(np.linalg.norm(sample.moments, axis=1))
                    center_distances.extend(_iter_center_distances(sample.centers))
                    accepted_seeds.append(sample.seed)
                    source_molecule_orders.append(sample.source_molecule_order)
                    candidate_attempts.append(sample.candidate_attempt)
                    rejection_counts.update(sample.candidate_rejections)
                    quality_observations.append(
                        dict(sample.quality["observed"])
                    )
                    generation_records.append(sample.generation)
                    force_residuals.append(sample.total_force_residual)
                    angular_residuals.append(sample.total_angular_residual)
                    configurations.append(
                        np.concatenate(
                            (sample.centers.reshape(-1), sample.rotations.reshape(-1))
                        )
                    )
                    if (sample_index + 1) % 100 == 0:
                        print(f"{sample_index + 1}/{settings.sample_count} samples written")
        if _sha256(generator_path) != generator_sha256:
            raise RuntimeError("generator source changed during dataset generation")
        partial_path.replace(csv_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    first_generation = generation_records[0]
    radius_values = np.asarray(
        [record["region_radius_A"] for record in generation_records],
        dtype=np.float64,
    )
    minimum_distances = np.asarray(
        [item["minimum_interatomic_distance_A"] for item in quality_observations],
        dtype=np.float64,
    )
    minimum_lj_ratios = np.asarray(
        [item["minimum_lj_r_over_sigma"] for item in quality_observations],
        dtype=np.float64,
    )
    maximum_pair_forces = np.asarray(
        [item["maximum_pair_force_kcal_mol_A"] for item in quality_observations],
        dtype=np.float64,
    )
    maximum_atomic_forces = np.asarray(
        [item["maximum_atomic_force_kcal_mol_A"] for item in quality_observations],
        dtype=np.float64,
    )
    metadata = {
        "schema_name": "tfenn_five_benzene_gnn_dataset",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": 1,
        "sample_count": settings.sample_count,
        "molecule_count": MOLECULE_COUNT,
        "row_count": settings.sample_count * MOLECULE_COUNT,
        "rows_per_sample": MOLECULE_COUNT,
        "columns": list(COLUMNS),
        "sampling": {
            "base_seed": settings.base_seed,
            "seed_strategy": "numpy_seed_sequence_from_base_sample_and_candidate",
            "accepted_seeds": accepted_seeds,
            "source_molecule_orders": [
                list(order) for order in source_molecule_orders
            ],
            "root_source_index_counts": {
                str(source_index): sum(
                    order[0] == source_index for order in source_molecule_orders
                )
                for source_index in range(MOLECULE_COUNT)
            },
            "accepted_candidate_attempts": candidate_attempts,
            "candidate_rejections": dict(sorted(rejection_counts.items())),
            "target_mass_density_g_cm3": settings.density_g_cm3,
            "source_sampling_center_world_A": [0.0, 0.0, 0.0],
            "molecule_labeling": (
                "deterministic_uniform_permutation_before_root_selection"
            ),
            "source_region_radius_A": float(radius_values[0]),
            "source_region_radius_range_A": [
                float(radius_values.min()),
                float(radius_values.max()),
            ],
            "equilibration_sweeps": 0,
            "production_sweeps": 0,
            "max_candidate_attempts": settings.max_candidate_attempts,
            "quality_limits": {
                "min_atom_distance_A": first_generation["min_atom_distance_A"],
                "min_lj_r_over_sigma": first_generation["min_lj_r_over_sigma"],
                "max_pair_force_kcal_mol_A": settings.max_pair_force,
                "max_atomic_force_kcal_mol_A": settings.max_atomic_force,
                "max_positive_pair_lj_energy_kcal_mol": first_generation[
                    "max_positive_pair_lj_energy_kcal_mol"
                ],
                "contact_distance_A": first_generation["contact_distance_A"],
            },
            "selection_limits": {
                "max_molecular_force_kcal_mol_A": settings.max_molecular_force,
                "max_molecular_torque_kcal_mol": settings.max_molecular_torque,
            },
        },
        "coordinate_convention": {
            "frame": "single_root_molecule_0_body_frame",
            "root_center_A": [0.0, 0.0, 0.0],
            "root_rotation": "identity",
            "center_transform": "transpose_O0_times_world_center_minus_world_center_0",
            "rotation_transform": "transpose_O0_times_world_orientation",
            "force_transform": "transpose_O0_times_world_force",
            "moment_transform": "transpose_O0_times_world_torque_about_molecule_center",
            "rotation_semantics": "active_body_to_root_frame",
        },
        "units": {
            "center": "angstrom",
            "force": "kcal_per_mol_per_angstrom",
            "moment": "kcal_per_mol",
        },
        "validation_limits": {
            "force_median_range_kcal_mol_A": [0.7, 1.5],
            "force_p95_max_kcal_mol_A": 3.0,
            "force_p99_max_kcal_mol_A": settings.max_molecular_force,
            "force_max_kcal_mol_A": settings.max_molecular_force,
            "force_label_median_spread_max_kcal_mol_A": 0.25,
            "moment_p99_max_kcal_mol": settings.max_molecular_torque,
            "moment_max_kcal_mol": settings.max_molecular_torque,
            "moment_label_median_spread_max_kcal_mol": 0.50,
            "nonroot_center_distance_median_spread_max_A": 0.50,
            "nonroot_center_direction_mean_max_A": 0.50,
            "nonroot_rotation_mean_absolute_max": 0.10,
            "root_source_chi_square_max": 18.5,
            "conservation_tolerance": 1.0e-10,
            "rotation_tolerance": 1.0e-10,
            "recompute_tolerance": 5.0e-10,
        },
        "statistics": {
            "force_norm_kcal_mol_A": _percentiles(np.asarray(force_norms)),
            "moment_norm_kcal_mol": _percentiles(np.asarray(moment_norms)),
            "center_pair_distance_A": _percentiles(np.asarray(center_distances)),
            "minimum_interatomic_distance_A": _percentiles(minimum_distances),
            "minimum_lj_r_over_sigma": _percentiles(minimum_lj_ratios),
            "maximum_pair_force_kcal_mol_A": _percentiles(maximum_pair_forces),
            "maximum_atomic_force_kcal_mol_A": _percentiles(
                maximum_atomic_forces
            ),
            "maximum_total_force_residual": float(max(force_residuals)),
            "maximum_total_angular_residual": float(max(angular_residuals)),
            "unique_configuration_count": int(
                np.unique(np.stack(configurations), axis=0).shape[0]
            ),
        },
        "generation": {
            "workers": workers,
            "candidate_count": settings.sample_count + sum(rejection_counts.values()),
        },
        "opls": opls_provenance,
        "tfenn": _git_provenance(Path(__file__).resolve().parents[3]),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "csv_sha256": _sha256(csv_path),
        "generator_sha256": generator_sha256,
    }
    metadata_partial = metadata_path.with_name(f"{metadata_path.name}.partial")
    try:
        if _sha256(generator_path) != generator_sha256:
            raise RuntimeError("generator source changed before metadata finalization")
        metadata_partial.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf_8",
        )
        if _sha256(generator_path) != generator_sha256:
            raise RuntimeError("generator source changed during metadata finalization")
        metadata_partial.replace(metadata_path)
    except BaseException:
        metadata_partial.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise
    return csv_path, metadata_path


def _load_dataset(csv_path: Path) -> tuple[np.ndarray, ...]:
    with csv_path.open("r", newline="", encoding="utf_8_sig") as stream:
        header = tuple(next(csv.reader(stream), ()))
    if header != COLUMNS:
        raise ValueError("dataset columns do not match the five benzene schema")
    values = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
        ndmin=2,
    )
    if values.shape[1] != len(COLUMNS) or values.shape[0] % MOLECULE_COUNT:
        raise ValueError("dataset row shape is invalid")
    sample_count = values.shape[0] // MOLECULE_COUNT
    expected_samples = np.repeat(np.arange(sample_count), MOLECULE_COUNT)
    expected_molecules = np.tile(np.arange(MOLECULE_COUNT), sample_count)
    if not np.array_equal(values[:, 0], expected_samples):
        raise ValueError("sample identifiers are invalid")
    if not np.array_equal(values[:, 1], expected_molecules):
        raise ValueError("molecule identifiers are invalid")
    data = values[:, 2:]
    centers = data[:, :3].reshape(sample_count, MOLECULE_COUNT, 3)
    rotations = data[:, 3:12].reshape(sample_count, MOLECULE_COUNT, 3, 3)
    forces = data[:, 12:15].reshape(sample_count, MOLECULE_COUNT, 3)
    moments = data[:, 15:18].reshape(sample_count, MOLECULE_COUNT, 3)
    return centers, rotations, forces, moments


_VALIDATION_ARRAYS: tuple[np.ndarray, ...] | None = None
_VALIDATION_ENGINE: Any = None
_VALIDATION_SPECIES: Any = None
_VALIDATION_QUALITY: Any = None
_VALIDATION_POLICY: Any = None
_VALIDATION_SOURCE_SEEDS: tuple[int, ...] = ()


def _initialize_validation(
    arrays: tuple[np.ndarray, ...],
    quality_values: dict[str, float],
    density_g_cm3: float,
    source_seeds: tuple[int, ...],
) -> None:
    global _VALIDATION_ARRAYS
    global _VALIDATION_ENGINE
    global _VALIDATION_SPECIES
    global _VALIDATION_QUALITY
    global _VALIDATION_POLICY
    global _VALIDATION_SOURCE_SEEDS
    from opls2020 import GenerationPolicy, QualityLimits, StaticEngine, benzene

    _VALIDATION_ARRAYS = arrays
    _VALIDATION_ENGINE = StaticEngine(use_neighbor_list=False)
    _VALIDATION_SPECIES = benzene()
    _VALIDATION_QUALITY = QualityLimits(**quality_values)
    _VALIDATION_POLICY = GenerationPolicy(
        target_mass_density_g_cm3=density_g_cm3,
        quality=_VALIDATION_QUALITY,
        equilibration_sweeps=0,
        production_sweeps=0,
    )
    _VALIDATION_SOURCE_SEEDS = source_seeds


def _recompute_sample(sample_index: int) -> dict[str, object]:
    from opls2020 import MoleculeInstance, Pose, SystemSpec, assess_configuration

    if (
        _VALIDATION_ARRAYS is None
        or _VALIDATION_ENGINE is None
        or _VALIDATION_SPECIES is None
        or _VALIDATION_QUALITY is None
    ):
        raise RuntimeError("validation worker is not initialized")
    centers, rotations, stored_forces, stored_moments = _VALIDATION_ARRAYS
    molecules = tuple(
        MoleculeInstance(
            f"benzene_{molecule_index:04d}",
            _VALIDATION_SPECIES.species_id,
            Pose.from_matrix(
                centers[sample_index, molecule_index],
                rotations[sample_index, molecule_index],
            ),
        )
        for molecule_index in range(MOLECULE_COUNT)
    )
    system = SystemSpec(
        configuration_id=f"five_benzene_validate_{sample_index:06d}",
        species={_VALIDATION_SPECIES.species_id: _VALIDATION_SPECIES},
        molecules=molecules,
        boundary="open",
        electrostatics="direct",
    )
    result = _VALIDATION_ENGINE.evaluate(system)
    report = assess_configuration(result, _VALIDATION_QUALITY)
    return {
        "force_error": float(
            np.max(
                np.abs(
                    result.molecular_forces_kcal_mol_A
                    - stored_forces[sample_index]
                )
            )
        ),
        "moment_error": float(
            np.max(
                np.abs(
                    result.molecular_torques_kcal_mol
                    - stored_moments[sample_index]
                )
            )
        ),
        "quality_passed": report.passed,
        "quality": report.observed.as_dict(),
    }


def _regenerate_source_sample(sample_index: int) -> dict[str, object]:
    from opls2020 import generate_molecular_cloud

    if (
        _VALIDATION_ARRAYS is None
        or _VALIDATION_ENGINE is None
        or _VALIDATION_SPECIES is None
        or _VALIDATION_POLICY is None
        or len(_VALIDATION_SOURCE_SEEDS) == 0
    ):
        raise RuntimeError("source replay worker is not initialized")
    stored_centers, stored_rotations, stored_forces, stored_moments = (
        _VALIDATION_ARRAYS
    )
    seed = _VALIDATION_SOURCE_SEEDS[sample_index]
    system = generate_molecular_cloud(
        f"five_benzene_{sample_index:06d}",
        ((_VALIDATION_SPECIES, MOLECULE_COUNT),),
        _VALIDATION_POLICY,
        seed,
    )
    result = _VALIDATION_ENGINE.evaluate(system)
    centers, rotations, forces, moments, source_order = _root_normalize(
        system,
        result,
        seed,
    )
    return {
        "sample_index": sample_index,
        "source_order": list(source_order),
        "center_error": float(
            np.max(np.abs(centers - stored_centers[sample_index]))
        ),
        "rotation_error": float(
            np.max(np.abs(rotations - stored_rotations[sample_index]))
        ),
        "force_error": float(
            np.max(np.abs(forces - stored_forces[sample_index]))
        ),
        "moment_error": float(
            np.max(np.abs(moments - stored_moments[sample_index]))
        ),
    }


def validate_dataset(
    csv_path: Path,
    report_path: Path,
    workers: int,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be positive")
    runtime_opls = _opls_provenance()
    metadata_path = csv_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf_8"))
    centers, rotations, forces, moments = _load_dataset(csv_path)
    sample_count = len(centers)
    identity = np.eye(3, dtype=np.float64)
    rotation_products = rotations @ np.swapaxes(rotations, -1, -2)
    determinants = np.linalg.det(rotations)
    force_norms = np.linalg.norm(forces, axis=2)
    moment_norms = np.linalg.norm(moments, axis=2)
    center_norms = np.linalg.norm(centers, axis=2)
    center_pair_distances = np.stack(
        [
            np.linalg.norm(centers[:, first] - centers[:, second], axis=1)
            for first in range(MOLECULE_COUNT)
            for second in range(first + 1, MOLECULE_COUNT)
        ],
        axis=1,
    )
    total_forces = np.sum(forces, axis=1)
    total_angular = np.sum(moments, axis=1) + np.sum(
        np.cross(centers, forces),
        axis=1,
    )
    configurations = np.concatenate(
        (centers.reshape(sample_count, -1), rotations.reshape(sample_count, -1)),
        axis=1,
    )
    force_statistics = _percentiles(force_norms.reshape(-1))
    moment_statistics = _percentiles(moment_norms.reshape(-1))
    force_statistics_by_label = {
        str(label): _percentiles(force_norms[:, label])
        for label in range(MOLECULE_COUNT)
    }
    moment_statistics_by_label = {
        str(label): _percentiles(moment_norms[:, label])
        for label in range(MOLECULE_COUNT)
    }
    force_label_medians = np.median(force_norms, axis=0)
    moment_label_medians = np.median(moment_norms, axis=0)
    nonroot_distance_medians = np.median(center_norms[:, 1:], axis=0)
    nonroot_direction_means = np.mean(centers[:, 1:], axis=0)
    nonroot_rotation_means = np.mean(rotations[:, 1:], axis=0)
    source_orders = np.asarray(
        metadata["sampling"].get("source_molecule_orders", ()),
        dtype=np.int64,
    )
    accepted_seeds = np.asarray(
        metadata["sampling"].get("accepted_seeds", ()),
        dtype=object,
    )
    accepted_candidate_attempts = np.asarray(
        metadata["sampling"].get("accepted_candidate_attempts", ()),
        dtype=np.int64,
    )
    candidate_metadata_shape_valid = bool(
        accepted_seeds.shape == (sample_count,)
        and accepted_candidate_attempts.shape == (sample_count,)
    )
    accepted_seed_replay_valid = bool(
        candidate_metadata_shape_valid
        and np.all(accepted_candidate_attempts >= 0)
        and np.all(
            accepted_candidate_attempts
            < int(metadata["sampling"]["max_candidate_attempts"])
        )
        and all(
            int(seed)
            == _candidate_seed(
                int(metadata["sampling"]["base_seed"]),
                sample_index,
                int(accepted_candidate_attempts[sample_index]),
            )
            for sample_index, seed in enumerate(accepted_seeds)
        )
    )
    source_order_shape_valid = source_orders.shape == (
        sample_count,
        MOLECULE_COUNT,
    )
    source_permutations_valid = bool(
        source_order_shape_valid
        and np.array_equal(
            np.sort(source_orders, axis=1),
            np.broadcast_to(np.arange(MOLECULE_COUNT), source_orders.shape),
        )
    )
    expected_source_orders = (
        np.asarray(
            [_source_molecule_order(int(seed)) for seed in accepted_seeds],
            dtype=np.int64,
        )
        if accepted_seeds.shape == (sample_count,)
        else np.empty((0, MOLECULE_COUNT), dtype=np.int64)
    )
    source_orders_deterministic = bool(
        source_permutations_valid
        and np.array_equal(source_orders, expected_source_orders)
    )
    root_source_counts = (
        np.bincount(source_orders[:, 0], minlength=MOLECULE_COUNT)
        if source_permutations_valid
        else np.zeros(MOLECULE_COUNT, dtype=np.int64)
    )
    expected_root_count = sample_count / MOLECULE_COUNT
    root_source_chi_square = float(
        np.sum((root_source_counts - expected_root_count) ** 2 / expected_root_count)
    )
    limits = metadata["validation_limits"]
    quality_values = {
        key: float(value)
        for key, value in metadata["sampling"]["quality_limits"].items()
    }
    arrays = (
        np.ascontiguousarray(centers),
        np.ascontiguousarray(rotations),
        np.ascontiguousarray(forces),
        np.ascontiguousarray(moments),
    )
    source_seed_tuple = tuple(int(seed) for seed in accepted_seeds)
    if len(source_seed_tuple) != sample_count:
        raise ValueError("accepted source seed metadata is incomplete")
    source_replay_indices = np.unique(
        np.linspace(
            0,
            sample_count - 1,
            min(64, sample_count),
            dtype=np.int64,
        )
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_validation,
        initargs=(
            arrays,
            quality_values,
            float(metadata["sampling"]["target_mass_density_g_cm3"]),
            source_seed_tuple,
        ),
    ) as executor:
        recomputed = list(
            executor.map(
                _recompute_sample,
                range(sample_count),
                chunksize=max(1, sample_count // (workers * 8)),
            )
        )
        source_replayed = list(
            executor.map(
                _regenerate_source_sample,
                (int(index) for index in source_replay_indices),
                chunksize=1,
            )
        )
    force_error = max(float(item["force_error"]) for item in recomputed)
    moment_error = max(float(item["moment_error"]) for item in recomputed)
    quality_passed_count = sum(bool(item["quality_passed"]) for item in recomputed)
    source_center_error = max(
        float(item["center_error"]) for item in source_replayed
    )
    source_rotation_error = max(
        float(item["rotation_error"]) for item in source_replayed
    )
    source_force_error = max(float(item["force_error"]) for item in source_replayed)
    source_moment_error = max(
        float(item["moment_error"]) for item in source_replayed
    )
    source_order_replay_passed = all(
        item["source_order"] == source_orders[int(item["sample_index"])].tolist()
        for item in source_replayed
    )
    minimum_distances = np.asarray(
        [item["quality"]["minimum_interatomic_distance_A"] for item in recomputed],
        dtype=np.float64,
    )
    minimum_lj_ratios = np.asarray(
        [item["quality"]["minimum_lj_r_over_sigma"] for item in recomputed],
        dtype=np.float64,
    )
    maximum_pair_forces = np.asarray(
        [item["quality"]["maximum_pair_force_kcal_mol_A"] for item in recomputed],
        dtype=np.float64,
    )
    maximum_atomic_forces = np.asarray(
        [item["quality"]["maximum_atomic_force_kcal_mol_A"] for item in recomputed],
        dtype=np.float64,
    )
    median_lower, median_upper = limits["force_median_range_kcal_mol_A"]
    conservation_tolerance = float(limits["conservation_tolerance"])
    rotation_tolerance = float(limits["rotation_tolerance"])
    recompute_tolerance = float(limits["recompute_tolerance"])
    checks = {
        "sample_count": sample_count == int(metadata["sample_count"]),
        "molecule_count": centers.shape[1] == MOLECULE_COUNT,
        "row_count": sample_count * MOLECULE_COUNT == int(metadata["row_count"]),
        "csv_sha256": _sha256(csv_path) == metadata["csv_sha256"],
        "generator_sha256": _sha256(Path(__file__).resolve())
        == metadata["generator_sha256"],
        "finite": bool(
            np.isfinite(centers).all()
            and np.isfinite(rotations).all()
            and np.isfinite(forces).all()
            and np.isfinite(moments).all()
        ),
        "root_center": float(np.max(np.abs(centers[:, 0])))
        <= rotation_tolerance,
        "root_rotation": float(np.max(np.abs(rotations[:, 0] - identity)))
        <= rotation_tolerance,
        "rotation_orthogonality": float(
            np.max(np.abs(rotation_products - identity))
        )
        <= rotation_tolerance,
        "rotation_determinant": float(np.max(np.abs(determinants - 1.0)))
        <= rotation_tolerance,
        "unique_configurations": int(np.unique(configurations, axis=0).shape[0])
        == sample_count,
        "source_molecule_permutations": source_permutations_valid,
        "accepted_seed_replay": accepted_seed_replay_valid,
        "source_molecule_order_replay": source_orders_deterministic,
        "source_geometry_replay": max(source_center_error, source_rotation_error)
        <= recompute_tolerance,
        "source_target_replay": max(source_force_error, source_moment_error)
        <= recompute_tolerance,
        "source_label_replay": source_order_replay_passed,
        "root_source_balance": root_source_chi_square
        <= float(limits["root_source_chi_square_max"]),
        "force_conservation": float(
            np.max(np.linalg.norm(total_forces, axis=1))
        )
        <= conservation_tolerance,
        "angular_conservation": float(
            np.max(np.linalg.norm(total_angular, axis=1))
        )
        <= conservation_tolerance,
        "force_median": float(median_lower)
        <= force_statistics["p50"]
        <= float(median_upper),
        "force_p95": force_statistics["p95"]
        <= float(limits["force_p95_max_kcal_mol_A"]),
        "force_p99": force_statistics["p99"]
        <= float(limits["force_p99_max_kcal_mol_A"]),
        "force_maximum": force_statistics["p100"]
        <= float(limits["force_max_kcal_mol_A"]) + recompute_tolerance,
        "force_label_balance": float(np.ptp(force_label_medians))
        <= float(limits["force_label_median_spread_max_kcal_mol_A"]),
        "moment_p99": moment_statistics["p99"]
        <= float(limits["moment_p99_max_kcal_mol"]),
        "moment_maximum": moment_statistics["p100"]
        <= float(limits["moment_max_kcal_mol"]) + recompute_tolerance,
        "moment_label_balance": float(np.ptp(moment_label_medians))
        <= float(limits["moment_label_median_spread_max_kcal_mol"]),
        "nonroot_center_distance_balance": float(
            np.ptp(nonroot_distance_medians)
        )
        <= float(limits["nonroot_center_distance_median_spread_max_A"]),
        "nonroot_center_direction_isotropy": float(
            np.max(np.linalg.norm(nonroot_direction_means, axis=1))
        )
        <= float(limits["nonroot_center_direction_mean_max_A"]),
        "nonroot_rotation_isotropy": float(
            np.max(np.abs(nonroot_rotation_means))
        )
        <= float(limits["nonroot_rotation_mean_absolute_max"]),
        "opls_quality": quality_passed_count == sample_count,
        "opls_force_recompute": force_error <= recompute_tolerance,
        "opls_moment_recompute": moment_error <= recompute_tolerance,
        "opls_version": metadata["opls"]["runtime_version"]
        == runtime_opls["runtime_version"]
        == OPLS_VERSION,
        "opls_source_commit": metadata["opls"]["source_commit"]
        == runtime_opls["source_commit"]
        == OPLS_COMMIT,
    }
    report = {
        "schema_name": "tfenn_five_benzene_gnn_validation",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "statistics": {
            "sample_count": sample_count,
            "molecule_count": MOLECULE_COUNT,
            "force_norm_kcal_mol_A": force_statistics,
            "force_norm_by_output_label_kcal_mol_A": force_statistics_by_label,
            "moment_norm_kcal_mol": moment_statistics,
            "moment_norm_by_output_label_kcal_mol": moment_statistics_by_label,
            "center_pair_distance_A": _percentiles(
                center_pair_distances.reshape(-1)
            ),
            "nonroot_center_distance_median_by_output_label_A": {
                str(label): float(nonroot_distance_medians[label - 1])
                for label in range(1, MOLECULE_COUNT)
            },
            "nonroot_center_direction_mean_by_output_label_A": {
                str(label): [
                    float(value) for value in nonroot_direction_means[label - 1]
                ]
                for label in range(1, MOLECULE_COUNT)
            },
            "nonroot_rotation_mean_by_output_label": {
                str(label): nonroot_rotation_means[label - 1].tolist()
                for label in range(1, MOLECULE_COUNT)
            },
            "root_source_index_counts": {
                str(label): int(root_source_counts[label])
                for label in range(MOLECULE_COUNT)
            },
            "root_source_chi_square": root_source_chi_square,
            "minimum_interatomic_distance_A": _percentiles(minimum_distances),
            "minimum_lj_r_over_sigma": _percentiles(minimum_lj_ratios),
            "maximum_pair_force_kcal_mol_A": _percentiles(maximum_pair_forces),
            "maximum_atomic_force_kcal_mol_A": _percentiles(
                maximum_atomic_forces
            ),
            "maximum_total_force_residual": float(
                np.max(np.linalg.norm(total_forces, axis=1))
            ),
            "maximum_total_angular_residual": float(
                np.max(np.linalg.norm(total_angular, axis=1))
            ),
            "maximum_rotation_orthogonality_error": float(
                np.max(np.abs(rotation_products - identity))
            ),
            "maximum_rotation_determinant_error": float(
                np.max(np.abs(determinants - 1.0))
            ),
            "unique_configuration_count": int(
                np.unique(configurations, axis=0).shape[0]
            ),
        },
        "recomputed": {
            "sample_count": sample_count,
            "quality_passed_count": quality_passed_count,
            "maximum_force_component_error": force_error,
            "maximum_moment_component_error": moment_error,
            "source_replay_sample_count": len(source_replayed),
            "source_maximum_center_component_error": source_center_error,
            "source_maximum_rotation_component_error": source_rotation_error,
            "source_maximum_force_component_error": source_force_error,
            "source_maximum_moment_component_error": source_moment_error,
        },
        "thresholds": limits,
        "opls_runtime": {
            "version": runtime_opls["runtime_version"],
            "source_commit": runtime_opls["source_commit"],
        },
        "csv_path": str(csv_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "csv_sha256": _sha256(csv_path),
        "metadata_sha256": _sha256(metadata_path),
        "workers": workers,
    }
    if report_path.exists():
        raise FileExistsError("validation report already exists")
    partial_path = report_path.with_name(f"{report_path.name}.partial")
    partial_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    partial_path.replace(report_path)
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"five benzene validation failed: {failed}")
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("csv_path", type=Path)
    generate.add_argument("--sample_count", type=int, default=1_000)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--base_seed", type=int, default=DEFAULT_BASE_SEED)
    generate.add_argument("--density_g_cm3", type=float, default=DEFAULT_DENSITY_G_CM3)
    generate.add_argument(
        "--max_molecular_force",
        type=float,
        default=DEFAULT_MAX_MOLECULAR_FORCE,
    )
    generate.add_argument(
        "--max_molecular_torque",
        type=float,
        default=DEFAULT_MAX_MOLECULAR_TORQUE,
    )
    generate.add_argument("--max_pair_force", type=float, default=DEFAULT_MAX_PAIR_FORCE)
    generate.add_argument(
        "--max_atomic_force",
        type=float,
        default=DEFAULT_MAX_ATOMIC_FORCE,
    )
    generate.add_argument("--max_candidate_attempts", type=int, default=100)
    validate = commands.add_parser("validate")
    validate.add_argument("csv_path", type=Path)
    validate.add_argument("report_path", type=Path)
    validate.add_argument("--workers", type=int, default=8)
    return parser


def main() -> int:
    arguments = _build_parser().parse_args()
    if arguments.command == "generate":
        settings = GenerationSettings(
            sample_count=arguments.sample_count,
            base_seed=arguments.base_seed,
            density_g_cm3=arguments.density_g_cm3,
            max_molecular_force=arguments.max_molecular_force,
            max_molecular_torque=arguments.max_molecular_torque,
            max_pair_force=arguments.max_pair_force,
            max_atomic_force=arguments.max_atomic_force,
            max_candidate_attempts=arguments.max_candidate_attempts,
        )
        csv_path, metadata_path = generate_dataset(
            arguments.csv_path,
            settings,
            arguments.workers,
        )
        print(csv_path)
        print(metadata_path)
        return 0
    report_path = validate_dataset(
        arguments.csv_path,
        arguments.report_path,
        arguments.workers,
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
