from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from importlib.metadata import distribution
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import numpy as np

from .benzene_cluster import BENZENE_CLUSTER_COLUMNS
from .benzene_pair import load_benzene_pair_csv


@dataclass(frozen=True, slots=True)
class BenzeneClusterGenerationConfig:
    sample_count: int = 10_000
    dataset_revision: int = 1
    molecule_count: int = 2
    seed: int = 20260810
    distance_range_A: tuple[float, float] = (5.0, 10.0)
    min_interatomic_distance_A: float = 3.0
    max_force_norm_kcal_mol_A: float | None = None
    max_moment_norm_kcal_mol: float | None = None
    max_attempts_per_sample: int = 500

    def __post_init__(self) -> None:
        minimum, maximum = self.distance_range_A
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, Integral)
            or self.sample_count < 1
        ):
            raise ValueError("sample_count must be a positive integer.")
        if (
            isinstance(self.dataset_revision, bool)
            or not isinstance(self.dataset_revision, Integral)
            or self.dataset_revision < 1
        ):
            raise ValueError("dataset_revision must be a positive integer.")
        if (
            isinstance(self.molecule_count, bool)
            or not isinstance(self.molecule_count, Integral)
            or self.molecule_count < 2
        ):
            raise ValueError("molecule_count must be at least two.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer.")
        if not np.isfinite((minimum, maximum)).all() or not 0.0 < minimum < maximum:
            raise ValueError(
                "distance_range_A must contain increasing positive values."
            )
        if (
            not np.isfinite(self.min_interatomic_distance_A)
            or self.min_interatomic_distance_A < 0.0
        ):
            raise ValueError("min_interatomic_distance_A cannot be negative.")
        for name, value in (
            ("max_force_norm_kcal_mol_A", self.max_force_norm_kcal_mol_A),
            ("max_moment_norm_kcal_mol", self.max_moment_norm_kcal_mol),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite when set.")
        if (
            isinstance(self.max_attempts_per_sample, bool)
            or not isinstance(self.max_attempts_per_sample, Integral)
            or self.max_attempts_per_sample < 1
        ):
            raise ValueError("max_attempts_per_sample must be positive.")


@dataclass(frozen=True, slots=True)
class BenzenePairGenerationConfig(BenzeneClusterGenerationConfig):
    molecule_count: int = field(default=2, init=False)


@dataclass(frozen=True, slots=True)
class BenzeneTripleGenerationConfig(BenzeneClusterGenerationConfig):
    molecule_count: int = field(default=3, init=False)


@dataclass(frozen=True, slots=True)
class BenzeneClusterSample:
    centers: np.ndarray
    rotations: np.ndarray
    forces: np.ndarray
    moments: np.ndarray


def _check_windows_openmp_runtime() -> None:
    if sys.platform != "win32" or "torch" not in sys.modules:
        return
    torch_module = sys.modules["torch"]
    torch_file = getattr(torch_module, "__file__", None)
    if torch_file is None:
        return
    torch_runtime = Path(torch_file).resolve().parent / "lib" / "libiomp5md.dll"
    conda_runtime = Path(sys.prefix) / "Library" / "bin" / "libiomp5md.dll"
    if torch_runtime.exists() and conda_runtime.exists():
        raise RuntimeError(
            "OPLS generation must run in a separate Windows process from torch "
            "because this conda environment contains two Intel OpenMP runtimes."
        )


class _OplsContext:
    def __init__(self) -> None:
        _check_windows_openmp_runtime()
        from opls2020 import (
            DEFAULT_MODEL,
            DEFAULT_PARAMETER_CATALOG,
            MoleculeInstance,
            Pose,
            StaticEngine,
            SystemSpec,
            benzene,
        )

        self.model = DEFAULT_MODEL
        self.parameters = DEFAULT_PARAMETER_CATALOG
        self.species = benzene()
        self.species_catalog = {self.species.species_id: self.species}
        self.reference_coordinates = np.asarray(
            self.species.reference_coordinates_A,
            dtype=np.float64,
        )
        self.engine = StaticEngine(
            model=self.model,
            parameters=self.parameters,
            use_neighbor_list=False,
        )
        self.MoleculeInstance = MoleculeInstance
        self.Pose = Pose
        self.SystemSpec = SystemSpec

    def evaluate(
        self,
        centers: np.ndarray,
        rotations: np.ndarray,
        configuration_id: str,
        random_seed: int | None,
    ) -> Any:
        molecules = tuple(
            self.MoleculeInstance(
                f"benzene_{index}",
                self.species.species_id,
                self.Pose.from_matrix(center, rotation),
            )
            for index, (center, rotation) in enumerate(
                zip(centers, rotations, strict=True)
            )
        )
        system = self.SystemSpec(
            configuration_id=configuration_id,
            species=self.species_catalog,
            molecules=molecules,
            random_seed=random_seed,
        )
        return self.engine.evaluate(system)


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    x = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    y = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    z = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    w = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.array(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ),
            (
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ),
            (
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def _minimum_interatomic_distance(
    first_positions: np.ndarray,
    second_positions: np.ndarray,
) -> float:
    difference = first_positions[:, None, :] - second_positions[None, :, :]
    return float(np.linalg.norm(difference, axis=2).min())


def _sample_poses(
    rng: np.random.Generator,
    config: BenzeneClusterGenerationConfig,
    reference_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(config.max_attempts_per_sample):
        centers = np.zeros((config.molecule_count, 3), dtype=np.float64)
        rotations = np.broadcast_to(
            np.eye(3, dtype=np.float64),
            (config.molecule_count, 3, 3),
        ).copy()
        for molecule_index in range(1, config.molecule_count):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            centers[molecule_index] = rng.uniform(*config.distance_range_A) * direction
            rotations[molecule_index] = _random_rotation_matrix(rng)

        atom_positions = (
            np.einsum(
                "aj,mij->mai",
                reference_coordinates,
                rotations,
            )
            + centers[:, None, :]
        )
        valid = all(
            _minimum_interatomic_distance(
                atom_positions[first_index],
                atom_positions[second_index],
            )
            >= config.min_interatomic_distance_A
            for first_index in range(config.molecule_count)
            for second_index in range(first_index + 1, config.molecule_count)
        )
        if valid:
            return centers, rotations
    raise RuntimeError(
        "Could not sample a valid configuration within "
        f"{config.max_attempts_per_sample} attempts."
    )


def _sample_with_context(
    rng: np.random.Generator,
    config: BenzeneClusterGenerationConfig,
    context: _OplsContext,
    configuration_id: str,
    random_seed: int | None,
) -> BenzeneClusterSample:
    centers, rotations = _sample_poses(
        rng,
        config,
        context.reference_coordinates,
    )
    result = context.evaluate(
        centers,
        rotations,
        configuration_id,
        random_seed,
    )
    sample = BenzeneClusterSample(
        centers,
        rotations,
        np.asarray(result.molecular_forces_kcal_mol_A, dtype=np.float64),
        np.asarray(result.molecular_torques_kcal_mol, dtype=np.float64),
    )
    target_arrays = (sample.forces, sample.moments)
    if not all(np.isfinite(values).all() for values in target_arrays):
        raise RuntimeError("OPLS target health check failed: nonfinite target values.")

    force_norm_max = float(np.linalg.norm(sample.forces, axis=1).max())
    moment_norm_max = float(np.linalg.norm(sample.moments, axis=1).max())
    if (
        config.max_force_norm_kcal_mol_A is not None
        and force_norm_max > config.max_force_norm_kcal_mol_A
    ):
        raise RuntimeError(
            "OPLS target health check failed: force norm "
            f"{force_norm_max:.16g} exceeds "
            f"{config.max_force_norm_kcal_mol_A:.16g} kcal per mol per angstrom."
        )
    if (
        config.max_moment_norm_kcal_mol is not None
        and moment_norm_max > config.max_moment_norm_kcal_mol
    ):
        raise RuntimeError(
            "OPLS target health check failed: moment norm "
            f"{moment_norm_max:.16g} exceeds "
            f"{config.max_moment_norm_kcal_mol:.16g} kcal per mol."
        )
    return sample


def sample_benzene_cluster(
    rng: np.random.Generator,
    config: BenzeneClusterGenerationConfig,
    *,
    configuration_id: str = "tfenn_benzene_sample",
) -> BenzeneClusterSample:
    return _sample_with_context(
        rng,
        config,
        _OplsContext(),
        configuration_id,
        config.seed,
    )


def sample_benzene_pair(
    rng: np.random.Generator,
    config: BenzeneClusterGenerationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if config.molecule_count != 2:
        raise ValueError("sample_benzene_pair requires molecule_count equal to two.")
    sample = sample_benzene_cluster(rng, config)
    return (
        sample.rotations[1],
        sample.centers[1],
        sample.forces[0],
        sample.moments[0],
    )


_WORKER_CONFIG: BenzeneClusterGenerationConfig | None = None
_WORKER_CONTEXT: _OplsContext | None = None


def _initialize_worker(config: BenzeneClusterGenerationConfig) -> None:
    global _WORKER_CONFIG, _WORKER_CONTEXT
    _WORKER_CONFIG = config
    _WORKER_CONTEXT = _OplsContext()


def _seed_for_sample(seed: int, sample_index: int) -> int:
    state = np.random.SeedSequence((seed, sample_index)).generate_state(1)
    return int(state[0])


def _sample_worker(sample_index: int) -> BenzeneClusterSample:
    if _WORKER_CONFIG is None or _WORKER_CONTEXT is None:
        raise RuntimeError("The generation worker was not initialized.")
    sample_seed = _seed_for_sample(_WORKER_CONFIG.seed, sample_index)
    return _sample_with_context(
        np.random.default_rng(sample_seed),
        _WORKER_CONFIG,
        _WORKER_CONTEXT,
        f"benzene_{_WORKER_CONFIG.molecule_count}_{sample_index:08d}",
        sample_seed,
    )


def _iter_samples(
    config: BenzeneClusterGenerationConfig,
    workers: int,
) -> Iterable[BenzeneClusterSample]:
    if workers == 1:
        context = _OplsContext()
        for sample_index in range(config.sample_count):
            sample_seed = _seed_for_sample(config.seed, sample_index)
            yield _sample_with_context(
                np.random.default_rng(sample_seed),
                config,
                context,
                f"benzene_{config.molecule_count}_{sample_index:08d}",
                sample_seed,
            )
        return

    chunksize = max(1, config.sample_count // (workers * 8))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(config,),
    ) as executor:
        yield from executor.map(
            _sample_worker,
            range(config.sample_count),
            chunksize=chunksize,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tfenn_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def run_git(*arguments: str) -> str | None:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run_git("status", "--porcelain")
    return {
        "version": "0.1.0",
        "commit": run_git("rev-parse", "HEAD"),
        "source_tree_dirty": bool(status) if status is not None else None,
    }


def _opls_provenance(context: _OplsContext) -> dict[str, Any]:
    import opls2020
    from opls2020.io import SCHEMA_VERSION, code_provenance

    package = distribution("opls2020-static")
    direct_url_text = package.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text is not None else None
    if direct_url is not None and not isinstance(direct_url, dict):
        raise RuntimeError("OPLS direct_url metadata must contain an object.")
    runtime_version = opls2020.__version__
    distribution_version = package.version
    if runtime_version != distribution_version:
        raise RuntimeError(
            "OPLS runtime and distribution versions differ: "
            f"{runtime_version} != {distribution_version}."
        )
    editable = bool(
        isinstance(direct_url, dict)
        and isinstance(direct_url.get("dir_info"), dict)
        and direct_url["dir_info"].get("editable") is True
    )
    package_code = code_provenance()
    resolved_code = dict(package_code)
    provenance_source = "opls2020.io.code_provenance"
    if editable and resolved_code.get("git_commit") is None:
        source_url = direct_url.get("url") if isinstance(direct_url, dict) else None
        if isinstance(source_url, str):
            parsed = urlparse(source_url)
            if parsed.scheme == "file":
                source_text = unquote(parsed.path)
                if sys.platform == "win32" and source_text.startswith("/"):
                    source_text = source_text[1:]
                source_path = Path(source_text).resolve()

                def run_git(*arguments: str) -> str | None:
                    result = subprocess.run(
                        (
                            "git",
                            "-c",
                            f"safe.directory={source_path}",
                            "-C",
                            str(source_path),
                            *arguments,
                        ),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    return result.stdout.strip() if result.returncode == 0 else None

                commit = run_git("rev-parse", "HEAD")
                status = run_git("status", "--porcelain")
                if commit is not None:
                    resolved_code["git_commit"] = commit
                    resolved_code["git_dirty"] = (
                        bool(status) if status is not None else None
                    )
                    provenance_source = "opls2020.io_with_editable_git_fallback"
    parameter_sets = context.parameters.sets_for(context.species.parameter_ids)
    return {
        "runtime_version": runtime_version,
        "distribution_version": distribution_version,
        "result_schema_version": SCHEMA_VERSION,
        "source_status": (
            "local_dirty_candidate" if editable else "installed_distribution"
        ),
        "direct_url": direct_url,
        "code_provenance": resolved_code,
        "code_provenance_source": provenance_source,
        "model": context.model.as_dict(),
        "species": context.species.as_dict(),
        "parameter_sets": [parameter_set.as_dict() for parameter_set in parameter_sets],
        "use_neighbor_list": False,
    }


def _dataset_metadata(
    config: BenzeneClusterGenerationConfig,
    csv_path: Path,
    workers: int,
) -> dict[str, Any]:
    context = _OplsContext()
    dataset_name = {2: "benzene_pair", 3: "benzene_triple"}.get(
        config.molecule_count,
        "benzene_cluster",
    )
    return {
        "schema_name": "tfenn_rigid_system",
        "schema_version": 2,
        "dataset": dataset_name,
        "dataset_revision": config.dataset_revision,
        "sample_count": config.sample_count,
        "molecule_count": config.molecule_count,
        "rows_per_sample": config.molecule_count,
        "row_count": config.sample_count * config.molecule_count,
        "columns": list(BENZENE_CLUSTER_COLUMNS),
        "sampling": {
            "seed": config.seed,
            "seed_strategy": "numpy_seed_sequence_from_dataset_seed_and_sample_id",
            "distance_range_A": list(config.distance_range_A),
            "min_interatomic_distance_A": config.min_interatomic_distance_A,
            "max_attempts_per_sample": config.max_attempts_per_sample,
            "target_health": {
                "require_finite": True,
                "max_force_norm_kcal_mol_A": config.max_force_norm_kcal_mol_A,
                "max_moment_norm_kcal_mol": config.max_moment_norm_kcal_mol,
            },
        },
        "conventions": {
            "root_molecule_id": 0,
            "root_center_A": [0.0, 0.0, 0.0],
            "root_rotation": "identity",
            "rotation": "active_body_to_root_frame",
            "force": "root_frame_force_on_molecule",
            "moment": "root_frame_torque_about_molecule_center",
            "position_unit": "angstrom",
            "force_unit": "kcal_per_mol_per_angstrom",
            "moment_unit": "kcal_per_mol",
        },
        "generation": {"workers": workers},
        "opls": _opls_provenance(context),
        "tfenn": _tfenn_provenance(),
        "csv_sha256": _sha256(csv_path),
    }


def generate_benzene_cluster_dataset(
    output_csv: str | Path,
    config: BenzeneClusterGenerationConfig,
    *,
    workers: int = 1,
    progress_every: int | None = 1_000,
) -> tuple[Path, Path]:
    _check_windows_openmp_runtime()
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    csv_path = Path(output_csv)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("output_csv must use the .csv suffix.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = csv_path.with_name(f"{csv_path.name}.partial")
    metadata_path = csv_path.with_suffix(".json")

    try:
        with partial_path.open("w", newline="", encoding="utf_8") as stream:
            writer = csv.writer(stream)
            writer.writerow(BENZENE_CLUSTER_COLUMNS)
            for sample_index, sample in enumerate(_iter_samples(config, workers)):
                for molecule_index in range(config.molecule_count):
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
                if progress_every and (sample_index + 1) % progress_every == 0:
                    print(f"{sample_index + 1}/{config.sample_count} samples written")
        partial_path.replace(csv_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    metadata = _dataset_metadata(config, csv_path, workers)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    return csv_path, metadata_path


def generate_benzene_pair_dataset(
    output_csv: str | Path,
    config: BenzeneClusterGenerationConfig,
    **kwargs: Any,
) -> tuple[Path, Path]:
    if config.molecule_count != 2:
        raise ValueError("Pair generation requires molecule_count equal to two.")
    return generate_benzene_cluster_dataset(output_csv, config, **kwargs)


def generate_benzene_triple_dataset(
    output_csv: str | Path,
    config: BenzeneClusterGenerationConfig,
    **kwargs: Any,
) -> tuple[Path, Path]:
    if config.molecule_count != 3:
        raise ValueError("Triple generation requires molecule_count equal to three.")
    return generate_benzene_cluster_dataset(output_csv, config, **kwargs)


def compare_legacy_pair_rows(
    legacy_csv: str | Path,
    output_json: str | Path,
    *,
    row_count: int = 3,
) -> Path:
    if row_count < 1:
        raise ValueError("row_count must be positive.")
    legacy_path = Path(legacy_csv)
    arrays = load_benzene_pair_csv(legacy_path)
    if row_count > len(arrays):
        raise ValueError("row_count exceeds the legacy dataset size.")

    context = _OplsContext()
    rows: list[dict[str, Any]] = []
    new_data_rows: list[tuple[Any, ...]] = []
    for row_index in range(row_count):
        centers = np.stack((np.zeros(3), arrays.displacement[row_index]))
        rotations = np.stack((np.eye(3), arrays.relative_rotation[row_index]))
        result = context.evaluate(
            centers,
            rotations,
            f"legacy_pose_{row_index:08d}",
            None,
        )
        old_force = arrays.force[row_index]
        old_moment = arrays.moment[row_index]
        new_force = np.asarray(result.molecular_forces_kcal_mol_A[0])
        new_moment = np.asarray(result.molecular_torques_kcal_mol[0])
        all_forces = np.asarray(result.molecular_forces_kcal_mol_A)
        all_moments = np.asarray(result.molecular_torques_kcal_mol)
        for molecule_index in range(2):
            new_data_rows.append(
                (
                    row_index,
                    molecule_index,
                    *centers[molecule_index],
                    *rotations[molecule_index].reshape(9),
                    *all_forces[molecule_index],
                    *all_moments[molecule_index],
                )
            )
        force_delta = new_force - old_force
        moment_delta = new_moment - old_moment
        old_force_norm = float(np.linalg.norm(old_force))
        old_moment_norm = float(np.linalg.norm(old_moment))
        force_delta_norm = float(np.linalg.norm(force_delta))
        moment_delta_norm = float(np.linalg.norm(moment_delta))
        rows.append(
            {
                "row_index": row_index,
                "relative_center_A": centers[1].tolist(),
                "relative_rotation": rotations[1].tolist(),
                "old_force_kcal_mol_A": old_force.tolist(),
                "new_force_kcal_mol_A": new_force.tolist(),
                "force_delta_kcal_mol_A": force_delta.tolist(),
                "old_force_norm": old_force_norm,
                "new_force_norm": float(np.linalg.norm(new_force)),
                "force_delta_norm": force_delta_norm,
                "force_relative_delta": (
                    force_delta_norm / old_force_norm if old_force_norm else None
                ),
                "old_moment_kcal_mol": old_moment.tolist(),
                "new_moment_kcal_mol": new_moment.tolist(),
                "moment_delta_kcal_mol": moment_delta.tolist(),
                "old_moment_norm": old_moment_norm,
                "new_moment_norm": float(np.linalg.norm(new_moment)),
                "moment_delta_norm": moment_delta_norm,
                "moment_relative_delta": (
                    moment_delta_norm / old_moment_norm if old_moment_norm else None
                ),
                "new_total_energy_kcal_mol": float(result.total_energy_kcal_mol),
            }
        )

    def statistics(key: str) -> dict[str, float | None]:
        values = np.asarray([row[key] for row in rows if row[key] is not None])
        if values.size == 0:
            return {"minimum": None, "mean": None, "maximum": None}
        return {
            "minimum": float(values.min()),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
        }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_data_path = output_path.with_name(f"{output_path.stem}_new_data.csv")
    with new_data_path.open("w", newline="", encoding="utf_8") as stream:
        writer = csv.writer(stream)
        writer.writerow(BENZENE_CLUSTER_COLUMNS)
        writer.writerows(new_data_rows)

    new_opls = _opls_provenance(context)
    comparison = {
        "schema_name": "tfenn_opls_same_pose_comparison",
        "schema_version": 1,
        "legacy_csv": str(legacy_path.as_posix()),
        "legacy_csv_sha256": _sha256(legacy_path),
        "new_data_csv": str(new_data_path.as_posix()),
        "new_data_csv_sha256": _sha256(new_data_path),
        "legacy_opls": {
            "package_version": "0.2.0",
            "version_provenance": "inferred_from_generator",
            "model": "legacy_open_shifted_force_linear_12_A",
            "geometry_profile": "legacy_project_v1",
        },
        "new_opls": new_opls,
        "comparison_note": (
            "Relative centers and rotations are identical. "
            f"The OPLS {new_opls['runtime_version']} default benzene geometry "
            "and model semantics are used."
        ),
        "summary": {
            "force_delta_norm": statistics("force_delta_norm"),
            "force_relative_delta": statistics("force_relative_delta"),
            "moment_delta_norm": statistics("moment_delta_norm"),
            "moment_relative_delta": statistics("moment_relative_delta"),
        },
        "rows": rows,
    }
    output_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("output_csv", type=Path)
    generate.add_argument("molecule_count", type=int, choices=(2, 3))
    generate.add_argument("--sample_count", type=int, default=10_000)
    generate.add_argument("--dataset_revision", type=int, default=1)
    generate.add_argument("--seed", type=int, default=20260810)
    generate.add_argument("--distance_min_A", type=float, default=5.0)
    generate.add_argument("--distance_max_A", type=float, default=10.0)
    generate.add_argument("--min_interatomic_distance_A", type=float, default=3.0)
    generate.add_argument("--max_force_norm_kcal_mol_A", type=float)
    generate.add_argument("--max_moment_norm_kcal_mol", type=float)
    generate.add_argument("--max_attempts_per_sample", type=int, default=500)
    generate.add_argument("--workers", type=int, default=1)

    compare = subparsers.add_parser("compare")
    compare.add_argument("legacy_csv", type=Path)
    compare.add_argument("output_json", type=Path)
    compare.add_argument("--row_count", type=int, default=3)
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    if arguments.command == "compare":
        output_path = compare_legacy_pair_rows(
            arguments.legacy_csv,
            arguments.output_json,
            row_count=arguments.row_count,
        )
        print(output_path)
        return

    config = BenzeneClusterGenerationConfig(
        sample_count=arguments.sample_count,
        dataset_revision=arguments.dataset_revision,
        molecule_count=arguments.molecule_count,
        seed=arguments.seed,
        distance_range_A=(arguments.distance_min_A, arguments.distance_max_A),
        min_interatomic_distance_A=arguments.min_interatomic_distance_A,
        max_force_norm_kcal_mol_A=arguments.max_force_norm_kcal_mol_A,
        max_moment_norm_kcal_mol=arguments.max_moment_norm_kcal_mol,
        max_attempts_per_sample=arguments.max_attempts_per_sample,
    )
    csv_path, metadata_path = generate_benzene_cluster_dataset(
        arguments.output_csv,
        config,
        workers=arguments.workers,
    )
    print(csv_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
