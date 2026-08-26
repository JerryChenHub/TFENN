from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.benzene_pair.data.benzene_cluster import (
    load_benzene_cluster_csv,
)


MOLECULE_COUNT = 5
PAIR_INDEX = tuple(itertools.combinations(range(MOLECULE_COUNT), 2))
OPLS_VERSION = "2.0.0"
OPLS_COMMIT = "a5f874ed00152b156cd2525c961bd81030237e31"
DEFAULT_TOLERANCE = 5.0e-10
DEFAULT_CSV = Path(__file__).with_name(
    "five_benzene_opls_2_0_0_1k_v1.csv"
)
DEFAULT_OUTPUT = Path(__file__).with_name(
    "five_benzene_opls_2_0_0_1k_v1_pair_forces.npz"
)


_WORKER_SPECIES: Any = None
_WORKER_ENGINE: Any = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    levels = (0, 1, 5, 25, 50, 75, 95, 99, 100)
    numbers = np.percentile(values, levels)
    return {
        f"p{level}": float(number)
        for level, number in zip(levels, numbers, strict=True)
    }


def _opls_provenance() -> dict[str, Any]:
    import opls2020

    package = importlib.metadata.distribution("opls2020-static")
    direct_text = package.read_text("direct_url.json")
    direct_url = json.loads(direct_text) if direct_text else None
    commit = None
    if isinstance(direct_url, dict):
        vcs_info = direct_url.get("vcs_info")
        if isinstance(vcs_info, dict):
            commit = vcs_info.get("commit_id")
    if opls2020.__version__ != OPLS_VERSION or package.version != OPLS_VERSION:
        raise RuntimeError("OPLS runtime version does not match the dataset")
    if commit != OPLS_COMMIT:
        raise RuntimeError("OPLS source commit does not match the dataset")
    return {
        "runtime_version": opls2020.__version__,
        "distribution_version": package.version,
        "source_commit": commit,
        "direct_url": direct_url,
        "engine": "StaticEngine",
        "use_neighbor_list": False,
        "boundary": "open",
        "electrostatics": "direct",
    }


def _initialize_worker() -> None:
    global _WORKER_SPECIES
    global _WORKER_ENGINE
    from opls2020 import StaticEngine, benzene

    _WORKER_SPECIES = benzene()
    _WORKER_ENGINE = StaticEngine(use_neighbor_list=False)


def _derive_sample(
    task: tuple[int, np.ndarray, np.ndarray],
) -> tuple[int, np.ndarray, float]:
    from opls2020 import MoleculeInstance, Pose, SystemSpec

    sample_index, centers, rotations = task
    if _WORKER_SPECIES is None or _WORKER_ENGINE is None:
        raise RuntimeError("pair force worker is not initialized")
    pair_forces = np.empty((len(PAIR_INDEX), 3), dtype=np.float64)
    reverse_residual = 0.0
    species = _WORKER_SPECIES
    for pair_id, (first, second) in enumerate(PAIR_INDEX):
        molecules = tuple(
            MoleculeInstance(
                f"molecule_{index}",
                species.species_id,
                Pose.from_matrix(centers[index], rotations[index]),
            )
            for index in (first, second)
        )
        system = SystemSpec(
            f"sample_{sample_index:06d}_pair_{first}_{second}",
            {species.species_id: species},
            molecules,
        )
        result = _WORKER_ENGINE.evaluate(system)
        force = np.asarray(
            result.molecular_forces_kcal_mol_A,
            dtype=np.float64,
        )
        if force.shape != (2, 3) or not np.isfinite(force).all():
            raise RuntimeError("StaticEngine returned an invalid pair force")
        pair_forces[pair_id] = force[0]
        reverse_residual = max(
            reverse_residual,
            float(np.max(np.abs(force[0] + force[1]))),
        )
    return sample_index, pair_forces, reverse_residual


def derive_pair_forces(
    csv_path: Path,
    output_path: Path,
    workers: int,
    tolerance: float,
) -> tuple[Path, Path]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    csv_path = csv_path.resolve()
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("output path must have npz suffix")
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("pair force output already exists")
    arrays = load_benzene_cluster_csv(csv_path)
    if arrays.molecule_count != MOLECULE_COUNT:
        raise ValueError("pair force derivation requires five molecules")
    provenance = _opls_provenance()
    sample_count = len(arrays)
    pair_forces = np.empty(
        (sample_count, len(PAIR_INDEX), 3),
        dtype=np.float64,
    )
    reverse_residual = 0.0
    tasks = (
        (sample_index, arrays.centers[sample_index], arrays.rotations[sample_index])
        for sample_index in range(sample_count)
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
    ) as executor:
        results = executor.map(
            _derive_sample,
            tasks,
            chunksize=max(1, sample_count // (workers * 8)),
        )
        for completed, (sample_index, force, residual) in enumerate(results, 1):
            pair_forces[sample_index] = force
            reverse_residual = max(reverse_residual, residual)
            if completed % 100 == 0 or completed == sample_count:
                print(f"{completed}/{sample_count} samples derived", flush=True)
    if not np.isfinite(pair_forces).all():
        raise RuntimeError("derived pair forces contain nonfinite values")
    reconstructed = np.zeros_like(arrays.forces, dtype=np.float64)
    for pair_id, (first, second) in enumerate(PAIR_INDEX):
        reconstructed[:, first] += pair_forces[:, pair_id]
        reconstructed[:, second] -= pair_forces[:, pair_id]
    reconstruction_residual = float(
        np.max(np.abs(reconstructed - arrays.forces))
    )
    if reverse_residual > tolerance:
        raise RuntimeError("pair force reversal residual exceeds tolerance")
    if reconstruction_residual > tolerance:
        raise RuntimeError("pair force reconstruction residual exceeds tolerance")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output_path.with_name(f"{output_path.name}.partial")
    partial_metadata = metadata_path.with_name(f"{metadata_path.name}.partial")
    try:
        with partial_output.open("wb") as stream:
            np.savez_compressed(
                stream,
                sample_id=np.arange(sample_count, dtype=np.int64),
                pair_index=np.asarray(PAIR_INDEX, dtype=np.int64),
                pair_force_kcal_mol_A=pair_forces,
            )
        partial_output.replace(output_path)
        norms = np.linalg.norm(pair_forces, axis=-1)
        metadata = {
            "schema_name": "tfenn_five_benzene_pair_force_supervision",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "csv_path": str(csv_path),
                "csv_sha256": _sha256(csv_path),
                "sample_count": sample_count,
                "molecule_count": arrays.molecule_count,
            },
            "opls_runtime": provenance,
            "pair_contract": {
                "pair_count": len(PAIR_INDEX),
                "pair_index": [list(pair) for pair in PAIR_INDEX],
                "orientation": "first index receives force from second index",
                "reverse_force": "negative of stored force",
                "aggregation": (
                    "add stored force to first index and subtract it from second index"
                ),
            },
            "arrays": {
                "sample_id": {
                    "shape": [sample_count],
                    "dtype": "int64",
                },
                "pair_index": {
                    "shape": [len(PAIR_INDEX), 2],
                    "dtype": "int64",
                },
                "pair_force_kcal_mol_A": {
                    "shape": [sample_count, len(PAIR_INDEX), 3],
                    "dtype": "float64",
                    "unit": "kcal_per_mol_per_angstrom",
                },
            },
            "validation": {
                "passed": True,
                "tolerance": tolerance,
                "maximum_reverse_component_residual": reverse_residual,
                "maximum_reaggregation_component_residual": (
                    reconstruction_residual
                ),
            },
            "statistics": {
                "pair_force_component_rms_kcal_mol_A": float(
                    np.sqrt(np.mean(pair_forces * pair_forces))
                ),
                "pair_force_norm_kcal_mol_A": _percentiles(norms),
            },
            "artifacts": {
                "npz_path": str(output_path),
                "npz_sha256": _sha256(output_path),
                "generator_path": str(Path(__file__).resolve()),
                "generator_sha256": _sha256(Path(__file__).resolve()),
            },
            "workers": workers,
        }
        partial_metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf_8",
        )
        partial_metadata.replace(metadata_path)
    except BaseException:
        partial_output.unlink(missing_ok=True)
        partial_metadata.unlink(missing_ok=True)
        raise
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return output_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    derive_pair_forces(
        arguments.csv,
        arguments.output,
        arguments.workers,
        arguments.tolerance,
    )
