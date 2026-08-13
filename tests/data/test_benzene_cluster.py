from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from TFENN.data import (
    BENZENE_CLUSTER_COLUMNS,
    BenzeneClusterGenerationConfig,
    BenzenePairGenerationConfig,
    BenzeneTripleGenerationConfig,
    load_benzene_cluster_csv,
    load_benzene_cluster_metadata,
)


OPLS_BASE_COMMIT = "319521f5551782c7f9016a51f84225578e583068"
OPLS_MODEL = "opls2020_open_direct_quintic_10_12_codata2022_v1"


ISOLATED_SCRIPT = r"""
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

from TFENN.data import (
    BenzeneClusterGenerationConfig,
    BenzenePairGenerationConfig,
    BenzeneTripleGenerationConfig,
    generate_benzene_cluster_dataset,
    sample_benzene_cluster,
)


def config_for(payload):
    values = {
        "sample_count": payload.get("sample_count", 1),
        "dataset_revision": payload.get("dataset_revision", 1),
        "seed": payload["seed"],
        "distance_range_A": tuple(payload.get("distance_range_A", (5.5, 7.0))),
        "min_interatomic_distance_A": payload.get(
            "min_interatomic_distance_A",
            2.5,
        ),
        "max_attempts_per_sample": payload.get("max_attempts_per_sample", 100),
    }
    for name in (
        "max_force_norm_kcal_mol_A",
        "max_moment_norm_kcal_mol",
    ):
        if name in payload:
            values[name] = payload[name]
    molecule_count = payload["molecule_count"]
    if payload.get("named_config") and molecule_count == 2:
        return BenzenePairGenerationConfig(**values)
    if payload.get("named_config") and molecule_count == 3:
        return BenzeneTripleGenerationConfig(**values)
    return BenzeneClusterGenerationConfig(
        molecule_count=molecule_count,
        **values,
    )


operation = sys.argv[1]
payload = json.loads(sys.argv[2])

if operation == "sample":
    from opls2020 import benzene

    config = config_for(payload)
    sample = sample_benzene_cluster(np.random.default_rng(config.seed), config)
    reference = np.asarray(benzene().reference_coordinates_A)
    atom_positions = np.einsum(
        "aj,mij->mai",
        reference,
        sample.rotations,
    ) + sample.centers[:, None, :]
    minimum_separation = min(
        float(
            np.linalg.norm(
                atom_positions[first, :, None, :]
                - atom_positions[second, None, :, :],
                axis=2,
            ).min()
        )
        for first, second in combinations(range(config.molecule_count), 2)
    )
    total_moment = (
        sample.moments + np.cross(sample.centers, sample.forces)
    ).sum(axis=0)
    report = {
        "centers_shape": list(sample.centers.shape),
        "rotations_shape": list(sample.rotations.shape),
        "forces_shape": list(sample.forces.shape),
        "moments_shape": list(sample.moments.shape),
        "finite": all(
            bool(np.isfinite(values).all())
            for values in (
                sample.centers,
                sample.rotations,
                sample.forces,
                sample.moments,
            )
        ),
        "root_center_max_abs": float(np.abs(sample.centers[0]).max()),
        "root_rotation_max_abs_error": float(
            np.abs(sample.rotations[0] - np.eye(3)).max()
        ),
        "rotation_orthogonality_max_abs_error": float(
            np.abs(
                sample.rotations @ np.swapaxes(sample.rotations, 1, 2)
                - np.eye(3)
            ).max()
        ),
        "rotation_determinant_max_abs_error": float(
            np.abs(np.linalg.det(sample.rotations) - 1.0).max()
        ),
        "minimum_interatomic_distance_A": minimum_separation,
        "force_sum_max_abs": float(np.abs(sample.forces.sum(axis=0)).max()),
        "total_moment_max_abs": float(np.abs(total_moment).max()),
    }
elif operation == "recompute":
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

    config = config_for(payload)
    sample = sample_benzene_cluster(np.random.default_rng(config.seed), config)
    species = benzene()
    molecules = tuple(
        MoleculeInstance(
            f"benzene_{index}",
            species.species_id,
            Pose.from_matrix(sample.centers[index], sample.rotations[index]),
        )
        for index in range(config.molecule_count)
    )
    system = SystemSpec(
        configuration_id="test_complete_cluster",
        species={species.species_id: species},
        molecules=molecules,
        random_seed=config.seed,
    )
    result = StaticEngine(
        model=DEFAULT_MODEL,
        parameters=DEFAULT_PARAMETER_CATALOG,
        use_neighbor_list=False,
    ).evaluate(system)
    report = {
        "package_version": opls2020.__version__,
        "model_semantics_id": DEFAULT_MODEL.model_semantics_id,
        "force_max_abs_error": float(
            np.abs(sample.forces - result.molecular_forces_kcal_mol_A).max()
        ),
        "moment_max_abs_error": float(
            np.abs(sample.moments - result.molecular_torques_kcal_mol).max()
        ),
    }
elif operation == "generate":
    config = config_for(payload)
    csv_path, metadata_path = generate_benzene_cluster_dataset(
        Path(payload["output"]),
        config,
        workers=1,
        progress_every=None,
    )
    report = {
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
    }
elif operation == "impossible":
    config = config_for(payload)
    try:
        sample_benzene_cluster(np.random.default_rng(config.seed), config)
    except RuntimeError as error:
        report = {"raised": True, "message": str(error)}
    else:
        report = {"raised": False, "message": ""}
else:
    raise ValueError(f"unknown isolated operation: {operation}")

print(json.dumps(report))
"""


def cluster_config(
    molecule_count: int,
    *,
    sample_count: int = 1,
    seed: int = 20260810,
) -> BenzeneClusterGenerationConfig:
    return BenzeneClusterGenerationConfig(
        sample_count=sample_count,
        molecule_count=molecule_count,
        seed=seed,
        distance_range_A=(5.5, 7.0),
        min_interatomic_distance_A=2.5,
        max_attempts_per_sample=100,
    )


def csv_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_isolated(operation: str, payload: dict[str, object]) -> dict[str, object]:
    environment = os.environ.copy()
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    source = str(Path(__file__).resolve().parents[2] / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not existing else source + os.pathsep + existing
    )
    completed = subprocess.run(
        (sys.executable, "-c", ISOLATED_SCRIPT, operation, json.dumps(payload)),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if completed.returncode:
        raise AssertionError(
            "isolated calculation failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError("isolated calculation produced no result")
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise AssertionError("isolated calculation result must be an object")
    return result


def test_cluster_config_supports_generic_and_named_counts() -> None:
    assert len(BENZENE_CLUSTER_COLUMNS) == 20
    assert BenzenePairGenerationConfig().molecule_count == 2
    assert BenzenePairGenerationConfig().distance_range_A == (5.0, 10.0)
    assert BenzenePairGenerationConfig().max_force_norm_kcal_mol_A is None
    assert BenzenePairGenerationConfig().max_moment_norm_kcal_mol is None
    assert BenzeneTripleGenerationConfig().molecule_count == 3
    assert BenzeneClusterGenerationConfig(molecule_count=4).molecule_count == 4
    with pytest.raises(ValueError, match="at least two"):
        BenzeneClusterGenerationConfig(molecule_count=1)


@pytest.mark.parametrize("dataset_revision", (True, 0, -1, 1.5))
def test_cluster_config_requires_positive_integer_dataset_revision(
    dataset_revision: object,
) -> None:
    with pytest.raises(ValueError, match="dataset_revision"):
        BenzeneClusterGenerationConfig(dataset_revision=dataset_revision)

    assert BenzeneClusterGenerationConfig(dataset_revision=7).dataset_revision == 7


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("max_force_norm_kcal_mol_A", 0.0),
        ("max_force_norm_kcal_mol_A", float("inf")),
        ("max_moment_norm_kcal_mol", -1.0),
        ("max_moment_norm_kcal_mol", float("nan")),
    ),
)
def test_cluster_config_requires_positive_finite_target_limits(
    name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=name):
        BenzeneClusterGenerationConfig(**{name: value})


@pytest.mark.parametrize("molecule_count", (2, 3))
def test_sample_shapes_rotations_separation_and_conservation(
    molecule_count: int,
) -> None:
    config = cluster_config(molecule_count, seed=17)
    report = run_isolated(
        "sample",
        {
            "molecule_count": molecule_count,
            "seed": config.seed,
            "distance_range_A": config.distance_range_A,
            "min_interatomic_distance_A": config.min_interatomic_distance_A,
            "max_attempts_per_sample": config.max_attempts_per_sample,
        },
    )
    assert report["centers_shape"] == [molecule_count, 3]
    assert report["rotations_shape"] == [molecule_count, 3, 3]
    assert report["forces_shape"] == [molecule_count, 3]
    assert report["moments_shape"] == [molecule_count, 3]
    assert report["finite"] is True
    assert report["root_center_max_abs"] == 0.0
    assert report["root_rotation_max_abs_error"] == 0.0
    assert report["rotation_orthogonality_max_abs_error"] <= 1.0e-12
    assert report["rotation_determinant_max_abs_error"] <= 1.0e-12
    assert report["minimum_interatomic_distance_A"] >= config.min_interatomic_distance_A
    assert report["force_sum_max_abs"] <= 1.0e-12
    assert report["total_moment_max_abs"] <= 1.0e-12


@pytest.mark.parametrize("molecule_count", (2, 3))
def test_sample_matches_complete_default_opls_evaluation(
    molecule_count: int,
) -> None:
    config = cluster_config(molecule_count, seed=29)
    report = run_isolated(
        "recompute",
        {
            "molecule_count": molecule_count,
            "seed": config.seed,
            "distance_range_A": config.distance_range_A,
            "min_interatomic_distance_A": config.min_interatomic_distance_A,
            "max_attempts_per_sample": config.max_attempts_per_sample,
        },
    )
    assert report["package_version"] == "2.0.0"
    assert report["model_semantics_id"] == OPLS_MODEL
    assert report["force_max_abs_error"] <= 1.0e-12
    assert report["moment_max_abs_error"] <= 1.0e-12


@pytest.mark.parametrize(
    ("config_type", "molecule_count", "dataset_name"),
    (
        (BenzenePairGenerationConfig, 2, "benzene_pair"),
        (BenzeneTripleGenerationConfig, 3, "benzene_triple"),
    ),
)
def test_generate_and_load_cluster_round_trip(
    tmp_path: Path,
    config_type: type[BenzeneClusterGenerationConfig],
    molecule_count: int,
    dataset_name: str,
) -> None:
    config = config_type(
        sample_count=2,
        dataset_revision=10 + molecule_count,
        seed=41 + molecule_count,
        distance_range_A=(5.5, 7.0),
        min_interatomic_distance_A=2.5,
        max_attempts_per_sample=100,
    )
    output = tmp_path / f"{dataset_name}.csv"
    generated = run_isolated(
        "generate",
        {
            "output": str(output),
            "molecule_count": molecule_count,
            "sample_count": config.sample_count,
            "dataset_revision": config.dataset_revision,
            "seed": config.seed,
            "distance_range_A": config.distance_range_A,
            "min_interatomic_distance_A": config.min_interatomic_distance_A,
            "max_attempts_per_sample": config.max_attempts_per_sample,
            "named_config": True,
        },
    )
    csv_path = Path(str(generated["csv_path"]))
    metadata_path = Path(str(generated["metadata_path"]))
    arrays = load_benzene_cluster_csv(csv_path)
    metadata = load_benzene_cluster_metadata(metadata_path)

    assert len(arrays) == config.sample_count
    assert arrays.molecule_count == molecule_count
    assert arrays.centers.shape == (2, molecule_count, 3)
    assert arrays.rotations.shape == (2, molecule_count, 3, 3)
    assert arrays.forces.shape == (2, molecule_count, 3)
    assert arrays.moments.shape == (2, molecule_count, 3)

    with csv_path.open("r", newline="", encoding="utf_8") as stream:
        rows = list(csv.reader(stream))
    assert tuple(rows[0]) == BENZENE_CLUSTER_COLUMNS
    assert len(rows[1:]) == config.sample_count * molecule_count
    identifiers = [tuple(map(int, row[:2])) for row in rows[1:]]
    assert identifiers == [
        (sample_id, molecule_id)
        for sample_id in range(config.sample_count)
        for molecule_id in range(molecule_count)
    ]

    assert metadata["schema_name"] == "tfenn_rigid_system"
    assert metadata["schema_version"] == 2
    assert metadata["dataset"] == dataset_name
    assert metadata["dataset_revision"] == config.dataset_revision
    assert metadata["sample_count"] == config.sample_count
    assert metadata["molecule_count"] == molecule_count
    assert metadata["rows_per_sample"] == molecule_count
    assert metadata["row_count"] == config.sample_count * molecule_count
    assert metadata["columns"] == list(BENZENE_CLUSTER_COLUMNS)
    assert metadata["sampling"]["seed"] == config.seed
    assert metadata["sampling"]["target_health"] == {
        "require_finite": True,
        "max_force_norm_kcal_mol_A": config.max_force_norm_kcal_mol_A,
        "max_moment_norm_kcal_mol": config.max_moment_norm_kcal_mol,
    }
    assert metadata["generation"]["workers"] == 1
    opls = metadata["opls"]
    assert opls["runtime_version"] == "2.0.0"
    assert opls["distribution_version"] == "2.0.0"
    assert opls["result_schema_version"] == "2.0.0"
    assert opls["source_status"] == "local_dirty_candidate"
    assert opls["direct_url"]["dir_info"]["editable"] is True
    assert opls["direct_url"]["url"].startswith("file:///")
    assert set(opls["code_provenance"]) == {
        "git_commit",
        "git_dirty",
        "source_tree_sha256",
    }
    assert opls["code_provenance"]["git_commit"] == OPLS_BASE_COMMIT
    assert opls["code_provenance"]["git_dirty"] is True
    assert opls["code_provenance_source"] == "opls2020.io_with_editable_git_fallback"
    source_tree_sha256 = opls["code_provenance"]["source_tree_sha256"]
    assert len(source_tree_sha256) == 64
    assert all(character in "0123456789abcdef" for character in source_tree_sha256)
    assert "release_tag" not in opls
    assert "release_commit" not in opls
    assert "source_archive_sha256" not in opls
    assert opls["model"]["model_semantics_id"] == OPLS_MODEL
    assert opls["species"]["geometry_profile_id"] == "opls2020_rigid"
    assert opls["parameter_sets"][0]["version"] == "S2_2023_10_01"
    assert metadata["csv_sha256"] == csv_sha256(csv_path)


def test_loader_rejects_incomplete_molecule_group(
    tmp_path: Path,
) -> None:
    config = BenzeneTripleGenerationConfig(
        sample_count=2,
        seed=53,
        distance_range_A=(5.5, 7.0),
        min_interatomic_distance_A=2.5,
        max_attempts_per_sample=100,
    )
    output = tmp_path / "triple.csv"
    generated = run_isolated(
        "generate",
        {
            "output": str(output),
            "molecule_count": config.molecule_count,
            "sample_count": config.sample_count,
            "seed": config.seed,
            "distance_range_A": config.distance_range_A,
            "min_interatomic_distance_A": config.min_interatomic_distance_A,
            "max_attempts_per_sample": config.max_attempts_per_sample,
            "named_config": True,
        },
    )
    csv_path = Path(str(generated["csv_path"]))
    lines = csv_path.read_text(encoding="utf_8").splitlines()
    csv_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf_8")
    with pytest.raises(ValueError, match="Invalid molecule grouping"):
        load_benzene_cluster_csv(csv_path)


def test_sample_rejects_impossible_separation() -> None:
    config = BenzeneClusterGenerationConfig(
        sample_count=1,
        molecule_count=2,
        seed=67,
        distance_range_A=(5.0, 5.1),
        min_interatomic_distance_A=100.0,
        max_attempts_per_sample=1,
    )
    report = run_isolated(
        "impossible",
        {
            "molecule_count": config.molecule_count,
            "seed": config.seed,
            "distance_range_A": config.distance_range_A,
            "min_interatomic_distance_A": config.min_interatomic_distance_A,
            "max_attempts_per_sample": config.max_attempts_per_sample,
        },
    )
    assert report["raised"] is True
    assert "Could not sample" in str(report["message"])


def test_pair_sample_rejects_target_above_health_limit() -> None:
    report = run_isolated(
        "impossible",
        {
            "molecule_count": 2,
            "named_config": True,
            "seed": 71,
            "distance_range_A": (6.0, 6.1),
            "min_interatomic_distance_A": 0.0,
            "max_force_norm_kcal_mol_A": 1.0e-12,
            "max_moment_norm_kcal_mol": 4.0,
            "max_attempts_per_sample": 10,
        },
    )
    assert report["raised"] is True
    assert "target health check failed: force norm" in str(report["message"])
