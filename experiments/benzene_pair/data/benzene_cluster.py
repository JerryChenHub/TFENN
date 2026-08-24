from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CENTER_COLUMNS = ("cx", "cy", "cz")
CLUSTER_ROTATION_COLUMNS = tuple(
    f"R{i}{j}" for i in range(1, 4) for j in range(1, 4)
)
CLUSTER_FORCE_COLUMNS = ("Fx", "Fy", "Fz")
CLUSTER_MOMENT_COLUMNS = ("Mx", "My", "Mz")
BENZENE_CLUSTER_COLUMNS = (
    ("sample_id", "molecule_id")
    + CENTER_COLUMNS
    + CLUSTER_ROTATION_COLUMNS
    + CLUSTER_FORCE_COLUMNS
    + CLUSTER_MOMENT_COLUMNS
)


@dataclass(frozen=True, slots=True)
class BenzeneClusterArrays:
    centers: np.ndarray
    rotations: np.ndarray
    forces: np.ndarray
    moments: np.ndarray

    def __len__(self) -> int:
        return self.centers.shape[0]

    @property
    def molecule_count(self) -> int:
        return self.centers.shape[1]

    @property
    def target(self) -> np.ndarray:
        return np.stack((self.forces, self.moments), axis=2)


def load_benzene_cluster_csv(
    path: str | Path,
    *,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
    validate_finite: bool = True,
) -> BenzeneClusterArrays:
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf_8_sig") as stream:
        header = tuple(next(csv.reader(stream), ()))

    if header != BENZENE_CLUSTER_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {csv_path}. "
            f"Expected {list(BENZENE_CLUSTER_COLUMNS)}, got {list(header)}."
        )

    values = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
        ndmin=2,
    )
    if values.shape[0] == 0 or values.shape[1] != len(BENZENE_CLUSTER_COLUMNS):
        raise ValueError(f"No complete samples were found in {csv_path}.")
    if validate_finite and not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values found in {csv_path}.")

    identifiers = values[:, :2]
    if not np.equal(identifiers, np.floor(identifiers)).all():
        raise ValueError(f"Sample and molecule identifiers must be integers in {csv_path}.")
    sample_ids = identifiers[:, 0].astype(np.int64)
    molecule_ids = identifiers[:, 1].astype(np.int64)
    if sample_ids[0] != 0 or molecule_ids[0] != 0:
        raise ValueError(f"Samples must begin with sample 0 and molecule 0 in {csv_path}.")

    molecule_count = int(np.count_nonzero(sample_ids == 0))
    if molecule_count < 2 or values.shape[0] % molecule_count:
        raise ValueError(f"Invalid molecule grouping in {csv_path}.")
    sample_count = values.shape[0] // molecule_count
    expected_samples = np.repeat(np.arange(sample_count), molecule_count)
    expected_molecules = np.tile(np.arange(molecule_count), sample_count)
    if not np.array_equal(sample_ids, expected_samples) or not np.array_equal(
        molecule_ids,
        expected_molecules,
    ):
        raise ValueError(f"Rows are not ordered as complete molecule groups in {csv_path}.")

    data = values[:, 2:].astype(dtype, copy=False)
    centers = np.ascontiguousarray(data[:, :3].reshape(sample_count, molecule_count, 3))
    rotations = np.ascontiguousarray(
        data[:, 3:12].reshape(sample_count, molecule_count, 3, 3)
    )
    forces = np.ascontiguousarray(
        data[:, 12:15].reshape(sample_count, molecule_count, 3)
    )
    moments = np.ascontiguousarray(
        data[:, 15:18].reshape(sample_count, molecule_count, 3)
    )
    return BenzeneClusterArrays(centers, rotations, forces, moments)


def load_benzene_cluster_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    if metadata_path.suffix.lower() != ".json":
        metadata_path = metadata_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf_8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")
    return metadata
