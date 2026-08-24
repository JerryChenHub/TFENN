from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROTATION_COLUMNS = tuple(f"R{i}{j}" for i in range(1, 4) for j in range(1, 4))
DISPLACEMENT_COLUMNS = ("x1", "x2", "x3")
FORCE_COLUMNS = ("F1", "F2", "F3")
MOMENT_COLUMNS = ("M1", "M2", "M3")
BENZENE_PAIR_COLUMNS = (
    ROTATION_COLUMNS
    + DISPLACEMENT_COLUMNS
    + FORCE_COLUMNS
    + MOMENT_COLUMNS
)


@dataclass(frozen=True, slots=True)
class BenzenePairArrays:
    relative_rotation: np.ndarray
    displacement: np.ndarray
    target: np.ndarray

    def __len__(self) -> int:
        return self.displacement.shape[0]

    @property
    def force(self) -> np.ndarray:
        return self.target[:, 0]

    @property
    def moment(self) -> np.ndarray:
        return self.target[:, 1]


def load_benzene_pair_csv(
    path: str | Path,
    *,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
    validate_finite: bool = True,
) -> BenzenePairArrays:
    """Load a benzene pair table with the canonical eighteen columns."""
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf_8_sig") as stream:
        header = tuple(next(csv.reader(stream), ()))

    if header != BENZENE_PAIR_COLUMNS:
        missing = [name for name in BENZENE_PAIR_COLUMNS if name not in header]
        unexpected = [name for name in header if name not in BENZENE_PAIR_COLUMNS]
        raise ValueError(
            f"Unexpected columns in {csv_path}. "
            f"Expected {list(BENZENE_PAIR_COLUMNS)}, got {list(header)}. "
            f"Missing {missing}, unexpected {unexpected}."
        )

    values = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=dtype, ndmin=2)
    if values.shape[1] != len(BENZENE_PAIR_COLUMNS):
        raise ValueError(
            f"Expected {len(BENZENE_PAIR_COLUMNS)} values per row in {csv_path}, "
            f"got {values.shape[1]}."
        )
    if validate_finite and not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values found in {csv_path}.")

    relative_rotation = np.ascontiguousarray(values[:, :9].reshape(-1, 3, 3))
    displacement = np.ascontiguousarray(values[:, 9:12])
    target = np.ascontiguousarray(values[:, 12:18].reshape(-1, 2, 3))
    return BenzenePairArrays(relative_rotation, displacement, target)


def load_benzene_pair_metadata(path: str | Path) -> dict[str, Any]:
    """Load the JSON sidecar for a CSV path or a direct JSON path."""
    metadata_path = Path(path)
    if metadata_path.suffix.lower() != ".json":
        metadata_path = metadata_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf_8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")
    return metadata
